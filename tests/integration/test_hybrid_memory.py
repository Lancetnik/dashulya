from dataclasses import replace
from unittest.mock import AsyncMock

import pytest

from dashulya.database import Database
from dashulya.domain.memory import OriginalMessage
from dashulya.memory.hybrid import HybridMemory, embedding_texts
from dashulya.memory.lexical import LexicalIndex
from dashulya.memory.vectors import EmbeddingCache
from tests.originals import original


@pytest.mark.usefixtures("_setup")
class TestHybrid:
    @pytest.fixture
    def _setup(self, database: Database) -> None:
        self.database = database
        self.queries: list[str] = []

    async def embed(self, text: str) -> list[float]:
        self.queries.append(text)
        return [1.0, 0.0]

    async def engine(
        self, messages: list[OriginalMessage], vectors: list[list[float]] | None = None
    ) -> HybridMemory:
        async def embed(texts: list[str]) -> list[list[float]]:
            return [(await self.embed(text)) + [0.0] * 510 for text in texts]

        cache = EmbeddingCache(self.database, embed)
        if vectors is not None:
            await cache.put({
                cache.key(text): values + [0.0] * 510 if len(values) == 2 else values
                for text, values in zip(embedding_texts(messages), vectors, strict=True)
            })
        return HybridMemory(messages, cache, LexicalIndex(self.database))

    async def test_lexical_recall_without_embeddings(self) -> None:
        messages = [original(1, "durable storage"), original(2, "voice input")]
        engine = await self.engine(messages)
        assert [h.sources[-1].message_id for h in await engine.search("storage")] == [1]
        assert self.queries == []
        assert await engine.coverage() == {
            "source_count": 2,
            "embedded_blocks": 0,
            "pending_blocks": 2,
        }

    async def test_dense_recall_and_rrf_use_native_cached_source_search(self) -> None:
        messages = [
            original(1, "store store store"),
            original(2, "synonym"),
            original(3, "store"),
        ]
        engine = await self.engine(messages, [[0, 1], [0.8, 0.2], [1, 0]])
        assert [h.sources[-1].message_id for h in await engine.search("store")] == [
            3,
            1,
            2,
        ]
        assert self.queries == ["store"]
        assert (await engine.coverage())["embedded_blocks"] == 3

    async def test_reply_parents_and_source_diversity(self) -> None:
        messages = [
            original(1, "first"),
            original(2, "second", parent=1),
            original(3, "third", parent=2),
            original(4, "fourth", parent=3),
        ]
        texts = embedding_texts(messages)
        assert "first" in texts[2]
        assert "first" not in texts[3]
        engine = await self.engine(messages, [[0, 1], [0, 1], [1, 0], [0, 1]])
        hits = await engine.search("unmatched")
        assert [m.message_id for m in hits[0].sources] == [1, 2, 3]
        assert [h.sources[-1].message_id for h in hits] == [3, 4]

    async def test_author_filter_excludes_parent_from_primary_evidence(self) -> None:
        messages = [
            original(1, "parent_keyword", author=10),
            original(2, "acknowledged", author=20, parent=1),
        ]
        engine = await self.engine(messages, [[0, 1], [1, 0]])
        hits = await engine.search("parent_keyword", author_id=20)
        assert [m.message_id for m in hits[0].sources] == [2]
        assert "parent_keyword" not in hits[0].text

    async def test_stale_parent_or_revision_vectors_are_not_used(self) -> None:
        old = original(1, "old secret")
        edited = replace(
            old, text="corrected tool", edited_at="2026-01-02T00:00:00+00:00"
        )
        cache = EmbeddingCache(self.database, AsyncMock())
        await cache.put({cache.key(embedding_texts([old])[0]): [1.0] + [0.0] * 511})
        engine = HybridMemory([edited], cache, LexicalIndex(self.database))
        assert (await engine.coverage())["pending_blocks"] == 1
        assert await engine.search("unmatched") == []
        assert (await engine.search("corrected"))[0].sources == (edited,)
        child = original(2, "acknowledged", parent=1)
        await cache.put({
            cache.key(embedding_texts([old, child])[1]): [1.0] + [0.0] * 511
        })
        child_engine = HybridMemory([child], cache, LexicalIndex(self.database))
        assert await child_engine.search("old secret") == []

    async def test_cross_chat_ids_and_future_parents_stay_distinct(self) -> None:
        foreign = original(1, "foreign_parent_secret", chat=-100999)
        child = original(2, "acknowledged", parent=1)
        engine = await self.engine([foreign, child])
        assert (await engine.search("acknowledged"))[0].sources == (child,)
        future = original(2, "later_secret")
        assert (
            "later_secret"
            not in embedding_texts([future, original(1, "child", parent=2)])[0]
        )

    async def test_author_date_filters_precede_the_native_candidate_cap(self) -> None:
        messages = [
            original(1, "topic", author=10),
            original(2, "topic", author=20),
            original(3, "topic", author=10),
        ]
        engine = await self.engine(messages, [[1, 0], [1, 0], [1, 0]])
        assert messages[0].sent_at is not None
        assert messages[2].sent_at is not None
        assert [
            h.sources[-1].message_id
            for h in await engine.search(
                "topic", author_id=10, after=messages[2].sent_at.isoformat()
            )
        ] == [3]
        assert [
            h.sources[-1].message_id
            for h in await engine.search("topic", before=messages[0].sent_at.isoformat())
        ] == [1]
        with pytest.raises(ValueError, match="timezone"):
            await engine.search("topic", after="2026-01-01")
