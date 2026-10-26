import unittest
from dataclasses import replace
from datetime import UTC, datetime
from operator import itemgetter
from unittest.mock import AsyncMock

from dashulya.domain.memory import OriginalMessage
from dashulya.memory.hybrid import HybridMemory, embedding_texts, query_terms
from dashulya.memory.lexical import LexicalIndex
from dashulya.memory.ranking import reciprocal_rank_fusion, select_source_diverse
from dashulya.memory.vectors import EmbeddingCache
from tests.postgres import PostgresTestCase


def original(
    mid: int,
    text: str,
    *,
    chat: int = -100123,
    author: int = 10,
    parent: int | None = None,
) -> OriginalMessage:
    return OriginalMessage(
        collection=f"telegram:{chat}",
        chat_id=chat,
        message_id=mid,
        text=text,
        sent_at=datetime(2026, 1, 1, 10, 0, mid, tzinfo=UTC),
        author_id=author,
        author="Alice",
        chat_title="Fixture",
        chat_username="fixture_chat",
        topic_id=None,
        reply_to_id=parent,
        edited_at=None,
        media_type=None,
        url=f"https://t.me/fixture_chat/{mid}",
        provenance="live",
        username="alice",
        is_bot=False,
    )


class RankingTests(unittest.TestCase):
    def test_rrf_uses_ordinal_ranks_and_deterministic_explicit_ties(self) -> None:
        fused = reciprocal_rank_fusion([[2, 1], [1, 3]], tie_order={1: 0, 2: 1, 3: 2})
        self.assertEqual([key for key, _ in fused], [1, 2, 3])
        self.assertEqual(fused[0][1], 1 / 61 + 1 / 62)
        self.assertEqual(
            reciprocal_rank_fusion([[1, 1], [2]], tie_order={1: 1, 2: 0})[0][0], 2
        )
        with self.assertRaisesRegex(ValueError, "tie position"):
            reciprocal_rank_fusion([[1]], tie_order={})

    def test_source_diversity_suppresses_covered_anchor_but_not_new_reply(self) -> None:
        candidates = [("a", ("parent", "a")), ("parent", ("parent",)), ("b", ("a", "b"))]
        selected = select_source_diverse(
            candidates, anchor=itemgetter(0), sources=itemgetter(1), limit=8
        )
        self.assertEqual([item[0] for item in selected], ["a", "b"])
        self.assertEqual(
            select_source_diverse(
                candidates,
                anchor=itemgetter(0),
                sources=itemgetter(1),
                limit=0,
            ),
            [],
        )

    def test_composite_source_keys_preserve_cross_chat_and_revision_identity(
        self,
    ) -> None:
        first = ("group", 1, 1)
        other_chat = ("private", 1, 1)
        other_revision = ("group", 1, 2)
        result = reciprocal_rank_fusion(
            [[first, other_chat], [other_revision]],
            tie_order={first: 0, other_chat: 1, other_revision: 2},
        )
        self.assertEqual({key for key, _ in result}, {first, other_chat, other_revision})


class HybridTests(PostgresTestCase):
    def setUp(self) -> None:
        super().setUp()
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
        self.assertEqual(
            [h.sources[-1].message_id for h in await engine.search("storage")], [1]
        )
        self.assertEqual(self.queries, [])
        self.assertEqual(
            await engine.coverage(),
            {"source_count": 2, "embedded_blocks": 0, "pending_blocks": 2},
        )

    async def test_dense_recall_and_rrf_use_native_cached_source_search(self) -> None:
        messages = [
            original(1, "store store store"),
            original(2, "synonym"),
            original(3, "store"),
        ]
        engine = await self.engine(messages, [[0, 1], [0.8, 0.2], [1, 0]])
        self.assertEqual(
            [h.sources[-1].message_id for h in await engine.search("store")], [3, 1, 2]
        )
        self.assertEqual(self.queries, ["store"])
        self.assertEqual((await engine.coverage())["embedded_blocks"], 3)

    async def test_reply_parents_and_source_diversity(self) -> None:
        messages = [
            original(1, "first"),
            original(2, "second", parent=1),
            original(3, "third", parent=2),
            original(4, "fourth", parent=3),
        ]
        texts = embedding_texts(messages)
        self.assertIn("first", texts[2])
        self.assertNotIn("first", texts[3])
        engine = await self.engine(messages, [[0, 1], [0, 1], [1, 0], [0, 1]])
        hits = await engine.search("unmatched")
        self.assertEqual([m.message_id for m in hits[0].sources], [1, 2, 3])
        self.assertEqual([h.sources[-1].message_id for h in hits], [3, 4])

    async def test_author_filter_excludes_parent_from_primary_evidence(self) -> None:
        messages = [
            original(1, "parent_keyword", author=10),
            original(2, "acknowledged", author=20, parent=1),
        ]
        engine = await self.engine(messages, [[0, 1], [1, 0]])
        hits = await engine.search("parent_keyword", author_id=20)
        self.assertEqual([m.message_id for m in hits[0].sources], [2])
        self.assertNotIn("parent_keyword", hits[0].text)

    async def test_stale_parent_or_revision_vectors_are_not_used(self) -> None:
        old = original(1, "old secret")
        edited = replace(
            old, text="corrected tool", edited_at="2026-01-02T00:00:00+00:00"
        )
        cache = EmbeddingCache(self.database, AsyncMock())
        await cache.put({cache.key(embedding_texts([old])[0]): [1.0] + [0.0] * 511})
        engine = HybridMemory([edited], cache, LexicalIndex(self.database))
        self.assertEqual((await engine.coverage())["pending_blocks"], 1)
        self.assertEqual(await engine.search("unmatched"), [])
        self.assertEqual((await engine.search("corrected"))[0].sources, (edited,))
        child = original(2, "acknowledged", parent=1)
        await cache.put({
            cache.key(embedding_texts([old, child])[1]): [1.0] + [0.0] * 511
        })
        child_engine = HybridMemory([child], cache, LexicalIndex(self.database))
        self.assertEqual(await child_engine.search("old secret"), [])

    async def test_cross_chat_ids_and_future_parents_stay_distinct(self) -> None:
        foreign = original(1, "foreign_parent_secret", chat=-100999)
        child = original(2, "acknowledged", parent=1)
        engine = await self.engine([foreign, child])
        self.assertEqual((await engine.search("acknowledged"))[0].sources, (child,))
        future = original(2, "later_secret")
        self.assertNotIn(
            "later_secret", embedding_texts([future, original(1, "child", parent=2)])[0]
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
        self.assertEqual(
            [
                h.sources[-1].message_id
                for h in await engine.search(
                    "topic", author_id=10, after=messages[2].sent_at.isoformat()
                )
            ],
            [3],
        )
        self.assertEqual(
            [
                h.sources[-1].message_id
                for h in await engine.search(
                    "topic", before=messages[0].sent_at.isoformat()
                )
            ],
            [1],
        )
        with self.assertRaisesRegex(ValueError, "timezone"):
            await engine.search("topic", after="2026-01-01")

    def test_query_terms_and_embedding_bounds(self) -> None:
        self.assertEqual(query_terms("и THE инструментами BB bb"), ["инструмент", "bb"])
        self.assertEqual(len(query_terms(" ".join(f"term{i}" for i in range(60)))), 48)
        self.assertTrue(
            all(
                len(text) <= 12000
                for text in embedding_texts([
                    original(1, "x" * 20000),
                    original(2, "y" * 20000, parent=1),
                ])
            )
        )
