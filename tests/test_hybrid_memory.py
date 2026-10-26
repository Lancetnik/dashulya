import asyncio
import threading
import time
import unittest
from collections.abc import Sequence
from dataclasses import replace
from datetime import UTC, datetime
from operator import itemgetter
from unittest.mock import patch

from dashulya.memory.history import OriginalMessage
from dashulya.memory.hybrid import HybridHit, HybridMemory, embedding_texts, query_terms
from dashulya.memory.ranking import reciprocal_rank_fusion, select_source_diverse


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


class HybridTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.queries: list[str] = []

    async def embed(self, text: str) -> list[float]:
        self.queries.append(text)
        return [1.0, 0.0]

    def engine(
        self, messages: list[OriginalMessage], vectors: list[list[float]] | None = None
    ) -> HybridMemory:
        cached = (
            {}
            if vectors is None
            else dict(zip(embedding_texts(messages), vectors, strict=True))
        )
        engine = HybridMemory(messages, cached, self.embed, dimensions=2)
        self.addCleanup(engine.close)
        return engine

    async def test_raw_lexical_is_available_without_vectors_or_bulk_provider_calls(
        self,
    ) -> None:
        messages = [original(1, "durable storage"), original(2, "voice input")]
        engine = self.engine(messages)
        hits = await engine.search("storage")
        self.assertEqual([h.sources[-1].message_id for h in hits], [1])
        self.assertIs(hits[0].sources[0], messages[0])
        self.assertEqual(self.queries, [])
        self.assertEqual(
            engine.coverage,
            {"source_count": 2, "embedded_blocks": 0, "pending_blocks": 2},
        )

    async def test_source_ranking_is_offloop_and_provider_stays_on_own_loop(self) -> None:
        messages = [original(1, "storage")]
        loop_thread = threading.get_ident()
        provider_threads = []
        worker_threads = []
        ticks = []

        async def provider(_query: str) -> list[float]:
            provider_threads.append(threading.get_ident())
            return [1, 0]

        engine = await asyncio.to_thread(
            HybridMemory,
            messages,
            {embedding_texts(messages)[0]: [1, 0]},
            provider,
            dimensions=2,
        )
        self.addCleanup(engine.close)
        search_sync = engine._search_sync

        def delayed(
            query: str,
            eligible: set[int],
            query_vector: Sequence[float] | None,
            limit: int,
        ) -> list[HybridHit]:
            worker_threads.append(threading.get_ident())
            time.sleep(0.03)
            return search_sync(query, eligible, query_vector, limit)

        async def tick() -> None:
            for _ in range(10):
                await asyncio.sleep(0.001)
                ticks.append(True)

        with patch.object(engine, "_search_sync", side_effect=delayed):
            hits, _ = await asyncio.gather(engine.search("storage"), tick())
        self.assertEqual(hits[0].sources, (messages[0],))
        self.assertEqual(provider_threads, [loop_thread])
        self.assertTrue(all(t != loop_thread for t in worker_threads))
        self.assertEqual(len(ticks), 10)

    async def test_dense_recall_and_rrf_use_cached_sources_and_one_query_embedding(
        self,
    ) -> None:
        messages = [
            original(1, "store store store"),
            original(2, "synonym"),
            original(3, "store"),
        ]
        engine = self.engine(messages, [[0, 1], [0.8, 0.2], [1, 0]])
        hits = await engine.search("store")
        self.assertEqual([h.sources[-1].message_id for h in hits], [3, 1, 2])
        self.assertEqual(self.queries, ["store"])
        self.assertEqual(engine.coverage["embedded_blocks"], 3)

    async def test_two_reply_parents_and_anchor_overlap_match_raw_policy(self) -> None:
        messages = [
            original(1, "first"),
            original(2, "second", parent=1),
            original(3, "third", parent=2),
            original(4, "fourth", parent=3),
        ]
        texts = embedding_texts(messages)
        self.assertIn("first", texts[2])
        self.assertNotIn("first", texts[3])
        engine = self.engine(messages, [[0, 1], [0, 1], [1, 0], [0, 1]])
        hits = await engine.search("unmatched")
        self.assertEqual([m.message_id for m in hits[0].sources], [1, 2, 3])
        self.assertEqual([h.sources[-1].message_id for h in hits], [3, 4])

    async def test_query_author_filter_hides_parent_sources_but_keeps_measured_index_context(
        self,
    ) -> None:
        messages = [
            original(1, "parent_keyword", author=10),
            original(2, "acknowledged", author=20, parent=1),
        ]
        engine = self.engine(messages, [[0, 1], [1, 0]])
        hits = await engine.search("parent_keyword", author_id=20)
        self.assertEqual([m.message_id for m in hits[0].sources], [2])
        self.assertNotIn("parent_keyword", hits[0].text)
        self.assertEqual(self.queries, ["parent_keyword"])

    async def test_hidden_or_removed_parent_cannot_enter_from_old_cached_vector(
        self,
    ) -> None:
        parent = original(1, "private_parent_secret")
        child = original(2, "acknowledged", parent=1)
        old_text = embedding_texts([parent, child])[1]
        engine = HybridMemory([child], {old_text: [1, 0]}, self.embed, dimensions=2)
        self.addCleanup(engine.close)
        self.assertEqual(engine.coverage["embedded_blocks"], 0)
        self.assertEqual(await engine.search("private_parent_secret"), [])
        self.assertEqual(self.queries, [])

    async def test_edits_change_embedding_key_and_never_reuse_stale_semantic_vector(
        self,
    ) -> None:
        old = original(1, "old tool")
        edited = replace(
            old, text="corrected tool", edited_at="2026-01-02T00:00:00+00:00"
        )
        engine = HybridMemory(
            [edited], {embedding_texts([old])[0]: [1, 0]}, self.embed, dimensions=2
        )
        self.addCleanup(engine.close)
        self.assertEqual(engine.coverage["pending_blocks"], 1)
        self.assertEqual(await engine.search("unmatched"), [])
        hits = await engine.search("corrected")
        self.assertEqual(hits[0].sources, (edited,))

    async def test_same_message_id_in_another_collection_never_becomes_reply_parent(
        self,
    ) -> None:
        foreign = original(1, "foreign_parent_secret", chat=-100999)
        child = original(2, "acknowledged", parent=1)
        texts = embedding_texts([foreign, child])
        child_text = next(text for text in texts if "message_id=2 " in text)
        self.assertNotIn("foreign_parent_secret", child_text)
        engine = self.engine([foreign, child])
        hits = await engine.search("acknowledged")
        self.assertEqual(hits[0].sources, (child,))

    async def test_future_reply_parent_is_not_indexed_into_child(self) -> None:
        child = original(1, "child", parent=2)
        parent = original(2, "later_secret")
        self.assertNotIn("later_secret", embedding_texts([parent, child])[0])

    async def test_author_time_filters_apply_before_candidate_cap_and_keep_boundaries(
        self,
    ) -> None:
        messages = [
            original(1, "topic", author=10),
            original(2, "topic", author=20),
            original(3, "topic", author=10),
        ]
        engine = self.engine(messages, [[1, 0], [1, 0], [1, 0]])
        assert messages[0].sent_at is not None
        assert messages[2].sent_at is not None
        hits = await engine.search(
            "topic", author_id=10, after=messages[2].sent_at.isoformat()
        )
        self.assertEqual([h.sources[-1].message_id for h in hits], [3])
        hits = await engine.search("topic", before=messages[0].sent_at.isoformat())
        self.assertEqual([h.sources[-1].message_id for h in hits], [1])
        with self.assertRaisesRegex(ValueError, "timezone"):
            await engine.search("topic", after="2026-01-01T00:00:00")

    async def test_vector_validation_rejects_malformed_cache_and_query_output(
        self,
    ) -> None:
        messages = [original(1, "text")]
        with self.assertRaisesRegex(ValueError, "dimensions"):
            self.engine(messages, [[1, 0, 0]])
        with self.assertRaisesRegex(ValueError, "finite"):
            self.engine(messages, [[float("nan"), 0]])

        async def bad_query(_text: str) -> list[float]:
            return [float("inf"), 0]

        engine = HybridMemory(
            messages, {embedding_texts(messages)[0]: [1, 0]}, bad_query, dimensions=2
        )
        self.addCleanup(engine.close)
        with self.assertRaisesRegex(ValueError, "finite"):
            await engine.search("query")

    def test_measured_tokenization_and_embedding_bounds_are_preserved(self) -> None:
        self.assertEqual(query_terms("и THE инструментами BB bb"), ["инструмент", "bb"])
        self.assertEqual(len(query_terms(" ".join(f"term{i}" for i in range(60)))), 48)
        messages = [
            original(1, "x" * 20000),
            original(2, "y" * 20000, parent=1),
            original(3, "z" * 20000, parent=2),
        ]
        self.assertTrue(all(len(text) <= 12000 for text in embedding_texts(messages)))


if __name__ == "__main__":
    unittest.main()
