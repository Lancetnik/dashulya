import os
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from unittest.mock import AsyncMock, patch

from evals.suites.memory.raw_hybrid import RawHybridVariant, RawLexicalVariant
from evals.suites.memory.types import QueryContext, SourceMessage


def source(
    mid: int,
    text: str,
    *,
    chat: str = "group",
    date: str = "2026-01-01T10:00:00Z",
    uid: int = 10,
    parent: int | None = None,
    bot: bool = False,
) -> SourceMessage:
    return SourceMessage(
        id=mid,
        chat_key=chat,
        date=date,
        text=text,
        user_id=uid,
        author=f"user_{uid}",
        user_name="Synthetic participant",
        reply_to_msg_id=parent,
        is_bot=bot,
    )


class RawHybridTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.workspace = Path(directory.name)
        self.context = QueryContext("group", "2026-01-02T00:00:00Z")
        self.embedded: list[str] = []

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.embedded.extend(texts)
        return [[1.0, 0.0] if "durability" in t else [0.0, 1.0] for t in texts]

    def backend(self, **kwargs: Any) -> RawHybridVariant:
        backend = RawHybridVariant(
            self.workspace, dimensions=2, embedder=self.embed, **kwargs
        )
        self.addAsyncCleanup(backend.close)
        return backend

    async def test_semantic_recall_finds_source_without_keyword_overlap(self) -> None:
        async def embed(texts: list[str]) -> list[list[float]]:
            return [
                [1.0, 0.0] if "database" in t or "durability" in t else [0.0, 1.0]
                for t in texts
            ]

        backend = RawHybridVariant(self.workspace, dimensions=2, embedder=embed)
        self.addAsyncCleanup(backend.close)
        await backend.prepare([source(1, "durability matters"), source(2, "UI colors")])
        hits = await backend.search("database", self.context, limit=1)
        self.assertEqual(hits[0].source_ids, (1,))
        self.assertIn("durability matters", hits[0].text)

    async def test_scope_cutoff_timezone_and_author_are_enforced(self) -> None:
        backend = self.backend()
        await backend.prepare([
            source(1, "durability allowed"),
            source(2, "durability other author", uid=20),
            source(1, "durability private secret", chat="private"),
            source(3, "durability future", date="2026-01-02T00:00:00Z"),
            source(4, "durability excluded old", date="2025-12-01T00:00:00Z"),
        ])
        context = QueryContext(
            "group",
            "2026-01-02T03:00:00+03:00",
            author_id=10,
            after="2026-01-01T09:00:00Z",
        )
        hits = await backend.search("durability", context)
        self.assertEqual([hit.source_ids for hit in hits], [(1,)])
        self.assertNotIn("secret", hits[0].text)

    async def test_equal_timestamp_uses_exclusive_message_id(self) -> None:
        backend = self.backend(semantic=False)
        await backend.prepare([source(1, "needle"), source(2, "needle")])
        context = QueryContext("group", "2026-01-01T10:00:00Z", before_id=2)
        self.assertEqual(
            [hit.source_ids for hit in await backend.search("needle", context)], [(1,)]
        )
        self.assertEqual(
            await backend.search("needle", replace(context, before_id=0)), []
        )

    async def test_ineligible_high_ranking_matches_cannot_exhaust_candidate_cap(
        self,
    ) -> None:
        backend = self.backend(semantic=False)
        await backend.prepare(
            [source(1, "needle accepted " + "other " * 100)]
            + [source(mid, "needle", chat="other") for mid in range(2, 202)]
        )
        hits = await backend.search("needle", self.context, limit=1)
        self.assertEqual(hits[0].source_ids, (1,))

    async def test_reply_context_preserves_original_ids_and_bot_label(self) -> None:
        backend = self.backend(semantic=False)
        await backend.prepare([
            source(1, "Which database?", uid=20),
            source(2, "Choose PostgreSQL", parent=1),
            source(3, "I invented this", parent=2, bot=True),
        ])
        hits = await backend.search("invented", self.context, limit=1)
        self.assertEqual(hits[0].source_ids, (1, 2, 3))
        self.assertIn("is_bot=True", hits[0].text)
        self.assertIn("reply_to=2", hits[0].text)

    async def test_reply_parents_outside_filter_are_not_returned(self) -> None:
        backend = self.backend(semantic=False)
        await backend.prepare([
            source(1, "confidential other author", uid=20),
            source(2, "needle", parent=1),
        ])
        hits = await backend.search(
            "needle", replace(self.context, author_id=10), limit=1
        )
        self.assertEqual(hits[0].source_ids, (2,))
        self.assertNotIn("confidential", hits[0].text)

    async def test_future_or_other_chat_cannot_supply_reply_context(self) -> None:
        backend = self.backend(semantic=False)
        await backend.prepare([
            source(1, "secret in another chat", chat="other"),
            source(2, "first needle", parent=1),
            source(3, "future secret", date="2026-02-01T00:00:00Z"),
            source(4, "second needle", parent=3),
        ])
        hits = await backend.search("needle", self.context)
        self.assertEqual({hit.source_ids for hit in hits}, {(2,), (4,)})
        self.assertTrue(all("secret" not in hit.text for hit in hits))

    async def test_reprepare_replaces_edits_and_removes_missing_sources(self) -> None:
        backend = self.backend(semantic=False)
        await backend.prepare([source(1, "old mistake"), source(2, "removed")])
        await backend.prepare([source(1, "final correction")])
        self.assertEqual(await backend.search("mistake removed", self.context), [])
        hits = await backend.search("correction", self.context)
        self.assertEqual(hits[0].source_ids, (1,))
        self.assertNotIn("mistake", hits[0].text)

    async def test_source_and_query_embeddings_resume_from_disk(self) -> None:
        backend = RawHybridVariant(self.workspace, dimensions=2, embedder=self.embed)
        await backend.prepare([source(1, "durability")])
        await backend.search("database", self.context)
        await backend.close()
        self.assertEqual(len(self.embedded), 2)
        reused = self.backend()
        await reused.prepare([source(1, "durability")])
        await reused.search("database", self.context)
        self.assertEqual(len(self.embedded), 2)
        await reused.prepare([source(1, "changed final text")])
        self.assertEqual(len(self.embedded), 3)

    async def test_long_reply_context_does_not_truncate_anchor_before_embedding(
        self,
    ) -> None:
        backend = self.backend()
        await backend.prepare([
            source(1, "long parent " * 3_000),
            source(2, "anchor fact", parent=1),
        ])
        reply_input = next(text for text in self.embedded if "anchor fact" in text)
        self.assertLessEqual(len(reply_input), 12_000)

    async def test_completed_batches_survive_later_embedding_failure(self) -> None:
        calls = 0

        async def flaky(texts: list[str]) -> list[list[float]]:
            nonlocal calls
            calls += 1
            if calls == 2:
                msg = "synthetic outage"
                raise RuntimeError(msg)
            return [[1.0, 0.0] for _ in texts]

        backend = RawHybridVariant(self.workspace, dimensions=2, embedder=flaky)
        self.addAsyncCleanup(backend.close)
        with (
            patch("evals.suites.memory.raw_hybrid._BATCH_SIZE", 1),
            patch("evals.suites.memory.raw_hybrid._PARALLEL_BATCHES", 1),
        ):
            with self.assertRaisesRegex(RuntimeError, "synthetic outage"):
                await backend.prepare([source(1, "first"), source(2, "second")])
            await backend.prepare([source(1, "first"), source(2, "second")])
        self.assertEqual(calls, 3)

    async def test_lexical_ablation_does_not_call_embeddings(self) -> None:
        backend = RawLexicalVariant(self.workspace)
        self.addAsyncCleanup(backend.close)
        with patch("evals.suites.memory.raw_hybrid.AsyncOpenAI") as client:
            await backend.prepare([source(1, "Needle")])
            hits = await backend.search("needle", self.context)
        client.assert_not_called()
        self.assertEqual(hits[0].source_ids, (1,))

    async def test_embedding_model_uses_environment_or_explicit_override(self) -> None:
        with patch.dict(os.environ, {"EMBEDDING_MODEL": "local-embedding-model"}):
            configured = self.backend()
            metadata = await configured.prepare([source(1, "needle")])
        self.assertEqual(metadata["embedding_model"], "local-embedding-model")
        explicit = RawHybridVariant(
            self.workspace / "explicit",
            model="explicit-model",
            dimensions=2,
            embedder=self.embed,
        )
        self.addAsyncCleanup(explicit.close)
        metadata = await explicit.prepare([source(1, "needle")])
        self.assertEqual(metadata["embedding_model"], "explicit-model")

    async def test_local_only_embeddings_fail_before_constructing_remote_client(
        self,
    ) -> None:
        backend = RawHybridVariant(self.workspace, dimensions=2)
        self.addAsyncCleanup(backend.close)
        with (
            patch.dict(
                os.environ,
                {
                    "MEMORY_EVAL_LOCAL_ONLY": "1",
                    "OPENAI_BASE_URL": "https://api.openai.com/v1",
                },
            ),
            patch("evals.suites.memory.raw_hybrid.AsyncOpenAI") as client,
            self.assertRaisesRegex(ValueError, "local HTTP"),
        ):
            await backend.prepare([source(1, "needle")])
        client.assert_not_called()

    async def test_invalid_vectors_and_naive_cutoffs_are_rejected(self) -> None:
        backend = RawHybridVariant(
            self.workspace, dimensions=2, embedder=AsyncMock(return_value=[[1.0]])
        )
        self.addAsyncCleanup(backend.close)
        with self.assertRaisesRegex(ValueError, "invalid vector"):
            await backend.prepare([source(1, "needle")])
        lexical = RawLexicalVariant(self.workspace / "lexical")
        self.addAsyncCleanup(lexical.close)
        await lexical.prepare([source(1, "needle")])
        with self.assertRaisesRegex(ValueError, "timezone"):
            await lexical.search("needle", QueryContext("group", "2026-01-02"))
