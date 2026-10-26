import json
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from unittest.mock import patch

from ag2.events import ToolCallEvent
from ag2.testing import TestConfig, TrackingConfig

from dashulya.domain.memory import MemoryAccess
from dashulya.memory.service import HybridChatMemory
from dashulya.memory.vectors import EmbeddingCache
from evals.suites.memory.candidate import (
    CandidateAnswerSession,
    fingerprints,
    prepare_candidate_history,
)
from evals.suites.memory.corpus import SourceArchive
from evals.suites.memory.types import QueryContext, SourceMessage
from tests.support import reply_json


def source(
    mid: int, text: str, *, date: str = "2026-01-01T10:00:00Z", bot: bool = False
) -> SourceMessage:
    return SourceMessage(
        mid, "test_group", date, text, 100 + mid, f"user_{mid}", f"Name {mid}", is_bot=bot
    )


class CandidateProductionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name)
        self.messages = [
            source(1, "Choose PostgreSQL for durability"),
            source(2, "Bot interpretation", bot=True),
            source(3, ""),
        ]
        self.archive = SourceArchive([
            *self.messages,
            source(4, "Future secret", date="2026-01-03T00:00:00Z"),
        ])
        self.context = QueryContext("test_group", "2026-01-02T00:00:00Z")
        self.embed_calls: list[list[str]] = []

        async def embed(texts: list[str]) -> list[list[float]]:
            self.embed_calls.append(texts)
            return [[1.0, *([0.0] * 511)] for _ in texts]

        self.cache = EmbeddingCache(self.path / "vectors.sqlite3", embed)
        self.live = await prepare_candidate_history(
            self.archive, self.context, self.path / "fixture"
        )
        self.service = HybridChatMemory(self.live, self.cache, owner_user_id=42)
        await self.service.start()
        self.addAsyncCleanup(self.service.close)

    async def test_actual_provider_dialogue_both_tools_raw_reply_and_canonical_trace(
        self,
    ) -> None:
        source_id = (await self.service.history(MemoryAccess(-1, False)).query())[
            "messages"
        ][0]["source_id"]
        config = TrackingConfig(
            TestConfig(
                ToolCallEvent("search_memory", arguments='{"query":"PostgreSQL"}'),
                ToolCallEvent(
                    "chat_history",
                    arguments=json.dumps({"action": "read", "source_id": source_id}),
                ),
                ToolCallEvent("chat_history", arguments='{"action":"count"}'),
                reply_json("Saved3messages. https://t.me/test_group/1"),
            )
        )
        session = CandidateAnswerSession(self.archive, self.service, self.context)
        result = await session.answer("Question", config)
        self.assertEqual(result["answer"]["text"], result["raw_reply"]["text"])
        self.assertIsNone(result["answer"]["aggregation_count"])
        self.assertIsNone(result["answer"]["abstained"])
        self.assertEqual(result["answer"]["cited_message_ids"], [1])
        self.assertEqual(
            [t["tool"] for t in result["trace"]],
            [
                "production_group_context",
                "search_memory",
                "search_memory",
                "chat_history",
                "chat_history",
            ],
        )
        self.assertEqual(result["trace"][1]["arguments"]["query"], "Question")
        self.assertEqual(result["trace"][1]["arguments"]["limit"], 8)
        self.assertEqual(result["trace"][2]["arguments"]["query"], "PostgreSQL")
        for call in result["trace"][1:3]:
            self.assertTrue(call["result"]["hits"])
            for hit in call["result"]["hits"]:
                for evidence in hit["sources"]:
                    original = self.archive.get(evidence["message_id"], self.context)
                    assert original is not None
                    offset = evidence.get("offset", 0)
                    self.assertEqual(
                        evidence["text"],
                        original.text[offset : offset + len(evidence["text"])],
                    )
        self.assertEqual(result["trace"][-1]["result"]["count"], 3)
        self.assertEqual(result["seen_source_ids"], [1, 2, 3])
        self.assertNotIn("Future secret", json.dumps(result))
        self.assertEqual(len(config.calls[0].tools), 2)
        access = config.calls[0].dependencies[MemoryAccess]
        self.assertEqual(access.chat_id, -1)
        self.assertFalse(access.is_private)
        self.assertTrue(result["production"]["chat_id_is_experiment_local"])
        self.assertTrue(result["production"]["automatic_initial_search"])
        self.assertIn("total", result["usage"])
        self.assertTrue(result["trace"][0]["result"]["messages"][1]["is_bot"])

    async def test_source_prefix_and_metadata_are_exact_and_cache_resume_makes_no_embeddings(
        self,
    ) -> None:
        originals = await self.service.history(MemoryAccess(42, True)).originals()
        self.assertEqual([m.message_id for m in originals], [1, 2, 3])
        self.assertEqual(originals[0].author, "Name 1")
        self.assertEqual(originals[0].username, "user_1")
        self.assertTrue(originals[1].is_bot)
        self.assertEqual(originals[2].text, "")
        previous = len(self.embed_calls)
        resumed = await prepare_candidate_history(
            self.archive, self.context, self.path / "fixture"
        )
        self.assertEqual(resumed.path, self.live.path)
        await self.service.refresh_once()
        self.assertEqual(len(self.embed_calls), previous)
        changed = SourceArchive([
            replace(m, text="Changed") if m.id == 1 else m for m in self.archive.messages
        ])
        with self.assertRaisesRegex(ValueError, "inputs changed"):
            await prepare_candidate_history(changed, self.context, self.path / "fixture")

    async def test_historical_request_original_is_not_available_as_prior_evidence(
        self,
    ) -> None:
        request = source(4, "Historical question", date=self.context.before)
        archive = SourceArchive([
            *self.messages,
            request,
            source(5, "Later secret", date="2026-01-03T00:00:00Z"),
        ])
        context = replace(self.context, before_id=4)
        historical = await prepare_candidate_history(
            archive, context, self.path / "historical"
        )
        service = HybridChatMemory(historical, self.cache, owner_user_id=42)
        session = CandidateAnswerSession(archive, service, context)
        config = TrackingConfig(
            TestConfig(
                ToolCallEvent("chat_history", arguments='{"action":"list"}'),
                reply_json("Done"),
            )
        )
        with (
            patch.object(service, "recall", wraps=service.recall) as recall,
            patch.object(service, "history", wraps=service.history) as history,
        ):
            result = await session.answer(
                request.text,
                config,
                input_context={"preceding_messages": [{"id": 3, "text": ""}]},
            )
        access = config.calls[0].dependencies[MemoryAccess]
        self.assertIs(recall.call_args.args[0], access)
        self.assertIs(history.call_args.args[0], access)
        self.assertEqual(result["trace"][1]["arguments"]["query"], request.text)
        self.assertFalse(result["production"]["synthetic_request"])
        self.assertEqual(
            result["production"]["current_request"]["user_id"], request.user_id
        )
        self.assertNotIn(4, result["seen_source_ids"])
        self.assertNotIn(5, result["seen_source_ids"])
        self.assertEqual(config.calls[0].dependencies[MemoryAccess].before_id, 4)

    async def test_forged_input_context_is_rejected_before_model(self) -> None:
        config = TrackingConfig(TestConfig(reply_json("Done")))
        session = CandidateAnswerSession(self.archive, self.service, self.context)
        with self.assertRaisesRegex(ValueError, "canonical original"):
            await session.answer(
                "Question",
                config,
                input_context={
                    "preceding_messages": [{"id": 4, "text": "Future secret"}]
                },
            )
        self.assertEqual(config.calls, [])

    async def test_noncanonical_tool_body_fails_closed_with_preserved_attempt_trace(
        self,
    ) -> None:
        class ForgedRecall:
            async def search(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
                return {
                    "hits": [
                        {
                            "sources": [
                                {
                                    "message_id": 1,
                                    "text": "Forged fact",
                                    "author_id": 101,
                                    "sent_at": "2026-01-01T10:00:00Z",
                                    "url": "https://t.me/test_group/1",
                                    "chat_id": -1,
                                }
                            ]
                        }
                    ]
                }

        session = CandidateAnswerSession(self.archive, self.service, self.context)
        with (
            patch.object(self.service, "recall", return_value=ForgedRecall()),
            self.assertRaisesRegex(ValueError, "noncanonical"),
        ):
            await session.answer(
                "Question",
                TestConfig(
                    ToolCallEvent("search_memory", arguments='{"query":"term"}'),
                    reply_json("Done"),
                ),
            )
        self.assertEqual(session.trace[-1]["error_type"], "ValueError")
        self.assertNotIn("result", session.trace[-1])
        self.assertNotIn("Forged fact", str(session.trace))

    def test_worktree_fingerprints_cover_actual_new_production_modules(self) -> None:
        hashes = fingerprints()
        for name in (
            "src/dashulya/ai/tools/history.py",
            "src/dashulya/main/providers/ai.py",
            "src/dashulya/memory/hybrid.py",
            "src/dashulya/memory/service.py",
            "evals/suites/memory/candidate.py",
        ):
            self.assertIn(name, hashes)
            self.assertEqual(len(hashes[name]), 64)
