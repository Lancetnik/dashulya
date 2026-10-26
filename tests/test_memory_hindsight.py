"""Source isolation and resumability checks without a Hindsight server or models."""

import json
import tempfile
import unittest
from collections.abc import Mapping
from pathlib import Path
from unittest.mock import AsyncMock, patch

from evals.suites.memory.hindsight import (
    HindsightChunksVariant,
    HindsightError,
    HindsightVariant,
)
from evals.suites.memory.types import QueryContext, SourceMessage


def message(
    identity: int = 1,
    *,
    chat: str = "example",
    author: int = 10,
    date: str = "2026-01-01T12:00:00Z",
) -> SourceMessage:
    return SourceMessage(
        id=identity,
        chat_key=chat,
        date=date,
        text=f"Person {author} tested project {identity} successfully.",
        user_id=author,
        author=f"person_{author}",
        user_name=f"Person {author}",
    )


async def successful_request(
    method: str, path: str, payload: Mapping[str, object] | None = None
) -> dict[str, object]:
    if method == "POST" and path.endswith("/memories"):
        assert payload is not None
        return {"operation_id": payload["operation_id"]}
    if method == "POST" and path.endswith("/consolidate"):
        return {"operation_id": "consolidation"}
    if method == "GET" and "/operations/" in path:
        return {"status": "completed"}
    return {}


class HindsightVariantTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.workspace = Path(self.directory.name)
        self.variant = HindsightVariant(self.workspace)
        self.request = AsyncMock(side_effect=successful_request)
        self.request_patch = patch.object(self.variant, "_request", self.request)
        self.request_patch.start()
        self.addCleanup(self.request_patch.stop)

    async def test_prepare_keeps_all_originals_and_resumes_completed_batches(
        self,
    ) -> None:
        originals = [message(i) for i in range(1, 20)]
        summary = await self.variant.prepare(originals)
        self.assertEqual(summary["source_count"], 19)
        self.assertEqual(summary["document_count"], 2)
        self.assertFalse(summary["consolidated"])
        self.assertEqual(summary["memory_mode"], "retained_facts")
        self.assertFalse(
            any(
                call.args[1].endswith("/consolidate")
                for call in self.request.call_args_list
            )
        )
        retains = [
            call
            for call in self.request.call_args_list
            if call.args[1].endswith("/memories")
        ]
        self.assertEqual(len(retains), 1)
        ids = [
            identity
            for item in retains[0].args[2]["items"]
            for identity in json.loads(item["metadata"]["source_ids"])
        ]
        self.assertEqual(ids, list(range(1, 20)))
        self.assertIn("user_id=10", retains[0].args[2]["items"][0]["content"])
        self.assertIn(
            "sent_at=2026-01-01T12:00:00Z", retains[0].args[2]["items"][0]["content"]
        )
        self.request.reset_mock()
        await self.variant.prepare(originals)
        self.assertFalse(
            any(
                call.args[1].endswith("/memories") for call in self.request.call_args_list
            )
        )

    async def test_reuses_completed_retains_after_cancelled_consolidation(self) -> None:
        await self.variant.prepare([message()])
        path = self.workspace / "hindsight-checkpoint.json"
        state = json.loads(path.read_text())
        bank = next(iter(state["banks"].values()))
        state.pop("prepared")
        state["consolidation"][bank] = {
            "operation_id": "cancelled-pilot",
            "completed": False,
        }
        path.write_text(json.dumps(state))
        self.request.reset_mock()
        result = await self.variant.prepare([message()])
        self.assertEqual(result["fingerprint"], state["fingerprint"])
        self.assertFalse(
            any(call.args[0] == "POST" for call in self.request.call_args_list)
        )
        updates = [
            call.args[2]["updates"]
            for call in self.request.call_args_list
            if call.args[0] == "PATCH"
        ]
        self.assertEqual(
            updates,
            [{"enable_auto_consolidation": False, "enable_observations": False}],
        )
        self.assertEqual(
            json.loads(path.read_text())["consolidation"][bank]["operation_id"],
            "cancelled-pilot",
        )

    async def test_native_chunks_uses_separate_bank_and_forces_extraction_mode(
        self,
    ) -> None:
        originals = [message(i) for i in range(1, 20)]
        facts = await self.variant.prepare(originals)
        chunks = HindsightChunksVariant(self.workspace / "chunks")
        request = AsyncMock(side_effect=successful_request)
        with (
            patch.object(chunks, "_request", request),
            patch.dict(
                "os.environ",
                {
                    "HINDSIGHT_EXTRACTION_MODE": "verbose",
                    "HINDSIGHT_CHUNK_CHARACTERS": "12000",
                },
            ),
        ):
            result = await chunks.prepare(originals)
        self.assertEqual(result["extraction_mode"], "chunks")
        self.assertEqual(result["variant"], "hindsight_chunks")
        self.assertEqual(result["memory_mode"], "native_raw_chunks")
        self.assertEqual(result["chunk_characters"], 1800)
        self.assertEqual(result["document_max_characters"], 12000)
        self.assertEqual(result["document_count"], 2)
        self.assertFalse(result["consolidated"])
        self.assertNotEqual(facts["fingerprint"], result["fingerprint"])
        self.assertNotEqual(self.variant._banks, chunks._banks)
        create = next(call for call in request.call_args_list if call.args[0] == "PUT")
        self.assertEqual(create.args[2]["retain_extraction_mode"], "chunks")
        self.assertEqual(create.args[2]["retain_chunk_size"], 1800)
        self.assertEqual(create.args[2]["retain_structured_chunk_size"], 1800)
        self.assertFalse(create.args[2]["enable_observations"])
        retain = next(
            call for call in request.call_args_list if call.args[1].endswith("/memories")
        )
        self.assertEqual(
            json.loads(retain.args[2]["items"][0]["metadata"]["source_ids"]),
            list(range(1, 17)),
        )
        self.assertFalse(
            any(
                call.args[1].endswith("/consolidate")
                for call in self.request.call_args_list
            )
        )

    async def test_rejects_different_prefix_in_existing_workspace(self) -> None:
        await self.variant.prepare([message()])
        with self.assertRaisesRegex(HindsightError, "different source prefix"):
            await self.variant.prepare([message(), message(2)])

    async def test_bank_identity_changes_with_source_prefix(self) -> None:
        await self.variant.prepare([message()])
        first = self.variant._banks["example"]
        other = HindsightVariant(self.workspace / "earlier")
        with patch.object(other, "_request", AsyncMock(side_effect=successful_request)):
            await other.prepare([message(), message(2)])
        self.assertNotEqual(first, other._banks["example"])

    async def test_recall_ignores_fabricated_metadata_and_foreign_documents(self) -> None:
        await self.variant.prepare([message(), message(1, chat="other")])
        document = next(
            identity
            for identity, (chat, _) in self.variant._documents.items()
            if chat == "example"
        )
        foreign = next(
            identity
            for identity, (chat, _) in self.variant._documents.items()
            if chat == "other"
        )
        self.request.side_effect = None
        self.request.return_value = {
            "results": [
                {"text": "unverified", "metadata": {"source_ids": "[999]"}},
                {"text": "foreign", "document_id": foreign},
                {"text": "grounded candidate", "document_id": document},
                {
                    "text": "partial pilot observation",
                    "type": "observation",
                    "document_id": document,
                },
            ]
        }
        hits = await self.variant.search(
            "project", QueryContext("example", "2026-01-02T00:00:00Z")
        )
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].source_ids, (1,))
        self.assertEqual(hits[0].text, "grounded candidate")
        self.assertEqual(hits[0].chat_key, "example")
        self.assertEqual(self.request.call_args.args[2]["types"], ["world", "experience"])

    async def test_observation_requires_every_source_fact_to_be_resolved(self) -> None:
        await self.variant.prepare([message()])
        document = next(iter(self.variant._documents))
        self.request.side_effect = None
        self.request.return_value = {
            "results": [
                {"text": "incomplete", "source_fact_ids": ["known", "missing"]},
                {"text": "complete", "source_fact_ids": ["known"]},
            ],
            "source_facts": {"known": {"text": "source", "document_id": document}},
        }
        hits = await self.variant.search(
            "project", QueryContext("example", "2026-01-02T00:00:00Z")
        )
        self.assertEqual([hit.text for hit in hits], ["complete"])
        self.assertEqual(hits[0].source_ids, (1,))

    async def test_older_cutoff_cannot_reuse_future_bank(self) -> None:
        await self.variant.prepare([message()])
        with self.assertRaisesRegex(HindsightError, "cutoff precedes"):
            await self.variant.search(
                "project", QueryContext("example", "2026-01-01T12:00:00Z", before_id=1)
            )

    async def test_author_filter_drops_claim_with_mixed_author_support(self) -> None:
        await self.variant.prepare([message(), message(2, author=20)])
        document = next(iter(self.variant._documents))
        self.request.side_effect = None
        self.request.return_value = {
            "results": [{"text": "mixed", "document_id": document}]
        }
        hits = await self.variant.search(
            "project", QueryContext("example", "2026-01-02T00:00:00Z", author_id=10)
        )
        self.assertEqual(hits, [])

    async def test_operation_failure_preserves_submitted_checkpoint(self) -> None:
        async def fail_operation(
            method: str, path: str, payload: Mapping[str, object] | None = None
        ) -> dict[str, object]:
            if "/operations/" in path:
                return {"status": "failed", "error_message": "provider-private-details"}
            return await successful_request(method, path, payload)

        self.request.side_effect = fail_operation
        with self.assertRaisesRegex(HindsightError, "is failed"):
            await self.variant.prepare([message()])
        checkpoint = json.loads(
            (self.workspace / "hindsight-checkpoint.json").read_text()
        )
        batch = next(iter(checkpoint["batches"].values()))
        self.assertTrue(batch["submitted"])
        self.assertFalse(batch["completed"])
        self.assertNotIn("prepared", checkpoint)
        self.assertNotIn("provider-private-details", json.dumps(checkpoint))

    async def test_lost_submission_acknowledgement_reuses_operation_identity(
        self,
    ) -> None:
        submitted: list[str] = []

        async def lose_first_ack(
            method: str, path: str, payload: Mapping[str, object] | None = None
        ) -> dict[str, object]:
            if method == "POST" and path.endswith("/memories"):
                assert payload is not None
                submitted.append(str(payload["operation_id"]))
                if len(submitted) == 1:
                    msg = "Hindsight request failed"
                    raise HindsightError(msg)
            return await successful_request(method, path, payload)

        self.request.side_effect = lose_first_ack
        with self.assertRaisesRegex(HindsightError, "request failed"):
            await self.variant.prepare([message()])
        await self.variant.prepare([message()])
        self.assertEqual(len(submitted), 2)
        self.assertEqual(submitted[0], submitted[1])

    async def test_groups_facts_by_source_set_with_best_score_and_unique_text(
        self,
    ) -> None:
        originals = [
            message(1, date="2026-01-01T12:00:02Z"),
            message(2, date="2026-01-01T12:00:01Z"),
            message(3, date="2026-01-01T12:00:03Z"),
        ]
        with patch.dict("os.environ", {"HINDSIGHT_DOCUMENT_MESSAGES": "2"}):
            await self.variant.prepare(originals)
        first, second = self.variant._documents
        self.request.side_effect = None
        self.request.return_value = {
            "results": [
                {
                    "text": "first qualified fact",
                    "document_id": first,
                    "scores": {"final": 0.2},
                },
                {
                    "text": "another document",
                    "document_id": second,
                    "scores": {"final": 0.7},
                },
                {
                    "text": "second qualified fact",
                    "document_id": first,
                    "scores": {"final": 0.9},
                },
                {
                    "text": "first qualified fact",
                    "document_id": first,
                    "scores": {"final": 0.8},
                },
            ]
        }
        hits = await self.variant.search(
            "project", QueryContext("example", "2026-01-02T00:00:00Z"), limit=2
        )
        self.assertEqual(len(hits), 2)
        self.assertEqual(hits[0].source_ids, (1, 2))
        self.assertEqual(hits[0].score, 0.9)
        self.assertEqual(hits[0].text, "first qualified fact\nsecond qualified fact")
        self.assertEqual(hits[1].source_ids, (3,))

    async def test_group_limit_omits_whole_long_fact_instead_of_its_qualification(
        self,
    ) -> None:
        await self.variant.prepare([message()])
        document = next(iter(self.variant._documents))
        self.request.side_effect = None
        self.request.return_value = {
            "results": [
                {"text": "A short qualified statement.", "document_id": document},
                {"text": "x" * 2200 + " only for this task", "document_id": document},
            ]
        }
        hits = await self.variant.search(
            "project", QueryContext("example", "2026-01-02T00:00:00Z")
        )
        self.assertEqual(len(hits), 1)
        self.assertLessEqual(len(hits[0].text), 2000)
        self.assertIn("A short qualified statement.", hits[0].text)
        self.assertIn("omitted", hits[0].text)
        self.assertNotIn("xxx", hits[0].text)
