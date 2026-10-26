"""Accumulated knowledge keeps source lineage and respects snapshot boundaries."""

import tempfile
import unittest
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Literal
from unittest.mock import AsyncMock, patch

from pydantic import ValidationError

from evals.suites.memory.claims import (
    ClaimDraft,
    ClaimExtractor,
    ClaimsVariant,
    ExtractionResult,
    SourceQuote,
)
from evals.suites.memory.providers import parse_structured, validate_local_endpoint
from evals.suites.memory.types import QueryContext, SourceMessage


def message(
    identifier: int,
    text: str,
    *,
    chat: str = "group-a",
    bot: bool = False,
    date: str | None = None,
    parent: int | None = None,
) -> SourceMessage:
    return SourceMessage(
        id=identifier,
        chat_key=chat,
        date=date or f"2026-01-{identifier:02}T10:00:00+00:00",
        text=text,
        user_id=42,
        author="participant",
        user_name="Participant",
        reply_to_msg_id=parent,
        is_bot=bot,
    )


def draft(
    source: SourceMessage,
    *,
    operation: Literal["add", "change", "correct", "uncertain"] = "add",
    value: str | None = None,
    text: str | None = None,
    kind: Literal["statement", "opinion", "observation", "outcome"] = "statement",
    replaces_values: list[str] | None = None,
) -> ClaimDraft:
    return ClaimDraft(
        subject="Participant",
        subject_key="person:42",
        predicate="tool_use",
        value=value or source.text,
        text=text or source.text,
        kind=kind,
        operation=operation,
        replaces_values=replaces_values or [],
        evidence=[SourceQuote(message_id=source.id, quote=source.text)],
    )


class ClaimsVariantTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.workspace = Path(self.directory.name)
        self.variants: list[ClaimsVariant] = []
        self.context = QueryContext("group-a", "2026-02-01T00:00:00+00:00")

    async def asyncTearDown(self) -> None:
        for variant in self.variants:
            await variant.close()
        self.directory.cleanup()

    def variant(
        self, extractor: ClaimExtractor, *, batch_size: int = 80, concurrency: int = 6
    ) -> ClaimsVariant:
        variant = ClaimsVariant(
            self.workspace,
            extractor=extractor,
            batch_size=batch_size,
            concurrency=concurrency,
        )
        self.variants.append(variant)
        return variant

    async def test_scope_source_time_and_author_filter_derived_knowledge(self) -> None:
        sources = [
            message(1, "Participant uses Cedar"),
            message(2, "Participant uses Birch", chat="group-b"),
        ]

        async def extract(batch: Sequence[SourceMessage]) -> ExtractionResult:
            return ExtractionResult(claims=[draft(item) for item in batch])

        variant = self.variant(extract)
        await variant.prepare(sources)
        hits = await variant.search("Participant", self.context)
        self.assertEqual([hit.source_ids for hit in hits], [(1,)])
        at_source = QueryContext("group-a", sources[0].date, before_id=1)
        self.assertEqual(await variant.search("Cedar", at_source), [])
        after_source = QueryContext("group-a", sources[0].date, before_id=2)
        self.assertEqual(len(await variant.search("Cedar", after_source)), 1)
        wrong_author = QueryContext("group-a", "2026-02-01T00:00:00+00:00", author_id=99)
        self.assertEqual(await variant.search("Cedar", wrong_author), [])

    async def test_every_cited_source_must_have_a_valid_exact_quote(self) -> None:
        first = message(1, "Participant likes Cedar")
        second = message(2, "Participant tried Birch")
        forged = draft(first)
        forged.evidence.append(SourceQuote(message_id=second.id, quote="not in source"))
        unknown = draft(first)
        unknown.evidence.append(SourceQuote(message_id=999, quote="unknown"))
        extractor = AsyncMock(
            return_value=ExtractionResult(claims=[forged, unknown, draft(second)])
        )
        variant = self.variant(extractor)
        stats = await variant.prepare([first, second])
        self.assertEqual(stats["claims"], 1)
        hits = await variant.search("Participant", self.context)
        self.assertEqual(hits[0].source_ids, (2,))

    async def test_change_retains_history_correction_replaces_error_uncertainty_keeps_both(
        self,
    ) -> None:
        sources = [
            message(1, "Participant uses Cedar"),
            message(2, "Participant moved to Birch"),
            message(3, "Participant never used either; correct tool is Elm"),
            message(4, "Another supported account says Ash; unclear which is right"),
        ]
        operations: dict[int, Literal["add", "change", "correct", "uncertain"]] = {
            1: "add",
            2: "change",
            3: "correct",
            4: "uncertain",
        }

        async def extract(batch: Sequence[SourceMessage]) -> ExtractionResult:
            return ExtractionResult(
                claims=[
                    draft(
                        item,
                        operation=operations[item.id],
                        replaces_values=["Cedar", "Birch"] if item.id == 3 else [],
                    )
                    for item in batch
                ]
            )

        variant = self.variant(extract, batch_size=1)
        await variant.prepare(sources[:2])
        hits = await variant.search("Participant", self.context)
        self.assertEqual(len(hits), 2)
        self.assertTrue(
            any("historical" in hit.text and "Cedar" in hit.text for hit in hits)
        )
        await variant.prepare(sources)
        hits = await variant.search("Participant account", self.context)
        self.assertEqual({hit.source_ids for hit in hits}, {(3,), (4,)})
        self.assertFalse(any("[statement; historical;" in hit.text for hit in hits))

    async def test_correction_does_not_remove_unrelated_supported_tool_experience(
        self,
    ) -> None:
        sources = [
            message(1, "Participant uses Cedar"),
            message(2, "Participant uses Birch"),
            message(3, "The Cedar attribution was wrong; Participant actually uses Elm"),
        ]

        async def extract(batch: Sequence[SourceMessage]) -> ExtractionResult:
            return ExtractionResult(
                claims=[
                    draft(
                        item,
                        operation="correct" if item.id == 3 else "add",
                        replaces_values=["Cedar"] if item.id == 3 else [],
                    )
                    for item in batch
                ]
            )

        variant = self.variant(extract, batch_size=1)
        await variant.prepare(sources)
        hits = await variant.search("Participant", self.context)
        self.assertEqual({hit.source_ids for hit in hits}, {(2,), (3,)})

    async def test_later_reply_parent_context_cannot_resurrect_a_corrected_claim(
        self,
    ) -> None:
        old = message(1, "Participant uses Cedar")
        correction = message(2, "That was mistaken; Participant uses Elm")
        reply = message(3, "Thank you", parent=1)

        async def extract(batch: Sequence[SourceMessage]) -> ExtractionResult:
            return ExtractionResult(
                claims=[
                    draft(
                        item,
                        operation="correct" if item.id == 2 else "add",
                        replaces_values=["Cedar"] if item.id == 2 else [],
                    )
                    for item in batch
                    if item.id != 3
                ]
            )

        variant = self.variant(extract, batch_size=1)
        await variant.prepare([old, correction, reply])
        hits = await variant.search("Participant", self.context)
        self.assertEqual({hit.source_ids for hit in hits}, {(2,)})

    async def test_same_batch_corrections_follow_source_time_not_model_output_order(
        self,
    ) -> None:
        old = message(1, "Participant uses Cedar")
        correction = message(2, "Participant actually uses Elm")
        extractor = AsyncMock(
            return_value=ExtractionResult(
                claims=[
                    draft(correction, operation="correct", replaces_values=["Cedar"]),
                    draft(old),
                ]
            )
        )
        variant = self.variant(extractor)
        await variant.prepare([old, correction])
        hits = await variant.search("Participant", self.context)
        self.assertEqual({hit.source_ids for hit in hits}, {(2,)})

    async def test_source_edit_and_removal_rebuild_without_retaining_deleted_fact(
        self,
    ) -> None:
        original = message(1, "Participant has 12 cores")
        revised = message(1, "Participant has 16 cores")

        async def extract(batch: Sequence[SourceMessage]) -> ExtractionResult:
            return ExtractionResult(claims=[draft(item) for item in batch])

        variant = self.variant(extract)
        await variant.prepare([original])
        await variant.prepare([revised])
        hits = await variant.search("Participant cores", self.context)
        self.assertEqual(len(hits), 1)
        self.assertIn("16 cores", hits[0].text)
        self.assertNotIn("12 cores", hits[0].text)
        self.assertNotIn("historical", hits[0].text)
        await variant.prepare([])
        self.assertEqual(await variant.search("Participant", self.context), [])

    async def test_consolidation_preserves_independent_source_after_one_is_removed(
        self,
    ) -> None:
        first = message(1, "Participant likes Cedar for tests")
        second = message(2, "Participant likes Cedar for local tests")

        async def extract(batch: Sequence[SourceMessage]) -> ExtractionResult:
            return ExtractionResult(
                claims=[
                    draft(item, value="Cedar for tests", kind="opinion") for item in batch
                ]
            )

        variant = self.variant(extract, batch_size=1)
        await variant.prepare([first, second])
        hits = await variant.search("Cedar", self.context)
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].source_ids, (1, 2))
        self.assertIn("opinion", hits[0].text)
        await variant.prepare([second])
        hits = await variant.search("Cedar", self.context)
        self.assertEqual(hits[0].source_ids, (2,))
        self.assertNotIn("Cedar for tests\n", hits[0].text)

    async def test_completed_batches_resume_after_processing_failure(self) -> None:
        sources = [
            message(1, "Participant uses Cedar"),
            message(2, "Participant uses Elm"),
        ]
        attempts: list[int] = []
        failed = False

        async def extract(batch: Sequence[SourceMessage]) -> ExtractionResult:
            nonlocal failed
            attempts.append(batch[0].id)
            if batch[0].id == 2 and not failed:
                failed = True
                msg = "temporary extraction failure"
                raise RuntimeError(msg)
            return ExtractionResult(claims=[draft(item) for item in batch])

        variant = self.variant(extract, batch_size=1, concurrency=1)
        with self.assertRaisesRegex(RuntimeError, "temporary extraction failure"):
            await variant.prepare(sources)
        stats = await variant.prepare(sources)
        self.assertEqual(attempts, [1, 2, 2])
        self.assertEqual(stats["reused_batches"], 1)
        self.assertEqual(stats["extraction_calls"], 1)
        self.assertEqual(len(await variant.search("Participant", self.context)), 2)

    async def test_bot_content_cannot_establish_facts_and_parents_stay_in_the_same_chat(
        self,
    ) -> None:
        parent = message(1, "Question about Cedar")
        response = message(2, "Participant confirms Cedar", parent=1)
        other_chat = message(3, "Unrelated secret", chat="group-b")
        bot = message(4, "Invented biography", bot=True)
        seen: list[list[tuple[str, int]]] = []

        async def extract(batch: Sequence[SourceMessage]) -> ExtractionResult:
            seen.append([(item.chat_key, item.id) for item in batch])
            return ExtractionResult(claims=[draft(item) for item in batch])

        variant = self.variant(extract, batch_size=1)
        await variant.prepare([parent, response, other_chat, bot])
        self.assertIn([("group-a", 1), ("group-a", 2)], seen)
        self.assertFalse(any(("group-a", 4) in batch for batch in seen))
        self.assertTrue(all(len({chat for chat, _ in batch}) == 1 for batch in seen))

    async def test_reopened_workspace_reuses_extraction_checkpoints(self) -> None:
        source = message(1, "Participant uses Cedar")
        extractor = AsyncMock(return_value=ExtractionResult(claims=[draft(source)]))
        first = self.variant(extractor)
        await first.prepare([source])
        await first.close()
        self.variants.remove(first)
        second = self.variant(extractor)
        stats = await second.prepare([source])
        self.assertEqual(stats["extraction_calls"], 0)
        self.assertEqual(extractor.await_count, 1)
        self.assertEqual((await second.search("Cedar", self.context))[0].source_ids, (1,))

    async def test_long_sources_are_fully_processed_as_bounded_labeled_excerpts(
        self,
    ) -> None:
        source = message(1, 'A long factual description with quotes: "Cedar". ' * 80)
        fragments: list[SourceMessage] = []

        async def extract(batch: Sequence[SourceMessage]) -> ExtractionResult:
            fragments.extend(batch)
            return ExtractionResult(claims=[])

        variant = ClaimsVariant(self.workspace, extractor=extract, batch_characters=700)
        self.variants.append(variant)
        stats = await variant.prepare([source])
        self.assertEqual("".join(item.text for item in fragments), source.text)
        self.assertGreater(len(fragments), 1)
        self.assertTrue(all(item.id == source.id for item in fragments))
        maximum = stats["max_batch_source_characters"]
        assert isinstance(maximum, int)
        self.assertLessEqual(maximum, 700)
        self.assertEqual(stats["sources"], 1)

    async def test_malformed_structured_output_retries_without_losing_supported_claim(
        self,
    ) -> None:
        source = message(1, "Participant uses Cedar")
        with self.assertRaises(ValidationError) as malformed:
            ExtractionResult.model_validate_json('{"claims": [')
        extractor = AsyncMock(
            side_effect=[malformed.exception, ExtractionResult(claims=[draft(source)])]
        )
        variant = self.variant(extractor)
        stats = await variant.prepare([source])
        self.assertEqual(extractor.await_count, 2)
        self.assertEqual(stats["extraction_calls"], 2)
        self.assertEqual(stats["structured_output_failures"], 1)
        self.assertTrue(stats["usage_incomplete_due_to_parse_failures"])
        self.assertEqual(
            (await variant.search("Cedar", self.context))[0].source_ids, (1,)
        )

    async def test_persistently_malformed_window_splits_and_preserves_all_source_coverage(
        self,
    ) -> None:
        sources = [
            message(1, "Participant uses Cedar"),
            message(2, "Participant uses Birch"),
        ]
        with self.assertRaises(ValidationError) as malformed:
            ExtractionResult.model_validate_json('{"claims": [')

        async def extract(batch: Sequence[SourceMessage]) -> ExtractionResult:
            if len(batch) > 1:
                raise malformed.exception
            return ExtractionResult(claims=[draft(batch[0])])

        variant = self.variant(extract)
        stats = await variant.prepare(sources)
        self.assertEqual(stats["extraction_calls"], 5)
        self.assertEqual(stats["structured_output_failures"], 3)
        self.assertEqual(stats["split_batches"], 1)
        self.assertEqual(stats["completed_new_batches"], 1)
        self.assertEqual(
            {hit.source_ids for hit in await variant.search("Participant", self.context)},
            {(1,), (2,)},
        )
        resumed = await variant.prepare(sources)
        self.assertEqual(resumed["extraction_calls"], 0)
        self.assertEqual(resumed["reused_batches"], 1)

    async def test_short_malformed_source_exhausts_bounded_retries_without_false_checkpoint(
        self,
    ) -> None:
        source = message(1, "Participant uses Cedar")
        with self.assertRaises(ValidationError) as malformed:
            ExtractionResult.model_validate_json('{"claims": [')
        extractor = AsyncMock(side_effect=malformed.exception)
        variant = self.variant(extractor)
        with self.assertRaises(ValidationError):
            await variant.prepare([source])
        self.assertEqual(extractor.await_count, 3)
        self.assertEqual(await variant.search("Participant", self.context), [])


class LocalProviderTests(unittest.IsolatedAsyncioTestCase):
    async def test_local_endpoint_check_rejects_remote_hosts_and_cloud_model_routes(
        self,
    ) -> None:
        with patch.dict("os.environ", {"MEMORY_EVAL_LOCAL_ONLY": "1"}):
            for endpoint in (
                None,
                "https://api.openai.com/v1",
                "http://ollama.example/v1",
            ):
                with self.assertRaises(ValueError):
                    validate_local_endpoint(endpoint, "qwen3:4b")
            with self.assertRaises(ValueError):
                validate_local_endpoint("http://ollama:11434/v1", "model:cloud")
            validate_local_endpoint("http://ollama:11434/v1", "qwen3:4b")

    async def test_structured_local_chat_disables_thinking_and_returns_usage(
        self,
    ) -> None:
        client = AsyncMock()
        client.base_url = "http://ollama:11434/v1"
        client.chat.completions.parse.return_value = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(parsed=ExtractionResult(claims=[]))
                )
            ],
            usage=SimpleNamespace(prompt_tokens=11, completion_tokens=3),
        )
        with patch.dict(
            "os.environ", {"MEMORY_EVAL_LOCAL_ONLY": "1", "MEMORY_EVAL_API": "chat"}
        ):
            result, usage = await parse_structured(
                client,
                "qwen3:4b",
                "Read fictional evidence.",
                "Example",
                ExtractionResult,
            )
        self.assertEqual(result.claims, [])
        self.assertEqual(usage["input_tokens"], 11)
        self.assertEqual(usage["output_tokens"], 3)
        self.assertEqual(
            client.chat.completions.parse.call_args.kwargs["reasoning_effort"], "none"
        )
        client.responses.parse.assert_not_awaited()

    async def test_unsafe_local_destination_is_rejected_before_model_call(self) -> None:
        client = AsyncMock()
        client.base_url = "https://api.openai.com/v1"
        with (
            patch.dict(
                "os.environ", {"MEMORY_EVAL_LOCAL_ONLY": "1", "MEMORY_EVAL_API": "chat"}
            ),
            self.assertRaises(ValueError),
        ):
            await parse_structured(
                client, "qwen3:4b", "Instruction", "Example", ExtractionResult
            )
        client.chat.completions.parse.assert_not_awaited()
        client.responses.parse.assert_not_awaited()
