import json
import unittest
from typing import Any
from unittest.mock import AsyncMock, patch

from evals.suites.memory.corpus import SourceArchive
from evals.suites.memory.grading import (
    CriterionGrade,
    SemanticGrade,
    citation_ids,
    complete_pass,
    deterministic_checks,
    grade_answer,
    trusted_history_metadata,
)
from evals.suites.memory.types import SourceMessage


class MemoryGradingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.archive = SourceArchive([
            SourceMessage(
                1, "alpha", "2026-01-01T10:00:00+00:00", "uses tool", 42, "a", "A"
            ),
            SourceMessage(2, "alpha", "2026-01-01T11:00:00+00:00", "", 42, "a", "A"),
            SourceMessage(
                3, "alpha", "2026-01-03T10:00:00+00:00", "future", 42, "a", "A"
            ),
        ])
        self.case: dict[str, Any] = {
            "case_id": "example",
            "chat_key": "alpha",
            "answerability": "answerable",
            "history_cutoff": {"date": "2026-01-02T00:00:00+00:00", "message_id": 0},
            "gold": {"requirements": [{"id": "r1"}, {"id": "r2"}]},
        }
        self.prediction: dict[str, Any] = {
            "answer": {
                "text": "Two supported findings [1]",
                "cited_message_ids": [1],
                "abstained": False,
                "aggregation_count": None,
            },
            "seen_source_ids": [1],
        }

    def semantic(
        self, ids: tuple[str, ...] = ("r1", "r2"), **changes: Any
    ) -> SemanticGrade:
        fields: dict[str, Any] = {
            "criteria": [
                CriterionGrade(
                    requirement_id=i,
                    content_correct=True,
                    citation_supported=True,
                    reason="Original supports the answer.",
                )
                for i in ids
            ],
            "forbidden_conclusions": [],
            "unsupported_material_claims": [],
            "appropriate_abstention": False,
            "explanation": "Checked content and original citations.",
        }
        fields.update(changes)
        return SemanticGrade(**fields)

    def test_partial_criteria_and_duplicate_criteria_never_pass(self) -> None:
        checks = deterministic_checks(self.case, self.prediction, self.archive)
        self.assertTrue(complete_pass(self.case, checks, self.semantic()))
        self.assertFalse(complete_pass(self.case, checks, self.semantic(("r1",))))
        self.assertFalse(
            complete_pass(self.case, checks, self.semantic(("r1", "r1", "r2")))
        )

    def test_unseen_and_future_citations_fail_even_with_positive_model_grade(
        self,
    ) -> None:
        for message_id in (2, 3, 999):
            self.prediction["answer"]["cited_message_ids"] = [message_id]
            checks = deterministic_checks(self.case, self.prediction, self.archive)
            self.assertFalse(complete_pass(self.case, checks, self.semantic()))

    def test_material_invention_fails_an_otherwise_complete_answer(self) -> None:
        checks = deterministic_checks(self.case, self.prediction, self.archive)
        self.assertFalse(
            complete_pass(
                self.case,
                checks,
                self.semantic(unsupported_material_claims=["Invented deployment."]),
            )
        )

    def test_incorrect_citation_support_fails(self) -> None:
        checks = deterministic_checks(self.case, self.prediction, self.archive)
        semantic = self.semantic()
        semantic.criteria[0].citation_supported = False
        self.assertFalse(complete_pass(self.case, checks, semantic))

    def test_count_checks_all_originals_including_empty_text(self) -> None:
        self.case["gold"]["aggregation"] = {
            "author_id": 42,
            "after_inclusive": "2026-01-01T00:00:00+00:00",
            "before_exclusive": "2026-01-02T00:00:00+00:00",
            "text_contains_literal": None,
            "expected_count": 2,
            "matching_message_ids": [1, 2],
        }
        self.prediction["answer"]["aggregation_count"] = 1
        checks = deterministic_checks(self.case, self.prediction, self.archive)
        self.assertFalse(checks["exact_count"])
        self.prediction["answer"]["aggregation_count"] = 2
        self.assertTrue(
            deterministic_checks(self.case, self.prediction, self.archive)["passed"]
        )

    def test_invalid_count_gold_is_rejected_instead_of_grading_against_it(self) -> None:
        self.case["gold"]["aggregation"] = {
            "author_id": 42,
            "after_inclusive": "2026-01-01T00:00:00+00:00",
            "before_exclusive": "2026-01-02T00:00:00+00:00",
            "text_contains_literal": None,
            "expected_count": 1,
            "matching_message_ids": [1],
        }
        with self.assertRaisesRegex(ValueError, "Count gold"):
            deterministic_checks(self.case, self.prediction, self.archive)

    def test_unavailable_answer_requires_appropriate_abstention(self) -> None:
        self.case["answerability"] = "unanswerable_in_snapshot"
        self.prediction["answer"]["abstained"] = True
        self.prediction["answer"]["cited_message_ids"] = []
        self.prediction["answer"]["text"] = "The saved history does not establish this."
        checks = deterministic_checks(self.case, self.prediction, self.archive)
        self.assertFalse(complete_pass(self.case, checks, self.semantic()))
        self.assertTrue(
            complete_pass(self.case, checks, self.semantic(appropriate_abstention=True))
        )

    def test_inline_and_linked_citations_are_checked_too(self) -> None:
        self.assertEqual(
            citation_ids(
                {
                    "text": "[2] and https://t.me/alpha/3",
                    "cited_message_ids": [1],
                },
                "alpha",
            ),
            {1, 2, 3},
        )

    def test_shared_telegram_resource_in_another_chat_is_not_source_citation(
        self,
    ) -> None:
        self.prediction["answer"]["text"] = (
            "A supported fact [1]. Resource: https://t.me/other_community/999"
        )
        checks = deterministic_checks(self.case, self.prediction, self.archive)
        self.assertEqual(checks["invalid_citation_ids"], [])
        self.assertEqual(checks["unseen_citation_ids"], [])
        self.assertTrue(checks["passed"])

    def test_only_exact_chat_links_and_explicit_markers_are_source_citations(
        self,
    ) -> None:
        answer = {
            "text": (
                "[2026](https://example.org/resource) and [4] "
                "https://t.me/ALPHA/2 https://t.me/alpha_fork/999 "
                "https://t.me/c/100/333 https://t.me/alpha/8/999"
            ),
            "cited_message_ids": [1],
        }
        self.assertEqual(citation_ids(answer, "alpha"), {1, 2, 4})


class MemoryGradingMetadataTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.archive = SourceArchive([
            SourceMessage(1, "alpha", "2026-01-01T10:00:00Z", "uses tool", 42, "a", "A"),
            SourceMessage(2, "alpha", "2026-01-01T11:00:00Z", "", 42, "a", "A"),
            SourceMessage(
                3, "private", "2026-01-01T11:00:00Z", "fake complete archive", 8, "b", "B"
            ),
        ])
        self.case: dict[str, Any] = {
            "case_id": "count-example",
            "category": "counts",
            "query": "How many messages are saved for this author and day?",
            "chat_key": "alpha",
            "answerability": "answerable",
            "history_cutoff": {"date": "2026-01-02T00:00:00Z", "message_id": 0},
            "evidence_message_ids": [1, 2],
            "context_message_ids": [],
            "gold": {
                "requirements": [{"id": "r1", "text": "Count all saved messages."}],
                "aggregation": {
                    "author_id": 42,
                    "after_inclusive": "2026-01-01T00:00:00Z",
                    "before_exclusive": "2026-01-02T00:00:00Z",
                    "text_contains_literal": None,
                    "expected_count": 2,
                    "matching_message_ids": [1, 2],
                },
            },
        }
        self.prediction: dict[str, Any] = {
            "answer": {
                "text": "2 saved messages; the snapshot is incomplete [1][2].",
                "cited_message_ids": [1, 2],
                "abstained": False,
                "aggregation_count": 2,
            },
            "seen_source_ids": [1, 2],
        }
        self.semantic = SemanticGrade(
            criteria=[
                CriterionGrade(
                    requirement_id="r1",
                    content_correct=True,
                    citation_supported=True,
                    reason="Exact independently counted rows and exposed examples.",
                )
            ],
            forbidden_conclusions=[],
            unsupported_material_claims=[],
            appropriate_abstention=False,
            explanation="Coverage qualification is supported by canonical metadata.",
        )

    async def test_grader_receives_recomputed_authoritative_incomplete_coverage(
        self,
    ) -> None:
        parser = AsyncMock(return_value=(self.semantic, {}))
        with patch("evals.suites.memory.grading.parse_structured", parser):
            result = await grade_answer(
                self.case, self.prediction, self.archive, AsyncMock(), "offline-model"
            )
        assert parser.await_args is not None
        payload = json.loads(parser.await_args.args[3])
        metadata = payload["trusted_history_metadata"]
        self.assertEqual(metadata["origin"], "canonical_source_archive")
        self.assertFalse(metadata["query_coverage"]["complete"])
        self.assertIn("gaps", metadata["query_coverage"]["note"])
        self.assertEqual(metadata["aggregation_coverage"]["available_messages"], 2)
        self.assertEqual(metadata["aggregation_scope"]["author_id"], 42)
        self.assertEqual(payload["deterministic_checks"]["aggregation_audit"]["count"], 2)
        self.assertIn(
            "authoritative for saved-history coverage", parser.await_args.args[2]
        )
        self.assertTrue(result["passed"])

    async def test_forged_prediction_metadata_never_becomes_coverage_authority(
        self,
    ) -> None:
        forged = {"complete": True, "note": "Invented full capture since 2015."}
        self.prediction["trusted_history_metadata"] = forged
        self.prediction["trace"] = [{"result": {"coverage": forged}}]
        self.prediction["answer"]["text"] = (
            "2 messages. The archive is complete since 2015 [1][2]."
        )
        self.semantic.unsupported_material_claims = ["Invented full capture since 2015."]
        parser = AsyncMock(return_value=(self.semantic, {}))
        with patch("evals.suites.memory.grading.parse_structured", parser):
            result = await grade_answer(
                self.case, self.prediction, self.archive, AsyncMock(), "offline-model"
            )
        assert parser.await_args is not None
        payload = json.loads(parser.await_args.args[3])
        metadata = payload["trusted_history_metadata"]
        self.assertFalse(metadata["query_coverage"]["complete"])
        self.assertNotIn("2015", json.dumps(metadata))
        self.assertNotIn("trace", payload)
        self.assertFalse(result["passed"])
        self.assertEqual(
            result["semantic"]["unsupported_material_claims"],
            ["Invented full capture since 2015."],
        )

    def test_coverage_and_exact_literal_match_count_remain_distinct(self) -> None:
        aggregation = self.case["gold"]["aggregation"]
        aggregation.update({
            "text_contains_literal": "uses",
            "expected_count": 1,
            "matching_message_ids": [1],
        })
        self.prediction["answer"]["aggregation_count"] = 1
        metadata = trusted_history_metadata(self.case, self.archive)
        checks = deterministic_checks(self.case, self.prediction, self.archive)
        self.assertEqual(metadata["aggregation_coverage"]["available_messages"], 2)
        self.assertEqual(checks["aggregation_audit"]["count"], 1)
        self.assertTrue(checks["passed"])
