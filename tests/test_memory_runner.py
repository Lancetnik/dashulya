import argparse
import asyncio
import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Self
from unittest.mock import AsyncMock, patch

from evals.suites.memory import runner
from evals.suites.memory.grading import (
    CriterionGrade,
    SemanticGrade,
    deterministic_checks,
)


class OfflineClient:
    def __init__(self, **kwargs: Any) -> None:
        pass

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *args: object) -> None:
        pass


class MemoryRunnerTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.output = self.root / "runs" / "synthetic"
        self.prediction_path = self.output / "predictions" / "Q1.json"
        self.grade_path = self.output / "grades" / "Q1.json"
        self.manifest_path = self.output / "manifest.json"
        self.corpus = self.root / "corpus.jsonl"
        self.queries = self.root / "queries.jsonl"
        self.cases = self.root / "cases.jsonl"
        code = self.root / "code"
        code.mkdir()
        self.grader_code = code / "grading.py"
        self.runner_code = code / "runner.py"
        self.grader_code.write_bytes(
            Path(runner.__file__).with_name("grading.py").read_bytes()
        )
        self.runner_code.write_bytes(Path(runner.__file__).read_bytes())
        module_path = patch.object(runner, "__file__", str(self.runner_code))
        module_path.start()
        self.addCleanup(module_path.stop)
        environment = patch.dict(os.environ, {"MEMORY_EVAL_JUDGE_MODEL": "offline-judge"})
        environment.start()
        self.addCleanup(environment.stop)
        self.query: dict[str, Any] = {
            "case_id": "Q1",
            "query": "What did Alice report?",
            "chat_key": "group",
            "history_cutoff": {"date": "2026-01-03T00:00:00Z", "message_id": 0},
        }
        self.case = self.query | {
            "category": "people",
            "answerability": "answerable",
            "evidence_message_ids": [1],
            "context_message_ids": [],
            "gold": {
                "requirements": [{"id": "r1", "text": "Report the supported fact."}],
                "expected_answer": "Alice uses the tool.",
                "forbidden_claims": [],
            },
        }
        self._jsonl(
            self.corpus,
            [
                {
                    "id": 1,
                    "chat_key": "group",
                    "date": "2026-01-01T00:00:00Z",
                    "text": "I use the tool.",
                    "user_id": 10,
                    "user_name": "Alice",
                }
            ],
        )
        self._jsonl(self.queries, [self.query])
        self._jsonl(self.cases, [self.case])
        self.args = argparse.Namespace(
            corpus=self.corpus,
            queries=self.queries,
            cases=self.cases,
            workspace=self.root,
            run_id="synthetic",
            variant="raw_lexical",
            concurrency=1,
        )
        self._manifest()
        self.prediction: dict[str, Any] = {
            "case_id": "Q1",
            "variant": "raw_lexical",
            "answer": {
                "text": "Alice uses the tool [1].",
                "cited_message_ids": [1],
                "abstained": False,
                "aggregation_count": None,
            },
            "seen_source_ids": [1],
            "trace": [],
            "elapsed_seconds": 1.0,
        }
        runner.write_json(self.prediction_path, self.prediction)

    @staticmethod
    def _jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    def _manifest(self) -> None:
        runner.write_json(
            self.manifest_path,
            {
                "run_id": "synthetic",
                "variant": "raw_lexical",
                "model": "offline-answerer",
                "code_sha256": "original-answerer-code",
                "corpus_sha256": runner.fingerprint(self.corpus),
                "queries_sha256": runner.fingerprint(self.queries),
                "expected_case_ids": [
                    q["case_id"] for q in runner.read_jsonl(self.queries)
                ],
            },
        )

    def _result(self, *, content_correct: bool = True) -> dict[str, Any]:
        captured = runner.evaluation_inputs(self.args)
        semantic = SemanticGrade(
            criteria=[
                CriterionGrade(
                    requirement_id="r1",
                    content_correct=content_correct,
                    citation_supported=True,
                    reason="Synthetic source check.",
                )
            ],
            forbidden_conclusions=[],
            unsupported_material_claims=[],
            appropriate_abstention=False,
            explanation="Offline fixture, no model call.",
        )
        return {
            "case_id": "Q1",
            "category": "people",
            "answerability": "answerable",
            "passed": True,
            "checks": deterministic_checks(self.case, self.prediction, captured.archive),
            "semantic": semantic.model_dump(),
            "judge_model": "offline-judge",
            "usage": {},
        }

    def _cache_grade(self, *, content_correct: bool = True) -> dict[str, Any]:
        captured = runner.evaluation_inputs(self.args)
        result = (
            self._result(content_correct=content_correct)
            | captured.hashes
            | {"prediction_sha256": runner.fingerprint(self.prediction_path)}
        )
        runner.write_json(self.grade_path, result)
        return result

    def _report(self) -> dict[str, Any]:
        runner.report(self.args)
        summary: dict[str, Any] = json.loads((self.output / "summary.json").read_text())
        return summary

    async def test_grading_hashes_describe_the_exact_unchanged_judged_files(self) -> None:
        captured = runner.evaluation_inputs(self.args)
        expected = captured.hashes | {
            "prediction_sha256": runner.fingerprint(self.prediction_path)
        }
        grader = AsyncMock(return_value=self._result())
        with (
            patch.object(runner, "AsyncOpenAI", OfflineClient),
            patch.object(runner, "grade_answer", grader),
        ):
            await runner.grade(self.args)
        result = json.loads(self.grade_path.read_text())
        for name, digest in expected.items():
            self.assertEqual(result[name], digest)
        assert grader.await_args is not None
        self.assertEqual(grader.await_args.args[1], self.prediction)
        self.assertTrue(self._report()["threshold_met"])

    async def test_prediction_replacement_during_judging_never_receives_old_pass(
        self,
    ) -> None:
        result = self._result()

        async def changing_judge(*args: Any) -> dict[str, Any]:
            runner.write_json(
                self.prediction_path, self.prediction | {"error": "new failure"}
            )
            return result

        with (
            patch.object(runner, "AsyncOpenAI", OfflineClient),
            patch.object(runner, "grade_answer", changing_judge),
            self.assertRaisesRegex(ValueError, "Prediction changed"),
        ):
            await runner.grade(self.args)
        self.assertFalse(self.grade_path.exists())

    async def test_changed_gold_grader_or_run_manifest_aborts_pending_grade(self) -> None:
        result = self._result()
        for artifact in (
            self.cases,
            self.grader_code,
            self.runner_code,
            self.manifest_path,
        ):
            with self.subTest(artifact=artifact.name):
                original = artifact.read_bytes()

                async def changing_judge(
                    *args: Any, artifact: Path = artifact, original: bytes = original
                ) -> dict[str, Any]:
                    await asyncio.to_thread(artifact.write_bytes, original + b"\n")
                    return result

                try:
                    with (
                        patch.object(runner, "AsyncOpenAI", OfflineClient),
                        patch.object(runner, "grade_answer", changing_judge),
                        self.assertRaisesRegex(ValueError, "Evaluation input changed"),
                    ):
                        await runner.grade(self.args)
                    self.assertFalse(self.grade_path.exists())
                finally:
                    artifact.write_bytes(original)

    async def test_current_grade_is_reused_without_a_judge_call(self) -> None:
        self._cache_grade()
        grader = AsyncMock()
        with (
            patch.object(runner, "AsyncOpenAI", OfflineClient),
            patch.object(runner, "grade_answer", grader),
        ):
            await runner.grade(self.args)
        grader.assert_not_awaited()

    def test_failed_current_prediction_cannot_reuse_an_old_passing_grade(self) -> None:
        self._cache_grade()
        runner.write_json(
            self.prediction_path, self.prediction | {"error": "CurrentRunFailed"}
        )
        summary = self._report()
        self.assertEqual(summary["passed"], 0)
        self.assertFalse(summary["complete"])
        self.assertFalse(summary["threshold_met"])
        self.assertEqual(summary["missing"], ["Q1"])

    def test_failed_prediction_without_timing_still_counts_against_accuracy(self) -> None:
        self._cache_grade()
        runner.write_json(
            self.prediction_path,
            {"case_id": "Q1", "variant": "raw_lexical", "error": "TimeoutError"},
        )
        summary = self._report()
        self.assertEqual(summary["total_required"], 1)
        self.assertEqual(summary["accuracy_all"], 0)
        self.assertFalse(summary["complete"])
        self.assertFalse(summary["threshold_met"])
        self.assertEqual(summary["latency_measured"], 0)
        self.assertEqual(summary["latency_missing"], ["Q1"])
        self.assertIsNone(summary["median_answer_seconds"])

    def test_changed_prediction_or_gold_invalidates_cached_grade(self) -> None:
        for artifact in (self.prediction_path, self.cases):
            with self.subTest(artifact=artifact.name):
                self._cache_grade()
                original = artifact.read_bytes()
                try:
                    artifact.write_bytes(original + b"\n")
                    summary = self._report()
                    self.assertEqual(summary["passed"], 0)
                    self.assertFalse(summary["complete"])
                    self.assertFalse(summary["threshold_met"])
                finally:
                    artifact.write_bytes(original)

    def test_changed_grader_or_manifest_invalidates_cached_grade(self) -> None:
        for artifact in (self.grader_code, self.manifest_path):
            with self.subTest(artifact=artifact.name):
                self._cache_grade()
                original = artifact.read_bytes()
                try:
                    artifact.write_bytes(original + b"\n")
                    summary = self._report()
                    self.assertEqual(summary["graded"], 0)
                    self.assertFalse(summary["threshold_met"])
                finally:
                    artifact.write_bytes(original)

    def test_report_recomputes_semantic_and_deterministic_pass_decisions(self) -> None:
        self._cache_grade(content_correct=False)
        summary = self._report()
        self.assertTrue(summary["complete"])
        self.assertEqual(summary["passed"], 0)
        self.assertFalse(summary["threshold_met"])
        result = self._cache_grade()
        self.prediction["answer"]["cited_message_ids"] = [999]
        runner.write_json(self.prediction_path, self.prediction)
        result["prediction_sha256"] = runner.fingerprint(self.prediction_path)
        runner.write_json(self.grade_path, result)
        self.assertEqual(self._report()["passed"], 0)

    def test_partial_report_keeps_all_answerable_cases_in_denominator(self) -> None:
        second_query = self.query | {"case_id": "Q2", "query": "Another question?"}
        second_case = self.case | second_query
        self._jsonl(self.queries, [self.query, second_query])
        self._jsonl(self.cases, [self.case, second_case])
        self._manifest()
        self._cache_grade()
        summary = self._report()
        self.assertEqual(summary["factual_required"], 2)
        self.assertEqual(summary["factual_graded"], 1)
        self.assertEqual(summary["accuracy_answerable"], 0.5)
        self.assertFalse(summary["complete"])
        self.assertFalse(summary["threshold_met"])

    def test_manifest_must_match_current_corpus_questions_and_variant(self) -> None:
        for name in ("corpus_sha256", "queries_sha256", "expected_case_ids", "variant"):
            with self.subTest(name=name):
                original = self.manifest_path.read_bytes()
                manifest = json.loads(original)
                manifest[name] = "different"
                try:
                    runner.write_json(self.manifest_path, manifest)
                    with self.assertRaisesRegex(ValueError, "Run manifest differs"):
                        self._report()
                finally:
                    self.manifest_path.write_bytes(original)

    async def test_gold_question_and_scope_must_match_answerer_inputs_before_api(
        self,
    ) -> None:
        for field, replacement in (
            ("query", "A different question"),
            ("chat_key", "private"),
            ("history_cutoff", {"date": "2027-01-01T00:00:00Z", "message_id": 0}),
        ):
            with self.subTest(field=field):
                self._jsonl(self.cases, [self.case | {field: replacement}])
                client = AsyncMock()
                with (
                    patch.object(runner, "AsyncOpenAI", client),
                    self.assertRaisesRegex(ValueError, "Gold question or source scope"),
                ):
                    await runner.grade(self.args)
                client.assert_not_called()
        self._jsonl(self.cases, [self.case])

    def test_duplicate_or_missing_gold_case_never_silently_reduces_benchmark(
        self,
    ) -> None:
        for rows in ([self.case, self.case], []):
            with self.subTest(count=len(rows)):
                self._jsonl(self.cases, rows)
                with self.assertRaisesRegex(ValueError, "exactly one case"):
                    self._report()
