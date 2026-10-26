import hashlib
import json
import subprocess
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from evals.suites.memory.orchestrate import (
    PHASES,
    VARIANTS,
    FrozenMismatchError,
    Orchestrator,
    Variant,
    digest,
    registry_names,
    write_json,
)


class MatrixFixture:
    def __init__(self, repo: Path) -> None:
        self.repo = repo
        self.workspace = repo / "evals/runs/memory"
        self.matrix = self.workspace / "matrix-v3"
        self.frozen = self.matrix / "frozen-code"
        corpus = repo / "evals/datasets/ostrikov/v1/messages.jsonl"
        queries = repo / "evals/datasets/ostrikov/v2/queries.jsonl"
        cases = repo / "evals/datasets/ostrikov/v2/cases.jsonl"
        corpus.parent.mkdir(parents=True)
        queries.parent.mkdir(parents=True)
        corpus.write_text(
            "\n".join(
                json.dumps({
                    "id": i,
                    "chat_key": "fixture",
                    "date": f"2026-01-0{i}T00:00:00+00:00",
                })
                for i in range(1, 5)
            )
            + "\n"
        )
        queries.write_text(
            "\n".join(
                json.dumps({
                    "case_id": f"Q{i:02}",
                    "chat_key": "fixture",
                    "query": "Synthetic request",
                    "history_cutoff": {
                        "date": "2026-01-05T00:00:00+00:00"
                        if i < 49
                        else "2026-01-03T00:00:00+00:00",
                        "message_id": 0,
                    },
                })
                for i in range(50)
            )
            + "\n"
        )
        # Host orchestration hashes these opaque bytes without parsing gold.
        cases.write_text("opaque evaluator-only gold; host must never parse it")
        files = {
            "src/stub.py": "pass\n",
            "evals/suites/memory/runner.py": "VARIANTS = "
            + repr({v.variant: ("unused", "Unused") for v in VARIANTS.values()})
            + "\n",
            "evals/suites/memory/matrix.py": "def register():\n    pass\n",
            "evals/suites/memory/grading.py": "pass\n",
            "deploy/openviking.conf": "{}\n",
        }
        for name, content in files.items():
            path = self.frozen / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
        overlay = repo / "deploy/memory-matrix-frozen.compose.yaml"
        overlay.parent.mkdir(parents=True)
        overlay.write_text("services: {}\n")
        code = hashlib.sha256()
        for path in sorted((self.frozen / "evals/suites/memory").glob("*.py")):
            code.update(path.name.encode())
            code.update(path.read_bytes())
        write_json(
            self.matrix / "snapshot-manifest.json",
            {
                "protocol": "v3",
                "frozen_root": str(self.frozen.relative_to(repo)),
                "files": {n: digest(self.frozen / n) for n in files},
                "code_sha256": code.hexdigest(),
                "corpus_sha256": digest(corpus),
                "queries_sha256": digest(queries),
                "overlay_sha256": digest(overlay),
                "production_config_sha256": digest(
                    self.frozen / "deploy/openviking.conf"
                ),
                "limits": {
                    "answer_concurrency": 4,
                    "tool_budget": 12,
                    "answer_evidence_character_budget": 0,
                },
            },
        )


class FakeCommands:
    def __init__(self, orchestrator: Orchestrator) -> None:
        self.orchestrator = orchestrator
        self.calls: list[list[str]] = []
        self.fail_prepare = False
        self.partial_run = False

    def prepare(self, variant: str, *, frozen: bool = True) -> None:
        o = self.orchestrator
        for key, prefix in o.prefixes.items():
            write_json(
                o.workspace / "indices" / variant / key / "preparation.json",
                {
                    "variant": variant,
                    "scope": prefix["scope"],
                    "source_count": prefix["source_count"],
                    "corpus_sha256": o.frozen["corpus_sha256"],
                    "code_sha256": o.frozen["code_sha256"]
                    if frozen
                    else "old reusable source index",
                },
            )

    def predictions(self, spec: Variant) -> None:
        o = self.orchestrator
        directory = o.workspace / "runs" / spec.run_id
        write_json(
            directory / "manifest.json",
            {
                "run_id": spec.run_id,
                "variant": spec.variant,
                "model": "gpt-6-luna",
                "corpus_sha256": o.frozen["corpus_sha256"],
                "queries_sha256": o.frozen["queries_sha256"],
                "code_sha256": o.frozen["code_sha256"],
                "expected_case_ids": o.case_ids,
                "tool_budget": 12,
                "evidence_character_budget": 0,
            },
        )
        for case_id in o.case_ids[: 49 if self.partial_run else 50]:
            write_json(
                directory / "predictions" / (case_id + ".json"),
                {
                    "case_id": case_id,
                    "variant": spec.variant,
                    "answer": {"text": "Synthetic saved answer"},
                },
            )

    def provenance(self, directory: Path) -> dict[str, str]:
        o = self.orchestrator
        return {
            "corpus_sha256": o.frozen["corpus_sha256"],
            "queries_sha256": o.frozen["queries_sha256"],
            "gold_sha256": o.frozen["gold_sha256"],
            "run_manifest_sha256": digest(directory / "manifest.json"),
            "grading_code_sha256": digest(
                o.frozen_root / "evals/suites/memory/grading.py"
            ),
            "grading_runner_sha256": digest(
                o.frozen_root / "evals/suites/memory/runner.py"
            ),
        }

    def __call__(self, argv: list[str], log: Path) -> int:
        self.calls.append(argv)
        operation = argv[argv.index("evals.suites.memory.matrix") + 1]
        variant = argv[argv.index("--variant") + 1]
        run_id = argv[argv.index("--run-id") + 1]
        spec = next(v for v in VARIANTS.values() if v.variant == variant)
        log.write_text("Synthetic command completed\n", encoding="utf-8")
        if operation == "prepare":
            if self.fail_prepare:
                return 1
            self.prepare(variant)
        elif operation == "run":
            self.predictions(spec)
        else:
            directory = self.orchestrator.workspace / "runs" / run_id
            provenance = self.provenance(directory)
            if operation == "grade":
                for case_id in self.orchestrator.case_ids:
                    write_json(
                        directory / "grades" / (case_id + ".json"),
                        {
                            "case_id": case_id,
                            "prediction_sha256": digest(
                                directory / "predictions" / (case_id + ".json")
                            ),
                            "judge_model": "gpt-6-luna",
                            **provenance,
                        },
                    )
            else:
                accuracy = 0.7 if run_id.endswith("-judge2") else 0.8
                write_json(
                    directory / "summary.json",
                    {
                        "run_id": run_id,
                        "variant": variant,
                        "complete": True,
                        "graded": 50,
                        "total_required": 50,
                        "missing": [],
                        "invalid_results": {},
                        "provenance": provenance,
                        "accuracy_all": accuracy,
                        "accuracy_answerable": accuracy,
                        "threshold_met": True,
                    },
                )
        return 0


class OrchestratorTests(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.repo = Path(directory.name)
        MatrixFixture(self.repo)
        self.orchestrator = Orchestrator(
            self.repo, aliases=["r"], max_jobs=2, scanner=dict, poll_seconds=0.01
        )
        self.commands = FakeCommands(self.orchestrator)
        self.orchestrator.runner = self.commands
        self.commands.prepare("raw_hybrid", frozen=False)

    def test_explicit_matrix_has_all40_frozen_registration_names(self) -> None:
        self.assertEqual(len(VARIANTS), 40)
        self.assertEqual(
            registry_names(self.orchestrator.frozen_root),
            {v.variant for v in VARIANTS.values()},
        )
        with self.assertRaisesRegex(ValueError, "known matrix aliases"):
            Orchestrator(self.repo, aliases=["r; touch injected"], scanner=dict)

    def test_dependencies_require_both_prefixes_and_restored_service_marker(self) -> None:
        spec = VARIANTS["o"]
        self.commands.prepare("openviking")
        self.assertEqual(
            self.orchestrator.readiness(spec), ["answering-config:openviking"]
        )
        marker = self.orchestrator.workspace / "openviking/answering-config-ready.json"
        write_json(
            marker,
            {
                "ready": True,
                "restored": True,
                "max_concurrent": 64,
                "config_sha256": "a" * 64,
            },
        )
        self.assertTrue(self.orchestrator.readiness(spec))
        write_json(
            marker,
            {
                "ready": True,
                "restored": True,
                "max_concurrent": 4,
                "config_sha256": "a" * 64,
            },
        )
        self.assertEqual(self.orchestrator.readiness(spec), [])
        first = next(iter(self.orchestrator.prefixes))
        (
            self.orchestrator.workspace
            / "indices/openviking"
            / first
            / "preparation.json"
        ).unlink()
        self.assertIn("prepare:openviking", self.orchestrator.readiness(spec))
        self.commands.prepare("hindsight_chunks")
        self.assertIn(
            "answering-config:hindsight", self.orchestrator.readiness(VARIANTS["hchunks"])
        )

    def test_gold_mount_exists_only_in_grade_and_report_commands(self) -> None:
        spec = VARIANTS["r"]
        for phase in ("prepare", "run", "grade1", "report1", "grade2", "report2"):
            argv = self.orchestrator.command(spec, phase, 1)
            mounts = [x for x in argv if ":/gold/" in x]
            self.assertEqual(bool(mounts), phase.startswith(("grade", "report")))
            files = [argv[i + 1] for i, part in enumerate(argv) if part == "-f"]
            self.assertTrue(files[-1].endswith("memory-matrix-frozen.compose.yaml"))
            self.assertEqual(argv[argv.index("--concurrency") + 1], "4")
            self.assertIn("--no-deps", argv)

    def test_all_phases_and_full_second_judge_preserve_saved_prediction_bytes(
        self,
    ) -> None:
        self.assertEqual(self.orchestrator.run(once=True), 0)
        spec = VARIANTS["r"]
        original = self.orchestrator.workspace / "runs" / spec.run_id
        copied = original.with_name(original.name + "-judge2")
        for case_id in self.orchestrator.case_ids:
            self.assertEqual(
                (original / "predictions" / (case_id + ".json")).read_bytes(),
                (copied / "predictions" / (case_id + ".json")).read_bytes(),
            )
        self.assertEqual(len(list((original / "grades").glob("*.json"))), 50)
        self.assertEqual(len(list((copied / "grades").glob("*.json"))), 50)
        marker = json.loads((copied / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(marker["evaluation_only_copy_of"], spec.run_id)
        state = self.orchestrator.ledger["variants"]["r"]
        self.assertEqual(set(state["phases"]), set(PHASES))
        summary = json.loads(
            (self.orchestrator.output / "summary.json").read_text(encoding="utf-8")
        )
        row = next(r for r in summary["variants"] if r["alias"] == "r")
        self.assertAlmostEqual(row["mean_accuracy_all"], 0.75)
        self.assertEqual(len(summary["variants"]), 40)
        before = {
            p.name: p.read_bytes() for p in (original / "predictions").glob("*.json")
        }
        self.orchestrator.clone_judge2(spec)
        self.assertEqual(
            before,
            {p.name: p.read_bytes() for p in (original / "predictions").glob("*.json")},
        )

    def test_partial_predictions_never_pass_despite_successful_process_exit(self) -> None:
        self.commands.partial_run = True
        self.assertEqual(self.orchestrator.run(once=True), 1)
        state = self.orchestrator.ledger["variants"]["r"]
        self.assertEqual(state["state"], "failed")
        self.assertEqual(state["phases"]["run"]["attempts"], 2)
        self.assertNotIn("grade1", state["phases"])
        operations = [
            a[a.index("evals.suites.memory.matrix") + 1] for a in self.commands.calls
        ]
        self.assertEqual(operations, ["prepare", "run", "run"])

    def test_prepare_failure_remains_visible_with_null_score_not_zero(self) -> None:
        self.commands.fail_prepare = True
        self.assertEqual(self.orchestrator.run(once=True), 1)
        state = self.orchestrator.ledger["variants"]["r"]
        self.assertEqual(state["phases"]["prepare"]["attempts"], 2)
        row = next(
            x
            for x in json.loads(
                (self.orchestrator.output / "summary.json").read_text(encoding="utf-8")
            )["variants"]
            if x["alias"] == "r"
        )
        self.assertIsNone(row["mean_accuracy_all"])
        self.assertEqual(row["state"], "failed")

    def test_complete_existing_manual_run_is_recovered_without_regeneration(self) -> None:
        self.assertEqual(self.orchestrator.run(once=True), 0)

        def forbidden(_argv: list[str], _log: Path) -> int:
            self.fail("Completed cached phases must not be regenerated")

        resumed = Orchestrator(
            self.repo,
            aliases=["r"],
            runner=forbidden,
            scanner=dict,
            poll_seconds=0.01,
        )
        self.assertEqual(resumed.run(once=True), 0)

    def test_frozen_mismatch_is_fatal_in_ledger_without_overwriting_old_run(self) -> None:
        self.commands.predictions(VARIANTS["r"])
        path = (
            self.orchestrator.workspace / "runs" / VARIANTS["r"].run_id / "manifest.json"
        )
        data = json.loads(path.read_text(encoding="utf-8"))
        data["code_sha256"] = "different"
        write_json(path, data)
        self.commands.prepare("raw_hybrid")
        self.assertEqual(self.orchestrator.run(once=True), 1)
        self.assertEqual(self.commands.calls, [])
        self.assertEqual(
            json.loads(path.read_text(encoding="utf-8"))["code_sha256"], "different"
        )
        self.assertTrue(
            self.orchestrator.ledger["variants"]["r"]["phases"]["validation"]["fatal"]
        )

    def test_grades_and_summary_must_be_complete_and_provenance_current(self) -> None:
        self.assertEqual(self.orchestrator.run(once=True), 0)
        spec = VARIANTS["r"]
        directory = self.orchestrator.workspace / "runs" / spec.run_id
        path = directory / "grades" / (self.orchestrator.case_ids[0] + ".json")
        grade = json.loads(path.read_text(encoding="utf-8"))
        grade["prediction_sha256"] = "stale"
        write_json(path, grade)
        self.assertFalse(self.orchestrator.phase_complete(spec, "grade1"))
        self.assertFalse(self.orchestrator.phase_complete(spec, "report1"))

    def test_command_runner_never_uses_a_shell(self) -> None:
        log = self.orchestrator.output / "test.log"
        with patch(
            "evals.suites.memory.orchestrate.subprocess.run",
            return_value=subprocess.CompletedProcess([], 0),
        ) as run:
            self.assertEqual(
                self.orchestrator._command(["docker", "compose", "version"], log), 0
            )
        self.assertIs(run.call_args.kwargs["shell"], False)
        self.assertIsInstance(run.call_args.args[0], list)

    def test_snapshot_hash_change_is_rejected_before_new_jobs(self) -> None:
        path = self.orchestrator.frozen_root / "src/stub.py"
        path.write_text("changed\n")
        with self.assertRaisesRegex(FrozenMismatchError, "snapshot changed"):
            self.orchestrator.run(once=True)
        self.assertEqual(self.commands.calls, [])

    def test_active_manual_generation_is_adopted_then_only_failed_cache_is_retried(
        self,
    ) -> None:
        self.commands.prepare("raw_hybrid")
        seen = []

        def scanner() -> dict[tuple[str, str], str]:
            seen.append(True)
            return {(VARIANTS["r"].run_id, "run"): "manual-r"} if len(seen) <= 2 else {}

        self.orchestrator.scanner = scanner
        self.assertEqual(self.orchestrator.run(once=True), 0)
        state = self.orchestrator.ledger["variants"]["r"]
        self.assertEqual(state["external_container"], "manual-r")
        self.assertEqual(state["phases"]["run"]["attempts"], 2)
        generated = [
            a
            for a in self.commands.calls
            if a[a.index("evals.suites.memory.matrix") + 1] == "run"
        ]
        self.assertEqual(len(generated), 1)

    def test_waiting_backend_is_preserved_without_starting_any_phase(self) -> None:
        self.orchestrator.aliases = ["o"]
        self.assertEqual(self.orchestrator.run(once=True), 2)
        self.assertEqual(self.commands.calls, [])
        state = self.orchestrator.ledger["variants"]["o"]
        self.assertEqual(state["state"], "waiting")
        self.assertIn("answering-config:openviking", state["wait_reasons"])

    def test_parallel_phase_limit_is_respected(self) -> None:
        self.orchestrator.aliases = ["r", "r-d", "rawlex"]
        self.commands.prepare("raw_lexical", frozen=False)
        active = 0
        maximum = 0
        lock = threading.Lock()

        def run(argv: list[str], log: Path) -> int:
            nonlocal active, maximum
            with lock:
                active += 1
                maximum = max(maximum, active)
            try:
                time.sleep(0.01)
                return self.commands(argv, log)
            finally:
                with lock:
                    active -= 1

        self.orchestrator.runner = run
        self.assertEqual(self.orchestrator.run(once=True), 0)
        self.assertGreater(maximum, 1)
        self.assertLessEqual(maximum, 2)


if __name__ == "__main__":
    unittest.main()
