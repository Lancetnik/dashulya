"""Host-only, resumable Docker orchestration for the frozen forty-variant matrix.

This module deliberately imports only the standard library. It never interprets
gold requirements or evaluates answer content; the frozen grader owns scoring.
"""

import argparse
import ast
import fcntl
import hashlib
import json
import statistics
import subprocess  # ruff: ignore[suspicious-subprocess-import] -- This host tool intentionally invokes Docker with argv and shell=False.
import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# ruff: file-ignore[subprocess-without-shell-equals-true, start-process-with-partial-path] -- Fixed Docker CLI operations; aliases are allowlisted and no shell is used.


@dataclass(frozen=True)
class Variant:
    alias: str
    variant: str
    dependencies: tuple[str, ...]

    @property
    def run_id(self) -> str:
        return f"{self.alias}-gpt6luna-v3"


# Read from matrix.py's explicit component_names/component_name declarations.
# Order is stable and independent of measurements, gold or model verdicts.
_MAIN = (
    ("r", "raw_hybrid", "raw_hybrid_decisions", ("raw_hybrid",)),
    ("c", "claims_dense", "claims_dense_decisions", ("raw_hybrid", "claims")),
    ("o", "openviking", "openviking_decisions", ("openviking",)),
    ("h", "hindsight_concise", "hindsight_concise_decisions", ("hindsight_concise",)),
    (
        "rc",
        "raw_claims_dense_rrf",
        "raw_claims_dense_decisions",
        ("raw_hybrid", "claims"),
    ),
    (
        "ro",
        "raw_openviking_rrf",
        "raw_openviking_decisions",
        ("raw_hybrid", "openviking"),
    ),
    (
        "rh",
        "raw_hindsight_concise_rrf",
        "raw_hindsight_concise_decisions",
        ("raw_hybrid", "hindsight_concise"),
    ),
    (
        "co",
        "claims_openviking_rrf",
        "claims_openviking_decisions",
        ("raw_hybrid", "claims", "openviking"),
    ),
    (
        "ch",
        "claims_hindsight_concise_rrf",
        "claims_hindsight_concise_decisions",
        ("raw_hybrid", "claims", "hindsight_concise"),
    ),
    (
        "oh",
        "openviking_hindsight_concise_rrf",
        "openviking_hindsight_concise_decisions",
        ("openviking", "hindsight_concise"),
    ),
    (
        "rco",
        "raw_claims_dense_openviking_rrf",
        "raw_claims_dense_openviking_decisions",
        ("raw_hybrid", "claims", "openviking"),
    ),
    (
        "rch",
        "raw_claims_hindsight_concise_rrf",
        "raw_claims_hindsight_concise_decisions",
        ("raw_hybrid", "claims", "hindsight_concise"),
    ),
    (
        "roh",
        "raw_openviking_hindsight_concise_rrf",
        "raw_openviking_hindsight_concise_decisions",
        ("raw_hybrid", "openviking", "hindsight_concise"),
    ),
    (
        "coh",
        "claims_openviking_hindsight_concise_rrf",
        "claims_openviking_hindsight_concise_decisions",
        ("raw_hybrid", "claims", "openviking", "hindsight_concise"),
    ),
    (
        "rcoh",
        "all_components_rrf",
        "all_components_decisions",
        ("raw_hybrid", "claims", "openviking", "hindsight_concise"),
    ),
)
VARIANTS = {
    spec.alias: spec
    for alias, plain, ranked, dependencies in _MAIN
    for spec in (
        Variant(alias, plain, dependencies),
        Variant(alias + "-d", ranked, dependencies),
    )
}
VARIANTS.update({
    spec.alias: spec
    for spec in (
        Variant("rawlex", "raw_lexical", ("raw_lexical",)),
        Variant("claimslex", "claims", ("claims",)),
        Variant("rawlex-d", "raw_lexical_decisions", ("raw_lexical",)),
        Variant("r-claimslex", "raw_claims_rrf", ("raw_hybrid", "claims")),
        Variant("hchunks", "hindsight_chunks", ("hindsight_chunks",)),
        Variant("hchunks-d", "hindsight_chunks_decisions", ("hindsight_chunks",)),
        Variant(
            "r-hchunks", "raw_hindsight_chunks_rrf", ("raw_hybrid", "hindsight_chunks")
        ),
        Variant(
            "r-hchunks-d",
            "raw_hindsight_chunks_decisions",
            ("raw_hybrid", "hindsight_chunks"),
        ),
        Variant("hverbose", "hindsight", ("hindsight",)),
        Variant("hverbose-d", "hindsight_decisions", ("hindsight",)),
    )
})
PHASES = ("prepare", "run", "grade1", "report1", "clone_judge2", "grade2", "report2")
CommandRunner = Callable[[list[str], Path], int]
ActiveScanner = Callable[[], dict[tuple[str, str], str]]


class FrozenMismatchError(ValueError):
    """An existing result cannot be mixed with the frozen experiment inputs."""


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return result


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def registry_names(root: Path) -> set[str]:
    """Inspect registrations without importing the model-dependent modules."""
    names: set[str] = set()
    for name in ("runner.py", "matrix.py"):
        tree = ast.parse(
            (root / "evals/suites/memory" / name).read_text(encoding="utf-8")
        )
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "VARIANTS"
                for target in node.targets
            ):
                names.update(ast.literal_eval(node.value))
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr != "update" or not node.args:
                continue
            target = node.func.value
            if isinstance(target, ast.Attribute) and target.attr == "VARIANTS":
                names.update(ast.literal_eval(node.args[0]))
    return names


class Orchestrator:
    def __init__(
        self,
        repo: Path,
        *,
        aliases: Sequence[str] | None = None,
        max_jobs: int = 2,
        runner: CommandRunner | None = None,
        scanner: ActiveScanner | None = None,
        poll_seconds: float = 10,
    ) -> None:
        if not 1 <= max_jobs <= 3:
            error = "max_jobs must be between one and three."
            raise ValueError(error)
        requested = list(VARIANTS) if aliases is None else list(aliases)
        if (
            not requested
            or len(set(requested)) != len(requested)
            or set(requested) - VARIANTS.keys()
        ):
            error = "Select unique known matrix aliases."
            raise ValueError(error)
        self.repo = repo.resolve()
        self.workspace = self.repo / "evals/runs/memory"
        self.output = self.workspace / "matrix-v3"
        self.snapshot_path = self.output / "snapshot-manifest.json"
        self.snapshot = read_json(self.snapshot_path)
        self.frozen_root = self.repo / self.snapshot["frozen_root"]
        self.corpus = self.repo / "evals/datasets/ostrikov/v1/messages.jsonl"
        self.queries = self.repo / "evals/datasets/ostrikov/v2/queries.jsonl"
        self.cases = self.repo / "evals/datasets/ostrikov/v2/cases.jsonl"
        self.overlay = self.repo / "deploy/memory-matrix-frozen.compose.yaml"
        self.max_jobs, self.poll_seconds = max_jobs, poll_seconds
        self.runner = runner or self._command
        self.scanner = scanner or self._active_commands
        self.aliases = requested
        self._lock = threading.Lock()
        # Completed artifacts are immutable during a run. Validate each phase
        # once per host process; restart recovery performs those checks again.
        self._checked_phases: set[tuple[str, str]] = set()
        self._verify_snapshot()
        if registry_names(self.frozen_root) != {v.variant for v in VARIANTS.values()}:
            error = "Frozen registry differs from the explicit forty-variant matrix."
            raise FrozenMismatchError(error)
        self.case_ids, self.prefixes = self._question_metadata()
        self.frozen = {
            "snapshot_manifest_sha256": digest(self.snapshot_path),
            "corpus_sha256": digest(self.corpus),
            "queries_sha256": digest(self.queries),
            "gold_sha256": digest(self.cases),
            "code_sha256": self.snapshot["code_sha256"],
            "overlay_sha256": digest(self.overlay),
        }
        self.ledger_path = self.output / "ledger.json"
        if self.ledger_path.exists():
            self.ledger = read_json(self.ledger_path)
            if self.ledger["frozen"] != self.frozen:
                error = "Matrix ledger belongs to different frozen inputs."
                raise FrozenMismatchError(error)
        else:
            self.ledger = {
                "protocol": "v3",
                "frozen": self.frozen,
                "expected_case_ids": self.case_ids,
                "prefixes": self.prefixes,
                "variants": {
                    alias: {
                        "variant": v.variant,
                        "run_id": v.run_id,
                        "dependencies": list(v.dependencies),
                        "state": "waiting",
                        "phases": {},
                    }
                    for alias, v in VARIANTS.items()
                },
            }
            self._save()

    def _verify_snapshot(self) -> None:
        for relative, expected in self.snapshot["files"].items():
            path = self.frozen_root / relative
            if (
                not path.resolve().is_relative_to(self.frozen_root.resolve())
                or digest(path) != expected
            ):
                error = "Frozen source snapshot changed."
                raise FrozenMismatchError(error)
        code = hashlib.sha256()
        for path in sorted((self.frozen_root / "evals/suites/memory").glob("*.py")):
            code.update(path.name.encode())
            code.update(path.read_bytes())
        expected = {
            "code_sha256": code.hexdigest(),
            "corpus_sha256": digest(self.corpus),
            "queries_sha256": digest(self.queries),
            "overlay_sha256": digest(self.overlay),
            "production_config_sha256": digest(
                self.frozen_root / "deploy/openviking.conf"
            ),
        }
        if any(self.snapshot.get(k) != v for k, v in expected.items()):
            error = "Frozen source, corpus, question or overlay hashes differ."
            raise FrozenMismatchError(error)

    def _assert_inputs(self) -> None:
        self._verify_snapshot()
        if (
            digest(self.snapshot_path) != self.frozen["snapshot_manifest_sha256"]
            or digest(self.cases) != self.frozen["gold_sha256"]
        ):
            error = "Evaluation snapshot manifest or gold bytes changed."
            raise FrozenMismatchError(error)

    def _question_metadata(self) -> tuple[list[str], dict[str, dict[str, Any]]]:
        questions = [
            json.loads(line)
            for line in self.queries.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        ids = [q["case_id"] for q in questions]
        if (
            len(ids) != 50
            or len(set(ids)) != 50
            or any("gold" in q or "evidence_message_ids" in q for q in questions)
        ):
            error = "Matrix requires exactly fifty unique gold-free query inputs."
            raise ValueError(error)
        prefixes: dict[str, dict[str, Any]] = {}
        for question in questions:
            context = {
                "chat_key": question["chat_key"],
                "before": question["history_cutoff"]["date"],
                "before_id": int(question["history_cutoff"]["message_id"]),
                "author_id": None,
                "after": None,
            }
            key = hashlib.sha256(
                json.dumps(context, sort_keys=True).encode()
            ).hexdigest()[:16]
            prefixes[key] = {"scope": context, "source_count": 0}
        for line in self.corpus.read_text(encoding="utf-8").splitlines():
            source = json.loads(line)
            sent = datetime.fromisoformat(source["date"])
            for prefix in prefixes.values():
                scope = prefix["scope"]
                if source["chat_key"] == scope["chat_key"] and (
                    sent,
                    int(source["id"]),
                ) < (datetime.fromisoformat(scope["before"]), scope["before_id"]):
                    prefix["source_count"] += 1
        if len(prefixes) != 2:
            error = "The v3 matrix requires the full snapshot and historical prefix."
            raise ValueError(error)
        return ids, prefixes

    def _save(self) -> None:
        self.ledger["updated_at"] = datetime.now(UTC).isoformat()
        write_json(self.ledger_path, self.ledger)

    def _prepared(self, variant: str, *, frozen_code: bool = False) -> bool:
        for key, prefix in self.prefixes.items():
            path = self.workspace / "indices" / variant / key / "preparation.json"
            if not path.exists():
                return False
            metadata = read_json(path)
            expected = {
                "variant": variant,
                "scope": prefix["scope"],
                "source_count": prefix["source_count"],
                "corpus_sha256": self.snapshot["corpus_sha256"],
            }
            if frozen_code:
                expected["code_sha256"] = self.snapshot["code_sha256"]
            if any(metadata.get(k) != v for k, v in expected.items()):
                return False
        return True

    def readiness(self, spec: Variant) -> list[str]:
        reasons = [
            f"prepare:{name}" for name in spec.dependencies if not self._prepared(name)
        ]
        services = []
        if "openviking" in spec.dependencies:
            services.append(("openviking", 4))
        if any(name.startswith("hindsight") for name in spec.dependencies):
            services.append(("hindsight", 8))
        for service, concurrency in services:
            path = self.workspace / service / "answering-config-ready.json"
            try:
                marker = read_json(path) if path.exists() else {}
            except json.JSONDecodeError:
                marker = {}
            if (
                marker.get("ready") is not True
                or marker.get("restored") is not True
                or marker.get("max_concurrent") != concurrency
                or not marker.get("config_sha256")
            ):
                reasons.append(f"answering-config:{service}")
        return reasons

    def command(self, spec: Variant, phase: str, attempt: int) -> list[str]:
        if phase == "clone_judge2":
            error = "Cloning is a local evaluation-only operation."
            raise ValueError(error)
        operation = phase.rstrip("12")
        run_id = spec.run_id + ("-judge2" if phase.endswith("2") else "")
        argv = ["docker", "compose", "--project-directory", str(self.repo)]
        for name in (
            "memory-eval.compose.yaml",
            "memory-hindsight.compose.yaml",
            "memory-openviking.compose.yaml",
            "memory-matrix-frozen.compose.yaml",
        ):
            argv.extend(["-f", str(self.repo / "deploy" / name)])
        argv.extend([
            "run",
            "--rm",
            "--no-deps",
            "-T",
            "--name",
            f"dashulya-matrix-{spec.alias}-{phase}-{attempt}",
            "-e",
            "OPENAI_MODEL=gpt-6-luna",
            "-e",
            "MEMORY_EVAL_JUDGE_MODEL=gpt-6-luna",
            "-e",
            f"MEMORY_EVAL_TOOL_BUDGET={self.snapshot['limits']['tool_budget']}",
            "-e",
            f"MEMORY_EVAL_EVIDENCE_CHARACTERS={self.snapshot['limits']['answer_evidence_character_budget']}",
        ])
        if operation in {"grade", "report"}:
            argv.extend(["-v", f"{self.cases}:/gold/cases.jsonl:ro"])
        argv.extend([
            "evaluator",
            "python",
            "-m",
            "evals.suites.memory.matrix",
            operation,
            "--variant",
            spec.variant,
            "--workspace",
            "/runs",
            "--run-id",
            run_id,
            "--concurrency",
            "4",
        ])
        return argv

    def _command(self, argv: list[str], log: Path) -> int:
        with log.open("ab") as stream:
            result = subprocess.run(
                argv,
                cwd=self.repo,
                stdout=stream,
                stderr=subprocess.STDOUT,
                shell=False,
                check=False,
            )
        return result.returncode

    def _active_commands(self) -> dict[tuple[str, str], str]:
        result = subprocess.run(
            [
                "docker",
                "ps",
                "--filter",
                "label=com.docker.compose.service=evaluator",
                "--format",
                "{{.Names}}",
            ],
            cwd=self.repo,
            capture_output=True,
            text=True,
            shell=False,
            check=True,
        )
        active = {}
        known = {v.run_id for v in VARIANTS.values()} | {
            v.run_id + "-judge2" for v in VARIANTS.values()
        }
        for name in result.stdout.splitlines():
            inspected = subprocess.run(
                ["docker", "inspect", "--format", "{{json .Config.Cmd}}", name],
                cwd=self.repo,
                capture_output=True,
                text=True,
                shell=False,
                check=False,
            )
            if inspected.returncode:
                continue
            args = json.loads(inspected.stdout)
            if "--run-id" not in args:
                continue
            run_id = args[args.index("--run-id") + 1]
            if run_id not in known or "evals.suites.memory.matrix" not in args:
                continue
            phase = args[args.index("evals.suites.memory.matrix") + 1]
            if phase in {"prepare", "run", "grade", "report"}:
                active[run_id, phase] = name
        return active

    def _run_path(self, spec: Variant, second: bool = False) -> Path:
        return self.workspace / "runs" / (spec.run_id + ("-judge2" if second else ""))

    def _manifest(self, spec: Variant, second: bool = False) -> dict[str, Any] | None:
        path = self._run_path(spec, second) / "manifest.json"
        if not path.exists():
            return None
        manifest = read_json(path)
        expected = {
            "run_id": path.parent.name,
            "variant": spec.variant,
            "code_sha256": self.snapshot["code_sha256"],
            "corpus_sha256": self.frozen["corpus_sha256"],
            "queries_sha256": self.frozen["queries_sha256"],
            "expected_case_ids": self.case_ids,
            "model": "gpt-6-luna",
            "tool_budget": self.snapshot["limits"]["tool_budget"],
            "evidence_character_budget": self.snapshot["limits"][
                "answer_evidence_character_budget"
            ],
        }
        if any(manifest.get(k) != v for k, v in expected.items()):
            error = "Existing run manifest differs from frozen v3 inputs."
            raise FrozenMismatchError(error)
        return manifest

    def _predictions(self, spec: Variant, second: bool = False) -> dict[str, str] | None:
        if self._manifest(spec, second) is None:
            return None
        directory = self._run_path(spec, second) / "predictions"
        if {p.stem for p in directory.glob("*.json")} - set(self.case_ids):
            error = "Prediction directory contains unknown case identities."
            raise FrozenMismatchError(error)
        hashes = {}
        for case_id in self.case_ids:
            path = directory / (case_id + ".json")
            if not path.exists():
                return None
            prediction = read_json(path)
            if (
                prediction.get("case_id") != case_id
                or prediction.get("variant") != spec.variant
            ):
                error = "Prediction identity differs from the selected variant."
                raise FrozenMismatchError(error)
            if prediction.get("error") or not isinstance(prediction.get("answer"), dict):
                return None
            hashes[case_id] = digest(path)
        return hashes

    def clone_judge2(self, spec: Variant) -> None:
        original = self._run_path(spec)
        hashes = self._predictions(spec)
        if hashes is None:
            error = "Cannot rejudge an incomplete prediction set."
            raise ValueError(error)
        manifest = read_json(original / "manifest.json")
        destination = self._run_path(spec, True)
        cloned = manifest | {
            "run_id": destination.name,
            "evaluation_only_copy_of": spec.run_id,
            "evaluation_only_prediction_sha256": hashes,
        }
        if (destination / "manifest.json").exists() and read_json(
            destination / "manifest.json"
        ) != cloned:
            error = "Existing judge2 copy differs from the original frozen answers."
            raise FrozenMismatchError(error)
        (destination / "predictions").mkdir(parents=True, exist_ok=True)
        for case_id, expected in hashes.items():
            data = (original / "predictions" / (case_id + ".json")).read_bytes()
            if hashlib.sha256(data).hexdigest() != expected:
                error = "Original prediction changed during evaluation-only copying."
                raise FrozenMismatchError(error)
            target = destination / "predictions" / (case_id + ".json")
            if target.exists() and target.read_bytes() != data:
                error = "Existing judge2 prediction bytes differ."
                raise FrozenMismatchError(error)
            if not target.exists():
                temporary = target.with_suffix(".json.tmp")
                temporary.write_bytes(data)
                temporary.replace(target)
        write_json(destination / "manifest.json", cloned)
        if self._predictions(spec) != hashes:
            error = "Original answers changed during evaluation-only copying."
            raise FrozenMismatchError(error)

    def phase_complete(self, spec: Variant, phase: str) -> bool:
        if phase == "prepare":
            return self._prepared(spec.variant, frozen_code=True)
        second = phase.endswith("2")
        predictions = self._predictions(spec, second)
        if predictions is None:
            return False
        if phase == "run":
            return True
        if phase == "clone_judge2":
            manifest = self._manifest(spec, True)
            return bool(
                manifest
                and manifest.get("evaluation_only_copy_of") == spec.run_id
                and predictions == self._predictions(spec)
                and manifest.get("evaluation_only_prediction_sha256") == predictions
            )
        directory = self._run_path(spec, second)
        expected_hashes = {
            k: self.frozen[k] for k in ("corpus_sha256", "queries_sha256", "gold_sha256")
        }
        expected_hashes.update(
            run_manifest_sha256=digest(directory / "manifest.json"),
            grading_code_sha256=digest(
                self.frozen_root / "evals/suites/memory/grading.py"
            ),
            grading_runner_sha256=digest(
                self.frozen_root / "evals/suites/memory/runner.py"
            ),
        )
        if {p.stem for p in (directory / "grades").glob("*.json")} != set(self.case_ids):
            return False
        for case_id in self.case_ids:
            grade = read_json(directory / "grades" / (case_id + ".json"))
            if (
                grade.get("case_id") != case_id
                or grade.get("prediction_sha256") != predictions[case_id]
                or grade.get("judge_model") != "gpt-6-luna"
                or any(grade.get(k) != v for k, v in expected_hashes.items())
            ):
                return False
        if phase.startswith("grade"):
            return True
        summary_path = directory / "summary.json"
        if not summary_path.exists():
            return False
        summary = read_json(summary_path)
        return bool(
            summary.get("complete") is True
            and summary.get("graded") == 50
            and summary.get("total_required") == 50
            and summary.get("run_id") == directory.name
            and summary.get("variant") == spec.variant
            and not summary.get("missing")
            and not summary.get("invalid_results")
            and summary.get("provenance") == expected_hashes
        )

    def _execute(self, alias: str, phase: str) -> None:
        spec = VARIANTS[alias]
        with self._lock:
            state = self.ledger["variants"][alias]
            entry = state["phases"].setdefault(phase, {"attempts": 0})
            entry.update(attempts=entry["attempts"] + 1, state="running")
            attempt = entry["attempts"]
            state["state"] = "running"
            log = self.output / "logs" / spec.run_id / f"{phase}-attempt{attempt}.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            log.touch(exist_ok=True)
            entry["log"] = str(log.relative_to(self.repo))
            entry["container_name"] = f"dashulya-matrix-{alias}-{phase}-{attempt}"
            self._save()
        try:
            code, complete = self._perform_phase(spec, phase, attempt, log)
            error = (
                None
                if complete
                else f"Phase exited {code}; complete artifacts={complete}."
            )
            fatal = False
        except Exception as exc:
            complete, code, error, fatal = (
                False,
                None,
                type(exc).__name__,
                isinstance(exc, FrozenMismatchError),
            )
        with self._lock:
            entry.update(
                state="complete" if complete else "failed",
                exit_code=code,
                error=error,
                fatal=fatal,
            )
            state["state"] = (
                "failed" if fatal or (not complete and attempt >= 2) else "pending"
            )
            if complete:
                self._checked_phases.add((alias, phase))
            self._save()

    def _perform_phase(
        self, spec: Variant, phase: str, attempt: int, log: Path
    ) -> tuple[int, bool]:
        self._assert_inputs()
        if phase == "clone_judge2":
            self.clone_judge2(spec)
            code = 0
        else:
            code = self.runner(self.command(spec, phase, attempt), log)
        self._assert_inputs()
        return code, code == 0 and self.phase_complete(spec, phase)

    def aggregate(self) -> None:
        rows = []
        for alias, spec in VARIANTS.items():
            state = self.ledger["variants"][alias]
            summaries = []
            for second in (False, True):
                path = self._run_path(spec, second) / "summary.json"
                try:
                    phase = "report2" if second else "report1"
                    valid = path.exists() and (
                        (alias, phase) in self._checked_phases
                        or self.phase_complete(spec, phase)
                    )
                    if valid:
                        self._checked_phases.add((alias, phase))
                except (FrozenMismatchError, json.JSONDecodeError):
                    valid = False
                summaries.append(read_json(path) if valid else None)
            complete = all(s is not None for s in summaries)
            rows.append({
                "alias": alias,
                "variant": spec.variant,
                "state": state["state"],
                "judge_passes": summaries,
                "mean_accuracy_all": statistics.mean(
                    s["accuracy_all"] for s in summaries if s is not None
                )
                if complete
                else None,
                "mean_accuracy_answerable": statistics.mean(
                    s["accuracy_answerable"] for s in summaries if s is not None
                )
                if complete
                else None,
                "both_judges_threshold_met": all(
                    s["threshold_met"] for s in summaries if s is not None
                )
                if complete
                else None,
                "failure_phases": {
                    k: v for k, v in state["phases"].items() if v["state"] == "failed"
                },
            })
        write_json(
            self.output / "summary.json",
            {
                "protocol": "v3",
                "frozen": self.frozen,
                "variants": rows,
                "aggregation": "Two complete independent grading passes over identical saved predictions; no per-case oracle selection. Incomplete candidates remain visible with null aggregate scores.",
            },
        )

    def _next_phase(self, alias: str, active: dict[tuple[str, str], str]) -> str | None:
        spec = VARIANTS[alias]
        state = self.ledger["variants"][alias]
        if state["state"] == "failed":
            return None
        try:
            phase = next(
                (
                    p
                    for p in PHASES
                    if (alias, p) not in self._checked_phases
                    and not self.phase_complete(spec, p)
                ),
                None,
            )
        except (FrozenMismatchError, json.JSONDecodeError) as exc:
            state["state"] = "failed"
            state["phases"]["validation"] = {
                "attempts": 0,
                "state": "failed",
                "fatal": True,
                "error": type(exc).__name__,
            }
            return None
        completed = PHASES if phase is None else PHASES[: PHASES.index(phase)]
        for previous in completed:
            self._checked_phases.add((alias, previous))
            state["phases"].setdefault(previous, {"attempts": 0}).update(
                state="complete", recovered=True
            )
        if phase is None:
            state["state"] = "complete"
            return None
        run_id = spec.run_id + ("-judge2" if phase.endswith("2") else "")
        adopted = active.get((run_id, phase.rstrip("12")))
        if adopted:
            state.update(state="external_running", external_container=adopted)
            state["phases"].setdefault(phase, {"attempts": 1}).update(
                state="external_running", container_name=adopted
            )
            return None
        reasons = self.readiness(spec)
        if reasons:
            state.update(state="waiting", wait_reasons=reasons)
            return None
        entry = state["phases"].get(phase, {"attempts": 0})
        if entry.get("fatal") or entry["attempts"] >= 2:
            state["state"] = "failed"
            return None
        return phase

    def _loop(self, executor: ThreadPoolExecutor, *, once: bool) -> int:
        running: dict[str, Future[None]] = {}
        while True:
            self._assert_inputs()
            completed = [a for a, f in running.items() if f.done()]
            for alias in completed:
                running.pop(alias).result()
            active = self.scanner()
            owned = {
                entry.get("container_name")
                for state in self.ledger["variants"].values()
                for entry in state["phases"].values()
                if entry.get("state") == "running"
            }
            occupied = len(running) + len(set(active.values()) - owned)
            for alias in self.aliases:
                if alias in running:
                    continue
                phase = self._next_phase(alias, active)
                if phase is not None and occupied < self.max_jobs:
                    running[alias] = executor.submit(self._execute, alias, phase)
                    occupied += 1
            with self._lock:
                self._save()
            if completed:
                self.aggregate()
            if not running:
                states = [self.ledger["variants"][a]["state"] for a in self.aliases]
                if all(s in {"complete", "failed"} for s in states) or (
                    once and not active
                ):
                    self.aggregate()
                    return (
                        1
                        if "failed" in states
                        else 0
                        if all(s == "complete" for s in states)
                        else 2
                    )
            time.sleep(min(60, max(0.01, self.poll_seconds)))

    def run(self, *, once: bool = False) -> int:
        """Drain ready work; normally keep waiting for root-controlled readiness."""
        self.output.mkdir(parents=True, exist_ok=True)
        with (self.output / "orchestrator.lock").open("a", encoding="utf-8") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with ThreadPoolExecutor(max_workers=self.max_jobs) as executor:
                return self._loop(executor, once=once)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--max-jobs", type=int, default=2, choices=[1, 2, 3])
    parser.add_argument("--poll-seconds", type=float, default=10)
    parser.add_argument(
        "--only", default="", help="Comma-separated aliases; default all forty."
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Drain currently ready candidates, then exit with code2 if others wait.",
    )
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args(argv)
    if args.list:
        print(  # ruff: ignore[print] -- Explicit CLI catalog, without source text or credentials.
            json.dumps(
                {
                    a: {
                        "variant": v.variant,
                        "dependencies": v.dependencies,
                        "run_id": v.run_id,
                    }
                    for a, v in VARIANTS.items()
                },
                indent=2,
            )
        )
        return
    orchestrator = Orchestrator(
        args.repo,
        aliases=args.only.split(",") if args.only else None,
        max_jobs=args.max_jobs,
        poll_seconds=args.poll_seconds,
    )
    raise SystemExit(orchestrator.run(once=args.once))


if __name__ == "__main__":
    main()
