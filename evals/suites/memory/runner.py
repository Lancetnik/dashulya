"""Prepare, run and grade memory variants against a separate question file."""

import argparse
import asyncio
import hashlib
import importlib
import importlib.metadata
import json
import logging
import os
import statistics
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any, cast

from openai import AsyncOpenAI

from evals.suites.memory.answerer import AnswerSession, model_config
from evals.suites.memory.corpus import SourceArchive, load_messages
from evals.suites.memory.grading import (
    SemanticGrade,
    complete_pass,
    deterministic_checks,
    grade_answer,
)
from evals.suites.memory.providers import configured_api
from evals.suites.memory.types import MemoryVariant, QueryContext

logger = logging.getLogger(__name__)
VARIANTS = {
    "raw_lexical": ("raw_hybrid", "RawLexicalVariant"),
    "raw_hybrid": ("raw_hybrid", "RawHybridVariant"),
    "claims": ("claims", "ClaimsVariant"),
    "hindsight": ("hindsight", "HindsightVariant"),
    "hindsight_chunks": ("hindsight", "HindsightChunksVariant"),
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def fingerprint(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def code_fingerprint() -> str:
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def context_for(query: dict[str, Any]) -> QueryContext:
    cutoff = query["history_cutoff"]
    return QueryContext(query["chat_key"], cutoff["date"], int(cutoff["message_id"]))


def prefix_key(context: QueryContext) -> str:
    return hashlib.sha256(
        json.dumps(asdict(context), sort_keys=True).encode()
    ).hexdigest()[:16]


def build_variant(name: str, path: Path) -> MemoryVariant:
    module_name, class_name = VARIANTS[name]
    module = importlib.import_module(f"evals.suites.memory.{module_name}")
    return cast("MemoryVariant", getattr(module, class_name)(path))


def inputs(args: argparse.Namespace) -> tuple[SourceArchive, list[dict[str, Any]]]:
    archive = SourceArchive(load_messages(args.corpus))
    queries = read_jsonl(args.queries)
    if not queries or len({q["case_id"] for q in queries}) != len(queries):
        error = "Question IDs must be unique."
        raise ValueError(error)
    for query in queries:
        if "gold" in query or "evidence_message_ids" in query:
            error = "Answering input must not contain benchmark gold. Use queries.jsonl."
            raise ValueError(error)
    return archive, queries


@dataclass(frozen=True)
class EvaluationInputs:
    """Loaded grading inputs bound to the exact bytes that were inspected."""

    archive: SourceArchive
    queries: list[dict[str, Any]]
    cases: dict[str, dict[str, Any]]
    artifacts: dict[str, tuple[Path, bytes]]
    manifest: dict[str, Any]

    @property
    def hashes(self) -> dict[str, str]:
        return {
            name: hashlib.sha256(content).hexdigest()
            for name, (_, content) in self.artifacts.items()
        }

    def assert_unchanged(self) -> None:
        for name, (path, content) in self.artifacts.items():
            if not path.exists() or path.read_bytes() != content:
                error = f"Evaluation input changed during grading/reporting: {name}."
                raise ValueError(error)


def evaluation_inputs(args: argparse.Namespace) -> EvaluationInputs:
    """Validate a run, its complete question set and the corresponding gold."""
    manifest_path = args.workspace / "runs" / args.run_id / "manifest.json"
    paths = {
        "corpus_sha256": args.corpus,
        "queries_sha256": args.queries,
        "gold_sha256": args.cases,
        "run_manifest_sha256": manifest_path,
        "grading_code_sha256": Path(__file__).with_name("grading.py"),
        "grading_runner_sha256": Path(__file__),
    }
    artifacts = {name: (path, path.read_bytes()) for name, path in paths.items()}
    archive, queries = inputs(args)
    manifest = json.loads(artifacts["run_manifest_sha256"][1])
    rows = [
        json.loads(line)
        for line in artifacts["gold_sha256"][1].decode("utf-8").splitlines()
        if line.strip()
    ]
    cases = {case["case_id"]: case for case in rows}
    if len(cases) != len(rows) or set(cases) != {q["case_id"] for q in queries}:
        error = "Gold must have exactly one case for every run question."
        raise ValueError(error)
    for query in queries:
        case = cases[query["case_id"]]
        if any(
            case[field] != query[field]
            for field in ("query", "chat_key", "history_cutoff")
        ):
            error = "Gold question or source scope differs from the answering input."
            raise ValueError(error)
        if case["answerability"] not in {"answerable", "unanswerable_in_snapshot"}:
            error = "Gold has an unknown answerability value."
            raise ValueError(error)
        requirement_ids = [r["id"] for r in case["gold"]["requirements"]]
        if not requirement_ids or len(set(requirement_ids)) != len(requirement_ids):
            error = "Gold requirements need nonempty, unique criterion IDs."
            raise ValueError(error)
    captured = EvaluationInputs(archive, queries, cases, artifacts, manifest)
    expected = {
        "run_id": args.run_id,
        "variant": args.variant,
        "corpus_sha256": captured.hashes["corpus_sha256"],
        "queries_sha256": captured.hashes["queries_sha256"],
        "expected_case_ids": [q["case_id"] for q in queries],
    }
    if any(manifest.get(name) != value for name, value in expected.items()):
        error = "Run manifest differs from the current corpus, questions or variant."
        raise ValueError(error)
    captured.assert_unchanged()
    return captured


def _prediction_identity(prediction: dict[str, Any], case_id: str, variant: str) -> bool:
    return prediction.get("case_id") == case_id and prediction.get("variant") == variant


def _current_grade(
    result: dict[str, Any],
    case: dict[str, Any],
    expected_hashes: dict[str, str],
    model: str,
) -> bool:
    return bool(
        result.get("case_id") == case["case_id"]
        and result.get("category") == case["category"]
        and result.get("answerability") == case["answerability"]
        and result.get("judge_model") == model
        and all(result.get(name) == value for name, value in expected_hashes.items())
    )


async def prepare(args: argparse.Namespace) -> None:
    archive, queries = inputs(args)
    prefixes = {prefix_key(context_for(q)): context_for(q) for q in queries}
    for key, context in sorted(prefixes.items(), key=lambda pair: pair[1].before):
        path = args.workspace / "indices" / args.variant / key
        variant = build_variant(args.variant, path)
        started = perf_counter()
        messages = archive.available(context)
        logger.info(
            "Preparing %s prefix=%s messages=%d", args.variant, key, len(messages)
        )
        try:
            result = await variant.prepare(messages)
            write_json(
                path / "preparation.json",
                {
                    "variant": args.variant,
                    "scope": asdict(context),
                    "source_count": len(messages),
                    "corpus_sha256": fingerprint(args.corpus),
                    "code_sha256": code_fingerprint(),
                    "elapsed_seconds": perf_counter() - started,
                    "result": result,
                },
            )
            logger.info("Prepared %s prefix=%s: %s", args.variant, key, result)
        finally:
            await variant.close()


async def prepared_variants(
    args: argparse.Namespace,
    archive: SourceArchive,
    queries: list[dict[str, Any]],
) -> dict[str, MemoryVariant]:
    variants: dict[str, MemoryVariant] = {}
    for query in queries:
        context = context_for(query)
        key = prefix_key(context)
        if key not in variants:
            path = args.workspace / "indices" / args.variant / key
            preparation = path / "preparation.json"
            if not preparation.exists():
                error = "Variant has no completed preparation for this source prefix."
                raise ValueError(error)
            metadata = json.loads(preparation.read_text())
            if metadata["corpus_sha256"] != fingerprint(args.corpus):
                error = "Prepared index belongs to a different corpus."
                raise ValueError(error)
            memory = build_variant(args.variant, path)
            await memory.prepare(archive.available(context))
            variants[key] = memory
    return variants


async def run(args: argparse.Namespace) -> None:
    archive, queries = inputs(args)
    output = args.workspace / "runs" / args.run_id
    output.mkdir(parents=True, exist_ok=True)
    model = os.getenv("OPENAI_MODEL", "gpt-6-luna")
    manifest = {
        "run_id": args.run_id,
        "variant": args.variant,
        "model": model,
        "model_base_url": os.getenv("OPENAI_BASE_URL"),
        "model_api": configured_api(),
        "local_only": os.getenv("MEMORY_EVAL_LOCAL_ONLY", "0") == "1",
        "corpus_sha256": fingerprint(args.corpus),
        "queries_sha256": fingerprint(args.queries),
        "code_sha256": code_fingerprint(),
        "ag2_version": importlib.metadata.version("ag2"),
        "expected_case_ids": [q["case_id"] for q in queries],
        "tool_budget": int(os.getenv("MEMORY_EVAL_TOOL_BUDGET", "12")),
        "evidence_character_budget": int(
            os.getenv("MEMORY_EVAL_EVIDENCE_CHARACTERS", "0")
        ),
        "runtime_profile": {
            name: os.getenv(name)
            for name in (
                "MEMORY_EVAL_CONTEXT_TOKENS",
                "MEMORY_EVAL_KV_CACHE_TYPE",
                "MEMORY_EVAL_ANSWER_MAX_TOKENS",
                "MEMORY_EVAL_CASE_TIMEOUT",
            )
        },
    }
    manifest_path = output / "manifest.json"
    if (
        manifest_path.exists()
        and json.loads(manifest_path.read_text(encoding="utf-8")) != manifest
    ):
        error = "Run inputs or code changed. Use a new run ID instead of mixing results."
        raise ValueError(error)
    write_json(manifest_path, manifest)
    configuration = model_config(os.environ["OPENAI_API_KEY"], model)
    variants = await prepared_variants(args, archive, queries)
    semaphore = asyncio.Semaphore(args.concurrency)

    async def one(query: dict[str, Any]) -> None:
        case_id = query["case_id"]
        path = output / "predictions" / f"{case_id}.json"
        if path.exists() and not json.loads(path.read_text()).get("error"):
            return
        async with semaphore:
            context = context_for(query)
            session = AnswerSession(
                archive,
                variants[prefix_key(context)],
                context,
                tool_budget=int(os.getenv("MEMORY_EVAL_TOOL_BUDGET", "12")),
            )
            started = perf_counter()
            logger.info("Answering %s %s", args.variant, case_id)
            try:
                result = await session.answer(
                    query["query"],
                    configuration,
                    input_context=query.get("input_context"),
                )
            except Exception as exc:
                logger.warning("Case %s failed (%s)", case_id, type(exc).__name__)
                result = {
                    "error": type(exc).__name__,
                    "trace": session.trace,
                    "seen_source_ids": sorted(session.seen_sources),
                    "elapsed_seconds": perf_counter() - started,
                }
            write_json(path, {"case_id": case_id, "variant": args.variant} | result)
            logger.info(
                "Answered %s %s in %.1fs", args.variant, case_id, perf_counter() - started
            )

    selected = queries
    if args.only:
        requested = set(args.only.split(","))
        selected = [q for q in queries if q["case_id"] in requested]
        if len(selected) != len(requested):
            error = "Unknown selected question ID."
            raise ValueError(error)
    try:
        await asyncio.gather(*(one(query) for query in selected))
    finally:
        for memory in variants.values():
            await memory.close()


async def grade(args: argparse.Namespace) -> None:
    captured = evaluation_inputs(args)
    hashes = captured.hashes
    output = args.workspace / "runs" / args.run_id
    model = os.getenv("MEMORY_EVAL_JUDGE_MODEL", os.getenv("OPENAI_MODEL", "gpt-6-luna"))
    semaphore = asyncio.Semaphore(args.concurrency)
    async with AsyncOpenAI(
        timeout=float(os.getenv("MEMORY_EVAL_TIMEOUT", "120")), max_retries=2
    ) as client:

        async def one(query: dict[str, Any]) -> None:
            case_id = query["case_id"]
            prediction_path = output / "predictions" / f"{case_id}.json"
            if not prediction_path.exists():
                return
            async with semaphore:
                captured.assert_unchanged()
                # The hash must describe the bytes passed to the judge, rather
                # than a later version that appeared while the request ran.
                prediction_bytes = prediction_path.read_bytes()
                prediction = json.loads(prediction_bytes)
                if not _prediction_identity(prediction, case_id, args.variant):
                    error = "Prediction identity differs from its run and question."
                    raise ValueError(error)
                if prediction.get("error"):
                    logger.info(
                        "Skipping failed prediction %s; rerun the answer first", case_id
                    )
                    return
                expected_hashes = hashes | {
                    "prediction_sha256": hashlib.sha256(prediction_bytes).hexdigest()
                }
                result_path = output / "grades" / f"{case_id}.json"
                if result_path.exists():
                    existing = json.loads(result_path.read_text())
                    if _current_grade(
                        existing, captured.cases[case_id], expected_hashes, model
                    ):
                        return
                result = await grade_answer(
                    captured.cases[case_id], prediction, captured.archive, client, model
                )
                captured.assert_unchanged()
                if prediction_path.read_bytes() != prediction_bytes:
                    error = "Prediction changed while its answer was being graded."
                    raise ValueError(error)
                write_json(result_path, result | expected_hashes)
                logger.info(
                    "Graded %s: %s", case_id, "PASS" if result["passed"] else "FAIL"
                )

        await asyncio.gather(*(one(query) for query in captured.queries))


async def recall(args: argparse.Namespace) -> None:
    archive, queries = inputs(args)
    variants = await prepared_variants(args, archive, queries)
    records = []
    try:
        for query in queries:
            context = context_for(query)
            hits = await variants[prefix_key(context)].search(
                query["query"], context, limit=8
            )
            sources = {
                i
                for hit in hits
                if hit.chat_key == context.chat_key
                and all(archive.get(i, context) is not None for i in hit.source_ids)
                for i in hit.source_ids
            }
            records.append({
                "case_id": query["case_id"],
                "source_ids": sorted(sources),
                "hits": len(hits),
            })
    finally:
        for memory in variants.values():
            await memory.close()
    write_json(args.workspace / "runs" / args.run_id / "recall.json", records)
    logger.info(
        "Saved retrieval-only results for %d queries; no answers graded.", len(records)
    )


def _report_case(
    args: argparse.Namespace,
    captured: EvaluationInputs,
    case_id: str,
    hashes: dict[str, str],
    model: str,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, str | None]:
    output = args.workspace / "runs" / args.run_id
    case = captured.cases[case_id]
    prediction_path = output / "predictions" / f"{case_id}.json"
    grade_path = output / "grades" / f"{case_id}.json"
    if not prediction_path.exists():
        return None, None, "Prediction is missing."
    prediction_bytes = prediction_path.read_bytes()
    prediction = json.loads(prediction_bytes)
    if not _prediction_identity(prediction, case_id, args.variant):
        return None, None, "Prediction identity does not match its run."
    if prediction.get("error"):
        return None, prediction, "Prediction execution failed."
    if not grade_path.exists():
        return None, prediction, "Grade is missing."
    result = json.loads(grade_path.read_text())
    expected_hashes = hashes | {
        "prediction_sha256": hashlib.sha256(prediction_bytes).hexdigest()
    }
    if not _current_grade(result, case, expected_hashes, model):
        return None, prediction, "Grade provenance is stale or belongs to another input."
    try:
        semantic = SemanticGrade.model_validate(result["semantic"])
    except (KeyError, ValueError):
        return None, prediction, "Grade has no valid structured semantic assessment."
    checks = deterministic_checks(case, prediction, captured.archive)
    # Recompute the decision instead of trusting cached booleans or old checks.
    return (
        result | {"checks": checks, "passed": complete_pass(case, checks, semantic)},
        prediction,
        None,
    )


def report(args: argparse.Namespace) -> None:
    captured = evaluation_inputs(args)
    hashes = captured.hashes
    model = os.getenv("MEMORY_EVAL_JUDGE_MODEL", os.getenv("OPENAI_MODEL", "gpt-6-luna"))
    grades: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    missing: list[str] = []
    invalid: dict[str, str] = {}
    for query in captured.queries:
        result, prediction, reason = _report_case(
            args, captured, query["case_id"], hashes, model
        )
        if prediction is not None:
            predictions.append(prediction)
        if result is None:
            missing.append(query["case_id"])
            invalid[query["case_id"]] = reason or "No current grade."
        else:
            grades.append(result)
    categories: dict[str, list[bool]] = defaultdict(list)
    required_categories: dict[str, int] = defaultdict(int)
    for case in captured.cases.values():
        required_categories[case["category"]] += 1
    for result in grades:
        categories[result["category"]].append(result["passed"])
    factual = [g for g in grades if g["answerability"] == "answerable"]
    factual_required = sum(
        c["answerability"] == "answerable" for c in captured.cases.values()
    )
    latency = [
        float(p["elapsed_seconds"])
        for p in predictions
        if isinstance(p.get("elapsed_seconds"), (int, float))
    ]
    summary: dict[str, Any] = {
        "run_id": args.run_id,
        "variant": args.variant,
        "created_at": datetime.now(UTC).isoformat(),
        "provenance": hashes,
        "answer_model": captured.manifest.get("model"),
        "answer_code_sha256": captured.manifest.get("code_sha256"),
        "judge_model": model,
        "total_required": len(captured.queries),
        "graded": len(grades),
        "passed": sum(g["passed"] for g in grades),
        "accuracy_all": sum(g["passed"] for g in grades) / len(captured.queries),
        "factual_required": factual_required,
        "factual_graded": len(factual),
        "factual_passed": sum(g["passed"] for g in factual),
        "accuracy_answerable": (
            sum(g["passed"] for g in factual) / factual_required
            if factual_required
            else 0.0
        ),
        "missing": missing,
        "invalid_results": invalid,
        "complete": not missing and len(grades) == len(captured.queries),
        "categories": {
            name: {
                "passed": sum(categories[name]),
                "graded": len(categories[name]),
                "required": count,
            }
            for name, count in required_categories.items()
        },
        "median_answer_seconds": statistics.median(latency) if latency else None,
        "latency_measured": len(latency),
        "latency_missing": [
            p["case_id"]
            for p in predictions
            if not isinstance(p.get("elapsed_seconds"), (int, float))
        ],
        "p95_answer_seconds": sorted(latency)[
            min(len(latency) - 1, int(len(latency) * 0.95))
        ]
        if latency
        else None,
        "tool_calls": {
            name: sum(t["tool"] == name for p in predictions for t in p.get("trace", []))
            for name in ["search_memory", "chat_history"]
        },
        "threshold": 0.70,
        "assessment": "Development-set model-graded accuracy; not held-out or human-certified.",
    }
    summary["threshold_met"] = bool(
        summary["complete"]
        and summary["accuracy_all"] >= 0.70
        and summary["accuracy_answerable"] >= 0.70
    )
    captured.assert_unchanged()
    write_json(args.workspace / "runs" / args.run_id / "summary.json", summary)
    logger.info("Summary: %s", json.dumps(summary))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "command", choices=["prepare", "run", "grade", "report", "recall"]
    )
    result.add_argument("--variant", choices=VARIANTS, default="raw_hybrid")
    result.add_argument("--corpus", type=Path, default=Path("/corpus/messages.jsonl"))
    result.add_argument("--queries", type=Path, default=Path("/inputs/queries.jsonl"))
    result.add_argument("--workspace", type=Path, default=Path("/runs"))
    result.add_argument("--cases", type=Path, default=Path("/gold/cases.jsonl"))
    result.add_argument("--run-id", default="raw-hybrid-v1")
    result.add_argument("--concurrency", type=int, default=4)
    result.add_argument("--only", default="")
    return result


def main(argv: Sequence[str] | None = None) -> None:
    args = parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    if args.command == "report":
        report(args)
    else:
        command = {"prepare": prepare, "run": run, "grade": grade, "recall": recall}[
            args.command
        ]
        asyncio.run(command(args))


if __name__ == "__main__":
    main()
