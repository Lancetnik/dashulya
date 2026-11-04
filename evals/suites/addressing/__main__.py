"""Prepare classifier inputs, collect Decisions probabilities and score them."""

import argparse
import asyncio
import hashlib
import json
import math
import os
import sys
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any
from zoneinfo import ZoneInfo

from ag2 import Agent, ResponseSchema
from dotenv import load_dotenv

from dashulya.ai.agents.models import create_models
from dashulya.ai.prompts import (
    GROUP_ADDRESSING_PROMPT,
    INPUT_TRUST_PROMPT,
    MESSAGE_FORMAT_PROMPT,
    TIMEZONE_PROMPT,
)
from evals.suites.addressing.benchmark import export_inputs, score_predictions


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def emit(value: Any) -> None:
    sys.stdout.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")


def output_directory(path: Path) -> Path:
    target = path.resolve()
    allowed = (Path(__file__).resolve().parents[2] / "runs").resolve()
    if not target.is_relative_to(allowed) or target == allowed:
        msg = "Evaluation outputs belong under evals/runs/."
        raise ValueError(msg)
    target.mkdir(parents=True, exist_ok=False)
    return target


def load_dataset(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    manifest = json.loads((path / "manifest.json").read_text())
    for name, expected in manifest["artifacts"].items():
        source = (path / name).resolve()
        if not source.is_relative_to(path.resolve()):
            msg = "Dataset artifacts must stay inside the dataset."
            raise ValueError(msg)
        if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
            msg = f"Dataset artifact hash mismatch: {name}"
            raise ValueError(msg)
    cases, inputs = read_jsonl(path / "cases.jsonl"), read_jsonl(path / "inputs.jsonl")
    export_inputs(cases, inputs, panel="all")
    return cases, inputs


def response_fields(reply: Any) -> dict[str, Any]:
    probability = reply.response.metadata.get("probability")
    if (
        type(probability) not in {int, float}
        or not math.isfinite(probability)
        or not 0 <= probability <= 1
    ):
        msg = "Decisions response has no valid predicate probability."
        raise ValueError(msg)
    return {
        "probability": probability,
        "model": reply.response.model,
        "usage": asdict(reply.response.usage),
    }


def model_agent(model: str, key: str, prompts: list[str]) -> Agent[bool]:
    return Agent(
        "group_addressing_eval",
        prompt=prompts,
        response_schema=ResponseSchema(
            bool, description="Does the current message call for a reply from Dashulya?"
        ),
        config=create_models(
            api_key=key, reply_model=model, decision_model=model
        ).decision,
    )


async def run_model(args: argparse.Namespace) -> int:
    # This command reads only exported model inputs, never evaluator cases.
    inputs = read_jsonl(args.inputs)
    if not inputs or any(set(r) != {"case_id", "input"} for r in inputs):
        msg = "Use an input-only export containing case_id and input."
        raise ValueError(msg)
    if len({r["case_id"] for r in inputs}) != len(inputs):
        msg = "Duplicate input identity."
        raise ValueError(msg)
    if args.concurrency < 1:
        msg = "Concurrency must be positive."
        raise ValueError(msg)
    if any(not isinstance(r["input"], str) for r in inputs):
        msg = "Each model input must be text."
        raise ValueError(msg)
    load_dotenv()
    key = os.environ.get("OPENAI_API_KEY", "")
    if not key:
        msg = "Set OPENAI_API_KEY for the explicit run command."
        raise ValueError(msg)
    model = args.model or os.environ.get("OPENAI_DECISION_MODEL", "gpt-6-luna")
    prompt = (
        args.prompt.read_text(encoding="utf-8").strip()
        if args.prompt
        else GROUP_ADDRESSING_PROMPT
    )
    timezone = ZoneInfo(args.timezone).key
    prompts = [
        INPUT_TRUST_PROMPT,
        prompt,
        TIMEZONE_PROMPT.format(timezone=timezone),
        MESSAGE_FORMAT_PROMPT,
    ]
    agent = model_agent(model, key, prompts)
    target = output_directory(args.out)
    (target / "classifier-prompt.txt").write_text(prompt)
    write_json(
        target / "run.json",
        {
            "status": "running",
            "model": model,
            "timezone": timezone,
            "input_sha256": hashlib.sha256(args.inputs.read_bytes()).hexdigest(),
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "effective_prompt_sha256": hashlib.sha256(
                "\n".join(prompts).encode()
            ).hexdigest(),
            "case_count": len(inputs),
            "concurrency": args.concurrency,
            "score": "Decisions predicate probability before the AG2 boolean threshold",
        },
    )
    semaphore = asyncio.Semaphore(args.concurrency)

    async def classify(row: dict[str, Any]) -> dict[str, Any]:
        async with semaphore:
            started = perf_counter()
            result: dict[str, Any] = {"case_id": row["case_id"]}
            try:
                async with asyncio.timeout(12):
                    reply = await agent.ask(row["input"])
                    result.update(response_fields(reply))
            except Exception as error:
                # Provider payloads and credentials stay out of exported errors.
                result.update(probability=None, error_type=type(error).__name__)
            result["duration_ms"] = round((perf_counter() - started) * 1000)
            return result

    failures = 0
    with (target / "predictions.jsonl").open("w") as stream:
        for finished in asyncio.as_completed([classify(row) for row in inputs]):
            result = await finished
            failures += result["probability"] is None
            stream.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
    metadata = json.loads((target / "run.json").read_text())
    metadata.update(status="failed" if failures else "complete", failed_cases=failures)
    write_json(target / "run.json", metadata)
    emit({"out": str(target), "cases": len(inputs), "failed": failures})
    return int(bool(failures))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("validate", "export", "score"):
        command = commands.add_parser(name)
        command.add_argument("dataset", type=Path)
        if name != "validate":
            command.add_argument("--partition", choices=("calibration", "test"))
            command.add_argument("--panel", choices=("core", "all"), default="core")
            command.add_argument("--out", type=Path, required=True)
        if name == "score":
            command.add_argument("--predictions", type=Path, required=True)
            command.add_argument("--threshold", type=float, default=0.5)
            command.add_argument("--min-recall", type=float, default=0.98)
    command = commands.add_parser(
        "run", help="Explicit command: calls the paid model API."
    )
    command.add_argument("--inputs", type=Path, required=True)
    command.add_argument("--prompt", type=Path)
    command.add_argument("--model")
    command.add_argument("--timezone", default="Europe/Moscow")
    command.add_argument("--concurrency", type=int, default=2)
    command.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "run":
        return asyncio.run(run_model(args))
    cases, inputs = load_dataset(args.dataset)
    if args.command == "validate":
        emit({
            "cases": len(cases),
            "core_cases": sum(c["metric_eligible"] for c in cases),
            "status": "valid",
        })
        return 0
    selected = export_inputs(cases, inputs, partition=args.partition, panel=args.panel)
    if args.command == "export":
        target = output_directory(args.out)
        (target / "inputs.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected)
        )
        write_json(
            target / "plan.json",
            {
                "partition": args.partition,
                "panel": args.panel,
                "cases": len(selected),
                "status": "prepared_not_run",
            },
        )
        emit({"out": str(target), "cases": len(selected), "status": "prepared_not_run"})
        return 0
    ids = {r["case_id"] for r in selected}
    all_ids = {c["case_id"] for c in cases}
    predictions = read_jsonl(args.predictions)
    if any(r["case_id"] not in all_ids for r in predictions):
        msg = "Predictions contain IDs outside this dataset."
        raise ValueError(msg)
    result = score_predictions(
        [c for c in cases if c["case_id"] in ids],
        [r for r in predictions if r["case_id"] in ids],
        threshold=args.threshold,
    )
    if not 0 <= args.min_recall <= 1:
        msg = "Minimum recall must be in [0, 1]."
        raise ValueError(msg)
    result.update(partition=args.partition, panel=args.panel)
    result["recommended_threshold"] = None
    if args.partition == "calibration":
        eligible = [
            p
            for p in result["thresholds"]
            if p["recall"] is not None and p["recall"] >= args.min_recall
        ]
        if eligible:
            result["recommended_threshold"] = min(
                eligible, key=lambda p: (p["fp"], -p["threshold"])
            )["threshold"]
    target = output_directory(args.out)
    write_json(target / "metrics.json", result)
    emit({k: v for k, v in result.items() if k != "thresholds"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
