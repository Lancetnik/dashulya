"""Evaluate the current OpenViking backend and production reply path separately."""

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import logging
import os
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any, cast

from openai import AsyncOpenAI
from pydantic import BaseModel, ConfigDict

from dashulya.ai.agents.models import create_models
from evals.suites.memory import runner
from evals.suites.memory.production import ProductionAnswerSession
from evals.suites.memory.providers import parse_structured

if TYPE_CHECKING:
    from ag2.config import ModelConfig

    from evals.suites.memory.openviking import OpenVikingVariant

logger = logging.getLogger(__name__)
PRODUCTION = "openviking_production"
NORMALIZATION_PROMPT = """Extract evaluation metadata from an already-written answer.
Do not answer the question, search, calculate totals, or add facts. You see no reference
answer or original-message corpus. Preserve uncertainty. Set aggregation_count only
when the answer explicitly states a single exact total number of chat messages.
Dates, citation IDs, counts of URLs/people/tools, ranges and approximate totals are not
an exact message count. Return an exact verbatim supporting count_quote, or null for
both fields. Set abstained when the answer explicitly says available evidence cannot
answer the request, with an exact verbatim abstention_quote. Ordinary qualifications
about incomplete coverage do not by themselves constitute abstention. These fields
are evaluator metadata and cannot alter the submitted answer text.
"""


class AnswerMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    aggregation_count: int | None
    count_quote: str | None
    abstained: bool
    abstention_quote: str | None


def validate_metadata(text: str, metadata: AnswerMetadata) -> None:
    """Require an exact answer-text witness; never infer evidence from benchmark gold."""
    if metadata.aggregation_count is not None and (
        not metadata.count_quote or metadata.count_quote not in text
    ):
        error = "Normalized message count needs a verbatim answer quote."
        raise ValueError(error)
    if metadata.abstained and (
        not metadata.abstention_quote or metadata.abstention_quote not in text
    ):
        error = "Normalized abstention needs a verbatim answer quote."
        raise ValueError(error)


def production_fingerprints() -> dict[str, str]:
    root = Path(__file__).resolve().parents[3]
    files = [
        "src/dashulya/ai/agents/factory.py",
        "src/dashulya/ai/agents/models.py",
        "src/dashulya/ai/prompts.py",
        "src/dashulya/ai/tools/memory.py",
        "src/dashulya/ai/dialogue.py",
        "src/dashulya/memory/index.py",
        "src/dashulya/memory/search.py",
        "src/dashulya/memory/openviking.py",
        "src/dashulya/memory/openviking_index.py",
        "deploy/openviking.conf",
    ]
    return {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in files
    }


def saved_raw_result(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    if not isinstance(value, dict):
        error = "Saved raw production reply must be an object."
        raise TypeError(error)
    return value


async def production_result(
    session: ProductionAnswerSession,
    query: dict[str, Any],
    config: "ModelConfig",
    raw_path: Path,
    client: AsyncOpenAI,
    model: str,
) -> dict[str, Any]:
    result = await asyncio.to_thread(saved_raw_result, raw_path)
    if result is None:
        result = await session.answer(
            query["query"],
            config,
            input_context=query.get("input_context"),
        )
        # Save the real agent output before evaluator-only parsing.
        runner.write_json(raw_path, result)
    metadata, usage = await parse_structured(
        client,
        model,
        NORMALIZATION_PROMPT,
        json.dumps(
            {
                "question": query["query"],
                "answer": result["answer"]["text"],
            },
            ensure_ascii=False,
        ),
        AnswerMetadata,
        max_tokens=1500,
    )
    validate_metadata(result["answer"]["text"], metadata)
    result["answer"]["aggregation_count"] = metadata.aggregation_count
    result["answer"]["abstained"] = metadata.abstained
    result["evaluation_metadata"] = metadata.model_dump()
    result["evaluation_metadata_usage"] = usage
    return result


async def run_production(args: argparse.Namespace) -> None:
    archive, queries = runner.inputs(args)
    output = args.workspace / "runs" / args.run_id
    output.mkdir(parents=True, exist_ok=True)
    model = os.getenv("OPENAI_MODEL", "gpt-6-luna")
    manifest = {
        "run_id": args.run_id,
        "variant": PRODUCTION,
        "model": model,
        "model_api": "responses",
        "corpus_sha256": runner.fingerprint(args.corpus),
        "queries_sha256": runner.fingerprint(args.queries),
        "code_sha256": runner.code_fingerprint(),
        "production_sha256": production_fingerprints(),
        "ag2_version": importlib.metadata.version("ag2"),
        "expected_case_ids": [q["case_id"] for q in queries],
        "backend_index_variant": "openviking",
        "tools": ["search_memory"],
        "automatic_initial_search": False,
        "tool_budget": None,
        "reply_timeout_seconds": 90,
        "model_timeout_seconds": 60,
        "model_max_retries": 1,
        "timezone": os.getenv("BOT_TIMEZONE", "Europe/Moscow"),
        "evaluation_normalizer_model": model,
        "evaluation_normalizer_has_gold_or_sources": False,
    }
    manifest_path = output / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        error = "Production baseline inputs/code changed; use a fresh run ID."
        raise ValueError(error)
    runner.write_json(manifest_path, manifest)
    preparation_args = argparse.Namespace(**{**vars(args), "variant": "openviking"})
    variants = await runner.prepared_variants(preparation_args, archive, queries)
    config = create_models(
        api_key=os.environ["OPENAI_API_KEY"], reply_model=model, decision_model=model
    ).reply
    semaphore = asyncio.Semaphore(args.concurrency)
    selected = queries
    if args.only:
        requested = set(args.only.split(","))
        selected = [q for q in queries if q["case_id"] in requested]
        if len(selected) != len(requested):
            error = "Unknown selected question ID."
            raise ValueError(error)

    async with AsyncOpenAI(timeout=120, max_retries=2) as client:

        async def one(query: dict[str, Any]) -> None:
            case_id = query["case_id"]
            path = output / "predictions" / f"{case_id}.json"
            if path.exists() and not json.loads(path.read_text()).get("error"):
                return
            async with semaphore:
                context = runner.context_for(query)
                memory = cast("OpenVikingVariant", variants[runner.prefix_key(context)])
                session = ProductionAnswerSession(
                    archive,
                    memory.get_backend(),
                    context,
                    timezone=manifest["timezone"],
                )
                started = perf_counter()
                logger.info("Answering production OpenViking %s", case_id)
                try:
                    result = await production_result(
                        session,
                        query,
                        config,
                        output / "raw" / f"{case_id}.json",
                        client,
                        model,
                    )
                except Exception as exc:
                    logger.warning(
                        "Production case %s failed (%s)", case_id, type(exc).__name__
                    )
                    result = {
                        "error": type(exc).__name__,
                        "trace": getattr(session, "trace", []),
                        "seen_source_ids": sorted(
                            getattr(session, "seen_sources", set())
                        ),
                        "elapsed_seconds": perf_counter() - started,
                    }
                runner.write_json(
                    path, {"case_id": case_id, "variant": PRODUCTION} | result
                )
                logger.info("Answered production OpenViking %s", case_id)

        try:
            await asyncio.gather(*(one(query) for query in selected))
        finally:
            for memory in variants.values():
                await memory.close()


def main() -> None:
    # Keep the original comparison runner/grade fingerprints unchanged.
    runner.VARIANTS.update({
        "openviking": ("openviking", "OpenVikingVariant"),
        PRODUCTION: ("openviking", "OpenVikingVariant"),
    })
    args = runner.parser().parse_args()
    if args.variant != PRODUCTION or args.command != "run":
        if args.variant == PRODUCTION and args.command in {"prepare", "recall"}:
            error = "Prepare the shared index with --variant openviking."
            raise ValueError(error)
        runner.main()
        return
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    asyncio.run(run_production(args))


if __name__ == "__main__":
    main()
