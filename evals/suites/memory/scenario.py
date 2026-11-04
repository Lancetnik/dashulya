"""Replay historical-summary conversations through Dialogue and production tools."""

import argparse
import asyncio
import json
import logging
import os
import re
import sys
from collections.abc import Callable, Sequence
from contextlib import AsyncExitStack
from dataclasses import asdict
from datetime import datetime, timedelta
from operator import itemgetter
from pathlib import Path
from time import perf_counter
from typing import Any, override

from ag2 import Context
from ag2.config import ModelConfig
from ag2.eval import Trace
from ag2.events import ToolCallEvent
from ag2.middleware import Middleware
from ag2.testing import TestClient, TestConfig, TrackingConfig
from dishka import Provider, from_context, make_async_container, provide
from dishka_ag2 import AG2Provider, AG2Scope, DishkaAsyncMiddleware
from dotenv import load_dotenv
from openai import AsyncOpenAI
from psycopg import sql
from pydantic import BaseModel, Field, field_validator

from dashulya.ai.agents.factory import create_reply_agent
from dashulya.ai.agents.models import create_models
from dashulya.ai.dialogue import Dialogue
from dashulya.ai.embeddings import MemoryEmbeddings
from dashulya.ai.message_format import MessageFormatter
from dashulya.ai.tools.history import chat_history
from dashulya.ai.tools.memory import search_memory
from dashulya.database import Database
from dashulya.domain.memory import (
    ChatMemoryService,
    HistoryReader,
    MemoryAccess,
    MemoryBackend,
    MemoryMessage,
    ReplyContext,
    message_source_id,
)
from dashulya.domain.messages import ConversationKey, ReplyRequest
from dashulya.memory.service import HybridChatMemory
from dashulya.memory.store import MessageStore
from dashulya.memory.vectors import Embedder, EmbeddingCache
from dashulya.observability import JsonLogFormatter

from .__main__ import output_directory, write_json
from .diagnostics import (
    DiagnosticLogs,
    DiagnosticTrace,
    Diagnostics,
    active_diagnostics,
    active_tool,
)
from .report import markdown
from .runtime import database_url, isolated_database
from .scoring import tool_results
from .snapshot import copy_snapshot


class Scenario(BaseModel):
    case_id: str = Field(pattern=r"^[A-Za-z0-9_-]+$")
    session: str
    query: str
    chat_id: int
    user_id: int
    target_chat_id: int
    sent_at: datetime
    after: datetime
    before: datetime
    author_id: int | None = None
    max_prompt_tokens: int = 50_000
    max_model_calls: int = 12
    max_tool_calls: int = 24

    @field_validator("sent_at", "after", "before")
    @classmethod
    def timezone_required(cls, value: datetime) -> datetime:
        if value.utcoffset() is None:
            error_message = "Scenario timestamps require a timezone."
            raise ValueError(error_message)
        return value


class ScenarioTools(Provider):
    scope = AG2Scope.APP
    service = from_context(ChatMemoryService)
    formatter = from_context(MessageFormatter)

    @provide(scope=AG2Scope.SESSION)
    def access(self, context: Context) -> MemoryAccess:  # ruff: ignore[no-self-use] -- Dishka provider.
        access = context.dependencies.get("memory_access")
        if (
            type(access) is not MemoryAccess
            or access.chat_id == 0
            or access.is_private != (access.chat_id > 0)
        ):
            error_message = "Replay requires trusted Dialogue memory access."
            raise ValueError(error_message)
        return access

    @provide(scope=AG2Scope.SESSION)
    def recall(self, service: ChatMemoryService, access: MemoryAccess) -> MemoryBackend:  # ruff: ignore[no-self-use] -- Dishka provider.
        return service.recall(access)

    @provide(scope=AG2Scope.SESSION)
    def history(self, service: ChatMemoryService, access: MemoryAccess) -> HistoryReader:  # ruff: ignore[no-self-use] -- Dishka provider.
        return service.history(access)


class ObservedStore(MessageStore):
    @override
    async def context(
        self,
        key: ConversationKey,
        through_message_id: int,
        *,
        limit: int,
        parent_hops: int,
    ) -> ReplyContext:
        started = perf_counter()
        result = await super().context(
            key, through_message_id, limit=limit, parent_hops=parent_hops
        )
        if (diagnostics := active_diagnostics.get()) is not None:
            diagnostics.context_ms += round((perf_counter() - started) * 1000)
            diagnostics.initial_sources.update(
                message_source_id(f"telegram:{m.chat_id}", m.message_id)
                for m in result.messages
                if m.message_id != through_message_id
            )
        return result


async def catalog(database: Database) -> dict[str, dict[str, Any]]:
    async with database.connection() as connection:
        cursor = await connection.execute(
            """SELECT collection,message_id,chat_id,sent_at,author_id,url,is_bot,
                encode(sha256(convert_to(body,'UTF8')),'hex') AS text_sha256
            FROM messages ORDER BY collection,message_id"""
        )
        return {
            message_source_id(row["collection"], row["message_id"]): row
            for row in await cursor.fetchall()
        }


def references(value: Any) -> set[str]:
    if isinstance(value, str):
        return {value} if value.startswith(("source:", "https://t.me/")) else set()
    if isinstance(value, dict):
        return set().union(
            *(
                references([key, child])
                for key, child in value.items()
                if key
                not in {"filters", "coverage", "truncated_threads", "truncated_thread"}
            )
        )
    if isinstance(value, list):
        return set().union(*(references(v) for v in value))
    return set()


def interval_ms(items: list[dict[str, Any]]) -> int:
    end, total = 0, 0
    for item in sorted(items, key=itemgetter("start_ms")):
        stop = item["start_ms"] + item["duration_ms"]
        total += max(0, stop - max(end, item["start_ms"]))
        end = max(end, stop)
    return total


def score(
    case: Scenario,
    diagnostics: Diagnostics,
    answer: str,
    sources: dict[str, dict[str, Any]],
    error: str | None,
) -> dict[str, Any]:
    seen = set(diagnostics.initial_sources)
    sql_errors: list[dict[str, Any]] = []
    for _, value in tool_results(
        Trace(
            events=diagnostics.results, exception=None, duration_ms=diagnostics.offset()
        )
    ):
        seen.update(references(value))
        sql_errors.extend(
            item for item in value.get("results", []) if item.get("outcome") == "error"
        )
    visible = {
        key: row
        for key, row in sources.items()
        if (case.chat_id > 0 or row["chat_id"] == case.chat_id)
        and row["sent_at"] is not None
        and (
            (row["sent_at"], row["message_id"])
            < (case.sent_at, diagnostics.request_message_id)
            if row["chat_id"] == case.chat_id
            and diagnostics.request_message_id is not None
            else row["sent_at"] < case.sent_at
        )
    }
    known_urls = {row["url"] for row in visible.values() if row["url"]}
    seen_urls = {visible[key]["url"] for key in seen & visible.keys()} | {
        value for value in seen if value.startswith("https://t.me/")
    }
    citations = set(re.findall(r"https://t\.me/(?:c/\d+|[\w]+)/\d+", answer))
    unbounded = []
    for call in diagnostics.tools:
        if call["name"] != "search_memory" or call["arguments"].get("memory_id"):
            continue
        arguments = call["arguments"]
        try:
            lower = datetime.fromisoformat(arguments["after"])
            upper = datetime.fromisoformat(arguments["before"])
            bounded = lower >= case.after and upper <= case.before and lower <= upper
        except KeyError, TypeError, ValueError:
            bounded = False
        if not bounded or (
            case.author_id is not None and arguments.get("author_id") != case.author_id
        ):
            unbounded.append(call["tool_call_id"])
    batches_valid = all(
        isinstance(call["arguments"].get("sql"), list)
        and 1 <= len(call["arguments"]["sql"]) <= 5
        for call in diagnostics.tools
        if call["name"] == "chat_history"
    )
    peak_tokens = max(
        (m.get("prompt_tokens") or 0 for m in diagnostics.models), default=0
    )
    usage_complete = bool(diagnostics.models) and all(
        m.get("prompt_tokens") is not None for m in diagnostics.models
    )
    checks = {
        "execution": error is None and bool(answer.strip()),
        "uses_memory_tools": bool(diagnostics.tools),
        "tool_calls_succeeded": all(t.get("outcome") == "ok" for t in diagnostics.tools)
        and not sql_errors,
        "history_batches_valid": batches_valid,
        "chat_authority_hidden": all(
            not ({"chat", "chats"} & call["arguments"].keys())
            for call in diagnostics.tools
        ),
        "recall_filters_present": not unbounded,
        "retrieved_source_scope": all(
            key in visible for key in seen if key.startswith("source:")
        ),
        "citations_exist_in_scope": bool(citations) and citations <= known_urls,
        "citations_seen_by_agent": bool(citations) and citations <= seen_urls,
        "cites_requested_chat": any(
            row["url"] in citations and row["chat_id"] == case.target_chat_id
            for row in visible.values()
        ),
        "context_token_budget": peak_tokens <= case.max_prompt_tokens
        if usage_complete
        else None,
    }
    return {
        "passed": False
        if any(value is False for value in checks.values())
        else None
        if any(value is None for value in checks.values())
        else True,
        "checks": checks,
        "unbounded_recall_calls": unbounded,
        "sql_errors": len(sql_errors),
        "peak_prompt_tokens": peak_tokens if usage_complete else None,
        "citations": sorted(citations),
        "seen_source_count": len({
            key
            for key, row in visible.items()
            if key in seen or (row["url"] is not None and row["url"] in seen_urls)
        }),
        "period_originals": sum(
            row["chat_id"] == case.target_chat_id
            and row["sent_at"] is not None
            and case.after <= row["sent_at"] < case.before
            and (case.author_id is None or row["author_id"] == case.author_id)
            for row in sources.values()
        ),
        "semantic_quality": "not_scored_no_human_gold",
    }


async def _turn(
    case: Scenario,
    dialogue: Dialogue,
    store: MessageStore,
    diagnostics: Diagnostics,
    bot_id: int,
) -> str:
    started = perf_counter()
    async with store.database.connection() as connection:
        cursor = await connection.execute(
            "SELECT coalesce(max(message_id),0)+1 AS next_id FROM messages WHERE chat_id=%s",
            (case.chat_id,),
        )
        row = await cursor.fetchone()
        assert row is not None
        message_id = row["next_id"]
        diagnostics.request_message_id = message_id
    await store.record(
        MemoryMessage(
            case.chat_id,
            message_id,
            case.query,
            case.sent_at,
            case.user_id,
            "Evaluation owner",
            "Replay",
        )
    )
    diagnostics.phases["request_capture_ms"] = round((perf_counter() - started) * 1000)
    started = perf_counter()
    result = await dialogue.reply(
        ReplyRequest(
            (case.chat_id, None),
            case.query,
            case.user_id,
            "Evaluation owner",
            case.sent_at,
            message_id,
        )
    )
    diagnostics.phases["reply_ms"] = round((perf_counter() - started) * 1000)
    answer = "\n\n".join(reply.text for reply in result.messages)
    started = perf_counter()
    for index, reply in enumerate(result.messages, 1):
        await store.record(
            MemoryMessage(
                case.chat_id,
                message_id + index,
                reply.text,
                case.sent_at,
                bot_id,
                "Dashulya",
                "Replay",
                reply_to_id=message_id,
                is_bot=True,
            )
        )
    diagnostics.phases["delivery_record_ms"] = round((perf_counter() - started) * 1000)
    return answer


async def run_case(
    case: Scenario,
    dialogue: Dialogue,
    store: MessageStore,
    sources: dict[str, dict[str, Any]],
    *,
    bot_id: int,
) -> dict[str, Any]:
    diagnostics = Diagnostics(
        max_model_calls=case.max_model_calls, max_tool_calls=case.max_tool_calls
    )
    token = active_diagnostics.set(diagnostics)
    error, answer = None, ""
    try:
        answer = await _turn(case, dialogue, store, diagnostics, bot_id)
    except Exception as failure:
        error = type(failure).__name__
    finally:
        active_diagnostics.reset(token)
    elapsed = diagnostics.offset()
    model_ms = sum(m["duration_ms"] for m in diagnostics.models)
    tool_ms = interval_ms(diagnostics.tools)
    return {
        "case": case.model_dump(mode="json"),
        "request_message_id": diagnostics.request_message_id,
        "answer": answer,
        "error": error,
        "duration_ms": elapsed,
        "timing": {
            "model_ms": model_ms,
            "tool_wall_ms": tool_ms,
            "other_ms": max(0, elapsed - model_ms - tool_ms),
            "initial_context_ms": diagnostics.context_ms,
            **diagnostics.phases,
        },
        "evaluation": score(case, diagnostics, answer, sources, error),
        "models": diagnostics.models,
        "tools": diagnostics.tools,
        "logs": diagnostics.logs,
        "embeddings": diagnostics.embeddings,
    }


async def replay(
    database: Database,
    cases: Sequence[Scenario],
    config: ModelConfig,
    embedder: Embedder,
    *,
    bot_id: int,
    on_result: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    store = ObservedStore(database)

    async def measured_embedding(texts: list[str]) -> list[list[float]]:
        started = perf_counter()
        try:
            return await embedder(texts)
        finally:
            if (diagnostics := active_diagnostics.get()) is not None:
                diagnostics.embeddings.append({
                    "tool_call_id": active_tool.get(),
                    "duration_ms": round((perf_counter() - started) * 1000),
                    "texts": len(texts),
                    "input_chars": sum(len(text) for text in texts),
                })

    memory = HybridChatMemory(
        store,
        EmbeddingCache(database, measured_embedding),
        owner_user_id=cases[0].user_id,
    )
    await memory.prepare_cached()
    formatter = MessageFormatter("Europe/Moscow", bot_id, owner_id=cases[0].user_id)
    container = make_async_container(
        ScenarioTools(),
        AG2Provider(),
        scopes=AG2Scope,
        context={ChatMemoryService: memory, MessageFormatter: formatter},
        lock_factory=asyncio.Lock,
    )
    tracking = TrackingConfig(config)
    agent = create_reply_agent(
        tracking,
        tools=[search_memory, chat_history],
        middleware=[
            Middleware(DishkaAsyncMiddleware, container=container),
            Middleware(DiagnosticTrace),
        ],
    )
    sessions: dict[str, Dialogue] = {}
    reports = []
    try:
        for case in cases:
            if isinstance(config, ScriptedReplayConfig):
                config.select(case.case_id)
            dialogue = sessions.setdefault(
                case.session,
                Dialogue(agent, history=store, formatter=formatter, bot_id=bot_id),
            )
            sources = await catalog(database)
            call_start = len(tracking.calls)
            report = await run_case(case, dialogue, store, sources, bot_id=bot_id)
            for model, call in zip(
                report["models"], tracking.calls[call_start:], strict=True
            ):
                model["tools"] = [asdict(schema) for schema in call.tools]
            reports.append(report)
            if on_result is not None:
                on_result(report)
    finally:
        await container.close()
        await memory.close()
    return reports


def clean_json(value: Any, secrets: Sequence[str]) -> Any:
    record = logging.LogRecord(
        "eval", logging.INFO, __file__, 1, json.dumps(value, ensure_ascii=False), (), None
    )
    return json.loads(json.loads(JsonLogFormatter(secrets).format(record))["message"])


class ScriptedReplayConfig(TestConfig):
    def __init__(self, scripts: dict[str, TestConfig]) -> None:
        super().__init__()
        self.scripts = scripts
        self.selected: str | None = None

    def select(self, case_id: str) -> None:
        self.selected = case_id

    @override
    def create(self) -> TestClient:
        if self.selected is None:
            error_message = (
                "Select a scripted replay case before starting its Dialogue turn."
            )
            raise ValueError(error_message)
        return self.scripts[self.selected].create()


def scripted_config(
    cases: Sequence[Scenario], sources: dict[str, dict[str, Any]]
) -> ModelConfig:
    """A local pipeline probe, explicitly without model choice or semantic grading."""
    scripts = {}
    for case in cases:
        turns: list[Any] = []
        arguments: dict[str, Any] = {
            "query": case.query,
            "after": case.after.isoformat(),
            "before": (case.before - timedelta(microseconds=1)).isoformat(),
        }
        if case.author_id is not None:
            arguments["author_id"] = case.author_id
        turns.append([
            ToolCallEvent(
                "search_memory",
                id=case.case_id + f"-recall-{i}",
                arguments=json.dumps(
                    {**arguments, "query": case.query + f" (probe {i})"},
                    ensure_ascii=False,
                ),
            )
            for i in range(2)
        ])
        predicate = sql.SQL(
            "chat_id={} AND sent_at>=TIMESTAMPTZ {} AND sent_at<TIMESTAMPTZ {}"
        ).format(
            sql.Literal(case.target_chat_id),
            sql.Literal(case.after.isoformat()),
            sql.Literal(case.before.isoformat()),
        )
        if case.author_id is not None:
            predicate += sql.SQL(" AND author_id={}").format(sql.Literal(case.author_id))
        turns.append(
            ToolCallEvent(
                "chat_history",
                id=case.case_id + "-history",
                arguments=json.dumps({
                    "sql": [
                        (
                            sql.SQL("SELECT count(*) AS total FROM messages WHERE ")
                            + predicate
                        ).as_string(),
                        (
                            sql.SQL(
                                "SELECT source_id,chat_id,message_id,sent_at,author_id,text,url FROM messages WHERE "
                            )
                            + predicate
                            + sql.SQL(" ORDER BY sent_at,message_id LIMIT 50")
                        ).as_string(),
                    ]
                }),
            )
        )
        url = next(
            (
                row["url"]
                for row in sources.values()
                if row["chat_id"] == case.target_chat_id
                and row["sent_at"] is not None
                and case.after <= row["sent_at"] < case.before
                and row["url"]
            ),
            None,
        )
        text = "Scripted pipeline probe; no semantic summary was generated."
        if url:
            text += f" [Original]({url})"
        turns.append(
            json.dumps({
                "messages": [{"text": text, "reply_to_message_id": None, "sticker": None}]
            })
        )
        scripts[case.case_id] = TestConfig(*turns)
    return ScriptedReplayConfig(scripts)


async def synthetic_query_vectors(texts: list[str]) -> list[list[float]]:
    """Disposable offline probe vectors, never a semantic-quality measurement."""
    return [[1.0] + [0.0] * 511 for _ in texts]


async def run(args: argparse.Namespace) -> Path:
    cases = [
        Scenario.model_validate_json(line)
        for line in args.cases.read_text().splitlines()
        if line.strip()
    ]
    if not cases or any(
        case.after >= case.before or case.before > case.sent_at for case in cases
    ):
        error_message = (
            "Scenarios require a nonempty historical period before the request."
        )
        raise ValueError(error_message)
    target = output_directory(args.out)
    source = os.environ.get("EVAL_SOURCE_DATABASE_URL") or os.environ.get(
        "DATABASE_URL", ""
    )
    key = os.environ.get("OPENAI_API_KEY", "")
    if not source or (not args.scripted and (not key or not args.model)):
        error_message = "Set source/evaluation database URLs, OPENAI_API_KEY and a model."
        raise ValueError(error_message)
    secrets = [
        value
        for name in (
            "OPENAI_API_KEY",
            "BOT_TOKEN",
            "POSTGRES_PASSWORD",
            "POSTGRES_APP_PASSWORD",
        )
        if (value := os.environ.get(name))
    ]
    handler = DiagnosticLogs(secrets)
    logger = logging.getLogger("dashulya")
    level = logger.level
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    try:
        async with (
            isolated_database(database_url()) as database,
            AsyncExitStack() as resources,
        ):
            snapshot = await copy_snapshot(
                source, database, before=min(case.sent_at for case in cases)
            )
            write_json(target / "snapshot.json", snapshot)
            if args.scripted:
                config = scripted_config(cases, await catalog(database))
                embedder = synthetic_query_vectors
            else:
                client = await resources.enter_async_context(
                    AsyncOpenAI(api_key=key, timeout=60, max_retries=1)
                )
                config = create_models(
                    api_key=key, reply_model=args.model, decision_model=args.model
                ).reply
                embedder = MemoryEmbeddings(client)

            def save_report(report: dict[str, Any]) -> None:
                write_json(
                    target / (report["case"]["case_id"] + ".json"),
                    clean_json(report, secrets),
                )
                sys.stdout.write(
                    json.dumps({
                        "case_id": report["case"]["case_id"],
                        "duration_ms": report["duration_ms"],
                        "checks": report["evaluation"]["checks"],
                        "error": report["error"],
                    })
                    + "\n"
                )
                sys.stdout.flush()

            reports = await replay(
                database,
                cases,
                config,
                embedder,
                bot_id=int(os.environ.get("BOT_TOKEN", "0:").split(":")[0]),
                on_result=save_report,
            )
            summary: dict[str, Any] = {
                "model": "scripted" if args.scripted else args.model,
                "mode": "scripted_pipeline_with_synthetic_query_vectors"
                if args.scripted
                else "live_model_and_query_embeddings",
                "cases": [
                    {
                        key: report[key]
                        for key in (
                            "case",
                            "duration_ms",
                            "timing",
                            "evaluation",
                            "error",
                        )
                    }
                    for report in reports
                ],
                "scope": "Dialogue + real model/tools/PostgreSQL; text only; Telegram delivery simulated; no semantic gold",
            }
            write_json(target / "summary.json", clean_json(summary, secrets))
            (target / "report.md").write_text(
                markdown(reports, snapshot, mode=summary["mode"])
            )
    finally:
        logger.removeHandler(handler)
        logger.setLevel(level)
        handler.close()
    return target


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL"))
    parser.add_argument(
        "--scripted",
        action="store_true",
        help="Local pipeline probe; no model/embedding API calls or semantic-quality measurement.",
    )
    args = parser.parse_args()
    try:
        target = asyncio.run(run(args))
    except Exception as error:
        sys.stderr.write(
            f"Replay failed ({type(error).__name__}); provider and database details omitted.\n"
        )
        return 1
    sys.stdout.write(f"Replay artifacts: {target}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
