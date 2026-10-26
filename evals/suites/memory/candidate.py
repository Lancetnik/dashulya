"""Evaluate the replacement production path, independently of frozen baselines."""

import argparse
import asyncio
import copy
import hashlib
import json
import logging
import os
import re
from collections.abc import AsyncIterator
from dataclasses import asdict, replace
from datetime import timedelta
from pathlib import Path
from time import perf_counter
from typing import Any, cast

from ag2 import Agent
from ag2.config import ModelConfig
from ag2.middleware import Middleware
from aiogram import Bot
from dishka import Provider, make_async_container, provide
from dishka_ag2 import AG2Provider, AG2Scope
from openai import AsyncOpenAI

from dashulya.ai.agents.models import AgentModels, create_models
from dashulya.ai.dialogue import Dialogue
from dashulya.ai.embeddings import MemoryEmbeddings
from dashulya.domain.memory import (
    ChatHistory,
    ChatMemoryService,
    HistoryReader,
    MemoryAccess,
    MemoryBackend,
    MemoryMessage,
)
from dashulya.domain.messages import ConversationKey, Reply
from dashulya.main.config import Settings
from dashulya.main.providers.ai import AgentProvider
from dashulya.memory.history import CanonicalHistory
from dashulya.memory.live import LiveMemory
from dashulya.memory.service import HybridChatMemory
from dashulya.memory.vectors import EmbeddingCache
from evals.suites.memory import runner
from evals.suites.memory.corpus import SourceArchive, aware_time, source_json
from evals.suites.memory.current_baseline import production_result
from evals.suites.memory.production import (
    ProductionAnswerSession,
    _CorpusHistory,
    _UsageCapture,
    _citations,
)
from evals.suites.memory.types import QueryContext, SourceMessage

logger = logging.getLogger(__name__)
CANDIDATE = "candidate_production"
BOT_ID = 99999999999


def fingerprints() -> dict[str, str]:
    root = Path(__file__).resolve().parents[3]
    paths = [*sorted((root / "src/dashulya").rglob("*.py")), Path(__file__).resolve()]
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in paths
    }


def _username(message: SourceMessage) -> str | None:
    return message.author if re.fullmatch(r"[A-Za-z0-9_]{1,32}", message.author) else None


def _message(message: SourceMessage, chat_id: int) -> MemoryMessage:
    return MemoryMessage(
        chat_id,
        message.id,
        message.text,
        aware_time(message.date),
        message.user_id,
        message.user_name or message.author,
        message.chat_key,
        chat_username=message.chat_key,
        reply_to_id=message.reply_to_msg_id,
        update_id=message.id,
        username=_username(message),
        is_bot=message.is_bot,
    )


async def prepare_candidate_history(
    archive: SourceArchive, context: QueryContext, workspace: Path, *, chat_id: int = -1
) -> LiveMemory:
    """Build/resume a source-only prefix journal; never read benchmark annotations."""
    await asyncio.to_thread(workspace.mkdir, parents=True, exist_ok=True)
    messages = archive.available(context)
    digest = hashlib.sha256(
        json.dumps(
            [asdict(m) for m in messages], sort_keys=True, ensure_ascii=False
        ).encode()
    ).hexdigest()
    expected = {
        "source_sha256": digest,
        "scope": asdict(context),
        "chat_id": chat_id,
        "source_count": len(messages),
    }
    manifest_path = workspace / "fixture.json"
    path = workspace / "live.sqlite3"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if previous["inputs"] != expected:
            error = "Candidate fixture inputs changed; use a fresh workspace."
            raise ValueError(error)
    elif path.exists():
        error = "An unmanifested candidate journal cannot be overwritten."
        raise FileExistsError(error)
    else:
        runner.write_json(manifest_path, {"inputs": expected, "complete": False})
    live = LiveMemory(path)
    await live.initialize()
    if not json.loads(manifest_path.read_text())["complete"]:
        for index, message in enumerate(messages, 1):
            await live.record(_message(message, chat_id))
            if index % 1000 == 0:
                logger.info("Prepared candidate sources %d/%d", index, len(messages))
        runner.write_json(manifest_path, {"inputs": expected, "complete": True})
    actual = await CanonicalHistory(live, MemoryAccess(42, True)).originals()
    wanted = {m.id: m for m in messages}
    if len(actual) != len(messages) or any(
        m.chat_id != chat_id
        or m.message_id not in wanted
        or m.text != wanted[m.message_id].text
        or m.author_id != wanted[m.message_id].user_id
        or m.sent_at != aware_time(wanted[m.message_id].date)
        or m.author != (wanted[m.message_id].user_name or wanted[m.message_id].author)
        or m.username != _username(wanted[m.message_id])
        or m.is_bot != wanted[m.message_id].is_bot
        or m.reply_to_id != wanted[m.message_id].reply_to_msg_id
        for m in actual
    ):
        error = "Candidate fixture canonical rows differ from their source prefix."
        raise ValueError(error)
    return live


class _CandidateHistory(_CorpusHistory):
    async def context(
        self,
        key: ConversationKey,
        through_message_id: int,
        *,
        limit: int,
        parent_hops: int,
    ) -> list[MemoryMessage]:
        messages = await super().context(
            key, through_message_id, limit=limit, parent_hops=parent_hops
        )
        return [
            replace(
                m,
                author=self.by_id[m.message_id].user_name
                or self.by_id[m.message_id].author,
                username=_username(self.by_id[m.message_id]),
                is_bot=self.by_id[m.message_id].is_bot,
            )
            for m in messages
        ]


class _RecordingService:
    def __init__(
        self,
        service: ChatMemoryService,
        archive: SourceArchive,
        context: QueryContext,
        chat_id: int,
        trace: list[dict[str, Any]],
        seen: set[int],
    ) -> None:
        self.service, self.archive, self.context, self.chat_id = (
            service,
            archive,
            context,
            chat_id,
        )
        self.trace, self.seen = trace, seen

    def _observe(self, value: Any) -> None:
        if isinstance(value, list):
            for item in value:
                self._observe(item)
        elif isinstance(value, dict):
            if "message_id" in value and "text" in value:
                original = self.archive.get(int(value["message_id"]), self.context)
                offset = value.get("offset", value.get("excerpt_start", 0))
                text = value["text"]
                error = "Candidate tool exposed noncanonical or out-of-scope original evidence."
                if (
                    original is None
                    or type(offset) is not int
                    or offset < 0
                    or not isinstance(text, str)
                ):
                    raise ValueError(error)
                observed = {
                    "author_id": value.get("author_id", value.get("user_id")),
                    "sent_at": aware_time(value["sent_at"]),
                    "chat_id": value.get("chat_id", self.chat_id),
                    "url": value.get("url"),
                }
                expected = {
                    "author_id": original.user_id,
                    "sent_at": aware_time(original.date),
                    "chat_id": self.chat_id,
                    "url": f"https://t.me/{self.context.chat_key}/{original.id}",
                }
                if (
                    original.text[offset : offset + len(text)] != text
                    or observed != expected
                ):
                    raise ValueError(error)
                self.seen.add(original.id)
            for item in value.values():
                if isinstance(item, (dict, list)):
                    self._observe(item)

    async def call(
        self, tool: str, action: Any, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        entry: dict[str, Any] = {"tool": tool, "arguments": arguments}
        self.trace.append(entry)
        try:
            result: dict[str, Any] = await action(**arguments)
            self._observe(result)
        except Exception as error:
            entry["error_type"] = type(error).__name__
            raise
        entry["result"] = copy.deepcopy(result)
        return result

    def recall(self, access: MemoryAccess) -> MemoryBackend:
        parent = self
        backend = self.service.recall(access)

        class Recall:
            def __init__(self) -> None:
                self.parent, self.backend = parent, backend

            async def search(
                self,
                query: str = "",
                *,
                memory_id: str | None = None,
                author_id: int | None = None,
                after: str | None = None,
                before: str | None = None,
                limit: int = 3,
            ) -> dict[str, Any]:
                return await self.parent.call(
                    "search_memory",
                    self.backend.search,
                    {
                        "query": query,
                        "memory_id": memory_id,
                        "author_id": author_id,
                        "after": after,
                        "before": before,
                        "limit": limit,
                    },
                )

        return Recall()

    def history(self, access: MemoryAccess) -> HistoryReader:
        parent = self
        backend = self.service.history(access)

        class History:
            def __init__(self) -> None:
                self.parent, self.backend = parent, backend

            async def query(self, action: str = "list", **kwargs: Any) -> dict[str, Any]:
                return await self.parent.call(
                    "chat_history", self.backend.query, {"action": action, **kwargs}
                )

        return History()


class _Dependencies(Provider):
    scope = AG2Scope.APP

    def __init__(
        self,
        service: ChatMemoryService,
        history: ChatHistory,
        config: ModelConfig,
        bot_id: int,
    ) -> None:
        super().__init__()
        self.service, self.history_store, self.config, self.bot_id = (
            service,
            history,
            config,
            bot_id,
        )

    @provide
    def memory(self) -> ChatMemoryService:
        return self.service

    @provide
    def history(self) -> ChatHistory:
        return self.history_store

    @provide(override=True)
    def models(self) -> AgentModels:
        return AgentModels(self.config, self.config)

    @provide
    async def bot(self) -> AsyncIterator[Bot]:
        bot = Bot(f"{self.bot_id}:TEST")
        try:
            yield bot
        finally:
            await bot.session.close()


class CandidateAnswerSession:
    def __init__(
        self,
        archive: SourceArchive,
        service: ChatMemoryService,
        context: QueryContext,
        *,
        timezone: str = "Europe/Moscow",
        bot_id: int = BOT_ID,
        requester_id: int = 42,
        requester_name: str = "Benchmark requester",
        chat_id: int = -1,
    ) -> None:
        self.archive, self.service, self.context = archive, service, context
        self.timezone, self.bot_id = timezone, bot_id
        self.requester_id, self.requester_name, self.chat_id = (
            requester_id,
            requester_name,
            chat_id,
        )
        self.trace: list[dict[str, Any]] = []
        self.seen_sources: set[int] = set()

    def _current(self, query: str) -> SourceMessage:
        if self.context.before_id:
            current = self.archive.by_key.get((
                self.context.chat_key,
                self.context.before_id,
            ))
            if (
                current is None
                or current.text != query
                or aware_time(current.date) != aware_time(self.context.before)
                or current.user_id is None
            ):
                error = (
                    "Historical candidate request differs from its canonical original."
                )
                raise ValueError(error)
            return current
        available = self.archive.available(self.context)
        return SourceMessage(
            max((m.id for m in available), default=0) + 1,
            self.context.chat_key,
            (aware_time(self.context.before) + timedelta(microseconds=1)).isoformat(),
            query,
            self.requester_id,
            self.requester_name,
            self.requester_name,
        )

    async def answer(
        self,
        query: str,
        config: ModelConfig,
        *,
        input_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        started = perf_counter()
        for field in ("preceding_messages", "reply_ancestors"):
            for item in (input_context or {}).get(field, []):
                original = self.archive.get(int(item["id"]), self.context)
                if original is None or original.text != (item.get("text") or ""):
                    error = (
                        "Candidate request context differs from its canonical original."
                    )
                    raise ValueError(error)
        current = self._current(query)
        history = _CandidateHistory(self.archive, self.context, current, self.chat_id)
        await history.context((self.chat_id, None), current.id, limit=20, parent_hops=5)
        selected = [m for m in history.selected if m.id != current.id]
        self.seen_sources.update(m.id for m in selected)
        self.trace.append({
            "tool": "production_group_context",
            "arguments": {"limit": 20, "parent_hops": 5},
            "result": {
                "messages": [source_json(m, max_chars=len(m.text)) for m in selected]
            },
        })
        recording = _RecordingService(
            self.service,
            self.archive,
            self.context,
            self.chat_id,
            self.trace,
            self.seen_sources,
        )
        container = make_async_container(
            AgentProvider(),
            AG2Provider(),
            _Dependencies(recording, history, config, self.bot_id),
            scopes=AG2Scope,
            lock_factory=asyncio.Lock,
            context={
                Settings: Settings(
                    bot_token=f"{self.bot_id}:TEST",
                    openai_api_key="test-key",  # pragma: allowlist secret -- Offline settings fixture; injected real model config owns its key.
                    owner_user_id=self.requester_id,
                    bot_timezone=self.timezone,
                )
            },
        )
        usage: dict[str, Any] = {}
        try:
            agent = await container.get(Agent[Reply])
            agent.add_middleware(Middleware(_UsageCapture, usage=usage))
            dialogue = await container.get(Dialogue)
            group: dict[str, object] = {"message_id": current.id}
            if current.reply_to_msg_id is not None:
                group["reply_to"] = {"message_id": current.reply_to_msg_id}
            reply = await dialogue.reply(
                (self.chat_id, None),
                query,
                user_id=current.user_id or self.requester_id,
                user_name=current.user_name or current.author,
                sent_at=aware_time(current.date),
                group_context=group,
                message_id=current.id,
            )
        finally:
            await container.close()
        return {
            "answer": {
                "text": reply.text,
                "cited_message_ids": _citations(reply.text, self.context.chat_key),
                "reply_to_message_id": reply.reply_to_message_id,
                "abstained": None,
                "aggregation_count": None,
            },
            "raw_reply": asdict(reply),
            "trace": self.trace,
            "seen_source_ids": sorted(self.seen_sources),
            "usage": usage,
            "elapsed_seconds": perf_counter() - started,
            "production": {
                "current_request": source_json(current, max_chars=len(current.text)),
                "chat_id": self.chat_id,
                "chat_id_is_experiment_local": True,
                "bot_id_is_experiment_local": True,
                "synthetic_request": not bool(self.context.before_id),
                "tools": ["search_memory", "chat_history"],
                "automatic_initial_search": False,
                "tool_budget": None,
                "topic_mapping": "No real forum-topic IDs supplied; thread_post_id is not a forum topic.",
            },
        }

    async def close(self) -> None:
        """The caller owns the selected memory service and its model clients."""


async def run_candidate(args: argparse.Namespace) -> None:
    archive, queries = runner.inputs(args)
    output = args.workspace / "runs" / args.run_id
    model = os.getenv("OPENAI_MODEL", "gpt-6-luna")
    manifest = {
        "run_id": args.run_id,
        "variant": CANDIDATE,
        "model": model,
        "model_api": "responses",
        "corpus_sha256": runner.fingerprint(args.corpus),
        "queries_sha256": runner.fingerprint(args.queries),
        "code_sha256": runner.code_fingerprint(),
        "production_sha256": fingerprints(),
        "expected_case_ids": [q["case_id"] for q in queries],
        "tools": ["search_memory", "chat_history"],
        "automatic_initial_search": False,
        "tool_budget": None,
        "reply_timeout_seconds": 90,
        "model_timeout_seconds": 60,
        "model_max_retries": 1,
        "timezone": os.getenv("BOT_TIMEZONE", "Europe/Moscow"),
        "evaluation_normalizer_has_gold_or_sources": False,
        "synthetic_chat_id": -1,
    }
    manifest_path = output / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        error = "Candidate run inputs/code changed; use a fresh run ID."
        raise ValueError(error)
    runner.write_json(manifest_path, manifest)
    selected = [
        q for q in queries if not args.only or q["case_id"] in set(args.only.split(","))
    ]
    if args.only and len(selected) != len(set(args.only.split(","))):
        error = "Unknown selected candidate question."
        raise ValueError(error)
    async with AsyncOpenAI(timeout=60, max_retries=1) as client:
        services = {}
        try:
            for query in queries:
                context = runner.context_for(query)
                key = runner.prefix_key(context)
                if key not in services:
                    live = await prepare_candidate_history(
                        archive, context, args.workspace / "candidate-fixtures" / key
                    )
                    vectors = EmbeddingCache(
                        args.workspace / "candidate-vectors.sqlite3",
                        MemoryEmbeddings(client),
                    )
                    service = HybridChatMemory(live, vectors, owner_user_id=42)
                    await service.start(require_ready=True)
                    services[key] = service
            if args.command == "prepare":
                return
            config = create_models(
                api_key=os.environ["OPENAI_API_KEY"],
                reply_model=model,
                decision_model=model,
            ).reply
            semaphore = asyncio.Semaphore(args.concurrency)

            async def one(query: dict[str, Any]) -> None:
                path = output / "predictions" / f"{query['case_id']}.json"
                if path.exists() and not json.loads(path.read_text()).get("error"):
                    return
                async with semaphore:
                    context = runner.context_for(query)
                    session = CandidateAnswerSession(
                        archive,
                        services[runner.prefix_key(context)],
                        context,
                        timezone=manifest["timezone"],
                    )
                    try:
                        result = await production_result(
                            cast("ProductionAnswerSession", session),
                            query,
                            config,
                            output / "raw" / f"{query['case_id']}.json",
                            client,
                            model,
                        )
                    except Exception as error:
                        result = {
                            "error": type(error).__name__,
                            "trace": session.trace,
                            "seen_source_ids": sorted(session.seen_sources),
                        }
                    if fingerprints() != manifest["production_sha256"]:
                        failure = "Production source changed during candidate evaluation."
                        raise ValueError(failure)
                    runner.write_json(
                        path,
                        {"case_id": query["case_id"], "variant": CANDIDATE, **result},
                    )
                    logger.info("Answered replacement production %s", query["case_id"])

            await asyncio.gather(*(one(query) for query in selected))
        finally:
            for service in services.values():
                await service.close()


def main() -> None:
    runner.VARIANTS[CANDIDATE] = ("raw_hybrid", "RawHybridVariant")
    args = runner.parser().parse_args()
    if args.variant != CANDIDATE or args.command in {"grade", "report"}:
        runner.main()
        return
    if args.command == "recall":
        error = "Candidate production evaluates actual tool use; use run or prepare."
        raise ValueError(error)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpx2").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    asyncio.run(run_candidate(args))


if __name__ == "__main__":
    main()
