"""Run the current production reply path against immutable offline chat sources.

No evaluation prompt, automatic recall, extra tool, Telegram delivery or access
classifier is added. The caller supplies the already authorized request and an
OpenVikingMemory backend built from the corresponding immutable source prefix.
"""

import asyncio
import copy
import re
from dataclasses import asdict
from datetime import timedelta
from time import perf_counter
from typing import Any

from ag2 import Context
from ag2.config import ModelConfig
from ag2.events import BaseEvent, ModelResponse
from ag2.middleware import AgentTurn, BaseMiddleware, Middleware
from ag2.usage import UsageReport
from dishka import Provider, make_async_container, provide
from dishka_ag2 import AG2Provider, AG2Scope, DishkaAsyncMiddleware

from dashulya.ai.agents.factory import create_reply_agent
from dashulya.ai.dialogue import Dialogue
from dashulya.ai.tools.memory import search_memory
from dashulya.domain.memory import MemoryBackend, MemoryMessage
from dashulya.domain.messages import ConversationKey
from evals.suites.memory.corpus import SourceArchive, aware_time, source_json
from evals.suites.memory.types import QueryContext, SourceMessage


class _BackendProvider(Provider):
    scope = AG2Scope.APP

    def __init__(self, backend: MemoryBackend) -> None:
        super().__init__()
        self.backend = backend

    @provide
    def memory(self) -> MemoryBackend:
        return self.backend


class _UsageCapture(BaseMiddleware):
    def __init__(
        self, event: BaseEvent, context: Context, *, usage: dict[str, Any]
    ) -> None:
        super().__init__(event, context)
        self.usage = usage

    async def on_turn(
        self, call_next: AgentTurn, event: BaseEvent, context: Context
    ) -> ModelResponse:
        result = await call_next(event, context)
        self.usage.update(
            asdict(UsageReport.from_events(await context.stream.history.get_events()))
        )
        return result


class _RecordingBackend:
    """Observe the production tool without expanding or rewriting its result."""

    def __init__(
        self, backend: MemoryBackend, archive: SourceArchive, scope: QueryContext
    ) -> None:
        self.backend = backend
        self.archive = archive
        self.scope = scope
        self.trace: list[dict[str, Any]] = []
        self.seen: set[int] = set()

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
        arguments = {
            "query": query,
            "memory_id": memory_id,
            "author_id": author_id,
            "after": after,
            "before": before,
            "limit": limit,
        }
        entry: dict[str, Any] = {"tool": "search_memory", "arguments": arguments}
        self.trace.append(entry)
        try:
            result = await self.backend.search(
                query,
                memory_id=memory_id,
                author_id=author_id,
                after=after,
                before=before,
                limit=limit,
            )
        except Exception as error:
            entry["error_type"] = type(error).__name__
            raise
        exposed: set[int] = set()
        for hit in result.get("hits", []):
            for field in ("sources", "context"):
                for evidence in hit.get(field, []):
                    message = self.archive.get(int(evidence["message_id"]), self.scope)
                    failure = "Production backend exposed noncanonical or out-of-scope evidence."
                    if message is None:
                        raise ValueError(failure)
                    expected = {
                        "user_id": message.user_id,
                        "reply_to_message_id": message.reply_to_msg_id,
                        "url": f"https://t.me/{self.scope.chat_key}/{message.id}",
                    }
                    sent_at = evidence.get("sent_at")
                    correct_date = isinstance(sent_at, str) and aware_time(
                        sent_at
                    ) == aware_time(message.date)
                    text = evidence.get("text")
                    if (
                        not isinstance(text, str)
                        or text not in message.text
                        or not correct_date
                        or any(evidence.get(k) != v for k, v in expected.items())
                    ):
                        raise ValueError(failure)
                    exposed.add(message.id)
        self.seen.update(exposed)
        entry["result"] = copy.deepcopy(result)
        return result


class _CorpusHistory:
    """Mirror LiveMemory.context's latest IDs and five reply-parent hops."""

    def __init__(
        self,
        archive: SourceArchive,
        scope: QueryContext,
        current: SourceMessage,
        chat_id: int,
    ) -> None:
        self.scope = scope
        self.current = current
        self.chat_id = chat_id
        # The export's thread_post_id is a channel-post relation, not a Telegram
        # forum message_thread_id. The benchmark models one ordinary group topic.
        self.by_id = {m.id: m for m in archive.available(scope)}
        self.by_id[current.id] = current
        self.selected: list[SourceMessage] = []

    async def context(
        self,
        key: ConversationKey,
        through_message_id: int,
        *,
        limit: int,
        parent_hops: int,
    ) -> list[MemoryMessage]:
        if key != (self.chat_id, None) or through_message_id != self.current.id:
            error = "Production history request differs from its fixed conversation."
            raise ValueError(error)
        recent = sorted(i for i in self.by_id if i <= through_message_id)[-limit:]
        selected = set(recent)
        for message_id in recent:
            child = self.by_id[message_id]
            for _ in range(parent_hops):
                parent = self.by_id.get(child.reply_to_msg_id or -1)
                if parent is None or parent.id >= child.id:
                    break
                selected.add(parent.id)
                child = parent
        self.selected = [self.by_id[i] for i in sorted(selected)]
        return [
            MemoryMessage(
                chat_id=self.chat_id,
                message_id=m.id,
                text=m.text,
                sent_at=aware_time(m.date),
                author_id=m.user_id,
                author=m.user_name or m.author,
                chat_title=self.scope.chat_key,
                chat_username=self.scope.chat_key,
                reply_to_id=m.reply_to_msg_id,
            )
            for m in self.selected
        ]


def _citations(text: str, chat_key: str) -> list[int]:
    ids = {
        int(match.group(1))
        for match in re.finditer(
            rf"https?://t\.me/{re.escape(chat_key)}/(\d+)(?:\b|/)", text
        )
    }
    # Numeric link labels on shared resources are not original-message citations.
    ids.update(int(i) for i in re.findall(r"\[(\d+)\](?!\()", text))
    return sorted(ids)


class ProductionAnswerSession:
    """One isolated production turn; backend ownership remains with the caller."""

    def __init__(
        self,
        archive: SourceArchive,
        backend: MemoryBackend,
        context: QueryContext,
        *,
        timezone: str = "Europe/Moscow",
        bot_id: int | None = None,
        requester_id: int = 42,
        requester_name: str = "Benchmark requester",
        chat_id: int = -1,
    ) -> None:
        self.archive = archive
        self.context = context
        self.backend = backend
        self.timezone = timezone
        self.bot_id = bot_id
        self.requester_id = requester_id
        self.requester_name = requester_name
        self.chat_id = chat_id
        self.trace: list[dict[str, Any]] = []
        self.seen_sources: set[int] = set()

    def _current(self, query: str) -> SourceMessage:
        existing = self.archive.by_key.get((
            self.context.chat_key,
            self.context.before_id,
        ))
        if self.context.before_id:
            if (
                existing is None
                or existing.text != query
                or aware_time(existing.date) != aware_time(self.context.before)
                or existing.user_id is None
            ):
                error = "Historical production request must match its canonical original."
                raise ValueError(error)
            return existing
        available = self.archive.available(self.context)
        return SourceMessage(
            id=max((m.id for m in available), default=0) + 1,
            chat_key=self.context.chat_key,
            date=(
                aware_time(self.context.before) + timedelta(microseconds=1)
            ).isoformat(),
            text=query,
            user_id=self.requester_id,
            author=self.requester_name,
            user_name=self.requester_name,
        )

    async def answer(
        self,
        query: str,
        config: ModelConfig,
        *,
        input_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        started = perf_counter()
        # Annotations never select production context: reconstruct it canonically.
        for field in ("preceding_messages", "reply_ancestors"):
            for supplied in (input_context or {}).get(field, []):
                original = self.archive.get(int(supplied["id"]), self.context)
                if original is None or original.text != (supplied.get("text") or ""):
                    error = "Supplied request context differs from its canonical source."
                    raise ValueError(error)
        current = self._current(query)
        history = _CorpusHistory(self.archive, self.context, current, self.chat_id)
        recording = _RecordingBackend(self.backend, self.archive, self.context)
        self.trace = recording.trace
        self.seen_sources = recording.seen
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
        usage: dict[str, Any] = {}
        container = make_async_container(
            _BackendProvider(recording),
            AG2Provider(),
            scopes=AG2Scope,
            lock_factory=asyncio.Lock,
        )
        try:
            agent = create_reply_agent(
                config,
                tools=[search_memory],
                middleware=[
                    Middleware(DishkaAsyncMiddleware, container=container),
                    Middleware(_UsageCapture, usage=usage),
                ],
            )
            dialogue = Dialogue(
                agent, timezone=self.timezone, history=history, bot_id=self.bot_id
            )
            group_context: dict[str, object] = {"message_id": current.id}
            if current.reply_to_msg_id is not None:
                group_context["reply_to"] = {"message_id": current.reply_to_msg_id}
            reply = await dialogue.reply(
                (self.chat_id, None),
                query,
                user_id=current.user_id or self.requester_id,
                user_name=current.user_name or current.author,
                sent_at=aware_time(current.date),
                group_context=group_context,
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
            "seen_source_ids": sorted(recording.seen),
            "usage": usage,
            "elapsed_seconds": perf_counter() - started,
            "production": {
                "current_request": source_json(current, max_chars=len(current.text)),
                "synthetic_request": self.context.before_id == 0,
                "chat_id": self.chat_id,
                "chat_id_is_experiment_local": self.chat_id == -1,
                "selected_context_ids": [m.id for m in history.selected],
                "bot_id": self.bot_id,
                "timezone": self.timezone,
                "assumptions": [
                    "Authorized direct group reply; no access/addressing classifier or Telegram delivery.",
                    "Offline snapshot supplies observed history; one ordinary group topic.",
                    "SourceMessage lacks attachment types, so historical media placeholders cannot be reconstructed.",
                    "Foreign bot flags are retained in traces; production history renders their sender identities.",
                    "Production Reply has no count or abstention fields; null means unmeasured.",
                ],
            },
        }

    async def close(self) -> None:
        """No persistent session resources; the caller owns the shared backend."""
