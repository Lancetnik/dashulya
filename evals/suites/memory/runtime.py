"""Run production memory tools against a disposable, cutoff-specific database."""

import asyncio
import json
import os
from collections.abc import AsyncGenerator, Sequence
from contextlib import asynccontextmanager
from datetime import timedelta
from time import perf_counter
from typing import Any
from uuid import uuid4

from ag2 import Agent, Context, MemoryStream
from ag2.config import ModelConfig
from ag2.eval import Trace
from ag2.events import (
    BaseEvent,
    ModelRequest,
    ModelResponse,
    TextInput,
    ToolCallEvent,
    ToolErrorEvent,
    ToolResultEvent,
    UsageEvent,
)
from ag2.middleware import Middleware
from dishka import Provider, from_context, make_async_container, provide
from dishka_ag2 import AG2Provider, AG2Scope, DishkaAsyncMiddleware
from openai import AsyncOpenAI
from psycopg import AsyncConnection, sql
from psycopg.conninfo import conninfo_to_dict, make_conninfo

from dashulya.ai.agents.factory import create_reply_agent
from dashulya.ai.embeddings import MemoryEmbeddings
from dashulya.ai.message_format import MessageFormatter, format_local_time
from dashulya.ai.tools.history import chat_history
from dashulya.ai.tools.memory import search_memory
from dashulya.database import Database
from dashulya.database.migrate import migrate
from dashulya.domain.memory import (
    ChatMemoryService,
    HistoryReader,
    MemoryAccess,
    MemoryBackend,
    OriginalMessage,
)
from dashulya.domain.messages import ReplyBatch
from dashulya.memory.service import HybridChatMemory
from dashulya.memory.store import MessageStore
from dashulya.memory.vectors import EmbeddingCache

from .dataset import Dataset, instant

# This is an evaluation owner identity, never a claimed Telegram group binding.
EVAL_OWNER_ID = 1


class MemoryToolsProvider(Provider):
    scope = AG2Scope.APP
    service = from_context(ChatMemoryService)
    formatter = from_context(MessageFormatter)

    @provide(scope=AG2Scope.SESSION)
    def access(self, context: Context) -> MemoryAccess:  # ruff: ignore[no-self-use] -- Dishka provider.
        access = context.dependencies.get("memory_access")
        if (
            not isinstance(access, MemoryAccess)
            or access.chat_id != EVAL_OWNER_ID
            or not access.is_private
        ):
            msg = "Evaluation tools require the trusted evaluation-owner session."
            raise ValueError(msg)
        return access

    @provide(scope=AG2Scope.SESSION)
    def recall(self, service: ChatMemoryService, access: MemoryAccess) -> MemoryBackend:  # ruff: ignore[no-self-use] -- Dishka provider.
        return service.recall(access)

    @provide(scope=AG2Scope.SESSION)
    def history(self, service: ChatMemoryService, access: MemoryAccess) -> HistoryReader:  # ruff: ignore[no-self-use] -- Dishka provider.
        return service.history(access)


def database_url() -> str:
    url = os.environ.get("EVAL_DATABASE_URL", "").strip()
    if not url:
        msg = "Set EVAL_DATABASE_URL to an isolated PostgreSQL evaluation database with pgvector and pg_search."
        raise ValueError(msg)
    app_url = os.environ.get("DATABASE_URL")
    if app_url:
        left, right = conninfo_to_dict(url), conninfo_to_dict(app_url)
        if all(
            left.get(k) == right.get(k) for k in ("host", "hostaddr", "port", "dbname")
        ):
            msg = "EVAL_DATABASE_URL must select a separate database from DATABASE_URL."
            raise ValueError(msg)
    return url


@asynccontextmanager
async def isolated_database(url: str) -> AsyncGenerator[Database]:
    schema = "eval_memory_" + uuid4().hex
    async with await AsyncConnection.connect(url, autocommit=True) as connection:
        # Refuse application data even when DATABASE_URL is absent or spelled differently.
        cursor = await connection.execute("SELECT to_regclass('public.messages')")
        row = await cursor.fetchone()
        if row and row[0] is not None:
            msg = "Evaluation database must not contain public application messages."
            raise ValueError(msg)
        await connection.execute(
            sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema))
        )
    scoped_url = make_conninfo(url, options=f"-c search_path={schema},public")
    database = Database(scoped_url)
    try:
        await migrate(scoped_url)
        await database.open()
        yield database
    finally:
        await database.close()
        async with await AsyncConnection.connect(url, autocommit=True) as connection:
            cursor = await connection.execute(
                "SELECT 'dashulya_bm25_' || md5(%s)", (schema,)
            )
            row = await cursor.fetchone()
            assert row is not None
            await connection.execute(
                sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(sql.Identifier(row[0]))
            )
            await connection.execute(
                sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema))
            )


def original(row: dict[str, Any]) -> OriginalMessage:
    key = row["chat_key"]
    return OriginalMessage(
        collection=f"archive:{key}",
        chat_id=None,
        message_id=row["id"],
        text=row["text"] or "",
        sent_at=instant(row["date"]),
        author_id=row["user_id"],
        author=row["user_name"] or row["author"] or "Unknown",
        chat_title=key,
        chat_username=key,
        topic_id=None,
        reply_to_id=row["reply_to_msg_id"],
        edited_at=None,
        media_type=row.get("media_type"),
        url=f"https://t.me/{key}/{row['id']}",
        provenance="import",
        archive_key=key,
        username=row.get("user_username"),
    )


@asynccontextmanager
async def memory_agent(
    dataset: Dataset,
    case: dict[str, Any],
    config: ModelConfig,
    *,
    api_key: str,
    url: str,
) -> AsyncGenerator[Agent[ReplyBatch]]:
    """Index only one prefix; neither gold evidence nor future rows are ingested."""
    async with (
        isolated_database(url) as database,
        AsyncOpenAI(
            api_key=api_key,
            timeout=60,
            max_retries=1,
        ) as client,
    ):
        store = MessageStore(database)
        for row in dataset.visible(case):
            await store.put(original(row))
        vectors = EmbeddingCache(database, MemoryEmbeddings(client))
        memory = HybridChatMemory(store, vectors, owner_user_id=EVAL_OWNER_ID)
        container = make_async_container(
            MemoryToolsProvider(),
            AG2Provider(),
            scopes=AG2Scope,
            context={
                ChatMemoryService: memory,
                MessageFormatter: MessageFormatter("Europe/Moscow"),
            },
            lock_factory=asyncio.Lock,
        )
        try:
            # Explicit preparation fails the run on errors; never hide partial indexing.
            await memory.refresh_once()
            yield create_reply_agent(
                config,
                tools=[search_memory, chat_history],
                middleware=[Middleware(DishkaAsyncMiddleware, container=container)],
            )
        finally:
            await container.close()
            await memory.close()


def request_input(query: dict[str, Any]) -> str:
    local = instant(query["history_cutoff"]["date"]).astimezone(
        MessageFormatter("Europe/Moscow").timezone
    )
    return (
        f"Sent at: {format_local_time(local, include_date=True)}\n"
        f"Evaluation owner (Telegram user ID: {EVAL_OWNER_ID}; interlocutor):\n"
        f"{query['query']}"
    )


def input_context(query: dict[str, Any]) -> Sequence[BaseEvent]:
    context = query.get("input_context", {})
    rows = {
        row["id"]: row
        for row in [
            *context.get("preceding_messages", []),
            *context.get("reply_ancestors", []),
        ]
    }
    if not rows:
        return ()
    # Render provided historical context with the exact same format as tool evidence.
    records = [
        original({**row, "chat_key": query["chat_key"]}).render(max_chars=12000)
        for row in sorted(rows.values(), key=lambda r: (instant(r["date"]), r["id"]))
    ]
    formatted = MessageFormatter("Europe/Moscow").result(
        {"messages": records},
        interlocutor_id=EVAL_OWNER_ID,
    )
    return [ModelRequest(parts=[TextInput(json.dumps(formatted, ensure_ascii=False))])]


async def capture(agent: Agent[ReplyBatch], query: dict[str, Any]) -> Trace:
    stream = MemoryStream()
    await stream.history.replace(input_context(query))
    events: list[BaseEvent] = []

    async def record(event: BaseEvent) -> None:
        # Keep the public eval substrate; lifecycle errors can carry provider bodies.
        if isinstance(event, ToolErrorEvent):
            event = ToolErrorEvent.from_call(
                ToolCallEvent(event.name or "unknown", id=event.parent_id),
                RuntimeError("Memory tool failed."),
            )
        if isinstance(
            event,
            (ModelRequest, ModelResponse, ToolCallEvent, ToolResultEvent, UsageEvent),
        ):
            events.append(event)

    stream.subscribe(record, sync_to_thread=False)
    error = None
    started = perf_counter()
    try:
        async with asyncio.timeout(90):
            result = await agent.ask(
                request_input(query),
                stream=stream,
                variables={"user_id": EVAL_OWNER_ID, "chat_id": EVAL_OWNER_ID},
                dependencies={
                    "memory_access": MemoryAccess(
                        EVAL_OWNER_ID,
                        is_private=True,
                        # Prefix ingestion enforces the full (time, ID) boundary.
                        before=instant(query["history_cutoff"]["date"])
                        + (
                            timedelta(microseconds=1)
                            if query["history_cutoff"]["message_id"] > 0
                            else timedelta()
                        ),
                    )
                },
            )
            await result.content()
    except Exception:
        error = RuntimeError("Agent request failed; provider details omitted.")
    return Trace(
        events=events, exception=error, duration_ms=int((perf_counter() - started) * 1000)
    )
