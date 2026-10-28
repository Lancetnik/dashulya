"""Local, read-only prototype for inspecting the bot's group reply context."""

import argparse
import asyncio
import json
import os
from dataclasses import asdict, replace
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from time import perf_counter
from typing import Any
from urllib.parse import parse_qs, urlsplit

from ag2 import Agent
from ag2.events import ModelRequest, TextInput
from dotenv import load_dotenv
from psycopg.conninfo import conninfo_to_dict, make_conninfo

from dashulya.ai.dialogue import Dialogue
from dashulya.ai.message_format import MessageFormatter
from dashulya.database import Database
from dashulya.domain.memory import MemoryMessage
from dashulya.domain.messages import Reply
from dashulya.memory.store import MessageStore

PAGE_SIZE = 80
ROOT = Path(__file__).resolve().parent


class ReadOnlyDatabase(Database):
    async def _initialize_schema(self) -> None:
        """The preview uses the existing schema and never runs migrations."""


def message_json(message: MemoryMessage, bot_id: int | None) -> dict[str, Any]:
    return {
        "id": message.message_id,
        "text": message.text,
        "sent_at": message.sent_at.isoformat(),
        "author_id": message.author_id,
        "author": message.author,
        "username": message.username,
        "is_self": bot_id is not None and message.author_id == bot_id,
        "is_bot": message.is_bot,
        "reply_to_id": message.reply_to_id,
        "topic_id": message.thread_id,
        "media_type": message.media_type,
    }


def saved_message(row: dict[str, Any]) -> MemoryMessage:
    return MemoryMessage(
        chat_id=row["chat_id"],
        message_id=row["message_id"],
        text=row["body"],
        sent_at=row["sent_at"],
        author_id=row["author_id"],
        author=row["author"],
        chat_title=row["chat_title"],
        chat_username=row["chat_username"],
        thread_id=row["topic_id"],
        reply_to_id=row["reply_to_id"],
        media_type=row["media_type"],
        username=row["author_username"],
        is_bot=row["is_bot"],
    )


class Preview:
    def __init__(self, database: Database, formatter: MessageFormatter) -> None:
        self.database = database
        self.formatter = formatter
        # Selection does not execute this agent or require model credentials.
        self.dialogue = Dialogue(
            Agent[Reply]("preview"), history=MessageStore(database), formatter=formatter
        )

    async def get(self, path: str, query: dict[str, list[str]]) -> dict[str, Any]:
        if path == "/api/chats":
            async with self.database.connection() as connection:
                cursor = await connection.execute(
                    """SELECT c.chat_id,c.title,c.username,count(*) AS message_count,
                    min(m.sent_at) AS first_at,max(m.sent_at) AS last_at,
                    array_remove(array_agg(DISTINCT m.topic_id),NULL) AS topics
                    FROM collections c JOIN messages m USING(collection)
                    WHERE c.chat_id<0 AND m.sent_at IS NOT NULL
                    GROUP BY c.collection ORDER BY max(m.sent_at) DESC"""
                )
                chats = await cursor.fetchall()
            return {
                "chats": chats,
                "timezone": self.formatter.timezone.key,
                "bot_id": self.formatter.bot_id,
            }
        chat_id = int(query["chat_id"][0])
        if chat_id >= 0:
            raise ValueError("Choose a group chat.")
        topic = int(query["topic_id"][0]) if query.get("topic_id") else None
        if path == "/api/messages":
            return await self.messages(chat_id, topic, query)
        if path == "/api/context":
            return await self.context(
                chat_id,
                topic,
                int(query["message_id"][0]),
                limit=int(query.get("limit", ["20"])[0]),
                parent_hops=int(query.get("parent_hops", ["5"])[0]),
            )
        raise LookupError("Page not found.")

    async def messages(
        self, chat_id: int, topic: int | None, query: dict[str, list[str]]
    ) -> dict[str, Any]:
        before = int(query["before_id"][0]) if query.get("before_id") else None
        around = int(query["around_id"][0]) if query.get("around_id") else None
        parameters = {
            "chat": chat_id,
            "topic": topic,
            "before": before,
            "around": around,
        }
        async with self.database.connection() as connection:
            cursor = await connection.execute(
                """WITH earlier AS (
                    SELECT * FROM messages WHERE chat_id=%(chat)s AND sent_at IS NOT NULL
                    AND (%(topic)s::bigint IS NULL OR topic_id=%(topic)s)
                    AND (%(before)s::bigint IS NULL OR message_id<%(before)s)
                    AND (%(around)s::bigint IS NULL OR message_id<=%(around)s)
                    ORDER BY message_id DESC LIMIT 81
                ), later AS (
                    SELECT * FROM messages WHERE chat_id=%(chat)s AND sent_at IS NOT NULL
                    AND (%(topic)s::bigint IS NULL OR topic_id=%(topic)s)
                    AND %(around)s::bigint IS NOT NULL AND message_id>%(around)s
                    ORDER BY message_id LIMIT 20
                ) SELECT * FROM earlier UNION ALL SELECT * FROM later ORDER BY message_id""",
                parameters,
            )
            rows = await cursor.fetchall()
        earlier = [row for row in rows if around is None or row["message_id"] <= around]
        more = len(earlier) > PAGE_SIZE
        if more:
            rows = rows[1:]
        return {
            "messages": [
                message_json(saved_message(row), self.formatter.bot_id) for row in rows
            ],
            "has_older": more,
        }

    async def context(
        self,
        chat_id: int,
        topic: int | None,
        message_id: int,
        *,
        limit: int = 20,
        parent_hops: int = 5,
    ) -> dict[str, Any]:
        if limit < 1 or parent_hops < 0:
            raise ValueError(
                "Use a positive message count and nonnegative reply-parent hops."
            )
        started = perf_counter()
        async with self.database.connection() as connection:
            cursor = await connection.execute(
                """SELECT * FROM messages WHERE chat_id=%s AND message_id=%s
                AND sent_at IS NOT NULL AND (%s::bigint IS NULL OR topic_id=%s)""",
                (chat_id, message_id, topic, topic),
            )
            row = await cursor.fetchone()
        if row is None:
            raise LookupError("This message is not saved in the selected chat/topic.")
        current = saved_message(row)
        # Match the reply path's input for captionless attachments.
        if not current.text:
            current = replace(
                current,
                text="Please respond to this media in the context of our conversation.",
            )
        selected = await self.dialogue.select_context(
            (chat_id, topic),
            message_id,
            current=current,
            limit=limit,
            parent_hops=parent_hops,
        )
        events = self.formatter.events(
            selected.messages,
            interlocutor_id=current.author_id,
            truncated_threads=selected.truncated_threads,
        )
        recent_ids = {item.message_id for item in selected.messages[-limit:]}
        event_json = []
        for index, event in enumerate(events):
            event_json.append(
                {
                    "type": type(event).__name__,
                    "role": "user" if isinstance(event, ModelRequest) else "assistant",
                    "text": "\n".join(
                        part.content
                        for part in event.parts
                        if isinstance(part, TextInput)
                    )
                    if isinstance(event, ModelRequest)
                    else event.content,
                    "message_id": None
                    if index == 0
                    else selected.messages[index - 1].message_id,
                }
            )
        return {
            "key": [chat_id, topic],
            "message_id": message_id,
            "parameters": {"limit": limit, "parent_hops": parent_hops},
            "messages": [
                {
                    **message_json(item, self.formatter.bot_id),
                    "selection": "current"
                    if item.message_id == message_id
                    else "recent"
                    if item.message_id in recent_ids
                    else "parent",
                }
                for item in selected.messages
            ],
            "events": event_json,
            "raw_events": [event.to_dict() for event in events],
            "truncated_threads": [asdict(item) for item in selected.truncated_threads],
            "elapsed_ms": round((perf_counter() - started) * 1000, 1),
        }


class Handler(BaseHTTPRequestHandler):
    def __init__(
        self,
        *args: Any,
        preview: Preview,
        loop: asyncio.AbstractEventLoop,
        **kwargs: Any,
    ) -> None:
        self.preview = preview
        self.loop = loop
        super().__init__(*args, **kwargs)

    def do_GET(self) -> None:
        url = urlsplit(self.path)
        if url.path in {"/", "/index.html"}:
            self.send_body(
                200, (ROOT / "index.html").read_bytes(), "text/html; charset=utf-8"
            )
            return
        if url.path == "/favicon.ico":
            self.send_body(204, b"", "image/x-icon")
            return
        try:
            if not url.path.startswith("/api/"):
                raise LookupError("Page not found.")
            future = asyncio.run_coroutine_threadsafe(
                self.preview.get(url.path, parse_qs(url.query)), self.loop
            )
            result = future.result(timeout=30)
            self.send_json(200, result)
        except (ValueError, KeyError, IndexError):
            self.send_json(
                400, {"error": "Invalid chat, topic, message ID or context parameters."}
            )
        except LookupError as error:
            self.send_json(404, {"error": str(error)})
        except Exception:
            self.send_json(
                503,
                {
                    "error": "Could not read PostgreSQL. Check the preview server and DATABASE_URL."
                },
            )

    def send_json(self, status: int, value: dict[str, Any]) -> None:
        self.send_body(
            status,
            json.dumps(value, ensure_ascii=False, default=str).encode(),
            "application/json; charset=utf-8",
        )

    def send_body(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)


async def serve(host: str, port: int) -> None:
    load_dotenv(Path.cwd() / ".env", override=False)
    url = os.getenv("DATABASE_URL", "")
    if not url:
        raise SystemExit("Set DATABASE_URL in .env or the environment.")
    options = str(conninfo_to_dict(url).get("options") or "")
    database = ReadOnlyDatabase(
        make_conninfo(
            url,
            options=options
            + " -c default_transaction_read_only=on -c statement_timeout=15000",
        )
    )
    token_id = os.getenv("BOT_TOKEN", "").partition(":")[0]
    bot_id = int(token_id) if token_id.isdecimal() else None
    owner_id = int(os.getenv("OWNER_USER_ID", "0"))
    author_ids = [
        int(part.strip())
        for part in os.getenv("OWNER_AUTHOR_IDS", "").split(",")
        if part.strip()
    ]
    formatter = MessageFormatter(
        os.getenv("BOT_TIMEZONE", "Europe/Moscow"),
        bot_id,
        author_users=dict.fromkeys(author_ids, owner_id) if owner_id else {},
    )
    try:
        await database.open()
        handler = partial(
            Handler,
            preview=Preview(database, formatter),
            loop=asyncio.get_running_loop(),
        )
        server = ThreadingHTTPServer((host, port), handler)
        print(
            f"Context preview: http://localhost:{port} (PostgreSQL, read-only)",
            flush=True,
        )
        try:
            await asyncio.to_thread(server.serve_forever)
        finally:
            await asyncio.to_thread(server.shutdown)
            await asyncio.to_thread(server.server_close)
    finally:
        await database.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    arguments = parser.parse_args()
    try:
        asyncio.run(serve(arguments.host, arguments.port))
    except KeyboardInterrupt:
        pass
