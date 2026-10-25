"""Canonical live messages and a durable, revision-aware indexing outbox."""

import json
import re
import sqlite3
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import aiosqlite

from dashulya.domain.memory import MemoryMessage
from dashulya.domain.messages import ConversationKey

LIVE_ROOT = "viking://user/default/resources/telegram"
BLOCK_SIZE = 8


async def _fetch(
    connection: aiosqlite.Connection, sql: str, parameters: tuple[Any, ...] = ()
) -> list[sqlite3.Row]:
    return list(await connection.execute_fetchall(sql, parameters))


class LiveMemory:
    def __init__(self, path: Path, *, root: str = LIVE_ROOT) -> None:
        self.path = path
        self.root = root.rstrip("/")

    @asynccontextmanager
    async def connection(self) -> AsyncGenerator[aiosqlite.Connection, None]:
        async with aiosqlite.connect(self.path) as connection:
            connection.row_factory = sqlite3.Row
            await connection.execute("PRAGMA busy_timeout = 5000")
            yield connection

    async def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        async with self.connection() as connection:
            await connection.executescript("""
                PRAGMA journal_mode = WAL;
                CREATE TABLE IF NOT EXISTS live_chats (
                    id INTEGER PRIMARY KEY, title TEXT NOT NULL, username TEXT
                );
                CREATE TABLE IF NOT EXISTS live_messages (
                    chat_id INTEGER NOT NULL, message_id INTEGER NOT NULL,
                    body TEXT NOT NULL, sent_at TEXT NOT NULL,
                    author_id INTEGER, author TEXT NOT NULL,
                    thread_id INTEGER, reply_to_id INTEGER, edited_at TEXT,
                    media_type TEXT, update_id INTEGER NOT NULL,
                    version_time REAL NOT NULL, revision INTEGER NOT NULL,
                    indexed_revision INTEGER NOT NULL DEFAULT 0,
                    attempts INTEGER NOT NULL DEFAULT 0, retry_at REAL NOT NULL DEFAULT 0,
                    PRIMARY KEY (chat_id, message_id)
                );
                CREATE INDEX IF NOT EXISTS live_pending ON live_messages(retry_at);
                CREATE INDEX IF NOT EXISTS live_conversation
                    ON live_messages(chat_id, thread_id, message_id);
            """)
            await connection.commit()

    async def context(
        self,
        key: ConversationKey,
        through_message_id: int,
        *,
        limit: int,
        parent_hops: int,
    ) -> list[MemoryMessage]:
        async with self.connection() as connection:
            rows = await _fetch(
                connection,
                """
                WITH RECURSIVE recent AS (
                    SELECT message_id FROM live_messages
                    WHERE chat_id=? AND thread_id IS ? AND message_id <= ?
                    ORDER BY message_id DESC LIMIT ?
                ), ancestors(message_id, depth) AS (
                    SELECT message_id, 0 FROM recent
                    UNION
                    SELECT parent.message_id, ancestors.depth + 1
                    FROM ancestors
                    JOIN live_messages child
                        ON child.chat_id=? AND child.message_id=ancestors.message_id
                    JOIN live_messages parent
                        ON parent.chat_id=child.chat_id
                        AND parent.message_id=child.reply_to_id
                    WHERE ancestors.depth < ? AND parent.thread_id IS ?
                        AND parent.message_id < child.message_id
                )
                SELECT m.*, c.title, c.username FROM live_messages m
                JOIN live_chats c ON c.id=m.chat_id
                WHERE m.chat_id=? AND m.message_id IN (
                    SELECT message_id FROM ancestors
                ) ORDER BY m.message_id
                """,
                (
                    key[0],
                    key[1],
                    through_message_id,
                    limit,
                    key[0],
                    parent_hops,
                    key[1],
                    key[0],
                ),
            )
        return [
            MemoryMessage(
                chat_id=row["chat_id"],
                message_id=row["message_id"],
                text=row["body"],
                sent_at=datetime.fromisoformat(row["sent_at"]),
                author_id=row["author_id"],
                author=row["author"],
                chat_title=row["title"],
                chat_username=row["username"],
                thread_id=row["thread_id"],
                reply_to_id=row["reply_to_id"],
                edited_at=datetime.fromisoformat(row["edited_at"])
                if row["edited_at"]
                else None,
                media_type=row["media_type"],
                update_id=row["update_id"],
            )
            for row in rows
        ]

    async def record(self, message: MemoryMessage, *, if_missing: bool = False) -> None:
        if message.sent_at.utcoffset() is None or (
            message.edited_at is not None and message.edited_at.utcoffset() is None
        ):
            error_message = "Live message dates must include a timezone."
            raise ValueError(error_message)
        version_time = (message.edited_at or message.sent_at).timestamp()
        sent_at = message.sent_at.astimezone(UTC).isoformat()
        edited_at = (
            message.edited_at.astimezone(UTC).isoformat() if message.edited_at else None
        )
        content = (
            message.text,
            sent_at,
            message.author_id,
            message.author,
            message.thread_id,
            message.reply_to_id,
            edited_at,
            message.media_type,
        )
        async with self.connection() as connection:
            await connection.execute("BEGIN IMMEDIATE")
            await connection.execute(
                "INSERT INTO live_chats VALUES (?, ?, ?) ON CONFLICT(id) DO UPDATE SET title=excluded.title, username=excluded.username",
                (message.chat_id, message.chat_title, message.chat_username),
            )
            async with connection.execute(
                "SELECT * FROM live_messages WHERE chat_id=? AND message_id=?",
                (message.chat_id, message.message_id),
            ) as cursor:
                existing = await cursor.fetchone()
            if if_missing and existing is not None:
                return
            stale = existing is not None and (
                existing["version_time"] > version_time
                or (
                    existing["version_time"] == version_time
                    and existing["update_id"] > message.update_id
                )
            )
            same = (
                existing is not None
                and tuple(
                    existing[key]
                    for key in (
                        "body",
                        "sent_at",
                        "author_id",
                        "author",
                        "thread_id",
                        "reply_to_id",
                        "edited_at",
                        "media_type",
                    )
                )
                == content
            )
            if not stale and not same:
                revision = existing["revision"] + 1 if existing else 1
                await connection.execute(
                    """
                    INSERT INTO live_messages (
                        chat_id, message_id, body, sent_at, author_id, author,
                        thread_id, reply_to_id, edited_at, media_type, update_id,
                        version_time, revision
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(chat_id, message_id) DO UPDATE SET
                        body=excluded.body, sent_at=excluded.sent_at,
                        author_id=excluded.author_id, author=excluded.author,
                        thread_id=excluded.thread_id, reply_to_id=excluded.reply_to_id,
                        edited_at=excluded.edited_at, media_type=excluded.media_type,
                        update_id=excluded.update_id, version_time=excluded.version_time,
                        revision=excluded.revision, attempts=0, retry_at=0
                """,
                    (
                        message.chat_id,
                        message.message_id,
                        *content,
                        message.update_id,
                        version_time,
                        revision,
                    ),
                )
            elif same and not stale:
                await connection.execute(
                    "UPDATE live_messages SET update_id=? WHERE chat_id=? AND message_id=?",
                    (message.update_id, message.chat_id, message.message_id),
                )
            await connection.commit()

    def block_uri(self, chat_id: int, message_id: int) -> str:
        return f"{self.root}/chat_{chat_id}_block_{message_id // BLOCK_SIZE}"

    def uri(self, row: dict[str, Any]) -> str:
        return (
            f"{self.block_uri(row['chat_id'], row['message_id'])}/{row['message_id']}.md"
        )

    @staticmethod
    def key(row: dict[str, Any]) -> str:
        return f"telegram:{row['chat_id']}:{row['message_id']}"

    @staticmethod
    def block_key(row: dict[str, Any]) -> str:
        return f"telegram-block:{row['chat_id']}:{row['message_id'] // BLOCK_SIZE}"

    async def pending(self, *, limit: int = 32) -> list[dict[str, Any]]:
        async with self.connection() as connection:
            rows = await _fetch(
                connection,
                """
                SELECT m.*, c.title, c.username FROM live_messages m
                JOIN live_chats c ON c.id=m.chat_id
                WHERE indexed_revision < revision AND retry_at <= ?
                ORDER BY sent_at, chat_id, message_id LIMIT ?
            """,
                (time.time(), limit),
            )
        return [dict(row) for row in rows]

    async def indexed(self, rows: list[dict[str, Any]]) -> None:
        async with self.connection() as connection:
            await connection.executemany(
                """
                UPDATE live_messages SET indexed_revision=?, attempts=0, retry_at=0
                WHERE chat_id=? AND message_id=? AND revision=?
            """,
                [
                    (r["revision"], r["chat_id"], r["message_id"], r["revision"])
                    for r in rows
                ],
            )
            await connection.commit()

    async def failed(self, rows: list[dict[str, Any]]) -> None:
        async with self.connection() as connection:
            await connection.executemany(
                """
                UPDATE live_messages SET attempts=attempts+1, retry_at=?
                WHERE chat_id=? AND message_id=? AND revision=?
            """,
                [
                    (
                        time.time() + min(60, 2 ** min(r["attempts"] + 1, 6)),
                        r["chat_id"],
                        r["message_id"],
                        r["revision"],
                    )
                    for r in rows
                ],
            )
            await connection.commit()

    async def coverage(self) -> dict[str, Any]:
        async with self.connection() as connection:
            rows = await _fetch(
                connection,
                """
                SELECT COUNT(*) message_count, COUNT(DISTINCT chat_id) chat_count,
                    COALESCE(SUM(indexed_revision=revision),0) indexed_count,
                    COALESCE(SUM(indexed_revision<revision),0) pending_count,
                    MIN(sent_at) first_message_at, MAX(sent_at) last_message_at
                FROM live_messages
            """,
            )
        return dict(rows[0])

    async def eligible(
        self, author: int | None, after: str | None, before: str | None
    ) -> dict[str, str]:
        async with self.connection() as connection:
            rows = await _fetch(
                connection, "SELECT * FROM live_messages ORDER BY chat_id, message_id"
            )
        records: dict[str, str] = {}
        blocks: dict[str, list[bool]] = {}
        block_keys: dict[str, str] = {}
        for source in rows:
            row = dict(source)
            valid = (
                row["revision"] == row["indexed_revision"]
                and (author is None or row["author_id"] == author)
                and (
                    after is None
                    or datetime.fromisoformat(row["sent_at"])
                    >= datetime.fromisoformat(after)
                )
                and (
                    before is None
                    or datetime.fromisoformat(row["sent_at"])
                    <= datetime.fromisoformat(before)
                )
            )
            block = self.block_uri(row["chat_id"], row["message_id"])
            blocks.setdefault(block, []).append(valid)
            block_keys[block] = self.block_key(row)
            if valid:
                records[self.uri(row)] = self.key(row)
        records.update(
            (block, block_keys[block]) for block, valid in blocks.items() if all(valid)
        )
        return records

    async def expand(
        self, key: str, author: int | None, after: str | None, before: str | None
    ) -> list[dict[str, Any]]:
        match = re.fullmatch(r"(telegram|telegram-block):(-?\d+):(\d+)", key)
        if match is None:
            return []
        kind, chat, identifier = match.groups()
        chat_id, number = int(chat), int(identifier)
        lower, upper = (
            (number * BLOCK_SIZE, (number + 1) * BLOCK_SIZE - 1)
            if kind == "telegram-block"
            else (number, number)
        )
        async with self.connection() as connection:
            sources = await _fetch(
                connection,
                """
                SELECT m.*, c.title, c.username FROM live_messages m
                JOIN live_chats c ON c.id=m.chat_id
                WHERE chat_id=? AND message_id BETWEEN ? AND ? ORDER BY message_id
            """,
                (chat_id, lower, upper),
            )
            if not sources or any(
                (author is not None and r["author_id"] != author)
                or (
                    after is not None
                    and datetime.fromisoformat(r["sent_at"])
                    < datetime.fromisoformat(after)
                )
                or (
                    before is not None
                    and datetime.fromisoformat(r["sent_at"])
                    > datetime.fromisoformat(before)
                )
                for r in sources
            ):
                return []
            source_ids = {r["message_id"] for r in sources}
            context: dict[int, sqlite3.Row] = {}
            for source in sources:
                parent_id = source["reply_to_id"]
                for _ in range(2):
                    rows = await _fetch(
                        connection,
                        """
                        SELECT m.*, c.title, c.username FROM live_messages m
                        JOIN live_chats c ON c.id=m.chat_id WHERE chat_id=? AND message_id=?
                    """,
                        (chat_id, parent_id),
                    )
                    if not rows:
                        break
                    parent = rows[0]
                    context[parent["message_id"]] = parent
                    parent_id = parent["reply_to_id"]
            for operator, order, anchor in (
                ("<", "DESC", sources[0]["message_id"]),
                (">", "ASC", sources[-1]["message_id"]),
            ):
                rows = await _fetch(
                    connection,
                    f"""
                    SELECT m.*, c.title, c.username FROM live_messages m
                    JOIN live_chats c ON c.id=m.chat_id
                    WHERE chat_id=? AND thread_id IS ? AND message_id {operator} ?
                    ORDER BY message_id {order} LIMIT 2
                """,  # ruff: ignore[hardcoded-sql-expression] -- Operators and orderings come from the fixed tuples above.
                    (chat_id, sources[0]["thread_id"], anchor),
                )
                context.update((r["message_id"], r) for r in rows)
        return [
            {
                "memory_id": key,
                "kind": "discussion" if kind == "telegram-block" else "message",
                "excerpt": "\n".join(r["body"] for r in sources)[:1500],
                "sources": [self.evidence(dict(r)) for r in sources],
                "context": [
                    self.evidence(dict(r))
                    for mid, r in sorted(context.items())
                    if mid not in source_ids
                ][:4],
                "related_memory_ids": [self.block_key(dict(sources[0]))]
                if kind == "telegram"
                else [],
            }
        ]

    @staticmethod
    def evidence(row: dict[str, Any]) -> dict[str, Any]:
        chat_id, message_id = row["chat_id"], row["message_id"]
        if row["username"]:
            url = f"https://t.me/{row['username']}/{message_id}"
        elif str(chat_id).startswith("-100"):
            url = f"https://t.me/c/{str(chat_id)[4:]}/{message_id}"
        else:
            url = ""
        return {
            "message_id": message_id,
            "chat_id": chat_id,
            "thread_id": row["thread_id"],
            "sent_at": row["sent_at"],
            "user_id": row["author_id"],
            "author": row["author"],
            "text": row["body"][:1200],
            "url": url,
            "reply_to_message_id": row["reply_to_id"],
            "truncated": len(row["body"]) > 1200,
            "excerpt_start": 0,
            "edited_at": row["edited_at"],
            "media_type": row["media_type"],
        }

    def document(self, row: dict[str, Any]) -> str:
        # Source text is quoted evidence, never a user's instruction or preference.
        metadata = self.evidence(row)
        metadata.pop("text")
        return f"# Telegram group message\n\n{row['body']}\n\n## Source metadata\n\n{json.dumps(metadata, ensure_ascii=False)}\n"
