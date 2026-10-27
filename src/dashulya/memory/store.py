"""One canonical PostgreSQL message store for imports, updates and reply context."""

import time
from datetime import UTC, datetime
from typing import Any

from dashulya.database import Database
from dashulya.domain.memory import (
    MemoryAccess,
    MemoryMessage,
    OriginalMessage,
    ReplyContext,
    TruncatedThread,
    message_source_id,
    message_url_template,
)
from dashulya.domain.messages import ConversationKey


def original(row: dict[str, Any]) -> OriginalMessage:
    edited = row["edited_at"]
    return OriginalMessage(
        collection=row["collection"],
        chat_id=row["chat_id"],
        message_id=row["message_id"],
        text=row["body"],
        sent_at=row["sent_at"],
        author_id=row["author_id"],
        author=row["author"],
        chat_title=row["chat_title"],
        chat_username=row["chat_username"],
        topic_id=row["topic_id"],
        reply_to_id=row["reply_to_id"],
        edited_at=edited.isoformat() if edited is not None else None,
        media_type=row["media_type"],
        url=row["url"],
        provenance=row["provenance"],
        archive_key=row["archive_key"],
        username=row["author_username"],
        is_bot=row["is_bot"],
    )


def message_url(chat_id: int, username: str | None, number: int) -> str | None:
    template = message_url_template(chat_id, username)
    return template.replace("{message_id}", str(number)) if template else None


class MessageStore:
    def __init__(self, database: Database) -> None:
        self.database = database

    async def initialize(self) -> None:
        await self.database.open()

    async def put(
        self,
        value: OriginalMessage,
        *,
        update_id: int = 0,
        version_time: float | None = None,
        if_missing: bool = False,
    ) -> None:
        edited = datetime.fromisoformat(value.edited_at) if value.edited_at else None
        instant = edited or value.sent_at
        if instant is not None and instant.utcoffset() is None:
            error_message = "Message timestamps must include a timezone."
            raise ValueError(error_message)
        parameters = {
            "collection": value.collection,
            "message_id": value.message_id,
            "chat_id": value.chat_id,
            "body": value.text,
            "sent_at": value.sent_at,
            "author_id": value.author_id,
            "author": value.author,
            "author_username": value.username,
            "is_bot": value.is_bot,
            "chat_title": value.chat_title,
            "chat_username": value.chat_username,
            "topic_id": value.topic_id,
            "reply_to_id": value.reply_to_id,
            "edited_at": edited,
            "media_type": value.media_type,
            "url": value.url,
            "provenance": value.provenance,
            "archive_key": value.archive_key,
            "update_id": update_id,
            "version_time": version_time
            if version_time is not None
            else instant.timestamp()
            if instant
            else -1e20,
            "body_folded": value.text.casefold(),
            "author_folded": value.author.casefold(),
            "username_folded": value.username.lstrip("@").casefold()
            if value.username
            else None,
            "if_missing": if_missing,
        }
        async with self.database.connection() as connection, connection.transaction():
            await connection.execute(
                """INSERT INTO collections VALUES (%(collection)s,%(chat_id)s,%(chat_title)s,%(chat_username)s)
                ON CONFLICT(collection) DO UPDATE SET title=excluded.title,username=excluded.username""",
                parameters,
            )
            await connection.execute(
                """INSERT INTO messages AS old (
                    collection,message_id,chat_id,body,sent_at,author_id,author,
                    author_username,is_bot,chat_title,chat_username,topic_id,
                    reply_to_id,edited_at,media_type,url,provenance,archive_key,
                    update_id,version_time,body_folded,author_folded,username_folded
                ) VALUES (
                    %(collection)s,%(message_id)s,%(chat_id)s,%(body)s,%(sent_at)s,
                    %(author_id)s,%(author)s,%(author_username)s,%(is_bot)s,
                    %(chat_title)s,%(chat_username)s,%(topic_id)s,%(reply_to_id)s,
                    %(edited_at)s,%(media_type)s,%(url)s,%(provenance)s,
                    %(archive_key)s,%(update_id)s,%(version_time)s,
                    %(body_folded)s,%(author_folded)s,%(username_folded)s
                ) ON CONFLICT(collection,message_id) DO UPDATE SET
                    body=excluded.body,sent_at=excluded.sent_at,author_id=excluded.author_id,
                    author=excluded.author,author_username=excluded.author_username,
                    is_bot=excluded.is_bot,chat_title=excluded.chat_title,
                    chat_username=excluded.chat_username,topic_id=excluded.topic_id,
                    reply_to_id=excluded.reply_to_id,edited_at=excluded.edited_at,
                    media_type=excluded.media_type,url=excluded.url,
                    provenance=excluded.provenance,archive_key=excluded.archive_key,
                    update_id=excluded.update_id,version_time=excluded.version_time,
                    body_folded=excluded.body_folded,author_folded=excluded.author_folded,
                    username_folded=excluded.username_folded,
                    revision=old.revision+CASE WHEN
                        (old.body,old.sent_at,old.author_id,old.author,old.author_username,
                         old.is_bot,old.topic_id,old.reply_to_id,old.edited_at,old.media_type,
                         old.provenance,old.archive_key) IS DISTINCT FROM
                        (excluded.body,excluded.sent_at,excluded.author_id,excluded.author,
                         excluded.author_username,excluded.is_bot,excluded.topic_id,
                         excluded.reply_to_id,excluded.edited_at,excluded.media_type,
                         excluded.provenance,excluded.archive_key)
                        THEN 1 ELSE 0 END,
                    attempts=0,retry_at=0
                WHERE NOT %(if_missing)s
                    AND (old.provenance!='live' OR excluded.provenance='live')
                    AND (old.version_time,old.update_id)<=(excluded.version_time,excluded.update_id)
                """,
                parameters,
            )

    async def record(self, message: MemoryMessage, *, if_missing: bool = False) -> None:
        if message.sent_at.utcoffset() is None or (
            message.edited_at is not None and message.edited_at.utcoffset() is None
        ):
            error_message = "Message timestamps must include a timezone."
            raise ValueError(error_message)
        await self.put(
            OriginalMessage(
                collection=f"telegram:{message.chat_id}",
                chat_id=message.chat_id,
                message_id=message.message_id,
                text=message.text,
                sent_at=message.sent_at.astimezone(UTC),
                author_id=message.author_id,
                author=message.author,
                chat_title=message.chat_title,
                chat_username=message.chat_username,
                topic_id=message.thread_id,
                reply_to_id=message.reply_to_id,
                edited_at=message.edited_at.astimezone(UTC).isoformat()
                if message.edited_at
                else None,
                media_type=message.media_type,
                url=message_url(
                    message.chat_id, message.chat_username, message.message_id
                ),
                provenance="live",
                username=message.username,
                is_bot=message.is_bot,
            ),
            update_id=message.update_id,
            if_missing=if_missing,
        )

    async def originals(
        self, access: MemoryAccess, *, chat: str | None = None
    ) -> list[OriginalMessage]:
        async with self.database.connection() as connection:
            cursor = await connection.execute(
                """SELECT * FROM messages WHERE
                (%(private)s OR chat_id=%(chat_id)s)
                AND (%(chat)s::text IS NULL OR collection=%(chat)s)
                AND (%(before)s::timestamptz IS NULL OR (
                    (edited_at IS NULL OR edited_at<%(before)s)
                    AND CASE WHEN %(before_id)s::bigint IS NULL OR chat_id IS DISTINCT FROM %(chat_id)s
                        THEN sent_at<%(before)s
                        ELSE (sent_at,message_id)<(%(before)s,%(before_id)s) END
                )) ORDER BY sent_at NULLS FIRST,collection,message_id""",
                {
                    "private": access.is_private,
                    "chat_id": access.chat_id,
                    "chat": chat,
                    "before": access.before,
                    "before_id": access.before_id,
                },
            )
            return [original(row) for row in await cursor.fetchall()]

    async def context(
        self,
        key: ConversationKey,
        through_message_id: int,
        *,
        limit: int,
        parent_hops: int,
    ) -> ReplyContext:
        """None selects the whole chat, including legacy ordinary reply threads."""
        async with self.database.connection() as connection:
            cursor = await connection.execute(
                """WITH RECURSIVE recent AS (
                    SELECT message_id FROM messages WHERE chat_id=%(chat)s
                        AND (%(topic)s::bigint IS NULL OR topic_id=%(topic)s)
                        AND message_id<=%(through)s AND sent_at IS NOT NULL
                        ORDER BY message_id DESC LIMIT %(limit)s
                ), ancestors(message_id,depth) AS (
                    SELECT message_id,0 FROM recent UNION
                    SELECT parent.message_id,a.depth+1 FROM ancestors a
                    JOIN messages child ON child.chat_id=%(chat)s AND child.message_id=a.message_id
                    JOIN messages parent ON parent.collection=child.collection
                        AND parent.message_id=child.reply_to_id
                    WHERE a.depth<%(hops)s
                        AND (%(topic)s::bigint IS NULL OR parent.topic_id=%(topic)s)
                        AND parent.message_id<child.message_id AND parent.sent_at IS NOT NULL
                ), selected AS MATERIALIZED (
                    SELECT message_id,min(depth) AS depth FROM ancestors GROUP BY message_id
                ), boundaries AS (
                    SELECT m.collection,m.message_id,m.reply_to_id FROM messages m
                    JOIN selected s USING(message_id) WHERE m.chat_id=%(chat)s
                        AND s.depth=%(hops)s AND m.reply_to_id<m.message_id
                        AND NOT EXISTS(SELECT 1 FROM selected p WHERE p.message_id=m.reply_to_id)
                ), lineage(boundary_id,collection,message_id,reply_to_id,hops) AS (
                    SELECT message_id,collection,message_id,reply_to_id,0 FROM boundaries
                    UNION ALL
                    SELECT l.boundary_id,p.collection,p.message_id,p.reply_to_id,l.hops+1
                    FROM lineage l JOIN messages p
                        ON p.collection=l.collection AND p.message_id=l.reply_to_id
                    WHERE p.message_id<l.message_id AND p.sent_at IS NOT NULL
                        AND (%(topic)s::bigint IS NULL OR p.topic_id=%(topic)s)
                ), roots AS (
                    SELECT boundary_id,min(message_id) FILTER(WHERE reply_to_id IS NULL) AS root_id,
                        min(hops) FILTER(WHERE reply_to_id IS NULL) AS remaining_hops
                    FROM lineage GROUP BY boundary_id
                ) SELECT m.*,r.boundary_id IS NOT NULL AS truncated,r.root_id,r.remaining_hops
                FROM messages m JOIN selected s USING(message_id)
                LEFT JOIN roots r ON r.boundary_id=m.message_id
                WHERE m.chat_id=%(chat)s ORDER BY m.message_id""",
                {
                    "chat": key[0],
                    "topic": key[1],
                    "through": through_message_id,
                    "limit": limit,
                    "hops": parent_hops,
                },
            )
            rows = await cursor.fetchall()
        return ReplyContext(
            tuple(
                MemoryMessage(
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
                    edited_at=row["edited_at"],
                    media_type=row["media_type"],
                    update_id=row["update_id"],
                    username=row["author_username"],
                    is_bot=row["is_bot"],
                )
                for row in rows
            ),
            tuple(
                TruncatedThread(
                    row["message_id"],
                    row["root_id"],
                    message_source_id(row["collection"], row["root_id"])
                    if row["root_id"] is not None
                    else None,
                    row["remaining_hops"],
                )
                for row in rows
                if row["truncated"]
            ),
        )

    async def pending(self, *, limit: int | None = 32) -> list[dict[str, Any]]:
        async with self.database.connection() as connection:
            cursor = await connection.execute(
                "SELECT * FROM messages WHERE indexed_revision<revision AND retry_at<=%s ORDER BY sent_at,collection,message_id LIMIT %s",
                (time.time(), limit),
            )
            return await cursor.fetchall()

    async def indexed(self, rows: list[dict[str, Any]]) -> None:
        async with (
            self.database.connection() as connection,
            connection.cursor() as cursor,
        ):
            await cursor.executemany(
                "UPDATE messages SET indexed_revision=%s,attempts=0,retry_at=0 WHERE collection=%s AND message_id=%s AND revision=%s",
                [
                    (r["revision"], r["collection"], r["message_id"], r["revision"])
                    for r in rows
                ],
            )

    async def coverage(self) -> dict[str, Any]:
        async with self.database.connection() as connection:
            cursor = await connection.execute(
                """SELECT COUNT(*) message_count,COUNT(DISTINCT collection) chat_count,
                    COUNT(*) FILTER(WHERE indexed_revision=revision) indexed_count,
                    COUNT(*) FILTER(WHERE indexed_revision<revision) pending_count,
                    MIN(sent_at) first_message_at,MAX(sent_at) last_message_at FROM messages"""
            )
            row = await cursor.fetchone()
            assert row is not None
            return row
