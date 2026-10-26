"""Explicit, resumable import of legacy SQLite files into unified PostgreSQL."""

import asyncio
import hashlib
import json
import sqlite3
from contextlib import closing
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from psycopg.types.json import Jsonb

from dashulya.database import Database
from dashulya.domain.groups import Group
from dashulya.domain.memory import MemoryAccess, OriginalMessage
from dashulya.memory.store import MessageStore, message_url
from dashulya.memory.vectors import EmbeddingCache
from dashulya.storage.groups import GroupStore


def _read(path: Path, table: str) -> list[dict[str, Any]]:
    allowed = {
        "messages",
        "metadata",
        "memories",
        "sources",
        "live_messages",
        "live_chats",
        "groups",
    }
    if table not in allowed:
        error_message = "Unknown legacy table."
        raise ValueError(error_message)
    with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as db:
        db.row_factory = sqlite3.Row
        return [
            dict(r)
            for r in db.execute(
                {
                    "messages": "SELECT * FROM messages",
                    "metadata": "SELECT * FROM metadata",
                    "memories": "SELECT * FROM memories",
                    "sources": "SELECT * FROM sources",
                    "live_messages": "SELECT * FROM live_messages",
                    "live_chats": "SELECT * FROM live_chats",
                    "groups": "SELECT * FROM groups",
                }[table]
            )
        ]


def _time(value: str | None) -> datetime | None:
    if value is None:
        return None
    instant = datetime.fromisoformat(value)
    if instant.utcoffset() is None:
        error_message = "Legacy timestamps must include a timezone."
        raise ValueError(error_message)
    return instant.astimezone(UTC)


def evidence_originals(
    path: Path, chat_id: int | None
) -> tuple[list[OriginalMessage], dict[str, Any]]:
    metadata = {r["key"]: json.loads(r["value"]) for r in _read(path, "metadata")}
    source = metadata["source_sha256"]
    username = metadata.get("chat_username")
    collection = f"telegram:{chat_id}" if chat_id is not None else f"archive:{source}"
    rows = [
        OriginalMessage(
            collection,
            chat_id,
            r["id"],
            r["body"],
            _time(r["sent_at"]),
            r["user_id"],
            r["author"],
            username or "Imported archive",
            username,
            None,
            r["reply_to_id"],
            None,
            None,
            r["url"],
            "imported",
            source,
            r.get("author_username") or r.get("username"),
            bool(r["is_bot"]) if r.get("is_bot") in {0, 1} else None,
        )
        for r in _read(path, "messages")
    ]
    return rows, metadata


def live_originals(path: Path) -> list[tuple[OriginalMessage, int, float]]:
    chats = {r["id"]: r for r in _read(path, "live_chats")}
    result = []
    for r in _read(path, "live_messages"):
        c = chats[r["chat_id"]]
        edited = _time(r["edited_at"])
        result.append((
            OriginalMessage(
                f"telegram:{r['chat_id']}",
                r["chat_id"],
                r["message_id"],
                r["body"],
                _time(r["sent_at"]),
                r["author_id"],
                r["author"],
                c["title"],
                c["username"],
                r["thread_id"],
                r["reply_to_id"],
                edited.isoformat() if edited is not None else None,
                r["media_type"],
                message_url(r["chat_id"], c["username"], r["message_id"]),
                "live",
                None,
                r.get("author_username"),
                bool(r["is_bot"]) if r.get("is_bot") in {0, 1} else None,
            ),
            r["update_id"],
            r["version_time"],
        ))
    return result


async def import_evidence(
    store: MessageStore, path: Path, chat_id: int | None = None
) -> list[OriginalMessage]:
    if chat_id == 0:
        error_message = "An imported chat needs a nonzero numeric Telegram ID."
        raise ValueError(error_message)
    rows, metadata = await asyncio.to_thread(evidence_originals, path, chat_id)
    async with store.database.connection() as connection:
        previous = await (
            await connection.execute(
                "SELECT metadata FROM source_snapshots WHERE source_key=%s",
                (metadata["source_sha256"],),
            )
        ).fetchone()
        if previous is not None:
            old_binding = previous["metadata"].get("binding_chat_id")
            if old_binding is not None and old_binding != chat_id:
                error_message = "The source archive is already bound to a different chat."
                raise ValueError(error_message)
    for row in rows:
        await store.put(row)
    metadata["binding_chat_id"] = chat_id
    async with store.database.connection() as connection:
        await connection.execute(
            "INSERT INTO source_snapshots VALUES (%s,%s,CURRENT_TIMESTAMP) ON CONFLICT(source_key) DO UPDATE SET metadata=excluded.metadata",
            (metadata["source_sha256"], Jsonb(metadata)),
        )
        sources: dict[int, list[int]] = {}
        for r in await asyncio.to_thread(_read, path, "sources"):
            sources.setdefault(r["memory_id"], []).append(r["message_id"])
        async with connection.cursor() as cursor:
            await cursor.executemany(
                "INSERT INTO legacy_records VALUES (%s,%s,%s,%s,%s) ON CONFLICT(collection,record_key) DO NOTHING",
                [
                    (
                        f"telegram:{chat_id}"
                        if chat_id is not None
                        else "archive:" + metadata["source_sha256"],
                        r["key"],
                        r["kind"],
                        r["body"],
                        sorted(sources.get(r["id"], [])),
                    )
                    for r in await asyncio.to_thread(_read, path, "memories")
                ],
            )
        if chat_id is not None:
            unbound = "archive:" + metadata["source_sha256"]
            async with connection.transaction():
                await connection.execute(
                    "DELETE FROM legacy_records WHERE collection=%s", (unbound,)
                )
                await connection.execute(
                    "DELETE FROM messages WHERE collection=%s AND provenance=%s AND archive_key=%s",
                    (unbound, "imported", metadata["source_sha256"]),
                )
                await connection.execute(
                    "DELETE FROM collections WHERE collection=%s AND NOT EXISTS(SELECT 1 FROM messages WHERE collection=%s)",
                    (unbound, unbound),
                )
    return rows


async def migrate_sqlite(
    database: Database,
    *,
    evidence: Path | None = None,
    live: Path | None = None,
    groups: Path | None = None,
    vectors: Path | None = None,
    chat_id: int | None = None,
    owner_user_id: int,
    verify: bool = True,
) -> dict[str, Any]:
    store = MessageStore(database)
    await store.initialize()
    expected = {}
    if evidence is not None:
        for row in await import_evidence(store, evidence, chat_id):
            expected[row.collection, row.message_id] = row
    if live is not None:
        for row, update, version in await asyncio.to_thread(live_originals, live):
            await store.put(row, update_id=update, version_time=version)
            expected[row.collection, row.message_id] = row
    registered = []
    if groups is not None:
        registry = GroupStore(database)
        registered = [
            Group(r["chat_id"], r["title"], r["username"])
            for r in await asyncio.to_thread(_read, groups, "groups")
        ]
        for group in registered:
            await registry.remember(group)
    vector_count = 0
    if vectors is not None:

        async def no_provider(_texts: list[str]) -> list[list[float]]:
            error_message = "Migration must never request new embeddings."
            raise AssertionError(error_message)

        cache = EmbeddingCache(database, no_provider)
        uri = await asyncio.to_thread(lambda: vectors.resolve().as_uri() + "?mode=ro")
        with closing(sqlite3.connect(uri, uri=True)) as source:
            cursor = source.execute(
                "SELECT cache_key,vector FROM embeddings ORDER BY cache_key"
            )
            while batch := cursor.fetchmany(256):
                await cache.put({key: json.loads(payload) for key, payload in batch})
                vector_count += len(batch)
    actual = await store.originals(MemoryAccess(owner_user_id, True))
    if verify:
        by_key = {(r.collection, r.message_id): r for r in actual}
        for key, row in expected.items():
            if by_key.get(key) != row:
                error_message = (
                    "Canonical PostgreSQL originals differ from the legacy snapshot."
                )
                raise ValueError(error_message)
        if len(by_key) != len(actual):
            error_message = "Migration created duplicate canonical originals."
            raise ValueError(error_message)
        if registered and not set(registered) <= set(
            await GroupStore(database).list_groups()
        ):
            error_message = "Migrated group registry differs from the source."
            raise ValueError(error_message)
    digest = hashlib.sha256(
        json.dumps(
            [asdict(r) for r in sorted(expected.values(), key=lambda r: r.order)],
            default=str,
            sort_keys=True,
        ).encode()
    ).hexdigest()
    return {
        "source_originals": len(expected),
        "available_originals": len(actual),
        "registered_groups": len(registered),
        "cached_embeddings_read": vector_count,
        "canonical_sha256": digest,
        "verified": verify,
        "provider_calls": 0,
    }
