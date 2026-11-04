"""Incremental, physically chat-isolated search documents and vector indexes."""

import asyncio
import hashlib
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any

from psycopg import AsyncConnection, sql

from dashulya.database import Database
from dashulya.memory._unicode61 import tokens
from dashulya.memory.hybrid import _blocks, _embedding_text, _render
from dashulya.memory.store import original
from dashulya.memory.vectors import EmbeddingCache

_DOCUMENT_COLUMNS = (
    "message_id",
    "depth",
    "chat_id",
    "author_id",
    "sent_at",
    "edited_at",
    "valid_after",
    "valid_until",
    "source_ids",
    "source_revisions",
    "flat",
    "vector_key",
    "payload",
)


def _documents(
    rows: list[dict[str, Any]], anchors: set[tuple[str, int]], vectors: EmbeddingCache
) -> list[dict[str, Any]]:
    revisions = {(r["collection"], r["message_id"]): r["revision"] for r in rows}
    documents = []
    for original_block in _blocks([original(row) for row in rows]):
        block = original_block
        for position in range(len(block) - 2, -1, -1):
            if block[position].chat_id != block[-1].chat_id:
                block = block[position + 1 :]
                break
        anchor = block[-1]
        if (anchor.collection, anchor.message_id) not in anchors:
            continue
        for size in range(1, len(block) + 1):
            selected = block[-size:]
            # A shorter block is needed only when the next parent's current
            # version is a later edit. Unedited parents precede their child.
            excluded = block[-size - 1] if size < len(block) else None
            if excluded is not None and excluded.edited_at is None:
                continue
            edits = [
                datetime.fromisoformat(m.edited_at) for m in selected[:-1] if m.edited_at
            ]
            payload = _embedding_text(selected)
            documents.append({
                "collection": anchor.collection,
                "message_id": anchor.message_id,
                "depth": size - 1,
                "chat_id": anchor.chat_id,
                "author_id": anchor.author_id,
                "sent_at": anchor.sent_at,
                "edited_at": datetime.fromisoformat(anchor.edited_at)
                if anchor.edited_at
                else None,
                "valid_after": max(edits) if edits else None,
                "valid_until": datetime.fromisoformat(excluded.edited_at)
                if excluded and excluded.edited_at
                else None,
                "source_ids": [m.message_id for m in selected],
                "source_revisions": [
                    revisions[m.collection, m.message_id] for m in selected
                ],
                "flat": " ".join(tokens("\n\n".join(_render(m) for m in selected))),
                "vector_key": vectors.key(payload),
                "payload": payload,
            })
    return documents


class SearchIndex:
    """Own incremental documents, per-chat native indexes and active readers.

    Preparation reads bounded dirty batches and their two reply parents.
    Retrieval never supplies a corpus or a list of eligible document keys.
    """

    def __init__(self, database: Database, vectors: EmbeddingCache) -> None:
        self.database = database
        self.vectors = vectors
        self.namespace = ""
        self.owner = ""
        self._init_lock = asyncio.Lock()
        self._prepare_lock = asyncio.Lock()
        self._closed = False
        self._users = 0
        self._idle = asyncio.Event()
        self._idle.set()

    async def initialize(self) -> None:
        async with self._init_lock:
            await self._initialize()

    async def _initialize(self) -> None:
        if self.namespace:
            return
        async with self.database.connection() as connection, connection.transaction():
            await connection.execute("SELECT pg_advisory_xact_lock(489627302)")
            row = await (
                await connection.execute(
                    "SELECT c.namespace,pg_get_userbyid(n.nspowner) AS owner FROM lexical_cache_config c JOIN pg_namespace n ON n.nspname=c.namespace"
                )
            ).fetchone()
            assert row is not None
            namespace = str(row["namespace"])
            self.owner = str(row["owner"])
            await connection.execute(
                sql.SQL(
                    "CREATE TABLE IF NOT EXISTS {} (collection text PRIMARY KEY,"
                    "table_name text UNIQUE NOT NULL,chat_id bigint)"
                ).format(sql.Identifier(namespace, "search_chats"))
            )
            await connection.execute(
                sql.SQL("ALTER TABLE {} OWNER TO {}").format(
                    sql.Identifier(namespace, "search_chats"), sql.Identifier(self.owner)
                )
            )
        self.namespace = namespace
        try:
            async with self.database.connection() as connection, connection.transaction():
                await connection.execute("SELECT pg_advisory_xact_lock(489627302)")
                chats = await (
                    await connection.execute(
                        sql.SQL("SELECT collection FROM {}").format(
                            sql.Identifier(namespace, "search_chats")
                        )
                    )
                ).fetchall()
                for row in chats:
                    await self._metadata(connection, row["collection"])
        except BaseException:
            self.namespace = ""
            raise

    def table(self, collection: str) -> sql.Identifier:
        name = "chat_" + hashlib.sha256(collection.encode()).hexdigest()[:40]
        return sql.Identifier(self.namespace, name)

    def metadata(self, collection: str) -> sql.Identifier:
        name = (
            "chat_" + hashlib.sha256(collection.encode()).hexdigest()[:40] + "_metadata"
        )
        return sql.Identifier(self.namespace, name)

    async def _metadata(
        self, connection: AsyncConnection[dict[str, Any]], collection: str
    ) -> None:
        table = self.metadata(collection)
        exists = await (
            await connection.execute(
                "SELECT to_regclass(%s) AS existing", (table.as_string(connection),)
            )
        ).fetchone()
        if exists and exists["existing"]:
            return
        await connection.execute(
            sql.SQL(
                "CREATE TABLE {} AS SELECT message_id,depth,chat_id,author_id,sent_at,edited_at,valid_after,valid_until,(embedding IS NOT NULL) AS embedded FROM {}"
            ).format(table, self.table(collection))
        )
        await connection.execute(
            sql.SQL("ALTER TABLE {} ADD PRIMARY KEY(message_id,depth)").format(table)
        )
        await connection.execute(
            sql.SQL("ALTER TABLE {} OWNER TO {}").format(
                table, sql.Identifier(self.owner)
            )
        )
        await connection.execute(
            sql.SQL("CREATE INDEX ON {} (author_id,sent_at)").format(table)
        )
        await connection.execute(sql.SQL("ANALYZE {}").format(table))

    @asynccontextmanager
    async def lease(self) -> AsyncGenerator[None]:
        if self._closed:
            message = "Memory retrieval is closed."
            raise RuntimeError(message)
        self._users += 1
        self._idle.clear()
        try:
            yield
        finally:
            self._users -= 1
            if not self._users:
                self._idle.set()

    async def close(self) -> None:
        self._closed = True
        await self._idle.wait()

    async def has_pending(self) -> bool:
        async with self.database.connection() as connection:
            row = await (
                await connection.execute(
                    "SELECT EXISTS(SELECT 1 FROM search_pending) AS pending"
                )
            ).fetchone()
            assert row is not None
            return bool(row["pending"])

    async def prepare(
        self, *, collection: str | None = None, batch_size: int = 512
    ) -> int:
        """Commit one bounded lexical batch; embeddings are attached separately."""
        await self.initialize()
        async with (
            self._prepare_lock,
            self.database.connection() as connection,
            connection.transaction(),
        ):
            await connection.execute("SELECT pg_advisory_xact_lock(489627302)")
            cursor = await connection.execute(
                "SELECT collection,message_id,version FROM search_pending "
                "WHERE (%s::text IS NULL OR collection=%s) ORDER BY version LIMIT %s",
                (collection, collection, batch_size),
            )
            pending = await cursor.fetchall()
            if not pending:
                return 0
            collections = sorted({r["collection"] for r in pending})
            for chat in collections:
                ids = [r["message_id"] for r in pending if r["collection"] == chat]
                cursor = await connection.execute(
                    """WITH RECURSIVE lineage AS (
                        SELECT m.*,0 AS depth FROM messages m WHERE collection=%s AND message_id=ANY(%s)
                        UNION ALL SELECT p.*,c.depth+1 FROM messages p JOIN lineage c
                            ON p.collection=c.collection AND p.message_id=c.reply_to_id
                            WHERE c.depth<2 AND p.message_id<>c.message_id
                    ) SELECT DISTINCT ON (message_id) * FROM lineage ORDER BY message_id,depth""",
                    (chat, ids),
                )
                rows = await cursor.fetchall()
                table = self.table(chat)
                name = "chat_" + hashlib.sha256(chat.encode()).hexdigest()[:40]
                exists = await (
                    await connection.execute(
                        sql.SQL("SELECT collection FROM {} WHERE collection=%s").format(
                            sql.Identifier(self.namespace, "search_chats")
                        ),
                        (chat,),
                    )
                ).fetchone()
                if exists is None:
                    await connection.execute(
                        sql.SQL("""CREATE TABLE {} (
                        id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
                        message_id bigint NOT NULL,depth smallint NOT NULL,
                        chat_id bigint,author_id bigint,sent_at timestamptz,edited_at timestamptz,
                        valid_after timestamptz,valid_until timestamptz,
                        source_ids bigint[] NOT NULL,source_revisions bigint[] NOT NULL,
                        flat text NOT NULL,vector_key text NOT NULL,payload text NOT NULL,
                        embedding vector(512),UNIQUE(message_id,depth))""").format(table)
                    )
                    await connection.execute(
                        sql.SQL("ALTER TABLE {} OWNER TO {}").format(
                            table, sql.Identifier(self.owner)
                        )
                    )
                    await connection.execute(
                        sql.SQL("INSERT INTO {} VALUES(%s,%s,%s)").format(
                            sql.Identifier(self.namespace, "search_chats")
                        ),
                        (chat, name, rows[0]["chat_id"] if rows else None),
                    )
                await connection.execute(
                    sql.SQL("DELETE FROM {} WHERE message_id=ANY(%s)").format(table),
                    (ids,),
                )
                await self._metadata(connection, chat)
                await connection.execute(
                    sql.SQL("DELETE FROM {} WHERE message_id=ANY(%s)").format(
                        self.metadata(chat)
                    ),
                    (ids,),
                )
                documents = await asyncio.to_thread(
                    _documents, rows, {(chat, number) for number in ids}, self.vectors
                )
                columns = sql.SQL(",").join(map(sql.Identifier, _DOCUMENT_COLUMNS))
                await connection.execute(
                    sql.SQL(
                        "CREATE TEMP TABLE search_documents ON COMMIT DROP AS SELECT {} FROM {} WITH NO DATA"
                    ).format(columns, table)
                )
                async with connection.cursor().copy(
                    sql.SQL("COPY search_documents ({}) FROM STDIN").format(columns)
                ) as writer:
                    for document in documents:
                        await writer.write_row(
                            tuple(document[column] for column in _DOCUMENT_COLUMNS)
                        )
                await connection.execute("ANALYZE search_documents")
                await connection.execute(
                    sql.SQL(
                        "INSERT INTO {} ({},embedding) SELECT s.*,e.embedding FROM search_documents s LEFT JOIN embeddings e ON e.cache_key=s.vector_key"
                    ).format(table, columns)
                )
                await connection.execute("DROP TABLE search_documents")
                await connection.execute(
                    sql.SQL(
                        "INSERT INTO {} SELECT message_id,depth,chat_id,author_id,sent_at,edited_at,valid_after,valid_until,(embedding IS NOT NULL) FROM {} WHERE message_id=ANY(%s)"
                    ).format(self.metadata(chat), table),
                    (ids,),
                )
                if exists is None:
                    await connection.execute(
                        sql.SQL(
                            "CREATE INDEX ON {} USING paradedb (id,(flat::pdb.whitespace('lowercase=false','pnorms=true','k1=2.0','b=0.75')),message_id,chat_id,author_id,sent_at,edited_at,valid_after,valid_until)"
                        ).format(table)
                    )
                    await connection.execute(
                        sql.SQL("CREATE INDEX ON {} (author_id,sent_at)").format(table)
                    )
                    await connection.execute(
                        sql.SQL(
                            "CREATE INDEX ON {} (vector_key) WHERE embedding IS NULL"
                        ).format(table)
                    )
                if exists is None:
                    await connection.execute(sql.SQL("ANALYZE {}").format(table))
            async with connection.cursor() as cursor:
                await cursor.executemany(
                    "DELETE FROM search_pending WHERE collection=%(collection)s AND message_id=%(message_id)s AND version=%(version)s",
                    pending,
                )
            return len(pending)

    async def bindings(
        self, *, chat_id: int | None = None, collection: str | None = None
    ) -> dict[str, int | None]:
        await self.initialize()
        async with self.database.connection() as connection:
            cursor = await connection.execute(
                sql.SQL(
                    "SELECT collection,chat_id FROM {} WHERE (%s::bigint IS NULL OR chat_id=%s) "
                    "AND (%s::text IS NULL OR collection=%s) ORDER BY collection"
                ).format(sql.Identifier(self.namespace, "search_chats")),
                (chat_id, chat_id, collection, collection),
            )
            return {row["collection"]: row["chat_id"] for row in await cursor.fetchall()}

    async def chats(
        self, *, chat_id: int | None = None, collection: str | None = None
    ) -> list[str]:
        return list(await self.bindings(chat_id=chat_id, collection=collection))

    async def embed_pending(self) -> None:
        """Reuse durable vectors; create only missing current full-block payloads."""
        for chat in await self.chats():
            table = self.table(chat)
            async with self.database.connection() as connection:
                cursor = await connection.execute(
                    sql.SQL(
                        "SELECT d.payload FROM {} d LEFT JOIN embeddings e ON e.cache_key=d.vector_key "
                        "WHERE d.valid_until IS NULL AND d.embedding IS NULL AND e.cache_key IS NULL"
                    ).format(table)
                )
                payloads = [r["payload"] for r in await cursor.fetchall()]
            if payloads:
                await self.vectors.vectors(payloads, load_vectors=False)
            async with self.database.connection() as connection:
                await connection.execute(
                    sql.SQL(
                        "WITH updated AS (UPDATE {} d SET embedding=e.embedding FROM embeddings e "
                        "WHERE d.embedding IS NULL AND d.vector_key=e.cache_key RETURNING d.message_id,d.depth) "
                        "UPDATE {} m SET embedded=true FROM updated u WHERE m.message_id=u.message_id AND m.depth=u.depth"
                    ).format(table, self.metadata(chat))
                )

    async def finish(self) -> None:
        """Build ANN indexes after bulk loading, then maintain them incrementally."""
        for chat in await self.chats():
            table = self.table(chat)
            name = (
                "chat_" + hashlib.sha256(chat.encode()).hexdigest()[:40] + "_vector_idx"
            )
            async with self.database.connection() as connection:
                existing = await (
                    await connection.execute(
                        "SELECT to_regclass(%s) AS existing",
                        (sql.Identifier(self.namespace, name).as_string(connection),),
                    )
                ).fetchone()
                if existing and existing["existing"]:
                    continue
                await connection.execute(
                    sql.SQL(
                        "CREATE INDEX IF NOT EXISTS {} ON {} USING hnsw (embedding vector_ip_ops) WHERE embedding IS NOT NULL"
                    ).format(sql.Identifier(name), table)
                )
                await connection.execute(sql.SQL("ANALYZE {}").format(table))
