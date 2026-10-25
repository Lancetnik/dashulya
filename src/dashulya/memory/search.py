"""Search messages and source-backed memories through one local index."""

import json
import re
import sqlite3
from contextlib import closing
from dataclasses import asdict, dataclass
from datetime import datetime
from operator import itemgetter
from pathlib import Path
from typing import Any, Self
from uuid import uuid4

from dashulya.memory._database import _connect, _read_connection


def _date(value: str | None) -> str | None:
    if value is None:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        error_message = "Dates must use ISO 8601 with a timezone."
        raise ValueError(error_message) from None
    if parsed.tzinfo is None:
        error_message = "Dates must include a timezone, for example 2026-10-25T00:00:00Z."
        raise ValueError(error_message)
    return parsed.isoformat()


def validate_search(
    query: str,
    memory_id: str | None,
    after: str | None,
    before: str | None,
    limit: int,
) -> tuple[list[str], str | None, str | None]:
    """Apply the same query and original-time bounds to both retrieval backends."""
    if not 1 <= limit <= 5:
        error_message = "limit must be between 1 and 5."
        raise ValueError(error_message)
    if bool(query.strip()) == bool(memory_id):
        error_message = "Provide either query keywords or a memory_id."
        raise ValueError(error_message)
    after, before = _date(after), _date(before)
    if (
        after
        and before
        and datetime.fromisoformat(after) > datetime.fromisoformat(before)
    ):
        error_message = "after must not be later than before."
        raise ValueError(error_message)
    tokens = re.findall(r"[^\W_]+", query, re.UNICODE)
    if not memory_id and (not tokens or len(tokens) > 20 or len(query) > 500):
        error_message = "Use 1 to 20 keywords, at most 500 characters."
        raise ValueError(error_message)
    return tokens, after, before


@dataclass(frozen=True)
class Evidence:
    message_id: int
    sent_at: str | None
    user_id: int | None
    author: str
    text: str
    url: str
    reply_to_message_id: int | None
    truncated: bool
    excerpt_start: int


@dataclass(frozen=True)
class MemoryHit:
    memory_id: str
    kind: str
    excerpt: str
    sources: tuple[Evidence, ...]
    context: tuple[Evidence, ...]
    related_memory_ids: tuple[str, ...]


def _evidence(row: sqlite3.Row, tokens: list[str]) -> Evidence:
    matches = [
        match.start()
        for token in tokens
        if (match := re.search(re.escape(token), row["body"], re.IGNORECASE))
    ]
    start = max(0, min(matches) - 200) if matches else 0
    return Evidence(
        row["id"],
        row["sent_at"],
        row["user_id"],
        row["author"],
        row["body"][start : start + 1200],
        row["url"],
        row["reply_to_id"],
        len(row["body"]) > 1200,
        start,
    )


class MemorySearch:
    """One search interface for the owner-only, single-chat experimental corpus."""

    def __init__(self, path: Path, coverage: dict[str, Any]) -> None:
        self.path = path
        self.coverage = coverage
        if self.coverage.get("version") != 1:
            error_message = "Unsupported memory index version."
            raise ValueError(error_message)

    @classmethod
    async def open(cls, path: Path) -> Self:
        async with _read_connection(path) as connection:
            rows = await connection.execute_fetchall("SELECT * FROM metadata")
        return cls(path, {row["key"]: json.loads(row["value"]) for row in rows})

    async def search(  # noqa: PLR0914 -- Keep SQL selection and evidence assembly together.
        self,
        query: str = "",
        *,
        memory_id: str | None = None,
        author_id: int | None = None,
        after: str | None = None,
        before: str | None = None,
        limit: int = 3,
    ) -> dict[str, Any]:
        """Search all levels, or expand a returned memory ID, with original evidence."""
        tokens, after, before = validate_search(query, memory_id, after, before, limit)
        expression = " AND ".join(f'"{token}"*' for token in tokens)
        async with _read_connection(self.path) as connection:
            # A derived memory is eligible only when every source satisfies the requested filters.
            filters = """
                NOT EXISTS (
                    SELECT 1 FROM sources s JOIN messages m ON m.id = s.message_id
                    WHERE s.memory_id = n.id AND (
                        (? IS NOT NULL AND (m.user_id IS NULL OR m.user_id != ?))
                        OR (? IS NOT NULL AND (m.sent_at IS NULL OR julianday(m.sent_at) < julianday(?)))
                        OR (? IS NOT NULL AND (m.sent_at IS NULL OR julianday(m.sent_at) > julianday(?)))
                    )
                )
            """
            parameters = [author_id, author_id, after, after, before, before]
            if memory_id:
                nodes = await connection.execute_fetchall(
                    f"SELECT n.* FROM memories n WHERE n.key = ? AND {filters}",  # noqa: S608 -- Fixed SQL fragment; values are bound.
                    [memory_id, *parameters],
                )
            else:
                nodes = await connection.execute_fetchall(
                    f"""SELECT n.* FROM memory_fts JOIN memories n ON n.id = memory_fts.rowid
                        WHERE memory_fts MATCH ? AND {filters}
                        ORDER BY bm25(memory_fts), n.id LIMIT 100""",  # noqa: S608 -- Fixed SQL fragment; values are bound.
                    [expression, *parameters],
                )
            hits, seen = [], set()
            for node in nodes:
                if node["conversation"] in seen:
                    continue
                sources = list(
                    await connection.execute_fetchall(
                        "SELECT m.* FROM sources s JOIN messages m ON m.id = s.message_id WHERE s.memory_id = ? ORDER BY m.id",
                        (node["id"],),
                    )
                )
                if not sources:
                    continue
                placeholders = ",".join("?" for _ in sources)
                related = await connection.execute_fetchall(
                    f"""SELECT DISTINCT parent.key FROM memories parent
                        JOIN sources s ON s.memory_id = parent.id
                        WHERE s.message_id IN ({placeholders})
                            AND parent.kind IN ('discussion', 'window') AND parent.id != ?
                        ORDER BY parent.key LIMIT 4""",  # noqa: S608 -- Only parameter placeholders are interpolated.
                    [*(source["id"] for source in sources), node["id"]],
                )
                seen.add(node["conversation"])
                context: dict[int, sqlite3.Row] = {}
                source_ids = {source["id"] for source in sources}
                for source in sources:
                    parent_id = source["reply_to_id"]
                    for _ in range(2):
                        if parent_id is None:
                            break
                        async with connection.execute(
                            "SELECT * FROM messages WHERE id = ?", (parent_id,)
                        ) as cursor:
                            parent = await cursor.fetchone()
                        if parent is None:
                            break
                        if parent["id"] not in source_ids:
                            context[parent["id"]] = parent
                        parent_id = parent["reply_to_id"]
                for operator, ordering, anchor in (
                    ("<", "DESC", sources[0]["id"]),
                    (">", "ASC", sources[-1]["id"]),
                ):
                    for neighbor in await connection.execute_fetchall(
                        f"SELECT * FROM messages WHERE conversation = ? AND id {operator} ? ORDER BY id {ordering} LIMIT 2",  # noqa: S608 -- Operator and ordering come from fixed tuples above.
                        (sources[0]["conversation"], anchor),
                    ):
                        if neighbor["id"] not in source_ids:
                            context[neighbor["id"]] = neighbor
                hits.append(
                    MemoryHit(
                        node["key"],
                        node["kind"],
                        node["body"][:1500],
                        tuple(_evidence(row, tokens) for row in sources),
                        tuple(
                            _evidence(row, tokens)
                            for row in sorted(context.values(), key=itemgetter("id"))[:4]
                        ),
                        tuple(row["key"] for row in related),
                    )
                )
                if len(hits) == limit:
                    break
        return {"coverage": self.coverage, "hits": [asdict(hit) for hit in hits]}

    def remember(self, text: str, source_ids: list[int], *, kind: str = "summary") -> str:
        """Store a derived record only when all its original messages exist."""
        source_ids = sorted(set(source_ids))
        if kind not in {"summary", "fact"} or not text.strip() or len(text) > 4000:
            error_message = "Provide a summary or fact with 1 to 4000 characters."
            raise ValueError(error_message)
        if not 1 <= len(source_ids) <= 8:
            error_message = "A memory must cite 1 to 8 source message IDs."
            raise ValueError(error_message)
        with closing(_connect(self.path, readonly=False)) as connection, connection:
            placeholders = ",".join("?" for _ in source_ids)
            rows = connection.execute(
                f"SELECT id, conversation FROM messages WHERE id IN ({placeholders})",  # noqa: S608 -- Only parameter placeholders are interpolated.
                source_ids,
            ).fetchall()
            if len(rows) != len(source_ids):
                error_message = "Every source message ID must exist in this index."
                raise ValueError(error_message)
            key = f"{kind}:{uuid4().hex}"
            conversations = {row["conversation"] for row in rows}
            conversation = next(iter(conversations)) if len(conversations) == 1 else key
            cursor = connection.execute(
                "INSERT INTO memories(key, kind, conversation, body) VALUES (?, ?, ?, ?)",
                (key, kind, conversation, text.strip()),
            )
            connection.executemany(
                "INSERT INTO sources VALUES (?, ?)",
                ((cursor.lastrowid, source_id) for source_id in source_ids),
            )
        return key
