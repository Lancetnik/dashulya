"""Build a local memory index from a historical chat snapshot."""

import hashlib
import json
import re
import sqlite3
import tempfile
from collections import defaultdict
from contextlib import closing
from datetime import datetime
from pathlib import Path
from typing import Any

from dashulya.memory._database import _connect

SCHEMA = """
PRAGMA foreign_keys = ON;
CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE messages (
    id INTEGER PRIMARY KEY, sent_at TEXT, user_id INTEGER, author TEXT NOT NULL,
    body TEXT NOT NULL, reply_to_id INTEGER, conversation TEXT NOT NULL, url TEXT NOT NULL
);
CREATE INDEX messages_conversation ON messages(conversation, id);
CREATE TABLE memories (
    id INTEGER PRIMARY KEY, key TEXT NOT NULL UNIQUE,
    kind TEXT NOT NULL CHECK(kind IN ('message', 'discussion', 'window', 'summary', 'fact')),
    conversation TEXT NOT NULL, body TEXT NOT NULL
);
CREATE TABLE sources (
    memory_id INTEGER NOT NULL REFERENCES memories(id),
    message_id INTEGER NOT NULL REFERENCES messages(id),
    PRIMARY KEY(memory_id, message_id)
);
CREATE VIRTUAL TABLE memory_fts USING fts5(
    body, content='memories', content_rowid='id', tokenize='unicode61', prefix='2 3 4'
);
CREATE TRIGGER memories_insert AFTER INSERT ON memories BEGIN
    INSERT INTO memory_fts(rowid, body) VALUES (new.id, new.body);
END;
"""


def _conversation(
    message: sqlite3.Row, messages: dict[int, sqlite3.Row], reply_targets: set[int]
) -> str:
    current, visited = message, set()
    while current["id"] not in visited:
        visited.add(current["id"])
        if current["thread_post_id"] is not None:
            return f"post:{current['thread_post_id']}"
        parent = messages.get(current["reply_to_msg_id"])
        if parent is None:
            break
        current = parent
    if len(visited) > 1 or message["id"] in reply_targets:
        return f"reply:{min(visited) if current['id'] in visited and current['reply_to_msg_id'] in visited else current['id']}"
    if message["date"]:
        bucket = int(datetime.fromisoformat(message["date"]).timestamp()) // 900
        return f"window:{bucket}"
    return f"message:{message['id']}"


def build_memory(
    source: Path, destination: Path, *, chat_username: str
) -> dict[str, Any]:
    """Create a derived index; never overwrite an existing database or alter the source."""
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{4,31}", chat_username):
        error_message = "chat_username must be a public Telegram username without @."
        raise ValueError(error_message)
    if destination.exists():
        error_message = f"Memory index already exists: {destination}"
        raise FileExistsError(error_message)
    with source.open("rb") as stream:
        source_checksum = hashlib.file_digest(stream, "sha256").hexdigest()
    with closing(_connect(source)) as original:
        rows = {
            row["id"]: row
            for row in original.execute("SELECT * FROM group_messages ORDER BY id")
        }
    if not rows:
        error_message = "The source contains no group messages."
        raise ValueError(error_message)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=destination.parent, suffix=".sqlite3", delete=False
    ) as file:
        temporary = Path(file.name)
    try:
        with closing(sqlite3.connect(temporary)) as index:
            index.executescript(SCHEMA)
            conversations: dict[str, list[sqlite3.Row]] = defaultdict(list)
            reply_targets = {
                row["reply_to_msg_id"]
                for row in rows.values()
                if row["reply_to_msg_id"] is not None
            }
            for row in rows.values():
                conversation = _conversation(row, rows, reply_targets)
                conversations[conversation].append(row)
                index.execute(
                    "INSERT INTO messages VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        row["id"],
                        row["date"],
                        row["user_id"],
                        row["author"] or "Unknown sender",
                        row["text"] or "",
                        row["reply_to_msg_id"],
                        conversation,
                        f"https://t.me/{chat_username}/{row['id']}",
                    ),
                )
                index.execute(
                    "INSERT INTO memories(id, key, kind, conversation, body) VALUES (?, ?, 'message', ?, ?)",
                    (row["id"], f"message:{row['id']}", conversation, row["text"] or ""),
                )
                index.execute("INSERT INTO sources VALUES (?, ?)", (row["id"], row["id"]))
            for conversation, messages in conversations.items():
                for offset in range(0, len(messages), 8):
                    chunk = messages[offset : offset + 8]
                    if len(chunk) < 2:
                        continue
                    kind = (
                        "window" if conversation.startswith("window:") else "discussion"
                    )
                    body = "\n".join(
                        f"{row['author'] or 'Unknown sender'}: {row['text'] or ''}"
                        for row in chunk
                    )
                    cursor = index.execute(
                        "INSERT INTO memories(key, kind, conversation, body) VALUES (?, ?, ?, ?)",
                        (f"{conversation}:{offset}", kind, conversation, body),
                    )
                    index.executemany(
                        "INSERT INTO sources VALUES (?, ?)",
                        ((cursor.lastrowid, row["id"]) for row in chunk),
                    )
            with source.open("rb") as stream:
                checksum = hashlib.file_digest(stream, "sha256").hexdigest()
            if checksum != source_checksum:
                error_message = (
                    "Source changed during indexing; retry from a stable SQLite snapshot."
                )
                raise ValueError(error_message)
            metadata = {
                "version": 1,
                "chat_username": chat_username,
                "source_sha256": checksum,
                "source_path": str(source.resolve()),
                "message_count": len(rows),
                "first_message_at": min(
                    (row["date"] for row in rows.values() if row["date"]), default=None
                ),
                "last_message_at": max(
                    (row["date"] for row in rows.values() if row["date"]), default=None
                ),
            }
            index.executemany(
                "INSERT INTO metadata VALUES (?, ?)",
                ((key, json.dumps(value)) for key, value in metadata.items()),
            )
            index.commit()
            assert index.execute("PRAGMA integrity_check").fetchall() == [("ok",)]
        # An exclusive hard link publishes the completed index without replacing another build.
        destination.hardlink_to(temporary)
        return metadata
    finally:
        temporary.unlink(missing_ok=True)
