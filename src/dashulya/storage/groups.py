import sqlite3
from itertools import starmap
from pathlib import Path

from dashulya.domain.groups import Group


class GroupStore:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(path)
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS groups (
                chat_id INTEGER PRIMARY KEY,
                title TEXT NOT NULL,
                username TEXT
            )
            """
        )
        self._connection.commit()

    def remember(self, group: Group, *, previous_chat_id: int | None = None) -> None:
        with self._connection:
            if previous_chat_id is not None:
                self._connection.execute(
                    "DELETE FROM groups WHERE chat_id = ?", (previous_chat_id,)
                )
            self._connection.execute(
                """
                INSERT INTO groups (chat_id, title, username) VALUES (?, ?, ?)
                ON CONFLICT(chat_id) DO UPDATE SET
                    title = excluded.title,
                    username = excluded.username
                WHERE groups.title != excluded.title
                    OR groups.username IS NOT excluded.username
                """,
                (group.chat_id, group.title, group.username),
            )

    def forget(self, chat_id: int) -> None:
        with self._connection:
            self._connection.execute("DELETE FROM groups WHERE chat_id = ?", (chat_id,))

    def list_groups(self) -> list[Group]:
        rows = self._connection.execute(
            "SELECT chat_id, title, username FROM groups ORDER BY title COLLATE NOCASE, chat_id"
        ).fetchall()
        return list(starmap(Group, rows))

    def close(self) -> None:
        self._connection.close()
