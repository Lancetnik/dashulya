import sqlite3
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import aiosqlite


def _connect(path: Path, *, readonly: bool = True) -> sqlite3.Connection:
    connection = sqlite3.connect(
        path.resolve().as_uri() + ("?mode=ro" if readonly else "?mode=rw"), uri=True
    )
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


@asynccontextmanager
async def _read_connection(path: Path) -> AsyncIterator[aiosqlite.Connection]:
    uri = path.resolve().as_uri()  # noqa: ASYNC240 -- Normalize the configured local filename; SQLite I/O uses aiosqlite.
    async with aiosqlite.connect(uri + "?mode=ro", uri=True) as connection:
        connection.row_factory = sqlite3.Row
        async with connection.execute("PRAGMA foreign_keys = ON"):
            yield connection
