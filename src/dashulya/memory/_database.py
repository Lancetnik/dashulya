import sqlite3
from collections.abc import AsyncGenerator
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
async def _read_connection(path: Path) -> AsyncGenerator[aiosqlite.Connection, None]:
    uri = path.resolve().as_uri()  # ruff: ignore[blocking-path-method-in-async-function] -- Normalize the configured local filename; SQLite I/O uses aiosqlite.
    async with aiosqlite.connect(uri + "?mode=ro", uri=True) as connection:
        connection.row_factory = sqlite3.Row
        async with connection.execute("PRAGMA foreign_keys = ON"):
            yield connection
