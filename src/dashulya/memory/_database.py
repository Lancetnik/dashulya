import sqlite3
from pathlib import Path


def _connect(path: Path, *, readonly: bool = True) -> sqlite3.Connection:
    connection = sqlite3.connect(
        path.resolve().as_uri() + ("?mode=ro" if readonly else "?mode=rw"), uri=True
    )
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection
