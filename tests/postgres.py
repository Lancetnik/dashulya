"""Real PostgreSQL fixtures with one disposable schema per test."""

import os
import unittest
from unittest.mock import patch
from uuid import uuid4

from psycopg import AsyncConnection, sql
from psycopg.conninfo import make_conninfo

from dashulya.database import Database
from dashulya.domain.memory import MemoryAccess
from dashulya.memory.history import CanonicalHistory
from dashulya.memory.migrate import import_evidence
from dashulya.memory.search import MemorySearch
from dashulya.memory.store import MessageStore


class PostgresTestCase(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.server_url = os.getenv("TEST_DATABASE_URL", "")
        if not self.server_url:
            self.skipTest("Set TEST_DATABASE_URL to run PostgreSQL integration tests.")
        self.schema = "test_" + uuid4().hex
        self.database_url = make_conninfo(
            self.server_url, options="-c search_path=" + self.schema + ",public"
        )
        self.environment = patch.dict(os.environ, {"DATABASE_URL": self.database_url})
        self.environment.start()
        self.addCleanup(self.environment.stop)

    async def asyncSetUp(self) -> None:
        async with await AsyncConnection.connect(
            self.server_url, autocommit=True
        ) as connection:
            await connection.execute(
                sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(self.schema))
            )

        async def drop() -> None:
            async with await AsyncConnection.connect(
                self.server_url, autocommit=True
            ) as connection:
                await connection.execute(
                    sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(self.schema))
                )

        self.addAsyncCleanup(drop)
        self.database = Database(self.database_url)
        await self.database.open()
        self.addAsyncCleanup(self.database.close)


async def imported_history(
    store: MessageStore,
    access: MemoryAccess,
    *,
    imported: MemorySearch,
    import_chat_id: int | None = None,
) -> CanonicalHistory:

    await import_evidence(store, imported.path, import_chat_id)
    return CanonicalHistory(store, access)
