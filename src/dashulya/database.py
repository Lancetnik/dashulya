"""Shared PostgreSQL resource; callers own operations, this module owns connections."""

import asyncio
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from psycopg import AsyncConnection
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool


class Database:
    def __init__(self, url: str) -> None:
        self.pool = AsyncConnectionPool[AsyncConnection[dict[str, Any]]](
            url,
            open=False,
            min_size=1,
            max_size=8,
            timeout=15,
            kwargs={"autocommit": True, "row_factory": dict_row},
        )
        self._lock = asyncio.Lock()
        self._ready = False

    async def open(self) -> None:
        async with self._lock:
            if self._ready:
                return
            try:
                await self.pool.open(wait=True, timeout=15)
                await self._initialize_schema()
                self._ready = True
            except BaseException:
                await self.pool.close()
                raise

    async def _initialize_schema(self) -> None:
        schema = await asyncio.to_thread(Path(__file__).with_name("schema.sql").read_text)
        async with self.pool.connection() as connection, connection.transaction():
            await connection.execute("SELECT pg_advisory_xact_lock(489627301)")
            await connection.execute(schema)

    @asynccontextmanager
    async def connection(self) -> AsyncGenerator[AsyncConnection[dict[str, Any]], None]:
        if not self._ready:
            await self.open()
        async with self.pool.connection() as connection:
            await connection.execute("SET TIME ZONE 'UTC'")
            yield connection

    async def close(self) -> None:
        await self.pool.close()
