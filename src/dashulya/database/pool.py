"""Shared PostgreSQL resource; callers own operations, this module owns connections."""

import asyncio
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

from psycopg import AsyncConnection
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from .migrate import require_schema


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
                async with self.pool.connection() as connection:
                    await require_schema(connection)
                self._ready = True
            except BaseException:
                await self.pool.close()
                raise

    @asynccontextmanager
    async def connection(self) -> AsyncGenerator[AsyncConnection[dict[str, Any]]]:
        if not self._ready:
            await self.open()
        async with self.pool.connection() as connection:
            await connection.execute("SET TIME ZONE 'UTC'")
            yield connection

    async def close(self) -> None:
        await self.pool.close()
