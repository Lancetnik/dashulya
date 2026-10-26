import asyncio
from datetime import UTC, datetime

from dashulya.domain.memory import MemoryAccess, MemoryMessage
from dashulya.memory.service import HybridChatMemory
from dashulya.memory.store import MessageStore
from dashulya.memory.vectors import EmbeddingCache
from tests.postgres import PostgresTestCase


class MemoryCancellationTests(PostgresTestCase):
    async def test_native_query_cancellation_returns_a_usable_connection(self) -> None:
        started = asyncio.Event()

        async def slow() -> None:
            async with self.database.connection() as connection:
                started.set()
                await connection.execute("SELECT pg_sleep(30)")

        task = asyncio.create_task(slow())
        await started.wait()
        await asyncio.sleep(0.02)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await asyncio.wait_for(task, 2)
        async with self.database.connection() as connection:
            row = await (await connection.execute("SELECT 42 AS value")).fetchone()
        self.assertEqual(row, {"value": 42})

    async def test_cancelled_recall_releases_lease_and_preserves_originals(self) -> None:
        store = MessageStore(self.database)
        await store.record(
            MemoryMessage(
                -123, 1, "needle", datetime(2026, 1, 1, tzinfo=UTC), 42, "Alice", "Group"
            )
        )
        started, cancelled = asyncio.Event(), asyncio.Event()

        async def embed(texts: list[str]) -> list[list[float]]:
            return [[1.0] + [0.0] * 511 for _ in texts]

        cache = EmbeddingCache(self.database, embed)
        memory = HybridChatMemory(store, cache, owner_user_id=42)
        self.addAsyncCleanup(memory.close)
        await memory.refresh_once()

        async def blocked(texts: list[str]) -> list[list[float]]:
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()
            return []

        cache.embedder = blocked
        task = asyncio.create_task(
            memory.recall(MemoryAccess(-123, False)).search("new query")
        )
        await asyncio.wait_for(started.wait(), 2)
        closing = asyncio.create_task(memory.close())
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        await asyncio.wait_for(closing, 2)
        self.assertTrue(cancelled.is_set())
        self.assertEqual(
            (await memory.history(MemoryAccess(-123, False)).query("count"))["count"], 1
        )
