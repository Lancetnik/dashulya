"""Cancellation must finish SQLite work before releasing a shared engine lease."""

import asyncio
import threading
import unittest
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from unittest.mock import patch

from dashulya.domain.memory import MemoryAccess, MemoryMessage
from dashulya.memory._workers import run_in_thread
from dashulya.memory.history import CanonicalHistory
from dashulya.memory.hybrid import HybridMemory, embedding_texts
from dashulya.memory.live import LiveMemory
from dashulya.memory.service import HybridChatMemory
from dashulya.memory.vectors import EmbeddingCache


class MemoryCancellationTests(unittest.IsolatedAsyncioTestCase):
    async def test_cancelled_search_drains_thread_before_lease_and_close(self) -> None:
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        live = LiveMemory(root / "live.sqlite3")
        await live.initialize()
        await live.record(
            MemoryMessage(
                -100123,
                1,
                "needle source",
                datetime(2026, 10, 25, tzinfo=UTC),
                42,
                "Owner",
                "Group",
            )
        )

        async def embed(texts: list[str]) -> list[list[float]]:
            return [[1.0, *([0.0] * 511)] for _ in texts]

        service = HybridChatMemory(
            live, EmbeddingCache(root / "vectors.sqlite3", embed), owner_user_id=42
        )
        self.addAsyncCleanup(service.close)
        originals = await service.history(MemoryAccess(-100123, False)).originals()
        started, release, finished = (
            threading.Event(),
            threading.Event(),
            threading.Event(),
        )
        self.addCleanup(release.set)
        close_before_finish: list[bool] = []
        real_search, real_close = HybridMemory._search_sync, HybridMemory.close

        def blocked(engine: HybridMemory, *args: Any) -> Any:
            started.set()
            if not release.wait(3):
                error = "Test did not release the SQLite worker."
                raise TimeoutError(error)
            try:
                return real_search(engine, *args)
            finally:
                finished.set()

        def close(engine: HybridMemory) -> None:
            close_before_finish.append(not finished.is_set())
            real_close(engine)

        async def search() -> None:
            async with service.engine(originals) as engine:
                await engine.search("needle")

        with (
            patch.object(HybridMemory, "_search_sync", blocked),
            patch.object(HybridMemory, "close", close),
        ):
            query = asyncio.create_task(search())
            self.assertTrue(await asyncio.to_thread(started.wait, 3))
            query.cancel()
            await asyncio.sleep(0)
            query.cancel()  # A second cancellation cannot interrupt worker draining.
            closing = asyncio.create_task(service.close())
            try:
                with self.assertRaises(TimeoutError):
                    await asyncio.wait_for(asyncio.shield(query), 0.03)
                self.assertFalse(closing.done())
                self.assertEqual(close_before_finish, [])
            finally:
                release.set()
            with self.assertRaises(asyncio.CancelledError):
                await query
            await asyncio.wait_for(closing, 3)
        self.assertTrue(finished.is_set())
        self.assertEqual(close_before_finish, [False])

    async def test_provider_embedding_remains_immediately_cancellable(self) -> None:
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        live = LiveMemory(root / "live.sqlite3")
        await live.initialize()
        await live.record(
            MemoryMessage(
                -100123,
                1,
                "needle",
                datetime(2026, 10, 25, tzinfo=UTC),
                42,
                "Owner",
                "Group",
            )
        )
        originals = await CanonicalHistory(live, MemoryAccess(-100123, False)).originals()
        provider_started = asyncio.Event()
        provider_cancelled = asyncio.Event()

        async def query_vector(text: str) -> list[float]:
            provider_started.set()
            try:
                await asyncio.Event().wait()
            finally:
                provider_cancelled.set()
            return [1.0, 0.0]

        engine = HybridMemory(
            originals,
            {embedding_texts(originals)[0]: [1.0, 0.0]},
            query_vector,
            dimensions=2,
        )
        self.addCleanup(engine.close)
        query = asyncio.create_task(engine.search("needle"))
        await asyncio.wait_for(provider_started.wait(), 3)
        query.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await asyncio.wait_for(query, 0.1)
        self.assertTrue(provider_cancelled.is_set())

    async def test_cancelled_constructor_drains_then_closes_returned_resource_once(
        self,
    ) -> None:
        started, release = threading.Event(), threading.Event()
        cleanup_started, cleanup_release = threading.Event(), threading.Event()
        self.addCleanup(release.set)
        self.addCleanup(cleanup_release.set)
        resource = object()
        closed = []

        def construct() -> object:
            started.set()
            if not release.wait(3):
                raise TimeoutError
            return resource

        def cleanup(value: object) -> None:
            cleanup_started.set()
            if not cleanup_release.wait(3):
                raise TimeoutError
            closed.append(value)

        operation = asyncio.create_task(run_in_thread(construct, on_cancel=cleanup))
        self.assertTrue(await asyncio.to_thread(started.wait, 3))
        operation.cancel()
        await asyncio.sleep(0)
        operation.cancel()
        release.set()
        self.assertTrue(await asyncio.to_thread(cleanup_started.wait, 3))
        operation.cancel()
        await asyncio.sleep(0)
        self.assertFalse(operation.done())
        cleanup_release.set()
        with self.assertRaises(asyncio.CancelledError):
            await operation
        self.assertEqual(closed, [resource])
