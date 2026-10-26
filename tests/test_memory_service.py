"""Scoped runtime retrieval through the same service the tools consume."""

import asyncio
import unittest
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import AsyncMock

from dashulya.domain.memory import MemoryAccess, MemoryMessage
from dashulya.memory.live import LiveMemory
from dashulya.memory.service import HybridChatMemory
from dashulya.memory.vectors import EmbeddingCache

NOW = datetime(2026, 10, 25, tzinfo=UTC)


class MemoryServiceTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        self.live = LiveMemory(root / "messages.sqlite3")
        await self.live.initialize()

        async def embed(texts: list[str]) -> list[list[float]]:
            return [[1.0] + [0.0] * 511 for _ in texts]

        self.embed = AsyncMock(side_effect=embed)
        self.service = HybridChatMemory(
            self.live,
            EmbeddingCache(root / "vectors.sqlite3", self.embed),
            owner_user_id=42,
        )
        self.addAsyncCleanup(self.service.close)

    @staticmethod
    def message(chat: int, text: str, *, number: int = 1) -> MemoryMessage:
        return MemoryMessage(
            chat_id=chat,
            message_id=number,
            text=text,
            sent_at=NOW,
            author_id=42,
            author="Alice",
            chat_title="Chat",
            is_bot=False,
        )

    async def test_group_scope_precedes_recall_and_private_access_federates(self) -> None:
        for chat, text in (
            (-1001, "Project Aurora"),
            (-1002, "Project Borealis"),
            (42, "Private secret"),
        ):
            await self.live.record(self.message(chat, text))
        await self.service.refresh_once()
        group = self.service.recall(MemoryAccess(-1001, False))
        result = await group.search("Project")
        self.assertEqual(len(result["hits"]), 1)
        self.assertEqual(result["hits"][0]["sources"][0]["chat_id"], -1001)
        private = self.service.recall(MemoryAccess(42, True))
        all_sources = await private.search("Project")
        self.assertEqual(len(all_sources["hits"]), 3)
        foreign = next(
            hit for hit in all_sources["hits"] if hit["sources"][0]["chat_id"] == 42
        )
        read = await group.search(memory_id=foreign["memory_id"])
        self.assertEqual(read["messages"], [])

    async def test_corrected_source_is_immediate_and_old_vectors_cannot_resurrect_it(
        self,
    ) -> None:
        original = self.message(-1001, "Uses obsolete tool")
        await self.live.record(original)
        await self.service.refresh_once()
        await self.live.record(
            replace(
                original,
                text="Uses current tool",
                edited_at=NOW + timedelta(seconds=1),
                update_id=2,
            )
        )
        result = await self.service.recall(MemoryAccess(-1001, False)).search(
            "current tool"
        )
        self.assertEqual(result["hits"][0]["sources"][0]["text"], "Uses current tool")
        self.assertEqual(result["coverage"]["pending_blocks"], 1)
        self.assertNotIn("obsolete", str(result))
        self.assertEqual(len(await self.live.pending()), 1)
        await self.service.refresh_once()
        refreshed = await self.service.recall(MemoryAccess(-1001, False)).search(
            "current tool"
        )
        self.assertEqual(refreshed["coverage"]["pending_blocks"], 0)
        self.assertNotIn("obsolete", str(refreshed))

    async def test_fixed_cutoff_excludes_future_rows_and_later_edits(self) -> None:
        await self.live.record(self.message(-1001, "Historical Aurora"))
        await self.live.record(
            replace(
                self.message(-1001, "Future Aurora", number=2),
                sent_at=NOW + timedelta(days=1),
            )
        )
        await self.service.refresh_once()
        scope = MemoryAccess(-1001, False, before=NOW + timedelta(hours=1), before_id=100)
        result = await self.service.recall(scope).search("Aurora")
        self.assertEqual(
            [s["message_id"] for hit in result["hits"] for s in hit["sources"]], [1]
        )

    async def test_exact_history_works_during_embedding_outage_and_clear_is_unrelated(
        self,
    ) -> None:
        await self.live.record(self.message(-1001, "Source survives restart"))
        self.embed.side_effect = RuntimeError("provider unavailable")
        with self.assertRaises(RuntimeError):
            await self.service.refresh_once()
        history = self.service.history(MemoryAccess(-1001, False))
        result = await history.query("count")
        self.assertEqual(result["count"], 1)
        self.assertEqual(len(await self.live.pending()), 1)

    async def test_shutdown_waits_for_active_recall_and_rejects_new_searches(
        self,
    ) -> None:
        await self.live.record(self.message(-1001, "Aurora source"))
        await self.service.refresh_once()
        started, release = asyncio.Event(), asyncio.Event()

        async def slow_embedding(texts: list[str]) -> list[list[float]]:
            started.set()
            await release.wait()
            return [[1.0] + [0.0] * 511 for _ in texts]

        self.embed.side_effect = slow_embedding
        search = asyncio.create_task(
            self.service.recall(MemoryAccess(-1001, False)).search("Aurora")
        )
        await asyncio.wait_for(started.wait(), timeout=2)
        closing = asyncio.create_task(self.service.close())
        await asyncio.sleep(0)
        self.assertFalse(closing.done())
        release.set()
        result = await asyncio.wait_for(search, timeout=2)
        self.assertEqual(result["hits"][0]["sources"][0]["text"], "Aurora source")
        await asyncio.wait_for(closing, timeout=2)
        with self.assertRaisesRegex(RuntimeError, "closed"):
            await self.service.recall(MemoryAccess(-1001, False)).search("Aurora")

    async def test_startup_outage_preserves_history_and_pending_work(self) -> None:
        await self.live.record(self.message(-1001, "Persisted original"))
        self.embed.side_effect = RuntimeError("embedding unavailable")
        with self.assertLogs("dashulya.memory.service", level="WARNING"):
            await self.service.start()
        result = await self.service.history(MemoryAccess(-1001, False)).query("count")
        self.assertEqual(result["count"], 1)
        self.assertEqual(len(await self.live.pending()), 1)
        await self.service.close()

    async def test_benchmark_startup_requires_complete_dense_preparation(self) -> None:
        await self.live.record(self.message(-1001, "Persisted original"))
        self.embed.side_effect = RuntimeError("embedding unavailable")
        with self.assertRaises(RuntimeError):
            await self.service.start(require_ready=True)
