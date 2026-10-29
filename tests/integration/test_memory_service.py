"""Scoped runtime retrieval through the same service the tools consume."""

import asyncio
from contextlib import AsyncExitStack
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, patch

import pytest

from dashulya.database import Database
from dashulya.domain.memory import MemoryAccess, MemoryMessage
from dashulya.memory.service import HybridChatMemory
from dashulya.memory.store import MessageStore
from dashulya.memory.vectors import EmbeddingCache

NOW = datetime(2026, 10, 25, tzinfo=UTC)


@pytest.mark.usefixtures("_setup")
class TestMemoryService:
    @pytest.fixture
    async def _setup(self, database: Database, cleanup: AsyncExitStack) -> None:
        self.database = database
        self.cleanup = cleanup
        self.live = MessageStore(self.database)
        await self.live.initialize()

        async def embed(texts: list[str]) -> list[list[float]]:
            return [[1.0] + [0.0] * 511 for _ in texts]

        self.embed = AsyncMock(side_effect=embed)
        self.service = HybridChatMemory(
            self.live,
            EmbeddingCache(self.database, self.embed),
            owner_user_id=42,
        )
        self.cleanup.push_async_callback(self.service.close)

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
        assert len(result["hits"]) == 1
        assert result["hits"][0]["sources"][0]["chat_id"] == -1001
        private = self.service.recall(MemoryAccess(42, True))
        all_sources = await private.search("Project")
        assert len(all_sources["hits"]) == 3
        foreign = next(
            hit for hit in all_sources["hits"] if hit["sources"][0]["chat_id"] == 42
        )
        read = await group.search(memory_id=foreign["memory_id"])
        assert read["messages"] == []

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
        assert result["hits"][0]["sources"][0]["text"] == "Uses current tool"
        assert result["coverage"]["pending_blocks"] == 1
        assert "obsolete" not in str(result)
        assert len(await self.live.pending()) == 1
        await self.service.refresh_once()
        refreshed = await self.service.recall(MemoryAccess(-1001, False)).search(
            "current tool"
        )
        assert refreshed["coverage"]["pending_blocks"] == 0
        assert "obsolete" not in str(refreshed)

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
        assert [s["message_id"] for hit in result["hits"] for s in hit["sources"]] == [1]

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
        assert not closing.done()
        release.set()
        result = await asyncio.wait_for(search, timeout=2)
        assert result["hits"][0]["sources"][0]["text"] == "Aurora source"
        await asyncio.wait_for(closing, timeout=2)
        with pytest.raises(RuntimeError, match="closed"):
            await self.service.recall(MemoryAccess(-1001, False)).search("Aurora")

    async def test_startup_outage_preserves_history_and_pending_work(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        await self.live.record(self.message(-1001, "Persisted original"))
        self.embed.side_effect = RuntimeError("embedding unavailable")
        with pytest.raises(RuntimeError):
            await self.service.refresh_once()
        history = self.service.history(MemoryAccess(-1001, False))
        assert (await history.query("count"))["count"] == 1
        assert len(await self.live.pending()) == 1
        caplog.clear()
        with caplog.at_level("WARNING", logger="dashulya.memory.service"):
            await self.service.start()
        assert any(
            record.name == "dashulya.memory.service"
            or record.name.startswith("dashulya.memory.service.")
            for record in caplog.records
        )
        result = await self.service.history(MemoryAccess(-1001, False)).query("count")
        assert result["count"] == 1
        assert len(await self.live.pending()) == 1
        await self.service.close()

    async def test_initial_and_background_outages_retry_without_losing_work(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        await self.live.record(self.message(-1001, "Imported original awaiting vectors"))
        await self.live.indexed(await self.live.pending())
        assert await self.live.pending() == []
        tick = asyncio.Event()
        sleeping: asyncio.Queue[None] = asyncio.Queue()
        attempts: list[list[str]] = []
        real_sleep = asyncio.sleep

        async def embed(texts: list[str]) -> list[list[float]]:
            attempts.append(texts)
            if len(attempts) in {1, 3}:
                error_message = "PRIVATE PROVIDER BODY"
                raise RuntimeError(error_message)
            return [[1.0] + [0.0] * 511 for _ in texts]

        async def sleep(delay: float) -> None:
            if delay > 0:
                sleeping.put_nowait(None)
                await tick.wait()
                tick.clear()
            await real_sleep(0)

        self.embed.side_effect = embed
        caplog.clear()
        with (
            patch("dashulya.memory.service.asyncio.sleep", side_effect=sleep),
            caplog.at_level("WARNING", logger="dashulya.memory.service"),
        ):
            try:
                await self.service.start()
                history = self.service.history(MemoryAccess(-1001, False))
                assert (await history.query("count"))["count"] == 1
                assert len(attempts) == 1
                await asyncio.wait_for(sleeping.get(), timeout=2)
                tick.set()
                await asyncio.wait_for(sleeping.get(), timeout=2)
                assert len(attempts) == 2
                assert attempts[1] == attempts[0]
                await self.live.record(self.message(-1001, "Later original", number=2))
                tick.set()
                await asyncio.wait_for(sleeping.get(), timeout=2)
                assert len(attempts) == 3
                assert (await history.query("count"))["count"] == 2
                assert len(await self.live.pending()) == 1
                for _ in range(3):
                    tick.set()
                    await asyncio.wait_for(sleeping.get(), timeout=2)
                    if not await self.live.pending():
                        break
                else:
                    pytest.fail("Background indexing did not recover after the outage")
                assert len(attempts) == 4
                assert attempts[3] == attempts[2]
                assert "PRIVATE PROVIDER BODY" not in str(caplog.messages)
            finally:
                await self.service.close()
        assert any(
            record.name == "dashulya.memory.service"
            or record.name.startswith("dashulya.memory.service.")
            for record in caplog.records
        )

    async def test_private_chat_selection_applies_before_engine_creation_and_cap(
        self,
    ) -> None:
        for chat, text in (
            (-1001, "needle one"),
            (-1002, "needle two"),
            (42, "needle private"),
        ):
            await self.live.record(self.message(chat, text))
        await self.service.refresh_once()
        private = self.service.recall(MemoryAccess(42, True))
        with patch.object(self.service, "engine", wraps=self.service.engine) as engine:
            result = await private.search("needle", chat="telegram:-1002", limit=1)
        assert {m.collection for m in engine.call_args.args[0]} == {"telegram:-1002"}
        assert len(result["hits"]) == 1
        assert result["hits"][0]["sources"][0]["chat_id"] == -1002
        coverage = result["coverage"]
        assert coverage["available_messages"] == 1
        assert coverage["first_saved_at"] == NOW.isoformat()
        assert coverage["last_saved_at"] == NOW.isoformat()
        assert {p["chat"] for p in coverage["known_periods"]} == {"telegram:-1002"}
        with pytest.raises(ValueError, match="outside"):
            await self.service.recall(MemoryAccess(-1001, False)).search(
                "needle", chat="telegram:-1002"
            )
        with pytest.raises(ValueError, match="Unknown saved chat"):
            await private.search("needle", chat="telegram:-100999")

    async def test_direct_id_honors_narrowed_chat_and_history_coverage_matches_search(
        self,
    ) -> None:
        await self.live.record(self.message(-1001, "needle first"))
        await self.live.record(self.message(-1002, "needle second"))
        await self.service.refresh_once()
        access = MemoryAccess(42, True)
        history = self.service.history(access)
        source = (await history.query(chat="telegram:-1001"))["messages"][0]
        recall = self.service.recall(access)
        wrong = await recall.search(memory_id=source["source_id"], chat="telegram:-1002")
        assert wrong["messages"] == []
        selected = await recall.search("needle", chat="telegram:-1001")
        canonical = (await history.query(chat="telegram:-1001"))["coverage"]
        assert {key: selected["coverage"][key] for key in canonical} == canonical

    async def test_source_expansion_preserves_inclusive_recall_upper_bound(
        self, subtests: pytest.Subtests
    ) -> None:
        await self.live.record(self.message(-1001, "needle boundary"))
        await self.live.record(
            replace(
                self.message(-1001, "needle later", number=2),
                sent_at=NOW + timedelta(microseconds=1),
            )
        )
        await self.service.refresh_once()
        access = MemoryAccess(-1001, False)
        history = self.service.history(access)
        recall = self.service.recall(access)
        result = await recall.search("needle", before=NOW.isoformat())
        assert len(result["hits"]) == 1
        source_id = result["hits"][0]["memory_id"]
        for upper in (NOW.isoformat(), "2026-10-25T03:00:00+03:00"):
            with subtests.test(before=upper):
                expanded = await recall.search(memory_id=source_id, before=upper)
                assert [m["message_id"] for m in expanded["messages"]] == [1]
        later = (await history.query())["messages"][-1]["source_id"]
        assert (await recall.search(memory_id=later, before=NOW.isoformat()))[
            "messages"
        ] == []
        # The separate canonical-history API retains its exclusive before bound.
        assert (await history.query("count", before=NOW.isoformat()))["count"] == 0
        with pytest.raises(ValueError, match="after must not be later"):
            await recall.search(
                memory_id=source_id,
                after=(NOW + timedelta(microseconds=1)).isoformat(),
                before=NOW.isoformat(),
            )

    async def test_source_expansion_accepts_maximum_upper_bound_without_widening_scope(
        self,
    ) -> None:
        await self.live.record(self.message(-1001, "needle boundary"))
        await self.live.record(
            replace(
                self.message(-1001, "needle future", number=2),
                sent_at=NOW + timedelta(microseconds=1),
            )
        )
        history = self.service.history(MemoryAccess(-1001, False))
        sources = (await history.query())["messages"]
        cutoff = MemoryAccess(-1001, False, before=NOW, before_id=2)
        recall = self.service.recall(cutoff)
        maximum = datetime.max.replace(tzinfo=UTC).isoformat()
        included = await recall.search(memory_id=sources[0]["source_id"], before=maximum)
        assert [m["message_id"] for m in included["messages"]] == [1]
        excluded = await recall.search(memory_id=sources[1]["source_id"], before=maximum)
        assert excluded["messages"] == []

    async def test_source_expansion_rejects_naive_and_unrepresentable_utc_bounds(
        self,
        subtests: pytest.Subtests,
    ) -> None:
        await self.live.record(self.message(-1001, "needle source"))
        access = MemoryAccess(-1001, False)
        original = (await self.service.history(access).query())["messages"][0]
        source_id = original["source_id"]
        recall = self.service.recall(access)
        for upper in (
            NOW.replace(tzinfo=None).isoformat(),
            "9999-12-31T23:59:59.999999-01:00",
        ):
            with (
                subtests.test(before=upper),
                pytest.raises(ValueError, match=r"timezone|supported UTC range"),
            ):
                await recall.search(memory_id=source_id, before=upper)
