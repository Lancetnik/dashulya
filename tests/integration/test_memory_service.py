"""Scoped runtime retrieval through the same service the tools consume."""

import asyncio
import json
from contextlib import AsyncExitStack
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, patch

import pytest
from psycopg.conninfo import conninfo_to_dict, make_conninfo
from psycopg.errors import FeatureNotSupported

from dashulya.database import Database
from dashulya.domain.memory import MemoryAccess, MemoryMessage, OriginalMessage
from dashulya.memory._selection import RecallSelection
from dashulya.memory.service import HybridChatMemory
from dashulya.memory.store import MessageStore
from dashulya.memory.vectors import EmbeddingCache
from dashulya.observability import JsonLogFormatter

NOW = datetime(2026, 10, 25, tzinfo=UTC)


@pytest.mark.usefixtures("_setup")
class TestMemoryService:
    async def test_dense_recall_remains_available_without_lexical_query_terms(
        self,
    ) -> None:
        await self.live.record(self.message(-1001, "Aurora semantic evidence"))
        await self.service.refresh_once()
        result = await self.service.recall(MemoryAccess(-1001, False)).search("what")
        assert [hit["sources"][0]["text"] for hit in result["hits"]] == [
            "Aurora semantic evidence"
        ]

    async def test_search_remains_usable_with_generic_prepared_plans(
        self, database_url: str
    ) -> None:
        options = str(conninfo_to_dict(database_url).get("options") or "")
        database = Database(
            make_conninfo(
                database_url, options=options + " -c plan_cache_mode=force_generic_plan"
            )
        )
        self.cleanup.push_async_callback(database.close)
        store = MessageStore(database)
        memory = HybridChatMemory(
            store, EmbeddingCache(database, self.embed), owner_user_id=42
        )
        self.cleanup.push_async_callback(memory.close)
        await store.record(self.message(-1001, "Aurora prepared search"))
        await memory.refresh_once()
        try:
            result = await memory.recall(MemoryAccess(-1001, False)).search("Aurora")
        except FeatureNotSupported:
            result = None
        assert result is not None, "Generic prepared plans broke native BM25 retrieval"
        assert result["hits"][0]["sources"][0]["text"] == "Aurora prepared search"

    @pytest.mark.parametrize("edited", [False, True])
    async def test_search_loads_only_bounded_database_candidates(
        self, caplog: pytest.LogCaptureFixture, edited: bool
    ) -> None:
        for number in range(1, 401):
            await self.live.record(
                self.message(-1001, f"Aurora decision {number}", number=number)
            )
        await self.service.refresh_once()
        if edited:
            await self.live.record(
                replace(
                    self.message(-1001, "Aurora revised decision", number=201),
                    edited_at=NOW + timedelta(minutes=1),
                    update_id=2,
                )
            )
        with caplog.at_level("INFO", logger="dashulya.memory._timing"):
            result = await self.service.recall(MemoryAccess(-1001, False)).search(
                "Aurora"
            )
        assert result["hits"]
        assert result["coverage"]["source_count"] == 400
        record = next(r for r in caplog.records if r.message == "Memory search completed")
        assert vars(record)["search_metrics"]["loaded_originals"] <= 24, (
            "Recall loaded the corpus instead of bounded database candidates"
        )

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

    async def test_parallel_searches_share_snapshot_without_serializing_embeddings(
        self,
    ) -> None:
        await self.live.record(self.message(-1001, "Aurora source"))
        await self.service.refresh_once()
        all_started, release = asyncio.Event(), asyncio.Event()
        started = 0

        async def embedding(texts: list[str]) -> list[list[float]]:
            nonlocal started
            started += 1
            if started == 4:
                all_started.set()
            await release.wait()
            return [[1.0] + [0.0] * 511 for _ in texts]

        self.embed.side_effect = embedding
        recall = self.service.recall(MemoryAccess(-1001, False))
        searches = [
            asyncio.create_task(recall.search(f"Aurora query {i}")) for i in range(4)
        ]
        try:
            await asyncio.wait_for(all_started.wait(), timeout=2)
            searches[0].cancel()
            with pytest.raises(asyncio.CancelledError):
                await searches[0]
            await self.service.refresh_once()
            closing = asyncio.create_task(self.service.close())
            await asyncio.sleep(0)
            assert not closing.done()
        finally:
            release.set()
            results = await asyncio.gather(*searches, return_exceptions=True)
        assert all(isinstance(result, dict) for result in results[1:])
        await asyncio.wait_for(closing, timeout=2)

    async def test_search_filters_before_loading_text_and_keeps_separate_context(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        parent = self.message(-1001, "Parent proposal", number=1)
        selected = replace(
            self.message(-1001, "Aurora decision", number=2),
            author_id=20,
            sent_at=NOW + timedelta(minutes=1),
            reply_to_id=1,
        )
        for message in (
            parent,
            selected,
            replace(
                self.message(-1001, "Nearby explanation", number=3),
                sent_at=NOW + timedelta(minutes=2),
            ),
            replace(
                self.message(-1001, "Unrelated old message", number=4),
                sent_at=NOW - timedelta(days=30),
            ),
            self.message(-1002, "Foreign secret"),
            replace(
                self.message(-1001, "Later edit", number=5),
                edited_at=NOW + timedelta(hours=2),
            ),
        ):
            await self.live.record(message)
        await self.service.refresh_once()
        access = MemoryAccess(-1001, False, before=NOW + timedelta(hours=1))
        with (
            patch.object(
                self.live, "originals", side_effect=AssertionError("Full history loaded")
            ),
            caplog.at_level("INFO", logger="dashulya.memory._timing"),
        ):
            result = await self.service.recall(access).search(
                "Aurora",
                author_id=20,
                after=selected.sent_at.isoformat(),
                before=selected.sent_at.isoformat(),
            )
        assert result["coverage"]["source_count"] == 1
        assert result["coverage"]["available_messages"] == 4
        assert [s["message_id"] for hit in result["hits"] for s in hit["sources"]] == [2]
        assert {m["message_id"] for m in result["hits"][0]["context"]} == {1, 3}
        assert "Foreign secret" not in str(result)
        assert "Later edit" not in str(result)
        record = next(r for r in caplog.records if r.message == "Memory search completed")
        logged = json.loads(JsonLogFormatter().format(record))
        assert logged["outcome"] == "ok"
        assert logged["search_metrics"]["candidate_count"] == 1
        assert logged["search_metrics"]["loaded_originals"] == 2
        assert {
            "selection_ms",
            "lexical_search_ms",
            "query_embedding_ms",
            "context_ms",
        } <= logged["search_metrics"].keys()
        assert "Parent proposal" not in str(logged)

    async def test_filtered_blocks_reuse_saved_vectors_and_preserve_reply_context(
        self,
    ) -> None:
        for number in range(1, 10):
            await self.live.record(
                replace(
                    self.message(-1001, f"Aurora source {number}", number=number),
                    author_id=20 if number == 8 else 42,
                    sent_at=NOW + timedelta(minutes=number),
                    reply_to_id=number - 1 if number > 1 else None,
                )
            )
        await self.service.refresh_once()
        access = MemoryAccess(-1001, False, before=NOW + timedelta(minutes=9))
        history = self.service.history(access)
        originals = await history.originals()
        target = next(m for m in originals if m.message_id == 8)
        expected = [m.render(max_chars=3000) for m in history.related(target, originals)]
        self.embed.reset_mock()
        result = await self.service.recall(access).search("Aurora", author_id=20)
        assert result["coverage"]["source_count"] == 1
        assert result["coverage"]["embedded_blocks"] == 1
        assert result["coverage"]["pending_blocks"] == 0
        assert result["hits"][0]["context"] == expected
        assert [s["message_id"] for s in result["hits"][0]["sources"]] == [8]
        self.embed.assert_awaited_once_with(["Aurora"])

    async def test_parallel_lexical_searches_prepare_missing_postings(self) -> None:
        await self.live.record(self.message(-1001, "Aurora durable storage"))
        await self.live.record(
            replace(self.message(-1001, "Aurora delivery", number=2), author_id=20)
        )
        recall = self.service.recall(MemoryAccess(-1001, False))
        results = await asyncio.wait_for(
            asyncio.gather(
                recall.search("Aurora"),
                recall.search("Aurora", author_id=20),
                recall.search("Aurora", author_id=42),
                recall.search("storage"),
            ),
            timeout=2,
        )
        assert [len(result["hits"]) for result in results] == [2, 1, 1, 1]
        assert [s["message_id"] for s in results[1]["hits"][0]["sources"]] == [2]
        self.embed.assert_not_awaited()

    async def test_native_snapshots_exclude_late_parent_edits_and_deleted_parents(
        self,
    ) -> None:
        parent = self.message(-1001, "oldparentword", number=1)
        child = replace(
            self.message(-1001, "acknowledged", number=2),
            sent_at=NOW + timedelta(minutes=1),
            reply_to_id=1,
        )
        await self.live.record(parent)
        await self.live.record(child)
        current = self.service.recall(MemoryAccess(-1001, False))
        assert (await current.search("oldparentword"))["hits"]
        await self.live.record(
            replace(
                parent,
                text="futureparentword",
                edited_at=NOW + timedelta(hours=2),
                update_id=2,
            )
        )
        assert (await current.search("futureparentword"))["hits"]
        historical = self.service.recall(
            MemoryAccess(-1001, False, before=NOW + timedelta(hours=1))
        )
        assert (await historical.search("futureparentword"))["hits"] == []
        assert (await historical.search("oldparentword"))["hits"] == []
        selected = await historical.search("acknowledged")
        assert [s["message_id"] for h in selected["hits"] for s in h["sources"]] == [2]
        assert "futureparentword" not in str(selected)
        async with self.database.connection() as connection:
            await connection.execute(
                "DELETE FROM messages WHERE collection='telegram:-1001' AND message_id=1"
            )
        assert (await current.search("futureparentword"))["hits"] == []
        assert (await current.search("acknowledged"))["hits"]
        self.embed.assert_not_awaited()

    async def test_shutdown_waits_for_supporting_context_loading(self) -> None:
        await self.live.record(self.message(-1001, "Aurora source"))
        await self.service.refresh_once()
        started, release = asyncio.Event(), asyncio.Event()
        real_context = RecallSelection.context

        async def context(
            selection: RecallSelection,
            targets: list[OriginalMessage],
            *,
            chat: str | None,
        ) -> list[OriginalMessage]:
            started.set()
            await release.wait()
            return await real_context(selection, targets, chat=chat)

        with patch.object(RecallSelection, "context", new=context):
            search = asyncio.create_task(
                self.service.recall(MemoryAccess(-1001, False)).search("Aurora")
            )
            try:
                await asyncio.wait_for(started.wait(), timeout=2)
                closing = asyncio.create_task(self.service.close())
                await asyncio.sleep(0)
                await asyncio.sleep(0)
                assert not closing.done()
            finally:
                release.set()
                await asyncio.wait_for(search, timeout=2)
                await asyncio.wait_for(closing, timeout=2)

    async def test_shutdown_drains_recall_across_refresh_and_rejects_new_searches(
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
        try:
            await asyncio.wait_for(started.wait(), timeout=2)
            # Preparing cached originals invalidates engines without disrupting
            # an active search or allowing shutdown to forget its lease.
            await asyncio.wait_for(self.service.refresh_once(), timeout=2)
            closing = asyncio.create_task(self.service.close())
            await asyncio.sleep(0)
            assert not closing.done()
        finally:
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
        with caplog.at_level("WARNING", logger="dashulya.memory"):
            await self.service.start()
        assert any(
            record.name == "dashulya.memory" or record.name.startswith("dashulya.memory.")
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
            patch("dashulya.memory._indexing._sleep", side_effect=sleep),
            caplog.at_level("WARNING", logger="dashulya.memory"),
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
            record.name == "dashulya.memory" or record.name.startswith("dashulya.memory.")
            for record in caplog.records
        )

    async def test_private_chat_selection_bounds_recall_and_coverage(
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
        result = await private.search("needle", chat="telegram:-1002", limit=1)
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
