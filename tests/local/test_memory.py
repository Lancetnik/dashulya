import asyncio
import hashlib
import json
import sqlite3
from contextlib import AsyncExitStack, closing
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from ag2 import Agent, MemoryStream
from ag2.events import ToolCallEvent, ToolResultEvent
from ag2.testing import TestConfig, TrackingConfig

from dashulya.domain.memory import MemoryAccess
from dashulya.domain.messages import Reply
from dashulya.main import create_container
from dashulya.main.config import Settings
from dashulya.memory.index import build_memory
from dashulya.memory.search import MemorySearch
from tests.support import TEST_API_KEY, MemoryOverride, ModelOverrides, reply_json


@pytest.mark.usefixtures("_setup")
class TestMemory:
    @pytest.fixture
    async def _setup(self, cleanup: AsyncExitStack) -> None:
        self.cleanup = cleanup
        self.directory = TemporaryDirectory()
        self.cleanup.callback(self.directory.cleanup)
        self.source = Path(self.directory.name) / "source.sqlite3"
        self.index = Path(self.directory.name) / "index.sqlite3"
        with closing(sqlite3.connect(self.source)) as connection:
            connection.execute("""
                CREATE TABLE group_messages (
                    id INTEGER PRIMARY KEY, date TEXT, text TEXT, user_id INTEGER,
                    author TEXT, reply_to_msg_id INTEGER, thread_post_id INTEGER
                )
            """)
            connection.executemany(
                "INSERT INTO group_messages VALUES (?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        1,
                        "2026-01-01T10:00:00+00:00",
                        "Consider Redis",
                        10,
                        "Alice",
                        None,
                        None,
                    ),
                    (
                        2,
                        "2026-01-01T10:01:00+00:00",
                        "Choose PostgreSQL for durability",
                        20,
                        "Bob",
                        1,
                        None,
                    ),
                    (
                        3,
                        "2026-01-01T10:02:00+00:00",
                        "Yes, this one",
                        10,
                        "Alice",
                        2,
                        None,
                    ),
                    (
                        4,
                        "2026-01-02T12:00:00+00:00",
                        "Unrecorded zebra detail",
                        30,
                        "Carol",
                        None,
                        None,
                    ),
                    (
                        5,
                        "2026-01-03T12:00:00+00:00",
                        "padding " * 1000 + "hiddenneedle",
                        30,
                        "Carol",
                        None,
                        None,
                    ),
                    (6, "2026-01-04T10:00:00+00:00", "alpha", 10, "Alice", None, 100),
                    (7, "2026-01-04T10:01:00+00:00", "beta", 20, "Bob", 6, 100),
                ],
            )
            connection.commit()
        self.original_bytes = self.source.read_bytes()
        self.coverage = build_memory(self.source, self.index, chat_username="test_chat")
        self.memory = await MemorySearch.open(self.index)

    async def test_locked_index_does_not_block_other_tasks(self) -> None:
        with closing(sqlite3.connect(self.index)) as writer:
            writer.execute("BEGIN EXCLUSIVE")
            searching = asyncio.create_task(self.memory.search("Redis"))
            try:
                await asyncio.sleep(0.02)
                assert not searching.done()
            finally:
                writer.rollback()
                result = await asyncio.wait_for(searching, timeout=2)
        assert result["hits"]

    async def test_search_matches_derived_wording_and_descends_to_original_messages(
        self,
    ) -> None:
        key = self.memory.remember("Persistent storage decision", [2])
        result = await self.memory.search("persistent storage")
        hit = result["hits"][0]
        assert hit["memory_id"] == key
        assert hit["kind"] == "summary"
        assert hit["sources"][0]["message_id"] == 2
        assert hit["sources"][0]["text"] == "Choose PostgreSQL for durability"
        assert hit["sources"][0]["url"] == "https://t.me/test_chat/2"
        assert (await self.memory.search(memory_id=key))["hits"][0]["memory_id"] == key
        original = (await self.memory.search("zebr"))["hits"][0]
        assert (original["kind"], original["sources"][0]["message_id"]) == ("message", 4)

    async def test_discussion_search_can_match_terms_in_different_messages(self) -> None:
        hit = (await self.memory.search("alpha beta"))["hits"][0]
        assert hit["kind"] == "discussion"
        assert [source["message_id"] for source in hit["sources"]] == [6, 7]

    async def test_reply_context_recovers_parent_and_grandparent(self) -> None:
        hit = (await self.memory.search(memory_id="message:3"))["hits"][0]
        assert hit["sources"][0]["text"] == "Yes, this one"
        assert {row["message_id"] for row in hit["context"]} == {1, 2}
        discussion = (await self.memory.search(memory_id=hit["related_memory_ids"][0]))[
            "hits"
        ][0]
        assert [row["message_id"] for row in discussion["sources"]] == [1, 2, 3]

    async def test_ranked_hits_collapse_repeated_records_from_one_conversation(
        self,
    ) -> None:
        self.memory.remember("Redis database overview", [1, 2])
        hits = (await self.memory.search("Redis"))["hits"]
        assert len(hits) == 1

    async def test_author_and_date_filters_do_not_misattribute_a_mixed_summary(
        self,
    ) -> None:
        self.memory.remember("Operational decision", [1, 2])
        assert (await self.memory.search("operational", author_id=10))["hits"] == []
        assert (await self.memory.search("operational", after="2026-01-01T10:00:30Z"))[
            "hits"
        ] == []
        assert (await self.memory.search("Redis", author_id=20))["hits"] == []
        assert (await self.memory.search("Redis", author_id=10))["hits"][0]["sources"][0][
            "user_id"
        ] == 10
        assert (await self.memory.search("zebra", before="2026-01-02T14:59:59+03:00"))[
            "hits"
        ] == []
        assert (await self.memory.search("zebra", after="2026-01-02T15:00:00+03:00"))[
            "hits"
        ]

    async def test_unknown_or_filtered_memory_id_returns_no_evidence(self) -> None:
        assert (await self.memory.search(memory_id="message:999"))["hits"] == []
        assert (await self.memory.search(memory_id="message:1", author_id=20))[
            "hits"
        ] == []

    async def test_long_message_excerpt_keeps_match_and_reports_truncation(self) -> None:
        source = (await self.memory.search("hiddenneedle"))["hits"][0]["sources"][0]
        assert "hiddenneedle" in source["text"]
        assert source["truncated"]
        assert source["excerpt_start"] > 0
        assert len(source["text"]) <= 1200

    async def test_memories_require_real_sources_and_persist_across_reopening(
        self,
    ) -> None:
        with pytest.raises(ValueError, match="Every source"):
            self.memory.remember("Unsupported claim", [999])
        with pytest.raises(ValueError, match="1 to 8 source"):
            self.memory.remember("Unsupported claim", [])
        key = self.memory.remember("Verified choice", [2], kind="fact")
        assert (await (await MemorySearch.open(self.index)).search("verified"))["hits"][
            0
        ]["memory_id"] == key
        assert (await self.memory.search("unsupported"))["hits"] == []

    async def test_query_input_is_bounded_and_not_interpreted_as_fts_or_sql_syntax(
        self,
        subtests: pytest.Subtests,
    ) -> None:
        assert (await self.memory.search('"Redis" OR zebra --'))["hits"] == []
        for arguments in (
            {"query": ""},
            {"query": "()"},
            {"query": "x", "memory_id": "message:1"},
            {"query": "x", "limit": 100},
            {"query": "x", "after": "2026-01-01"},
            {
                "query": "x",
                "after": "2026-02-01T00:00:00Z",
                "before": "2026-01-01T00:00:00Z",
            },
        ):
            with (
                subtests.test(arguments=arguments),
                pytest.raises(ValueError, match=r"query|keywords|limit|timezone|after"),
            ):
                (await self.memory.search(**arguments))

    async def test_build_preserves_snapshot_and_refuses_to_overwrite_memory(self) -> None:
        self.memory.remember("Retained record", [1])
        with pytest.raises(FileExistsError):
            build_memory(self.source, self.index, chat_username="test_chat")
        assert self.source.read_bytes() == self.original_bytes
        assert (
            self.coverage["source_sha256"]
            == hashlib.sha256(self.original_bytes).hexdigest()
        )
        assert self.coverage["message_count"] == 7
        assert (await self.memory.search("retained"))["hits"]

    async def test_legacy_offline_index_explicitly_rejects_chat_selection(self) -> None:
        with pytest.raises(ValueError, match="does not support chat selection"):
            await self.memory.search("PostgreSQL", chat="telegram:-100123")

    async def test_real_ag2_tool_roundtrip_reads_local_evidence_without_model_or_network(
        self,
    ) -> None:
        config = TrackingConfig(
            TestConfig(
                ToolCallEvent(
                    "search_memory",
                    arguments=json.dumps({"query": "PostgreSQL", "limit": 3}),
                ),
                reply_json("The choice was PostgreSQL: https://t.me/test_chat/2"),
            )
        )
        container = create_container(
            Settings(bot_token="123:TEST", openai_api_key=TEST_API_KEY, owner_user_id=42),
            MemoryOverride(self.memory),
            ModelOverrides(reply=config, decision=TestConfig("false")),
        )
        self.cleanup.push_async_callback(container.close)
        agent = await container.get(Agent[Reply])
        stream = MemoryStream()
        reply = await (
            await agent.ask(
                "What database did we choose?",
                stream=stream,
                dependencies={"memory_access": MemoryAccess(42, True)},
            )
        ).content()
        assert reply is not None
        assert "PostgreSQL" in reply.text
        events = await stream.history.get_events()
        results = [event for event in events if isinstance(event, ToolResultEvent)]
        assert len(results) == 1
        assert "https://t.me/test_chat/2" in str(results[0].result)
        assert results[0].name == "search_memory"
        assert len(config.calls[0].tools) == 3
