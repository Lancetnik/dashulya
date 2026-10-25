import hashlib
import json
import sqlite3
import unittest
from contextlib import closing
from pathlib import Path
from tempfile import TemporaryDirectory

from ag2 import Agent, MemoryStream
from ag2.events import ToolCallEvent, ToolResultEvent
from ag2.testing import TestConfig, TrackingConfig

from dashulya.ai.tools.memory import create_memory_tool
from dashulya.memory.index import build_memory
from dashulya.memory.search import MemorySearch


class MemoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
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
        self.memory = MemorySearch(self.index)

    def test_search_matches_derived_wording_and_descends_to_original_messages(
        self,
    ) -> None:
        key = self.memory.remember("Persistent storage decision", [2])
        result = self.memory.search("persistent storage")
        hit = result["hits"][0]
        self.assertEqual(hit["memory_id"], key)
        self.assertEqual(hit["kind"], "summary")
        self.assertEqual(hit["sources"][0]["message_id"], 2)
        self.assertEqual(hit["sources"][0]["text"], "Choose PostgreSQL for durability")
        self.assertEqual(hit["sources"][0]["url"], "https://t.me/test_chat/2")
        self.assertEqual(self.memory.search(memory_id=key)["hits"][0]["memory_id"], key)

    def test_original_messages_remain_searchable_without_derived_records(self) -> None:
        self.memory.remember("Persistent storage decision", [2])
        result = self.memory.search("zebr")
        self.assertEqual(result["hits"][0]["kind"], "message")
        self.assertEqual(result["hits"][0]["sources"][0]["message_id"], 4)

    def test_discussion_search_can_match_terms_in_different_messages(self) -> None:
        hit = self.memory.search("alpha beta")["hits"][0]
        self.assertEqual(hit["kind"], "discussion")
        self.assertEqual([source["message_id"] for source in hit["sources"]], [6, 7])

    def test_reply_context_recovers_parent_and_grandparent(self) -> None:
        hit = self.memory.search(memory_id="message:3")["hits"][0]
        self.assertEqual(hit["sources"][0]["text"], "Yes, this one")
        self.assertEqual({row["message_id"] for row in hit["context"]}, {1, 2})
        discussion = self.memory.search(memory_id=hit["related_memory_ids"][0])["hits"][0]
        self.assertEqual([row["message_id"] for row in discussion["sources"]], [1, 2, 3])

    def test_ranked_hits_collapse_repeated_records_from_one_conversation(self) -> None:
        self.memory.remember("Redis database overview", [1, 2])
        hits = self.memory.search("Redis")["hits"]
        self.assertEqual(len(hits), 1)

    def test_author_and_date_filters_do_not_misattribute_a_mixed_summary(self) -> None:
        self.memory.remember("Operational decision", [1, 2])
        self.assertEqual(self.memory.search("operational", author_id=10)["hits"], [])
        self.assertEqual(
            self.memory.search("operational", after="2026-01-01T10:00:30Z")["hits"], []
        )
        self.assertEqual(self.memory.search("Redis", author_id=20)["hits"], [])
        self.assertEqual(
            self.memory.search("Redis", author_id=10)["hits"][0]["sources"][0]["user_id"],
            10,
        )
        self.assertEqual(
            self.memory.search("zebra", before="2026-01-02T14:59:59+03:00")["hits"], []
        )
        self.assertTrue(
            self.memory.search("zebra", after="2026-01-02T15:00:00+03:00")["hits"]
        )

    def test_unknown_or_filtered_memory_id_returns_no_evidence(self) -> None:
        self.assertEqual(self.memory.search(memory_id="message:999")["hits"], [])
        self.assertEqual(
            self.memory.search(memory_id="message:1", author_id=20)["hits"], []
        )

    def test_long_message_excerpt_keeps_match_and_reports_truncation(self) -> None:
        source = self.memory.search("hiddenneedle")["hits"][0]["sources"][0]
        self.assertIn("hiddenneedle", source["text"])
        self.assertTrue(source["truncated"])
        self.assertGreater(source["excerpt_start"], 0)
        self.assertLessEqual(len(source["text"]), 1200)

    def test_memories_require_real_sources_and_persist_across_reopening(self) -> None:
        with self.assertRaisesRegex(ValueError, "Every source"):
            self.memory.remember("Unsupported claim", [999])
        with self.assertRaises(ValueError):
            self.memory.remember("Unsupported claim", [])
        key = self.memory.remember("Verified choice", [2], kind="fact")
        self.assertEqual(
            MemorySearch(self.index).search("verified")["hits"][0]["memory_id"], key
        )
        self.assertEqual(self.memory.search("unsupported")["hits"], [])

    def test_query_input_is_bounded_and_not_interpreted_as_fts_or_sql_syntax(
        self,
    ) -> None:
        self.assertEqual(self.memory.search('"Redis" OR zebra --')["hits"], [])
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
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                self.memory.search(**arguments)

    def test_build_preserves_snapshot_and_refuses_to_overwrite_memory(self) -> None:
        self.memory.remember("Retained record", [1])
        with self.assertRaises(FileExistsError):
            build_memory(self.source, self.index, chat_username="test_chat")
        self.assertEqual(self.source.read_bytes(), self.original_bytes)
        self.assertEqual(
            self.coverage["source_sha256"],
            hashlib.sha256(self.original_bytes).hexdigest(),
        )
        self.assertEqual(self.coverage["message_count"], 7)
        self.assertTrue(self.memory.search("retained")["hits"])


class MemoryToolTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_ag2_tool_roundtrip_reads_local_evidence_without_model_or_network(
        self,
    ) -> None:
        fixture = MemoryTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        config = TrackingConfig(
            TestConfig(
                ToolCallEvent(
                    "search_memory", arguments=json.dumps({"query": "PostgreSQL"})
                ),
                "The choice was PostgreSQL: https://t.me/test_chat/2",
            )
        )
        agent = Agent("test", config=config, tools=[create_memory_tool(fixture.memory)])
        stream = MemoryStream()
        reply = await agent.ask("What database did we choose?", stream=stream)
        assert reply.body is not None
        self.assertIn("PostgreSQL", reply.body)
        events = await stream.history.get_events()
        results = [event for event in events if isinstance(event, ToolResultEvent)]
        self.assertEqual(len(results), 1)
        self.assertIn("https://t.me/test_chat/2", str(results[0].result))
        self.assertEqual(results[0].name, "search_memory")
        self.assertEqual(len(config.calls[0].tools), 1)
