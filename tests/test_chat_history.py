import hashlib
import sqlite3
import unittest
from contextlib import closing
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from unittest.mock import patch

from dashulya.domain.memory import MemoryAccess, MemoryMessage
from dashulya.memory.history import CanonicalHistory
from dashulya.memory.live import LiveMemory
from dashulya.memory.search import MemorySearch

NOW = datetime(2026, 10, 25, 12, tzinfo=UTC)
GROUP = -100123
OTHER = -100456


class CanonicalHistoryTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name)
        self.live = LiveMemory(self.path / "live.sqlite3")
        await self.live.initialize()
        self.group = CanonicalHistory(self.live, MemoryAccess(GROUP, is_private=False))
        self.private = CanonicalHistory(self.live, MemoryAccess(42, is_private=True))

    @staticmethod
    def message(mid: int, **changes: Any) -> MemoryMessage:
        return replace(
            MemoryMessage(
                GROUP,
                mid,
                f"Original {mid}",
                NOW,
                10,
                "Alex",
                "Group",
                chat_username="public_group",
                update_id=mid,
            ),
            **changes,
        )

    def imported(self, *, dated: bool = True) -> MemorySearch:
        path = self.path / "import.sqlite3"
        with closing(sqlite3.connect(path)) as connection:
            connection.executescript("""
                CREATE TABLE messages (
                    id INTEGER PRIMARY KEY, sent_at TEXT, user_id INTEGER, author TEXT,
                    body TEXT, reply_to_id INTEGER, conversation TEXT, url TEXT
                );
                CREATE TABLE private_knowledge (body TEXT);
                INSERT INTO private_knowledge VALUES ('SECRET_DERIVED_MEMORY');
            """)
            connection.executemany(
                "INSERT INTO messages VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        mid,
                        NOW.isoformat() if dated else None,
                        10,
                        "Archive Alex",
                        f"Archive original {mid}",
                        None,
                        "post:999",
                        f"https://t.me/public_group/{mid}",
                    )
                    for mid in (1, 2)
                ],
            )
            connection.commit()
        return MemorySearch(
            path,
            {"version": 1, "source_sha256": "a" * 64, "chat_username": "public_group"},
        )

    async def test_group_isolation_private_federation_and_composite_identity(
        self,
    ) -> None:
        for msg in (
            self.message(1, text="Group original"),
            self.message(1, chat_id=OTHER, text="Other group secret"),
            self.message(1, chat_id=42, text="Private source", chat_username="owner"),
        ):
            await self.live.record(msg)
        group = await self.group.query()
        self.assertEqual([m["text"] for m in group["messages"]], ["Group original"])
        private = await self.private.query()
        self.assertEqual(len(private["messages"]), 3)
        self.assertEqual(len({m["source_id"] for m in private["messages"]}), 3)
        secret = next(m for m in private["messages"] if m["chat_id"] == 42)
        self.assertIsNone(secret["url"])
        self.assertEqual(
            (await self.group.query("read", source_id=secret["source_id"]))["messages"],
            [],
        )
        with self.assertRaisesRegex(ValueError, "outside"):
            await self.group.query(chat=f"telegram:{OTHER}")
        narrowed = await self.private.query(chat=f"telegram:{OTHER}")
        self.assertEqual(
            [m["text"] for m in narrowed["messages"]], ["Other group secret"]
        )

    async def test_topic_filter_narrows_primary_but_group_defaults_all_topics(
        self,
    ) -> None:
        await self.live.record(self.message(1, thread_id=7))
        await self.live.record(self.message(2, thread_id=8))
        await self.live.record(self.message(3))
        self.assertEqual((await self.group.query("count"))["count"], 3)
        page = await self.group.query(topic_id=7)
        self.assertEqual([m["message_id"] for m in page["messages"]], [1])
        self.assertEqual(page["messages"][0]["topic_id"], 7)

    async def test_exact_count_blank_unicode_casefold_and_sensitive_option(self) -> None:
        for mid in range(1, 114):
            await self.live.record(
                self.message(mid, text="" if mid == 7 else "Straße TEST")
            )
        self.assertEqual((await self.group.query("count"))["count"], 113)
        self.assertEqual(
            (await self.group.query("count", text_contains="STRASSE"))["count"], 112
        )
        self.assertEqual(
            (await self.group.query("count", text_contains="test"))["count"], 112
        )
        self.assertEqual(
            (await self.group.query("count", text_contains="test", case_sensitive=True))[
                "count"
            ],
            0,
        )
        self.assertEqual(
            (await self.group.query("count", text_contains="TEST", case_sensitive=True))[
                "count"
            ],
            112,
        )

    async def test_utc_after_inclusive_before_exclusive_and_unknown_archive_dates(
        self,
    ) -> None:
        await self.live.record(self.message(1, sent_at=NOW - timedelta(seconds=1)))
        await self.live.record(self.message(2, sent_at=NOW))
        await self.live.record(self.message(3, sent_at=NOW + timedelta(seconds=1)))
        result = await self.group.query(
            after="2026-10-25T15:00:00+03:00", before="2026-10-25T15:00:01+03:00"
        )
        self.assertEqual([m["message_id"] for m in result["messages"]], [2])
        self.assertEqual(result["messages"][0]["sent_at"], NOW.isoformat())
        imported = self.imported(dated=False)
        reader = CanonicalHistory(self.live, MemoryAccess(42, True), imported=imported)
        self.assertEqual((await reader.query("count"))["count"], 5)
        self.assertEqual((await reader.query("count", after=NOW.isoformat()))["count"], 2)
        with self.assertRaisesRegex(ValueError, "timezone"):
            await reader.query(after="2026-10-25T00:00:00")

    async def test_cursor_pages_full_set_without_dupes_and_cannot_change_scope(
        self,
    ) -> None:
        for mid in range(1, 114):
            await self.live.record(self.message(mid))
        cursor = None
        found = []
        while True:
            page = await self.group.query(cursor=cursor, limit=17)
            ids = [m["message_id"] for m in page["messages"]]
            self.assertEqual(ids, sorted(ids))
            found.extend(ids)
            cursor = page["next_cursor"]
            if cursor is None:
                break
        self.assertEqual(len(found), 113)
        self.assertEqual(set(found), set(range(1, 114)))
        first = await self.group.query(limit=17)
        await self.live.record(self.message(114))
        second = await self.group.query(cursor=first["next_cursor"], limit=17)
        self.assertTrue(
            {m["message_id"] for m in first["messages"]}.isdisjoint(
                m["message_id"] for m in second["messages"]
            )
        )
        with self.assertRaisesRegex(ValueError, "cursor"):
            await self.group.query(cursor=first["next_cursor"], text_contains="different")
        with self.assertRaisesRegex(ValueError, "cursor"):
            await self.private.query(cursor=first["next_cursor"])
        shifted = await self.group.query(offset=10, limit=3)
        self.assertEqual([m["message_id"] for m in shifted["messages"]], [102, 103, 104])

    async def test_live_edit_is_visible_next_query_before_indexing_and_shadows_snapshot(
        self,
    ) -> None:
        imported = self.imported()
        reader = CanonicalHistory(
            self.live, MemoryAccess(GROUP, False), imported=imported, import_chat_id=GROUP
        )
        original = (
            await reader.query(
                "read", source_id=(await reader.query())["messages"][0]["source_id"]
            )
        )["messages"][0]
        await self.live.record(
            self.message(
                1,
                text="Corrected canonical original",
                edited_at=NOW + timedelta(seconds=5),
            )
        )
        with patch.object(
            self.live, "pending", side_effect=RuntimeError("Knowledge unavailable")
        ) as pending:
            result = await reader.query("read", source_id=original["source_id"])
        pending.assert_not_called()
        self.assertEqual(result["messages"][0]["text"], "Corrected canonical original")
        self.assertEqual(result["messages"][0]["provenance"], "live")
        self.assertEqual((await reader.query("count"))["count"], 2)
        self.assertEqual(
            (await reader.query("count", text_contains="Archive original 1"))["count"], 0
        )
        self.assertEqual(
            (await reader.query("count", text_contains="SECRET_DERIVED_MEMORY"))["count"],
            0,
        )
        self.assertGreater(len(await self.live.pending()), 0)

    async def test_unknown_numeric_archive_is_private_only_and_binding_is_explicit(
        self,
    ) -> None:
        imported = self.imported()
        checksum = hashlib.sha256(imported.path.read_bytes()).hexdigest()
        await self.live.record(self.message(1, text="Live same ID"))
        group = CanonicalHistory(self.live, MemoryAccess(GROUP, False), imported=imported)
        self.assertEqual((await group.query("count"))["count"], 1)
        private = CanonicalHistory(self.live, MemoryAccess(42, True), imported=imported)
        sources = (await private.query())["messages"]
        self.assertEqual(len(sources), 3)
        archive = [m for m in sources if m["provenance"] == "imported"]
        self.assertTrue(all(m["chat_id"] is None for m in archive))
        self.assertTrue(all(m["chat"].startswith("archive:") for m in archive))
        self.assertTrue(all(m["topic_id"] is None for m in archive))
        self.assertEqual(len({m["source_id"] for m in sources}), 3)
        bound = CanonicalHistory(
            self.live, MemoryAccess(GROUP, False), imported=imported, import_chat_id=GROUP
        )
        self.assertEqual((await bound.query("count"))["count"], 2)
        foreign_bound = CanonicalHistory(
            self.live, MemoryAccess(GROUP, False), imported=imported, import_chat_id=OTHER
        )
        self.assertEqual((await foreign_bound.query("count"))["count"], 1)
        self.assertEqual(hashlib.sha256(imported.path.read_bytes()).hexdigest(), checksum)

    async def test_read_context_may_cross_query_filters_but_never_fixed_cutoff_or_chat(
        self,
    ) -> None:
        await self.live.record(
            self.message(
                1,
                text="Qualified original parent",
                author_id=11,
                sent_at=NOW - timedelta(days=1),
            )
        )
        await self.live.record(self.message(2, text="Yes", reply_to_id=1, author_id=10))
        await self.live.record(
            self.message(3, text="Future secret", sent_at=NOW + timedelta(seconds=1))
        )
        await self.live.record(
            self.message(1, chat_id=OTHER, text="Foreign parent secret")
        )
        access = MemoryAccess(GROUP, False, before=NOW, before_id=3)
        reader = CanonicalHistory(self.live, access)
        target = (
            await reader.query(text_contains="Yes", author_id=10, after=NOW.isoformat())
        )["messages"][0]
        result = await reader.query(
            "read",
            source_id=target["source_id"],
            text_contains="Yes",
            author_id=10,
            after=NOW.isoformat(),
        )
        self.assertEqual([m["message_id"] for m in result["context"]], [1])
        self.assertEqual(result["context"][0]["author_id"], 11)
        self.assertNotIn("Future secret", str(result))
        self.assertNotIn("Foreign parent secret", str(result))

    async def test_read_chunks_reassemble_full_original_and_context_is_bounded(
        self,
    ) -> None:
        body = "Original Unicode \u042f\n" * 1500
        await self.live.record(self.message(20, text=body, reply_to_id=19))
        for mid in range(1, 20):
            await self.live.record(self.message(mid, reply_to_id=mid - 1))
        for mid in range(21, 30):
            await self.live.record(self.message(mid))
        target = next(
            m for m in (await self.group.query())["messages"] if m["message_id"] == 20
        )
        offset = 0
        chunks = []
        while True:
            page = await self.group.query(
                "read", source_id=target["source_id"], offset=offset
            )
            record = page["messages"][0]
            chunks.append(record["text"])
            self.assertLessEqual(len(page["context"]), 9)
            self.assertLessEqual(len(record["text"]), 12000)
            offset = record["next_offset"]
            if offset is None:
                break
        self.assertEqual("".join(chunks), body)
        self.assertEqual(record["text_length"], len(body))

    async def test_authors_use_numeric_identity_chats_and_unknown_legacy_bot_metadata(
        self,
    ) -> None:
        await self.live.record(self.message(1, author_id=10))
        await self.live.record(self.message(2, author_id=11))
        await self.live.record(
            self.message(1, chat_id=OTHER, author_id=10, author="Alex renamed")
        )
        authors = (await self.private.query("authors"))["authors"]
        self.assertEqual(len(authors), 2)
        author = next(a for a in authors if a["author_id"] == 10)
        self.assertEqual(author["names"], ["Alex", "Alex renamed"])
        self.assertEqual(author["message_count"], 2)
        self.assertEqual((await self.group.query("chats"))["total"], 1)
        self.assertEqual((await self.private.query("chats"))["total"], 2)
        self.assertIsNone((await self.group.query())["messages"][0]["is_bot"])

    async def test_optional_sender_metadata_does_not_replace_chat_username(self) -> None:
        await self.live.record(self.message(1))
        async with self.live.connection() as connection:
            columns = {
                row[1]
                for row in await connection.execute_fetchall(
                    "PRAGMA table_info(live_messages)"
                )
            }
            if "username" not in columns:
                await connection.execute(
                    "ALTER TABLE live_messages ADD COLUMN username TEXT"
                )
            if "is_bot" not in columns:
                await connection.execute(
                    "ALTER TABLE live_messages ADD COLUMN is_bot INTEGER"
                )
            await connection.execute(
                "UPDATE live_messages SET username='agent_handle', is_bot=1"
            )
            await connection.commit()
        record = (await self.group.query())["messages"][0]
        self.assertEqual(record["username"], "agent_handle")
        self.assertTrue(record["is_bot"])
        self.assertEqual(record["chat_username"], "public_group")
        self.assertEqual(record["url"], "https://t.me/public_group/1")

    async def test_fixed_cutoff_rejects_naive_time_and_requires_timestamp_for_id(
        self,
    ) -> None:
        with self.assertRaisesRegex(ValueError, "timezone"):
            MemoryAccess(GROUP, False, before=NOW.replace(tzinfo=None))
        with self.assertRaisesRegex(ValueError, "timestamp"):
            MemoryAccess(GROUP, False, before_id=10)

    async def test_private_fixed_time_does_not_order_other_chats_by_request_id(
        self,
    ) -> None:
        await self.live.record(self.message(1, chat_id=42))
        await self.live.record(self.message(2, chat_id=42))
        await self.live.record(self.message(1, chat_id=GROUP))
        reader = CanonicalHistory(
            self.live, MemoryAccess(42, True, before=NOW, before_id=2)
        )
        records = (await reader.query())["messages"]
        self.assertEqual([(m["chat_id"], m["message_id"]) for m in records], [(42, 1)])

    async def test_later_edit_is_not_historical_evidence_and_old_import_stays_shadowed(
        self,
    ) -> None:
        imported = self.imported()
        await self.live.record(
            self.message(
                1, text="Future corrected fact", edited_at=NOW + timedelta(seconds=2)
            )
        )
        reader = CanonicalHistory(
            self.live,
            MemoryAccess(GROUP, False, before=NOW + timedelta(seconds=1)),
            imported=imported,
            import_chat_id=GROUP,
        )
        result = await reader.query()
        self.assertEqual([m["message_id"] for m in result["messages"]], [2])
        self.assertNotIn("Future corrected fact", str(result))
        self.assertEqual(
            (await reader.query("count", text_contains="Archive original 1"))["count"], 0
        )

    async def test_malformed_cursor_and_action_arguments_fail_without_scope_changes(
        self,
    ) -> None:
        for value in ("not-a-cursor", "a" * 3000):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "cursor"):
                await self.group.query(cursor=value)
        for arguments in (
            {"action": "read"},
            {"source_id": "source:unknown:1"},
            {"limit": 0},
            {"offset": -1},
            {"action": "unknown"},
            {"after": "2026-10-26T00:00:00Z", "before": "2026-10-25T00:00:00Z"},
        ):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                await self.group.query(**arguments)
