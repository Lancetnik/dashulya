import base64
import hashlib
import sqlite3
from contextlib import AsyncExitStack, closing
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from unittest.mock import patch

import pytest

from dashulya.database import Database
from dashulya.domain.memory import MemoryAccess, MemoryMessage
from dashulya.memory.history import CanonicalHistory
from dashulya.memory.search import MemorySearch
from dashulya.memory.store import MessageStore
from tests.postgres import imported_history

NOW = datetime(2026, 10, 25, 12, tzinfo=UTC)
GROUP = -100123
OTHER = -100456


@pytest.mark.usefixtures("_setup")
class TestCanonicalHistory:
    @pytest.fixture
    async def _setup(self, database: Database, cleanup: AsyncExitStack) -> None:
        self.database = database
        self.cleanup = cleanup
        self.directory = TemporaryDirectory()
        self.cleanup.callback(self.directory.cleanup)
        self.path = Path(self.directory.name)
        self.live = MessageStore(self.database)
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
                CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                INSERT INTO metadata VALUES ('version','1');
                INSERT INTO metadata VALUES ('source_sha256','"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"');
                INSERT INTO metadata VALUES ('chat_username','"public_group"');
                CREATE TABLE memories (id INTEGER PRIMARY KEY, key TEXT, kind TEXT, body TEXT);
                CREATE TABLE sources (memory_id INTEGER, message_id INTEGER);
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
        assert [m["text"] for m in group["messages"]] == ["Group original"]
        private = await self.private.query()
        assert len(private["messages"]) == 3
        assert len({m["source_id"] for m in private["messages"]}) == 3
        secret = next(m for m in private["messages"] if m["chat_id"] == 42)
        assert secret["url"] is None
        assert (await self.group.query("read", source_id=secret["source_id"]))[
            "messages"
        ] == []
        with pytest.raises(ValueError, match="outside"):
            await self.group.query(chat=f"telegram:{OTHER}")
        narrowed = await self.private.query(chat=f"telegram:{OTHER}")
        assert [m["text"] for m in narrowed["messages"]] == ["Other group secret"]

    async def test_topic_filter_narrows_primary_but_group_defaults_all_topics(
        self,
    ) -> None:
        await self.live.record(self.message(1, thread_id=7))
        await self.live.record(self.message(2, thread_id=8))
        await self.live.record(self.message(3))
        assert (await self.group.query("count"))["count"] == 3
        page = await self.group.query(topic_id=7)
        assert [m["message_id"] for m in page["messages"]] == [1]
        assert page["messages"][0]["topic_id"] == 7

    async def test_exact_count_blank_unicode_casefold_and_sensitive_option(self) -> None:
        for mid in range(1, 114):
            await self.live.record(
                self.message(mid, text="" if mid == 7 else "Straße TEST")
            )
        assert (await self.group.query("count"))["count"] == 113
        assert (await self.group.query("count", text_contains="STRASSE"))["count"] == 112
        assert (await self.group.query("count", text_contains="test"))["count"] == 112
        assert (
            await self.group.query("count", text_contains="test", case_sensitive=True)
        )["count"] == 0
        assert (
            await self.group.query("count", text_contains="TEST", case_sensitive=True)
        )["count"] == 112

    async def test_aggregation_examples_preserve_bodies_blanks_and_bounded_excerpts(
        self,
    ) -> None:
        bodies = {1: "A nonempty original", 2: "", 3: "Длинный текст " * 100}
        for mid, text in bodies.items():
            await self.live.record(self.message(mid, text=text))
        for action in ("count", "authors", "chats"):
            result = await self.group.query(action)
            if action == "count":
                assert result["count"] == 3
                examples = result["examples"]
            else:
                assert result[action][0]["message_count"] == 3
                examples = result[action][0]["examples"]
            assert [m["message_id"] for m in examples] == [1, 2, 3]
            for example in examples:
                body = bodies[example["message_id"]]
                assert example["text"] == body[:500]
                assert example["text_length"] == len(body)
                assert example["truncated"] == (len(body) > 500)
            assert examples[1]["text_length"] == 0
            assert examples[2]["next_offset"] == 500

    async def test_utc_after_inclusive_before_exclusive_and_unknown_archive_dates(
        self,
    ) -> None:
        await self.live.record(self.message(1, sent_at=NOW - timedelta(seconds=1)))
        await self.live.record(self.message(2, sent_at=NOW))
        await self.live.record(self.message(3, sent_at=NOW + timedelta(seconds=1)))
        result = await self.group.query(
            after="2026-10-25T15:00:00+03:00", before="2026-10-25T15:00:01+03:00"
        )
        assert [m["message_id"] for m in result["messages"]] == [2]
        assert result["messages"][0]["sent_at"] == NOW.isoformat()
        imported = self.imported(dated=False)
        reader = await imported_history(
            self.live, MemoryAccess(42, True), imported=imported
        )
        assert (await reader.query("count"))["count"] == 5
        assert (await reader.query("count", after=NOW.isoformat()))["count"] == 2
        with pytest.raises(ValueError, match="timezone"):
            await reader.query(after="2026-10-25T00:00:00")

    async def test_cursor_pages_full_set_without_dupes_and_cannot_change_scope(
        self,
    ) -> None:
        for mid in range(1, 114):
            await self.live.record(self.message(mid))
        cursor = None
        found: list[int] = []
        seen_cursors: set[str] = set()
        for _ in range((113 + 16) // 17):
            page = await self.group.query(cursor=cursor, limit=17)
            ids = [m["message_id"] for m in page["messages"]]
            assert ids, "Pagination must advance through nonempty pages"
            assert ids == sorted(ids)
            assert set(found).isdisjoint(ids), "A page repeated message IDs"
            found.extend(ids)
            cursor = page["next_cursor"]
            if cursor is None:
                break
            assert cursor not in seen_cursors, "Pagination repeated a cursor"
            seen_cursors.add(cursor)
        else:
            pytest.fail("Pagination did not terminate within the fixture's seven pages")
        assert len(found) == 113
        assert set(found) == set(range(1, 114))
        first = await self.group.query(limit=17)
        await self.live.record(self.message(114))
        second = await self.group.query(cursor=first["next_cursor"], limit=17)
        assert {m["message_id"] for m in first["messages"]}.isdisjoint(
            m["message_id"] for m in second["messages"]
        )
        with pytest.raises(ValueError, match="cursor"):
            await self.group.query(cursor=first["next_cursor"], text_contains="different")
        with pytest.raises(ValueError, match="cursor"):
            await self.private.query(cursor=first["next_cursor"])
        shifted = await self.group.query(offset=10, limit=3)
        assert [m["message_id"] for m in shifted["messages"]] == [102, 103, 104]

    async def test_default_page_returns_twenty_latest_originals(self) -> None:
        for mid in range(1, 22):
            await self.live.record(self.message(mid))
        first = await self.group.query()
        assert [m["message_id"] for m in first["messages"]] == list(range(2, 22))
        assert first["next_cursor"] is not None
        last = await self.group.query(cursor=first["next_cursor"])
        assert [m["message_id"] for m in last["messages"]] == [1]
        assert last["next_cursor"] is None

    async def test_cursor_accepts_2048_characters_and_rejects_longer_cursors(
        self,
        subtests: pytest.Subtests,
    ) -> None:
        for mid in range(1, 4):
            await self.live.record(self.message(mid))
        first = await self.group.query(limit=1)
        payload = base64.urlsafe_b64decode(first["next_cursor"])
        cursor = base64.urlsafe_b64encode(payload.ljust(1536, b" ")).decode()
        assert len(cursor) == 2048
        page = await self.group.query(cursor=cursor, limit=1)
        assert [m["message_id"] for m in page["messages"]] == [2]
        # The 2052-character encoding is valid JSON/base64, with the same query
        # and scope. Extra '=' padding is also rejected, independently of the
        # Python version's strict base64 decoder behavior.
        oversized_valid = base64.urlsafe_b64encode(payload.ljust(1539, b" ")).decode()
        for oversized in (cursor + "=", oversized_valid):
            with (
                subtests.test(length=len(oversized)),
                pytest.raises(ValueError, match="cursor"),
            ):
                await self.group.query(cursor=oversized, limit=1)

    async def test_live_edit_is_visible_next_query_before_indexing_and_shadows_snapshot(
        self,
    ) -> None:
        imported = self.imported()
        reader = await imported_history(
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
        assert result["messages"][0]["text"] == "Corrected canonical original"
        assert result["messages"][0]["provenance"] == "live"
        assert (await reader.query("count"))["count"] == 2
        assert (await reader.query("count", text_contains="Archive original 1"))[
            "count"
        ] == 0
        assert (await reader.query("count", text_contains="SECRET_DERIVED_MEMORY"))[
            "count"
        ] == 0
        assert len(await self.live.pending()) > 0

    async def test_unknown_numeric_archive_is_private_only_and_binding_is_explicit(
        self,
    ) -> None:
        imported = self.imported()
        checksum = hashlib.sha256(imported.path.read_bytes()).hexdigest()
        await self.live.record(self.message(1, text="Live same ID"))
        group = await imported_history(
            self.live, MemoryAccess(GROUP, False), imported=imported
        )
        assert (await group.query("count"))["count"] == 1
        private = await imported_history(
            self.live, MemoryAccess(42, True), imported=imported
        )
        sources = (await private.query())["messages"]
        assert len(sources) == 3
        archive = [m for m in sources if m["provenance"] == "imported"]
        assert all(m["chat_id"] is None for m in archive)
        assert all(m["chat"].startswith("archive:") for m in archive)
        assert all(m["topic_id"] is None for m in archive)
        assert len({m["source_id"] for m in sources}) == 3
        bound = await imported_history(
            self.live, MemoryAccess(GROUP, False), imported=imported, import_chat_id=GROUP
        )
        assert (await bound.query("count"))["count"] == 2
        assert hashlib.sha256(imported.path.read_bytes()).hexdigest() == checksum

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
        assert [m["message_id"] for m in result["context"]] == [1]
        assert result["context"][0]["author_id"] == 11
        assert "Future secret" not in str(result)
        assert "Foreign parent secret" not in str(result)

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
        for _ in range((len(body) + 11999) // 12000):
            page = await self.group.query(
                "read", source_id=target["source_id"], offset=offset
            )
            record = page["messages"][0]
            assert record["offset"] == offset, "Read ignored the requested offset"
            assert record["text"] == body[offset : offset + 12000]
            chunks.append(record["text"])
            assert len(page["context"]) <= 9
            assert len(record["text"]) <= 12000
            end = offset + len(record["text"])
            if end == len(body):
                assert record["next_offset"] is None, "Final chunk must terminate"
                break
            assert end > offset, "Read returned no new text"
            assert record["next_offset"] == end
            offset = end
        else:
            pytest.fail(
                "Read did not terminate within the fixture's expected chunk count"
            )
        assert "".join(chunks) == body
        assert record["text_length"] == len(body)

    async def test_authors_use_numeric_identity_chats_and_unknown_legacy_bot_metadata(
        self,
    ) -> None:
        await self.live.record(self.message(1, author_id=10))
        await self.live.record(self.message(2, author_id=11))
        await self.live.record(
            self.message(1, chat_id=OTHER, author_id=10, author="Alex renamed")
        )
        for action, key, identities in (
            ("authors", "author_id", {10, 11}),
            ("chats", "chat", {f"telegram:{GROUP}", f"telegram:{OTHER}"}),
        ):
            first = await self.private.query(action, limit=1)
            assert first["total"] == 2
            assert len(first[action]) == 1
            assert first["next_offset"] == 1
            last = await self.private.query(action, offset=first["next_offset"], limit=1)
            assert last["total"] == 2
            assert len(last[action]) == 1
            assert last["next_offset"] is None
            rows = first[action] + last[action]
            assert {row[key] for row in rows} == identities
            if action == "authors":
                author = next(a for a in rows if a["author_id"] == 10)
                assert author["names"] == ["Alex", "Alex renamed"]
                assert author["message_count"] == 2
            else:
                assert {row["chat"]: row["chat_id"] for row in rows} == {
                    f"telegram:{GROUP}": GROUP,
                    f"telegram:{OTHER}": OTHER,
                }
        assert (await self.group.query("chats"))["total"] == 1
        assert (await self.group.query())["messages"][0]["is_bot"] is None

    async def test_optional_sender_metadata_does_not_replace_chat_username(self) -> None:
        await self.live.record(self.message(1))
        await self.live.record(self.message(1, username="agent_handle", is_bot=True))
        record = (await self.group.query())["messages"][0]
        assert record["username"] == "agent_handle"
        assert record["is_bot"]
        assert record["chat_username"] == "public_group"
        assert record["url"] == "https://t.me/public_group/1"

    async def test_fixed_cutoff_rejects_naive_time_and_requires_timestamp_for_id(
        self,
    ) -> None:
        with pytest.raises(ValueError, match="timezone"):
            MemoryAccess(GROUP, False, before=NOW.replace(tzinfo=None))
        with pytest.raises(ValueError, match="timestamp"):
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
        assert [(m["chat_id"], m["message_id"]) for m in records] == [(42, 1)]

    async def test_later_edit_is_not_historical_evidence_and_old_import_stays_shadowed(
        self,
    ) -> None:
        imported = self.imported()
        await self.live.record(
            self.message(
                1, text="Future corrected fact", edited_at=NOW + timedelta(seconds=2)
            )
        )
        reader = await imported_history(
            self.live,
            MemoryAccess(GROUP, False, before=NOW + timedelta(seconds=1)),
            imported=imported,
            import_chat_id=GROUP,
        )
        result = await reader.query()
        assert [m["message_id"] for m in result["messages"]] == [2]
        assert "Future corrected fact" not in str(result)
        assert (await reader.query("count", text_contains="Archive original 1"))[
            "count"
        ] == 0

    async def test_malformed_cursor_and_action_arguments_fail_without_scope_changes(
        self,
        subtests: pytest.Subtests,
    ) -> None:
        for value in ("not-a-cursor", "a" * 3000):
            with subtests.test(value=value), pytest.raises(ValueError, match="cursor"):
                await self.group.query(cursor=value)
        for arguments in (
            {"action": "read"},
            {"source_id": "source:unknown:1"},
            {"limit": 0},
            {"offset": -1},
            {"action": "unknown"},
            {"after": "2026-10-26T00:00:00Z", "before": "2026-10-25T00:00:00Z"},
        ):
            with (
                subtests.test(arguments=arguments),
                pytest.raises(ValueError, match=r"source_id|limit|offset|action|after"),
            ):
                await self.group.query(**arguments)

    async def test_private_chat_selection_exposes_only_known_selected_collection(
        self,
    ) -> None:
        imported = self.imported()
        await self.live.record(self.message(1, text="Live current group"))
        await self.live.record(self.message(1, chat_id=OTHER, text="Other chat"))
        reader = await imported_history(
            self.live, MemoryAccess(42, True), imported=imported
        )
        archive = f"archive:{'a' * 64}"
        assert len(await reader.originals(chat=archive)) == 2
        selected = await reader.query(chat=archive)
        assert {m["chat"] for m in selected["messages"]} == {archive}
        assert selected["coverage"]["available_messages"] == 2
        assert {p["chat"] for p in selected["coverage"]["known_periods"]} == {archive}
        with pytest.raises(ValueError, match="Unknown saved chat"):
            await reader.originals(chat="telegram:-100999")
        with pytest.raises(ValueError, match="outside"):
            await self.group.originals(chat=archive)

    async def test_coverage_reports_scoped_saved_imported_bounds_without_global_metadata(
        self,
    ) -> None:
        imported = self.imported()
        imported.coverage.update(
            first_message_at="1900-01-01T00:00:00Z",
            last_message_at="2999-01-01T00:00:00Z",
        )
        await self.live.record(self.message(3, sent_at=NOW + timedelta(seconds=1)))
        await self.live.record(self.message(4, sent_at=NOW + timedelta(days=50)))
        await self.live.record(
            self.message(1, chat_id=OTHER, sent_at=NOW - timedelta(days=100))
        )
        access = MemoryAccess(GROUP, False, before=NOW + timedelta(seconds=2))
        reader = await imported_history(
            self.live, access, imported=imported, import_chat_id=GROUP
        )
        coverage = (await reader.query("count"))["coverage"]
        assert coverage["available_messages"] == 3
        assert coverage["first_saved_at"] == NOW.isoformat()
        assert coverage["last_saved_at"] == (NOW + timedelta(seconds=1)).isoformat()
        assert {p["provenance"] for p in coverage["known_periods"]} == {
            "live",
            "imported",
        }
        assert {p["chat"] for p in coverage["known_periods"]} == {f"telegram:{GROUP}"}
        assert "1900-" not in str(coverage)
        assert "2999-" not in str(coverage)
        assert str(OTHER) not in str(coverage)
        assert not coverage["complete"]

    async def test_name_lookup_matches_unicode_names_usernames_and_preserves_body_predicate(
        self,
    ) -> None:
        for message in (
            self.message(
                1, author_id=10, author="Straße", username="old_handle", text="first"
            ),
            self.message(
                2, author_id=10, author="Renamed", username="new_handle", text="second"
            ),
            self.message(
                3, author_id=11, author="Straße", username="other_handle", text="third"
            ),
            self.message(
                4, author_id=12, author="Unrelated", text="Straße in message body"
            ),
        ):
            await self.live.record(message)
        result = await self.group.query("authors", name_contains="STRASSE")
        assert {a["author_id"] for a in result["authors"]} == {10, 11}
        author = next(a for a in result["authors"] if a["author_id"] == 10)
        assert author["names"] == ["Renamed", "Straße"]
        assert author["usernames"] == ["new_handle", "old_handle"]
        assert author["message_count"] == 2
        username = await self.group.query("authors", name_contains="@OLD_HANDLE")
        assert [a["author_id"] for a in username["authors"]] == [10]
        body = await self.group.query("authors", text_contains="STRASSE")
        assert [a["author_id"] for a in body["authors"]] == [12]
        both = await self.group.query(
            "count", name_contains="@old_handle", text_contains="second"
        )
        assert both["count"] == 1

    async def test_identity_lookup_does_not_learn_future_alias_or_merge_unknown_people(
        self,
    ) -> None:
        await self.live.record(
            self.message(1, author_id=10, author="Old", username="old_handle")
        )
        await self.live.record(
            self.message(
                2,
                author_id=10,
                author="Future alias",
                username="future_handle",
                sent_at=NOW + timedelta(days=1),
            )
        )
        await self.live.record(self.message(3, author_id=None, author="Unknown A"))
        await self.live.record(self.message(4, author_id=None, author="Unknown B"))
        reader = CanonicalHistory(
            self.live, MemoryAccess(GROUP, False, before=NOW + timedelta(seconds=1))
        )
        assert (await reader.query("authors", name_contains="future_handle"))[
            "authors"
        ] == []
        unknown = (await reader.query("authors", name_contains="unknown"))["authors"]
        assert len(unknown) == 2
        assert all(a["author_id"] is None for a in unknown)
        assert (await reader.query("count", name_contains="unknown A"))["count"] == 1
