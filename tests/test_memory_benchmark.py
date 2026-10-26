import random
import unittest
from collections.abc import Sequence
from dataclasses import replace
from datetime import UTC, datetime, timedelta, timezone
from typing import Any
from unittest.mock import patch

from evals.suites.memory.answerer import AnswerSession
from evals.suites.memory.corpus import SourceArchive, aware_time, source_json
from evals.suites.memory.types import MemoryHit, QueryContext, SourceMessage


def message(
    mid: int,
    text: str = "Original source",
    *,
    chat: str = "group",
    date: str = "2026-01-01T10:00:00Z",
    uid: int = 10,
    parent: int | None = None,
    bot: bool = False,
    thread: int | None = None,
) -> SourceMessage:
    return SourceMessage(
        id=mid,
        chat_key=chat,
        date=date,
        text=text,
        user_id=uid,
        author=f"participant_{uid}",
        user_name="Shared name",
        reply_to_msg_id=parent,
        thread_post_id=thread,
        is_bot=bot,
    )


class ScriptedMemory:
    def __init__(self, hits: list[MemoryHit] | None = None) -> None:
        self.hits = hits or []
        self.contexts: list[QueryContext] = []

    async def prepare(self, messages: Sequence[SourceMessage]) -> dict[str, object]:
        return {"sources": len(messages)}

    async def search(
        self, query: str, context: QueryContext, *, limit: int = 8
    ) -> list[MemoryHit]:
        self.contexts.append(context)
        return self.hits[:limit]

    async def close(self) -> None:
        pass


def naive_neighbors(
    archive: SourceArchive,
    message_id: int,
    scope: QueryContext,
    *,
    radius: int = 2,
    max_seconds: int = 180,
) -> list[SourceMessage]:
    """Independent copy of the pre-index chronological scan, for equivalence checks."""
    target = archive.get(message_id, scope)
    if target is None or radius <= 0 or max_seconds < 0:
        return []
    radius = min(radius, 2)
    max_seconds = min(max_seconds, 180)
    instant = aware_time(target.date)
    position = (instant, target.id)
    preceding = []
    following = []
    for candidate in archive.available(scope):
        if candidate.id == target.id:
            continue
        if (
            target.thread_post_id is not None
            and candidate.thread_post_id is not None
            and target.thread_post_id != candidate.thread_post_id
        ):
            continue
        candidate_time = aware_time(candidate.date)
        if abs((candidate_time - instant).total_seconds()) > max_seconds:
            continue
        if (candidate_time, candidate.id) < position:
            preceding.append(candidate)
        else:
            following.append(candidate)
    return preceding[-radius:] + following[:radius]


class SourceArchiveTests(unittest.TestCase):
    def setUp(self) -> None:
        self.scope = QueryContext("group", "2026-01-03T00:00:00Z")

    def test_indexed_neighbors_equal_naive_across_random_scopes_and_time_offsets(
        self,
    ) -> None:
        rng = random.Random(20261026)
        start = datetime(2026, 1, 1, 10, tzinfo=UTC)
        messages = [
            message(
                mid,
                chat=chat,
                date=(start + timedelta(seconds=30 * rng.randint(-20, 20)))
                .astimezone(timezone(timedelta(minutes=rng.choice([-330, 0, 180, 330]))))
                .isoformat(),
                uid=rng.choice([10, 20, 30]),
                thread=rng.choice([None, 11, 12]),
            )
            for chat in ["group", "another_group", "private"]
            for mid in range(1, 121)
        ]
        rng.shuffle(messages)
        archive = SourceArchive(messages)
        nonempty = 0
        for iteration in range(1000):
            target = rng.choice(messages)
            instant = aware_time(target.date)
            cutoff = (
                instant
                if iteration % 4 < 2
                else instant + timedelta(seconds=rng.randrange(300))
            )
            before_id = (
                target.id + (iteration % 4) if iteration % 4 < 2 else rng.randrange(150)
            )
            after = rng.choice([
                None,
                None,
                (instant - timedelta(seconds=180)).isoformat(),
                (instant + timedelta(seconds=1)).isoformat(),
            ])
            scope = QueryContext(
                rng.choice([target.chat_key] * 9 + ["absent_chat"]),
                cutoff.astimezone(timezone(timedelta(hours=3))).isoformat(),
                before_id=before_id,
                after=after,
                author_id=rng.choice([None, None, target.user_id, 99]),
            )
            radius = rng.choice([-1, 0, 1, 2, 9])
            seconds = rng.choice([-1, 0, 90, 180, 999])
            expected = naive_neighbors(
                archive, target.id, scope, radius=radius, max_seconds=seconds
            )
            actual = archive.neighbors(
                target.id, scope, radius=radius, max_seconds=seconds
            )
            with self.subTest(iteration=iteration):
                self.assertEqual(actual, expected)
            nonempty += bool(actual)
        self.assertGreater(nonempty, 30)

    def test_neighbors_parse_query_bounds_without_reparsing_original_dates(self) -> None:
        archive = SourceArchive([
            message(i, date=f"2026-01-01T10:00:{i:02}Z") for i in range(1, 31)
        ])
        with patch("evals.suites.memory.corpus.aware_time", wraps=aware_time) as parse:
            result = archive.neighbors(
                15, replace(self.scope, after="2026-01-01T00:00:00Z")
            )
        self.assertEqual([m.id for m in result], [13, 14, 16, 17])
        self.assertLessEqual(parse.call_count, 2)

    def test_same_numeric_id_in_other_chat_does_not_replace_group_original(self) -> None:
        archive = SourceArchive([
            message(1, "group original"),
            message(1, "private secret", chat="private"),
        ])
        selected = archive.get(1, self.scope)
        self.assertIsNotNone(selected)
        assert selected is not None
        self.assertEqual(selected.text, "group original")
        self.assertEqual(
            [m.text for m in archive.available(self.scope)], ["group original"]
        )

    def test_duplicate_identity_in_one_chat_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate"):
            SourceArchive([message(1), message(1, "another version")])

    def test_timezone_after_inclusive_and_before_exclusive(self) -> None:
        archive = SourceArchive([
            message(1, date="2026-01-01T11:59:59+03:00"),
            message(2, date="2026-01-01T12:00:00+03:00"),
            message(3, date="2026-01-01T09:30:00Z"),
            message(4, date="2026-01-01T13:00:00+03:00"),
        ])
        scope = QueryContext(
            "group", "2026-01-01T10:00:00Z", after="2026-01-01T09:00:00Z"
        )
        self.assertEqual([m.id for m in archive.available(scope)], [2, 3])

    def test_equal_time_excludes_current_request_and_future_ids(self) -> None:
        archive = SourceArchive([message(i) for i in range(1, 5)])
        scope = QueryContext("group", "2026-01-01T10:00:00Z", before_id=3)
        self.assertEqual([m.id for m in archive.available(scope)], [1, 2])
        self.assertIsNone(archive.get(3, scope))

    def test_all_pages_cover_every_matching_record_once_including_blank_text(
        self,
    ) -> None:
        archive = SourceArchive(
            [message(i, "" if i == 17 else "needle") for i in range(1, 114)]
            + [message(400, "secret", chat="private")]
        )
        cursor = None
        found: list[int] = []
        sizes: list[int] = []
        while True:
            page, cursor = archive.list_messages(self.scope, cursor=cursor, limit=50)
            sizes.append(len(page))
            self.assertEqual([m.id for m in page], sorted(m.id for m in page))
            found.extend(m.id for m in page)
            if cursor is None:
                break
        self.assertEqual(sizes, [50, 50, 13])
        self.assertEqual(len(found), 113)
        self.assertEqual(set(found), set(range(1, 114)))
        self.assertEqual(len(archive.matching(self.scope, text_contains="needle")), 112)
        self.assertFalse(archive.coverage(self.scope)["complete"])

    def test_cursor_must_belong_to_matching_authorized_result_set(self) -> None:
        archive = SourceArchive([
            message(1, "needle"),
            message(2, "different"),
            message(3, chat="private"),
        ])
        for cursor in (2, 3, 999):
            with (
                self.subTest(cursor=cursor),
                self.assertRaisesRegex(ValueError, "cursor"),
            ):
                archive.list_messages(self.scope, text_contains="needle", cursor=cursor)

    def test_literal_unicode_search_uses_casefold_only_when_requested(self) -> None:
        archive = SourceArchive([message(1, "Straße — Память 📚"), message(2, "strasse")])
        self.assertEqual(
            [m.id for m in archive.matching(self.scope, text_contains="STRASSE")], [1, 2]
        )
        self.assertEqual(
            archive.matching(self.scope, text_contains="STRASSE", case_sensitive=True), []
        )
        self.assertEqual(
            [m.id for m in archive.matching(self.scope, text_contains="📚")], [1]
        )

    def test_context_excludes_query_future_and_foreign_chat_parents(self) -> None:
        archive = SourceArchive([
            message(1, "first request", parent=5),
            message(2, "second request", parent=6),
            message(5, "future", date="2026-02-01T00:00:00Z"),
            message(6, "private secret", chat="private"),
        ])
        self.assertEqual(archive.context(1, self.scope), [])
        self.assertEqual(archive.context(2, self.scope), [])

    def test_context_rejects_parent_sent_after_its_child(self) -> None:
        archive = SourceArchive([
            message(1, "child", parent=2),
            message(2, "not a possible parent", date="2026-01-02T00:00:00Z"),
        ])
        self.assertEqual(archive.context(1, self.scope), [])

    def test_author_lookup_keeps_identical_names_as_distinct_numeric_people(self) -> None:
        archive = SourceArchive([
            message(1, uid=10),
            message(2, uid=20),
            message(3, chat="private", uid=30),
        ])
        found = archive.authors("Shared name", self.scope)
        self.assertEqual({person["user_id"] for person in found}, {10, 20})

    def test_original_unicode_bot_flag_and_source_link_survive_serialization(
        self,
    ) -> None:
        row = source_json(message(7, "Память 📚", bot=True))
        self.assertEqual(row["text"], "Память 📚")
        self.assertTrue(row["is_bot"])
        self.assertEqual(row["url"], "https://t.me/group/7")

    def test_neighbors_recover_a_following_link_and_preceding_qualification(self) -> None:
        archive = SourceArchive([
            message(
                1, "Tried this only for small local tasks.", date="2026-01-01T09:59:30Z"
            ),
            message(2, "Here is the useful project.", date="2026-01-01T10:00:00Z"),
            message(3, "https://example.test/project", date="2026-01-01T10:00:15Z"),
        ])
        selected = archive.neighbors(2, self.scope)
        self.assertEqual([row.id for row in selected], [1, 3])
        self.assertIn("small local tasks", selected[0].text)
        self.assertEqual(selected[1].text, "https://example.test/project")
        self.assertEqual(archive.context(2, self.scope), [])

    def test_neighbors_return_at_most_two_each_side_in_chronological_order(self) -> None:
        archive = SourceArchive([
            message(6, date="2026-01-01T10:02:00Z"),
            message(1, date="2026-01-01T09:58:30Z"),
            message(4, date="2026-01-01T10:00:00Z"),
            message(2, date="2026-01-01T09:59:00Z"),
            message(7, date="2026-01-01T10:02:30Z"),
            message(3, date="2026-01-01T09:59:30Z"),
            message(5, date="2026-01-01T10:01:00Z"),
        ])
        self.assertEqual(
            [row.id for row in archive.neighbors(4, self.scope)], [2, 3, 5, 6]
        )
        self.assertEqual(
            [row.id for row in archive.neighbors(4, self.scope, radius=99)], [2, 3, 5, 6]
        )
        self.assertEqual(
            [row.id for row in archive.neighbors(4, self.scope, radius=1)], [3, 5]
        )
        self.assertEqual(archive.neighbors(4, self.scope, radius=0), [])

    def test_neighbors_respect_cutoff_ids_and_exclude_foreign_chat_collision(
        self,
    ) -> None:
        archive = SourceArchive([
            message(1, "preceding"),
            message(2, "selected"),
            message(3, "current request"),
            message(4, "later source"),
            message(2, "private target collision", chat="private"),
            message(5, "private neighbor", chat="private"),
        ])
        scope = QueryContext("group", "2026-01-01T10:00:00Z", before_id=3)
        self.assertEqual([row.id for row in archive.neighbors(2, scope)], [1])
        self.assertEqual(archive.neighbors(3, scope), [])
        self.assertEqual(archive.neighbors(5, scope), [])

    def test_neighbors_preserve_fixed_author_and_after_bounds(self) -> None:
        archive = SourceArchive([
            message(1, "too old", date="2026-01-01T09:59:00Z"),
            message(2, "selected", date="2026-01-01T10:00:00Z"),
            message(3, "other author", date="2026-01-01T10:00:30Z", uid=20),
            message(4, "same author", date="2026-01-01T10:01:00Z"),
            message(5, "future", date="2026-01-01T10:04:00Z"),
        ])
        scope = QueryContext(
            "group", "2026-01-01T10:03:00Z", author_id=10, after="2026-01-01T10:00:00Z"
        )
        self.assertEqual([row.id for row in archive.neighbors(2, scope)], [4])
        self.assertEqual(archive.neighbors(3, scope), [])

    def test_neighbor_time_window_is_inclusive_and_never_exceeds_three_minutes(
        self,
    ) -> None:
        archive = SourceArchive([
            message(1, date="2026-01-01T09:59:59Z"),
            message(2, date="2026-01-01T10:00:00Z"),
            message(3, date="2026-01-01T10:03:00Z"),
            message(4, date="2026-01-01T10:06:00Z"),
            message(5, date="2026-01-01T10:06:01Z"),
        ])
        self.assertEqual([row.id for row in archive.neighbors(3, self.scope)], [2, 4])
        self.assertEqual(
            [row.id for row in archive.neighbors(3, self.scope, max_seconds=999)], [2, 4]
        )
        self.assertEqual(archive.neighbors(3, self.scope, max_seconds=179), [])

    def test_neighbors_exclude_known_thread_mismatches_but_keep_unknown_threads(
        self,
    ) -> None:
        archive = SourceArchive([
            message(1, date="2026-01-01T09:59:00Z", thread=90),
            message(2, date="2026-01-01T09:59:30Z", thread=80),
            message(3, date="2026-01-01T09:59:45Z"),
            message(4, date="2026-01-01T10:00:00Z", thread=80),
            message(5, date="2026-01-01T10:00:15Z", thread=90),
            message(6, date="2026-01-01T10:00:30Z"),
            message(7, date="2026-01-01T10:01:00Z", thread=80),
        ])
        self.assertEqual(
            [row.id for row in archive.neighbors(4, self.scope)], [2, 3, 6, 7]
        )


class BenchmarkHarnessTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.scope = QueryContext("group", "2026-01-03T00:00:00Z")

    async def history(self, session: AnswerSession, **overrides: Any) -> dict[str, Any]:
        arguments: dict[str, Any] = {
            "action": "list",
            "text_contains": "",
            "author_id": None,
            "after": None,
            "before": None,
            "message_id": None,
            "cursor": None,
            "limit": 20,
            "case_sensitive": False,
            "offset": 0,
        }
        arguments.update(overrides)
        return await session.history(**arguments)

    async def test_invalid_backend_sources_drop_whole_multi_source_hit(self) -> None:
        archive = SourceArchive([
            message(1, "allowed original"),
            message(1, "same ID private secret", chat="private"),
            message(2, "other chat only", chat="other"),
            message(3, "future", date="2026-02-01T00:00:00Z"),
        ])
        memory = ScriptedMemory([
            MemoryHit("valid", (1,), chat_key="group"),
            MemoryHit("foreign", (2,), chat_key="group"),
            MemoryHit("future", (3,), chat_key="group"),
            MemoryHit("mixed valid and future", (1, 3), chat_key="group"),
            MemoryHit("invented ID", (999,), chat_key="group"),
            MemoryHit("no original support", (), chat_key="group"),
            MemoryHit("same ID private leak", (1,), chat_key="private"),
            MemoryHit("unknown origin", (1,)),
        ])
        session = AnswerSession(archive, memory, self.scope)
        output = await session.search("question", None, None, 12)
        self.assertEqual(len(output["hits"]), 1)
        self.assertEqual(output["hits"][0]["sources"][0]["text"], "allowed original")
        self.assertEqual(session.seen_sources, {1})
        self.assertEqual(memory.contexts, [self.scope])

    async def test_search_never_returns_parents_after_query_cutoff(self) -> None:
        archive = SourceArchive([
            message(1, "allowed", parent=2),
            message(2, date="2026-02-01T00:00:00Z"),
        ])
        session = AnswerSession(
            archive,
            ScriptedMemory([MemoryHit("valid", (1,), chat_key="group")]),
            self.scope,
        )
        output = await session.search("question", None, None, 8)
        self.assertEqual(output["hits"][0]["reply_context"], [])
        self.assertEqual(session.seen_sources, {1})

    async def test_nearby_continuations_are_labeled_and_only_exposed_records_are_seen(
        self,
    ) -> None:
        archive = SourceArchive([
            message(
                9, "Only used for local tasks.", date="2026-01-01T09:59:30Z", thread=7
            ),
            message(
                10, "Here is the project I tried.", date="2026-01-01T10:00:00Z", thread=7
            ),
            message(
                11, "https://example.test/project", date="2026-01-01T10:00:30Z", thread=7
            ),
            message(
                15, "Additional context " * 1000, date="2026-01-01T10:00:40Z", thread=7
            ),
            message(12, "Different topic", date="2026-01-01T10:00:45Z", thread=99),
            message(13, "Private content", chat="private", date="2026-01-01T10:00:50Z"),
            message(14, "After query cutoff", date="2026-01-01T10:01:00Z", thread=7),
        ])
        scope = QueryContext("group", "2026-01-01T10:01:00Z")
        memory = ScriptedMemory([
            MemoryHit("Project experience", (10,), chat_key="group")
        ])
        with patch.dict("os.environ", {"MEMORY_EVAL_EVIDENCE_CHARACTERS": "0"}):
            search_session = AnswerSession(archive, memory, scope)
            read_session = AnswerSession(archive, ScriptedMemory(), scope)
        search = await search_session.search("project experience", None, None, 8)
        hit = search["hits"][0]
        self.assertEqual([row["id"] for row in hit["sources"]], [10])
        self.assertEqual(hit["reply_context"], [])
        self.assertEqual([row["id"] for row in hit["nearby_context"]], [9, 11, 15])
        self.assertEqual(hit["nearby_context"][1]["text"], "https://example.test/project")
        self.assertEqual(search_session.seen_sources, {9, 10, 11, 15})
        read = await self.history(read_session, action="read", message_id=10)
        self.assertEqual([row["id"] for row in read["nearby_context"]], [9, 11, 15])
        self.assertEqual(read_session.seen_sources, {9, 10, 11, 15})

        with patch.dict("os.environ", {"MEMORY_EVAL_EVIDENCE_CHARACTERS": "1100"}):
            limited = AnswerSession(archive, memory, scope)
        output = await limited.search("project experience", None, None, 8)
        self.assertTrue(output.get("evidence_truncated"))
        exposed = {
            row["id"]
            for record in output.get("hits", [])
            for field in ("sources", "reply_context", "nearby_context")
            for row in record.get(field, [])
        }
        self.assertEqual(limited.seen_sources, exposed)
        self.assertNotIn(15, limited.seen_sources)
        self.assertTrue(exposed.isdisjoint({12, 13, 14}))

    async def test_exact_count_scans_113_sources_including_blank(self) -> None:
        archive = SourceArchive(
            [message(i, "" if i == 17 else "needle") for i in range(1, 114)]
            + [message(200, "needle", chat="other")]
        )
        session = AnswerSession(archive, ScriptedMemory(), self.scope)
        output = await self.history(session, action="count", limit=1)
        self.assertEqual(output["count"], 113)
        self.assertEqual(len(output["example_sources"]), 5)
        self.assertEqual(output["coverage"]["available_messages"], 113)
        needle_only = await self.history(session, action="count", text_contains="needle")
        self.assertEqual(needle_only["count"], 112)

    async def test_history_pagination_exposes_complete_result_set(self) -> None:
        archive = SourceArchive([message(i) for i in range(1, 114)])
        session = AnswerSession(archive, ScriptedMemory(), self.scope)
        cursor = None
        collected: list[int] = []
        while True:
            output = await self.history(session, limit=50, cursor=cursor)
            collected.extend(row["id"] for row in output["messages"])
            cursor = output["next_cursor"]
            if cursor is None:
                break
        self.assertEqual(sorted(collected), list(range(1, 114)))
        self.assertEqual(len(collected), len(set(collected)))

    async def test_read_offsets_preserve_unicode_and_can_continue_long_original(
        self,
    ) -> None:
        text = "📚" * 12_000 + "continuation"
        archive = SourceArchive([message(1, text, bot=True)])
        session = AnswerSession(archive, ScriptedMemory(), self.scope)
        first = await self.history(session, action="read", message_id=1)
        second = await self.history(session, action="read", message_id=1, offset=12_000)
        self.assertTrue(first["source"]["truncated"])
        self.assertEqual(first["source"]["text"], "📚" * 12_000)
        self.assertEqual(second["source"]["text"], "continuation")
        self.assertFalse(second["source"]["truncated"])
        self.assertTrue(second["source"]["is_bot"])

    async def test_optional_filters_cannot_clear_fixed_author_or_after_scope(
        self,
    ) -> None:
        archive = SourceArchive([
            message(1, "old", uid=10),
            message(2, "allowed", date="2026-01-02T00:00:00Z", uid=10),
            message(3, "different author", date="2026-01-02T00:00:00Z", uid=20),
        ])
        fixed = replace(self.scope, author_id=10, after="2026-01-02T00:00:00Z")
        memory = ScriptedMemory([
            MemoryHit(str(i), (i,), chat_key="group") for i in range(1, 4)
        ])
        session = AnswerSession(archive, memory, fixed)
        output = await session.search("question", None, None, 8)
        self.assertEqual([s["id"] for h in output["hits"] for s in h["sources"]], [2])
        history = await self.history(session)
        self.assertEqual([s["id"] for s in history["messages"]], [2])

    async def test_requested_filters_cannot_widen_fixed_time_bounds(self) -> None:
        archive = SourceArchive([
            message(1, "old"),
            message(2, "allowed", date="2026-01-02T00:00:00Z"),
            message(3, "future", date="2026-01-04T00:00:00Z"),
        ])
        scope = replace(self.scope, after="2026-01-02T00:00:00Z")
        session = AnswerSession(archive, ScriptedMemory(), scope)
        output = await self.history(
            session, after="2025-01-01T00:00:00Z", before="2027-01-01T00:00:00Z"
        )
        self.assertEqual([s["id"] for s in output["messages"]], [2])

    async def test_tool_budget_is_enforced_without_calling_memory_again(self) -> None:
        memory = ScriptedMemory()
        session = AnswerSession(SourceArchive([]), memory, self.scope, tool_budget=1)
        initial = await session.search("question", None, None, 8)
        self.assertEqual(initial["tool_calls_remaining"], 0)
        exhausted = await session.search("another question", None, None, 8)
        self.assertIn("error", exhausted)
        self.assertEqual(len(memory.contexts), 1)
