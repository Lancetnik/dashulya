import asyncio
import unittest
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock

from dashulya.ai.message_format import MessageFormatter
from dashulya.domain.memory import MemoryAccess, OriginalMessage
from dashulya.memory.history import CanonicalHistory
from dashulya.memory.store import MessageStore
from tests.support import group_records, group_rows

GROUP = -100123
NOW = datetime(2026, 10, 25, 12, tzinfo=UTC)


def original(number: int, **changes: Any) -> OriginalMessage:
    return replace(
        OriginalMessage(
            collection=f"telegram:{GROUP}",
            chat_id=GROUP,
            message_id=number,
            text=f"Original {number}",
            sent_at=NOW + timedelta(seconds=number),
            author_id=42,
            author="Alice",
            chat_title="Group",
            chat_username="group",
            topic_id=None,
            reply_to_id=number - 1 if number > 1 else None,
            edited_at=None,
            media_type=None,
            url=f"https://t.me/group/{number}",
            provenance="observed",
        ),
        **changes,
    )


def cut_header(text: str) -> dict[str, Any]:
    return {
        str(record["id"]): record["truncated_thread"]
        for record in group_records(text)
        if "truncated_thread" in record
    }


class HistoryReplyHopsTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.store = AsyncMock(spec=MessageStore)
        self.originals = [original(number) for number in range(1, 10)]
        self.store.originals.return_value = self.originals
        self.access = MemoryAccess(GROUP, False, before=NOW + timedelta(minutes=1))
        self.reader = CanonicalHistory(self.store, self.access)
        self.formatter = MessageFormatter()

    async def test_read_hop_depth_zero_default_and_full_chain(self) -> None:
        for hops in (0, 1, 3, 5, 8, 100):
            with self.subTest(hops=hops):
                result = await self.reader.query(
                    "read", source_id=original(9).source_id, parent_hops=hops
                )
                self.assertEqual([m["message_id"] for m in result["messages"]], [9])
                self.assertEqual(
                    [m["message_id"] for m in result["context"]],
                    list(range(max(1, 9 - hops), 9)),
                )
                if hops < 8:
                    boundary = original(9 - hops)
                    cut = result["truncated_threads"][boundary.source_id]
                    self.assertEqual(cut["root_id"], 1)
                    self.assertEqual(cut["remaining_hops"], 8 - hops)
                else:
                    self.assertEqual(result["truncated_threads"], {})
        default = await self.reader.query("read", source_id=original(9).source_id)
        self.assertEqual(default["parent_hops"], 5)
        self.assertEqual([m["message_id"] for m in default["context"]], [4, 5, 6, 7, 8])

    async def test_navigation_fetches_every_omitted_parent_in_one_call(self) -> None:
        first = self.formatter.result(
            await self.reader.query(
                "read", source_id=original(9).source_id, parent_hops=2
            )
        )
        cut = cut_header(first["message_blocks"][0]["text"])["7"]
        self.assertEqual(cut["remaining_hops"], 6)
        self.assertEqual(cut["boundary_source_id"], original(7).source_id)
        self.assertEqual(cut["source_id"], original(1).source_id)
        second = self.formatter.result(
            await self.reader.query(
                "read",
                source_id=cut["boundary_source_id"],
                parent_hops=cut["remaining_hops"],
            )
        )
        self.assertEqual(
            [row[0] for row in group_rows(second["message_blocks"][0]["text"])],
            list(range(1, 8)),
        )
        self.assertNotIn("truncated_threads", second["message_blocks"][0]["text"])
        self.assertNotIn("truncated_threads", first)

    async def test_list_preserves_primary_pagination_counts_and_deduplicates_parents(
        self,
    ) -> None:
        first = await self.reader.query(limit=2, parent_hops=2)
        self.assertEqual([m["message_id"] for m in first["messages"]], [8, 9])
        self.assertEqual([m["message_id"] for m in first["context"]], [6, 7])
        self.assertEqual(first["matching_count"], 9)
        self.assertEqual(
            first["truncated_threads"][original(6).source_id]["remaining_hops"], 5
        )
        second = await self.reader.query(
            limit=2, cursor=first["next_cursor"], parent_hops=0
        )
        self.assertEqual([m["message_id"] for m in second["messages"]], [6, 7])
        self.assertEqual(second["context"], [])
        self.assertEqual(second["matching_count"], 9)
        formatted = self.formatter.result(first)
        rows = group_rows(formatted["message_blocks"][0]["text"])
        self.assertEqual([row[0] for row in rows], [6, 7, 8, 9])

    async def test_minimum_distance_from_all_seeds_prevents_false_cut(self) -> None:
        self.store.originals.return_value = self.originals[3:]
        result = await self.reader.query(limit=6, parent_hops=5)
        self.assertEqual(result["context"], [])
        self.assertEqual(result["truncated_threads"], {})
        self.assertEqual(result["matching_count"], 6)

    async def test_primary_author_text_and_time_filters_do_not_drop_ancestors(
        self,
    ) -> None:
        target = original(9, author_id=99, text="Requested message")
        self.store.originals.return_value = [*self.originals[:-1], target]
        result = await self.reader.query(
            text_contains="Requested",
            author_id=99,
            after=target.sent_at.isoformat() if target.sent_at else None,
            parent_hops=3,
        )
        self.assertEqual([m["message_id"] for m in result["messages"]], [9])
        self.assertEqual([m["message_id"] for m in result["context"]], [6, 7, 8])
        self.assertEqual({m["author_id"] for m in result["context"]}, {42})
        self.assertEqual(result["matching_count"], 1)
        self.store.originals.assert_awaited_once_with(self.access, chat=None)

    async def test_gaps_cycles_and_other_topic_roots_keep_distance_unknown(self) -> None:
        cases = [
            self.originals[2:],
            [original(1, reply_to_id=2), *self.originals[1:]],
            [original(2, topic_id=8), *(original(n, topic_id=7) for n in range(3, 10))],
        ]
        for number, messages in enumerate(cases):
            with self.subTest(case=number):
                self.store.originals.return_value = messages
                result = await asyncio.wait_for(
                    self.reader.query(
                        "read",
                        source_id=original(9).source_id,
                        parent_hops=5,
                        topic_id=7 if number == 2 else None,
                    ),
                    timeout=1,
                )
                cut = result["truncated_threads"][original(4).source_id]
                self.assertIsNone(cut["root_id"])
                self.assertIsNone(cut["remaining_hops"])
                formatted = self.formatter.result(result)
                metadata = cut_header(formatted["message_blocks"][0]["text"])["4"]
                self.assertIsNone(metadata["remaining_hops"])

    async def test_foreign_chats_cannot_supply_a_root_despite_numeric_id_collision(
        self,
    ) -> None:
        foreign = original(
            2, collection="telegram:-100456", chat_id=-100456, reply_to_id=None
        )
        self.store.originals.return_value = [foreign, *self.originals[2:]]
        private = CanonicalHistory(self.store, MemoryAccess(42, True))
        result = await private.query("read", source_id=original(9).source_id)
        cut = result["truncated_threads"][original(4).source_id]
        self.assertIsNone(cut["root_id"])
        self.assertIsNone(cut["remaining_hops"])
        self.assertEqual({m["chat_id"] for m in result["context"]}, {GROUP})

    async def test_cross_chat_cuts_and_parent_rows_keep_separate_identities(self) -> None:
        foreign = [
            original(n, collection="telegram:-100456", chat_id=-100456)
            for n in range(1, 10)
        ]
        self.store.originals.return_value = sorted(
            [*self.originals, *foreign], key=lambda message: message.order
        )
        private = CanonicalHistory(self.store, MemoryAccess(42, True))
        result = self.formatter.result(await private.query(limit=2, parent_hops=2))
        self.assertEqual(len(result["message_blocks"]), 2)
        for block in result["message_blocks"]:
            self.assertEqual([row[0] for row in group_rows(block["text"])], [7, 8, 9])
            cut = cut_header(block["text"])["7"]
            boundary = next(
                ref for ref, value in block["sources"].items() if value["message_id"] == 7
            )
            self.assertEqual(cut["boundary_source_id"], boundary)
            self.assertEqual(cut["remaining_hops"], 6)

    async def test_aggregations_do_not_expand_examples_and_invalid_depths_fail(
        self,
    ) -> None:
        for action in ("count", "authors", "chats"):
            with self.subTest(action=action):
                result = await self.reader.query(action, parent_hops=100)
                self.assertNotIn("context", result)
                self.assertNotIn("truncated_threads", result)
        self.assertEqual((await self.reader.query("count"))["count"], 9)
        for invalid in (-1, True, 1.5):
            with (
                self.subTest(invalid=invalid),
                self.assertRaisesRegex(ValueError, "parent_hops"),
            ):
                await self.reader.query(parent_hops=invalid)  # type: ignore[arg-type]
