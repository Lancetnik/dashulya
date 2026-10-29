import asyncio
from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

import pytest
from ag2.events import ModelRequest, TextInput
from ag2.testing import TestConfig

from dashulya.ai.agents.factory import create_reply_agent
from dashulya.ai.dialogue import Dialogue
from dashulya.ai.message_format import MessageFormatter
from dashulya.database import Database
from dashulya.domain.memory import MemoryAccess, MemoryMessage
from dashulya.memory.history import CanonicalHistory
from dashulya.memory.store import MessageStore
from tests.support import group_records, group_rows, reply_json

GROUP = -100123
NOW = datetime(2026, 10, 25, 12, tzinfo=UTC)


@pytest.mark.usefixtures("_setup")
class TestContextTruncation:
    @pytest.fixture
    async def _setup(self, database: Database) -> None:
        self.database = database
        self.store = MessageStore(self.database)

    async def chain(
        self, start: int = 1, end: int = 9, *, topic: int | None = None
    ) -> None:
        for number in range(start, end + 1):
            await self.store.record(
                MemoryMessage(
                    GROUP,
                    number,
                    f"Message {number}",
                    NOW,
                    42,
                    "Alice",
                    "Group",
                    thread_id=topic,
                    reply_to_id=number - 1 if number > 1 else None,
                )
            )

    async def test_cut_keeps_only_five_parents_but_finds_and_reads_the_true_saved_root(
        self,
    ) -> None:
        await self.chain()
        context = await self.store.context((GROUP, None), 9, limit=1, parent_hops=5)
        assert [message.message_id for message in context.messages] == [4, 5, 6, 7, 8, 9]
        assert len(context.truncated_threads) == 1
        cut = context.truncated_threads[0]
        assert (cut.boundary_id, cut.root_id) == (4, 1)
        assert cut.remaining_hops == 3
        result = await CanonicalHistory(self.store, MemoryAccess(GROUP, False)).query(
            "read", source_id=cut.source_id
        )
        assert result["messages"][0]["message_id"] == 1
        assert result["messages"][0]["text"] == "Message 1"

    async def test_chain_ending_exactly_at_limit_and_union_covered_links_are_not_cut(
        self,
        subtests: pytest.Subtests,
    ) -> None:
        await self.chain(end=6)
        for limit in [1, 3, 20]:
            with subtests.test(limit=limit):
                context = await self.store.context(
                    (GROUP, None), 6, limit=limit, parent_hops=5
                )
                assert context.truncated_threads == ()

    async def test_omitted_hops_count_edges_and_fetch_the_entire_tail_once(self) -> None:
        identifiers = [10, 40, 90, 100, 300, 900, 1000]
        parent = None
        for number in identifiers:
            await self.store.record(
                MemoryMessage(
                    GROUP,
                    number,
                    f"Message {number}",
                    NOW,
                    42,
                    "Alice",
                    "Group",
                    reply_to_id=parent,
                )
            )
            parent = number
        context = await self.store.context((GROUP, None), 1000, limit=1, parent_hops=2)
        rendered = MessageFormatter().context(
            context.messages, truncated_threads=context.truncated_threads
        )
        cut = next(
            record["truncated_thread"]
            for item in rendered
            for record in group_records(item["text"])
            if "truncated_thread" in record
        )
        assert cut["root_id"] == 10
        assert cut["remaining_hops"] == 4
        result = await CanonicalHistory(self.store, MemoryAccess(GROUP, False)).query(
            "read", source_id=cut["boundary_source_id"], parent_hops=cut["remaining_hops"]
        )
        assert [m["message_id"] for m in result["messages"]] == [300]
        assert [m["message_id"] for m in result["context"]] == [10, 40, 90, 100]
        assert result["truncated_threads"] == {}

    async def test_missing_deeper_history_has_unknown_root_instead_of_first_available_id(
        self,
    ) -> None:
        await self.chain(start=3)
        context = await self.store.context((GROUP, None), 9, limit=1, parent_hops=5)
        cut = context.truncated_threads[0]
        assert cut.boundary_id == 4
        assert cut.root_id is None
        assert cut.source_id is None
        assert cut.remaining_hops is None

    async def test_other_chat_and_other_forum_topic_cannot_supply_a_root(self) -> None:
        await self.chain(start=3, topic=7)
        await self.store.record(
            MemoryMessage(
                -100456, 2, "Foreign root", NOW, 99, "Other", "Other", thread_id=7
            )
        )
        await self.store.record(
            MemoryMessage(GROUP, 2, "Other topic", NOW, 99, "Other", "Group", thread_id=8)
        )
        context = await self.store.context((GROUP, 7), 9, limit=1, parent_hops=5)
        assert context.truncated_threads[0].root_id is None
        assert all(message.thread_id == 7 for message in context.messages)

    async def test_invalid_cycle_does_not_loop_or_invent_a_root(self) -> None:
        await self.chain()
        await self.store.record(
            MemoryMessage(
                GROUP,
                1,
                "Malformed",
                NOW,
                42,
                "Alice",
                "Group",
                reply_to_id=2,
                update_id=10,
            )
        )
        context = await asyncio.wait_for(
            self.store.context((GROUP, None), 9, limit=1, parent_hops=5), 2
        )
        assert context.truncated_threads[0].root_id is None

    async def test_minimum_distance_from_any_seed_avoids_false_truncation_on_a_missing_link(
        self,
    ) -> None:
        await self.chain(start=4)
        context = await self.store.context((GROUP, None), 9, limit=6, parent_hops=5)
        assert context.truncated_threads == ()

    async def test_root_metadata_reaches_reply_and_classifier_without_becoming_a_reply_target(
        self,
    ) -> None:
        await self.chain(end=30)
        agent = create_reply_agent(TestConfig(reply_json("Answer", 1)), tools=[])
        dialogue = Dialogue(agent, history=self.store, bot_id=123)
        agent_ask = AsyncMock(wraps=agent.ask)
        with patch.object(agent, "ask", agent_ask):
            result = await dialogue.reply(
                (GROUP, None),
                "Current request",
                user_id=42,
                user_name="Alice",
                sent_at=NOW,
                group_context={"message_id": 30},
            )
        assert result.reply_to_message_id is None
        stream = agent_ask.call_args.kwargs["stream"]
        rendered = "\n".join(
            part.content
            for event in await stream.history.get_events()
            if isinstance(event, ModelRequest)
            for part in event.parts
            if isinstance(part, TextInput)
        )
        assert "truncated_thread" in rendered
        recent = await dialogue.recent_context((GROUP, None), through_message_id=30)
        cut = next(
            record["truncated_thread"]
            for item in recent
            for record in group_records(item["text"])
            if record["id"] == 6
        )
        assert cut["root_id"] == 1
        assert cut["remaining_hops"] == 5
        assert 1 not in [row[0] for item in recent for row in group_rows(item["text"])]
