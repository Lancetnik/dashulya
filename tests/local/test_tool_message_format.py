import asyncio
import copy
import json
from contextlib import AsyncExitStack
from dataclasses import replace
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from ag2 import Agent, MemoryStream
from ag2.events import DataInput, ModelRequest, TextInput, ToolCallEvent, ToolResultEvent
from ag2.testing import TestConfig
from dirty_equals import IsStr

from dashulya.ai.agents.factory import create_reply_agent
from dashulya.ai.dialogue import Dialogue
from dashulya.ai.message_format import MessageFormatter
from dashulya.domain.memory import (
    ChatHistory,
    HistoryReader,
    MemoryAccess,
    MemoryBackend,
    MemoryMessage,
    OriginalMessage,
    ReplyContext,
)
from dashulya.domain.messages import Reply, ReplyRequest
from dashulya.main import create_container
from dashulya.main.config import Settings
from tests.support import (
    TEST_API_KEY,
    MemoryOverride,
    ModelOverrides,
    group_records,
    group_rows,
    reply_json,
)


def original(number: int = 1, **changes: Any) -> OriginalMessage:
    return replace(
        OriginalMessage(
            collection="telegram:-100123",
            chat_id=-100123,
            message_id=number,
            text=f"Original {number}",
            sent_at=datetime(2026, 10, 25, 21, number, tzinfo=UTC),
            author_id=42,
            author="Alice",
            chat_title="Group",
            chat_username="group",
            topic_id=None,
            reply_to_id=None,
            edited_at=None,
            media_type=None,
            url=f"https://t.me/group/{number}",
            provenance="observed",
            username="alice",
            is_bot=False,
        ),
        **changes,
    )


def recall_result() -> dict[str, Any]:
    primary = original(
        2,
        reply_to_id=1,
        author_id=123,
        author="Dashulya",
        username="dashulya_bot",
        is_bot=True,
    )
    parent = original(1)
    return {
        "coverage": {"complete": False, "pending_blocks": 7},
        "hits": [
            {
                "kind": "message",
                "memory_id": primary.source_id,
                "memory": "DUPLICATE RETRIEVAL PAYLOAD",
                "score": 0.5,
                "sources": [primary.render()],
                "context": [parent.render()],
            },
            {
                "kind": "message",
                "memory_id": parent.source_id,
                "memory": "DUPLICATE RETRIEVAL PAYLOAD",
                "score": 0.4,
                "sources": [parent.render()],
                "context": [primary.render()],
            },
        ],
    }


@pytest.mark.usefixtures("_setup")
class TestToolMessageFormat:
    @pytest.fixture
    def _setup(self) -> None:
        self.formatter = MessageFormatter(bot_id=123)

    def test_interlocutor_follows_numeric_identity_across_chats_and_aliases(self) -> None:
        foreign = original(
            2,
            collection="telegram:-100456",
            chat_id=-100456,
            author="Alicia",
            username="new_alice",
        )
        namesake = original(3, author_id=99)
        raw = {"messages": [original().render(), foreign.render(), namesake.render()]}
        result = self.formatter.result(raw, interlocutor_id=42)
        first, second = result["message_blocks"]
        assert "User 42: Alice (@alice)" in first["text"]
        assert "99: Alice (@alice)" in first["text"]
        assert "42: Alicia (@new_alice)" in second["text"]
        assert "Interlocutor user id: 42" in first["text"]
        assert "Interlocutor user id: 42" in second["text"]
        assert "Interlocutor" not in str(self.formatter.result(raw))

    def test_search_deduplicates_rows_without_losing_hit_roles_or_source_metadata(
        self,
    ) -> None:
        raw = recall_result()
        baseline = copy.deepcopy(raw)
        result = self.formatter.result(raw)
        assert raw == baseline
        assert result["coverage"] == raw["coverage"]
        assert result["hits"][0]["sources"] == [original(2).source_id]
        assert result["hits"][0]["context"] == [original(1).source_id]
        assert result["hits"][0]["score"] == pytest.approx(0.5)
        block = result["message_blocks"][0]
        assert block["chat"] == "telegram:-100123"
        assert "messages:" in block["text"]
        assert "[id,sender,time,text,reply_to?,attachment?]" not in block["text"]
        assert group_rows(block["text"]) == [
            [1, 42, "00:01:00", "Original 1"],
            [2, 123, "00:02:00", "Original 2", 1],
        ]
        assert "42: Alice (@alice)" in block["text"]
        assert "User 123:" not in block["text"]
        assert group_records(block["text"])[1]["role"] == "assistant"
        assert "Europe/Moscow" not in block["text"]
        assert "26 October 2026 (Monday)" in block["text"]
        assert "DUPLICATE RETRIEVAL PAYLOAD" not in json.dumps(result)
        assert block["sources"][original(1).source_id] == {
            "message_id": 1,
            "url": "https://t.me/group/1",
            "provenance": "observed",
        }

    def test_all_history_actions_format_primary_context_and_nested_examples(
        self, subtests: pytest.Subtests
    ) -> None:
        self.formatter = MessageFormatter("Asia/Tokyo", bot_id=123)
        first, second = original().source_id, original(2).source_id
        cases: list[tuple[str, dict[str, Any], dict[str, Any]]] = [
            (
                "list",
                {
                    "messages": [original().render()],
                    "next_cursor": "opaque",
                    "matching_count": 9,
                },
                {"messages": [first], "next_cursor": "opaque", "matching_count": 9},
            ),
            (
                "read",
                {"messages": [original().render()], "context": [original(2).render()]},
                {"messages": [first], "context": [second]},
            ),
            (
                "count",
                {"count": 9, "examples": [original().render()]},
                {"count": 9, "examples": [first]},
            ),
            (
                "authors",
                {
                    "authors": [
                        {
                            "author_id": 42,
                            "message_count": 9,
                            "examples": [original().render()],
                        }
                    ],
                    "next_offset": 1,
                },
                {
                    "authors": [
                        {"author_id": 42, "message_count": 9, "examples": [first]}
                    ],
                    "next_offset": 1,
                },
            ),
            (
                "chats",
                {
                    "chats": [
                        {"chat": "telegram:-100123", "examples": [original().render()]}
                    ]
                },
                {"chats": [{"chat": "telegram:-100123", "examples": [first]}]},
            ),
        ]
        for action, body, expected in cases:
            with subtests.test(action=action):
                raw = {"action": action, "filters": {"author_id": 42}, **body}
                result = self.formatter.result(raw)
                (block,) = result.pop("message_blocks")
                assert result == {
                    "action": action,
                    "filters": {"author_id": 42},
                    **expected,
                }
                numbers = [1, 2] if action == "read" else [1]
                assert block == {
                    "chat": "telegram:-100123",
                    "chat_id": -100123,
                    "titles": ["Group"],
                    "usernames": ["group"],
                    "text": IsStr,
                    "sources": {
                        original(number).source_id: {
                            "message_id": number,
                            "url": f"https://t.me/group/{number}",
                            "provenance": "observed",
                        }
                        for number in numbers
                    },
                }
                assert group_rows(block["text"]) == [
                    [number, 42, f"06:0{number}:00", f"Original {number}"]
                    for number in numbers
                ]

    def test_minimal_source_has_a_reference_without_invented_chat_identity(self) -> None:
        result = self.formatter.result({
            "messages": [
                {
                    "message_id": 17,
                    "text": "Original without collection metadata",
                    "author": "Alice",
                    "sent_at": "2026-10-25T21:00:00+00:00",
                }
            ]
        })
        assert result == {
            "messages": ["message:17"],
            "message_blocks": [
                {
                    "chat": "unknown",
                    "chat_id": None,
                    "titles": [],
                    "usernames": [],
                    "text": IsStr,
                    "sources": {"message:17": {"message_id": 17}},
                }
            ],
        }
        assert group_rows(result["message_blocks"][0]["text"]) == [
            [17, "unknown_17", "00:00:00", "Original without collection metadata"]
        ]

    def test_cross_chat_message_id_collisions_stay_in_separate_blocks(self) -> None:
        foreign = original(
            collection="telegram:-100456",
            chat_id=-100456,
            text="Foreign",
            chat_title="Other",
            chat_username=None,
            url=None,
        )
        private = original(collection="telegram:42", chat_id=42, text="Private", url=None)
        result = self.formatter.result({
            "messages": [original().render(), foreign.render(), private.render()]
        })
        assert len(set(result["messages"])) == 3
        assert len(result["message_blocks"]) == 3
        for block, body, template in zip(
            result["message_blocks"],
            ["Original 1", "Foreign", "Private"],
            ["https://t.me/group/{message_id}", "https://t.me/c/456/{message_id}", None],
            strict=True,
        ):
            assert group_rows(block["text"])[0][3] == body
            header = next(
                line for line in block["text"].splitlines() if line.startswith("chat: ")
            )
            chat = json.loads(header.removeprefix("chat: "))
            assert chat["id"] == block["chat_id"]
            assert chat["message_url_template"] == template
        assert "url" not in result["message_blocks"][2]["sources"][private.source_id]

    def test_unbound_archive_preserves_explicit_source_url_without_inventing_a_template(
        self,
    ) -> None:
        source = original(collection="archive:group", chat_id=None)
        result = self.formatter.result({"messages": [source.render()]})
        block = result["message_blocks"][0]
        header = next(
            line for line in block["text"].splitlines() if line.startswith("chat: ")
        )
        chat = json.loads(header.removeprefix("chat: "))
        assert chat["message_url_template"] is None
        assert block["sources"][source.source_id]["url"] == source.url

    def test_excerpts_preserve_offsets_length_edits_topics_and_attachments(self) -> None:
        source = original(
            text='A\n"quoted"\n😀 B',
            reply_to_id=9,
            media_type="photo",
            topic_id=7,
            edited_at="2026-10-26T01:00:00Z",
            archive_key="archive:1",
        )
        result = self.formatter.result({
            "messages": [source.render(offset=2, max_chars=5)]
        })
        block = result["message_blocks"][0]
        row = group_rows(block["text"])[0]
        assert row == [1, 42, "00:01:00", source.text[2:7], 9, "photo"]
        metadata = block["sources"][source.source_id]
        assert metadata["offset"] == 2
        assert metadata["text_length"] == len(source.text)
        assert metadata["next_offset"] == 7
        assert metadata["truncated"]
        assert metadata["topic_id"] == 7
        assert metadata["edited_at"] == source.edited_at
        assert metadata["archive_key"] == source.archive_key

    def test_longest_overlapping_preview_is_used_even_when_primary_text_budget_is_zero(
        self,
        subtests: pytest.Subtests,
    ) -> None:
        source = original(text="Full original")
        for sources, context in (
            (source.render(max_chars=0), source.render()),
            (source.render(), source.render(max_chars=0)),
        ):
            with subtests.test(primary_text=sources["text"]):
                result = self.formatter.result({
                    "sources": [sources],
                    "context": [context],
                })
                assert result["sources"] == result["context"]
                block = result["message_blocks"][0]
                assert group_rows(block["text"])[0][3] == source.text
                assert "truncated" not in block["sources"][source.source_id]

    def test_unknown_timestamps_and_senders_are_explicit_and_dates_sort_by_instant(
        self,
    ) -> None:
        missing = original(sent_at=None, author_id=None, username=None, is_bot=None)
        later = original(2).render()
        earlier = original(3).render()
        earlier["sent_at"] = "2026-10-26T00:01:00+04:00"
        result = self.formatter.result({"messages": [later, missing.render(), earlier]})
        block = result["message_blocks"][0]
        rows = group_rows(block["text"])
        assert [row[0] for row in rows] == [1, 3, 2]
        assert rows[0][1:3] == ["unknown_1", None]
        assert "Unknown send date" in block["text"]
        assert "25 October 2026 (Sunday)" in block["text"]
        assert "26 October 2026 (Monday)" in block["text"]

    def test_empty_results_and_derived_memories_keep_their_meaning(self) -> None:
        empty = {"hits": [], "messages": [], "count": 0, "coverage": {"complete": False}}
        assert self.formatter.result(empty) == empty
        result = self.formatter.result({
            "hits": [
                {
                    "kind": "summary",
                    "memory": "A derived conclusion",
                    "sources": [original().render()],
                }
            ]
        })
        assert result["hits"][0]["memory"] == "A derived conclusion"


@pytest.mark.usefixtures("_setup")
class TestToolMessageRoundtrip:
    @pytest.fixture
    def _setup(self, cleanup: AsyncExitStack) -> None:
        self.cleanup = cleanup

    async def test_current_group_sender_is_marked_in_the_actual_agent_input(self) -> None:
        agent = create_reply_agent(TestConfig(reply_json("Done")), tools=[])
        current = MemoryMessage(
            -100123,
            3,
            "Question",
            datetime(2026, 10, 27, tzinfo=UTC),
            99,
            "Alice",
            "Group",
        )
        previous = replace(current, message_id=1, author_id=42, text="Earlier")
        own_previous = replace(current, message_id=2, text="My earlier message")
        history = AsyncMock(spec=ChatHistory)
        history.context.return_value = ReplyContext((previous, own_previous, current))
        dialogue = Dialogue(agent, history=history, bot_id=123)
        with patch.object(agent, "ask", wraps=agent.ask) as ask:
            await dialogue.reply(
                ReplyRequest(
                    (-100123, None),
                    current.text,
                    user_id=99,
                    user_name="Alice",
                    sent_at=current.sent_at,
                    message_id=3,
                )
            )
        stream = ask.call_args.kwargs["stream"]
        rendered = "\n".join(
            part.content
            for event in await stream.history.get_events()
            if isinstance(event, ModelRequest)
            for part in event.parts
            if isinstance(part, TextInput)
        )
        assert "User 99: Alice" in rendered
        assert "Interlocutor user id: 99" in rendered
        assert "42: Alice" in rendered
        assert "Interlocutor user id: 42" not in rendered
        recent = await dialogue.recent_context((-100123, None), through_message_id=3)
        assert "Interlocutor user id: 99" in str(recent)
        assert "Interlocutor user id: 42" not in str(recent)

    async def test_concurrent_tool_sessions_do_not_share_interlocutor_markers(
        self,
    ) -> None:
        sources = [original().render(), original(3, author_id=99).render()]
        backend = AsyncMock(spec=MemoryBackend)
        backend.search.return_value = {
            "hits": [{"kind": "message", "sources": sources, "context": []}]
        }
        reader = AsyncMock(spec=HistoryReader)
        reader.query.return_value = {"messages": sources}
        container = create_container(
            Settings(bot_token="123:TEST", openai_api_key=TEST_API_KEY, owner_user_id=42),
            MemoryOverride(backend, reader=reader),
            ModelOverrides(
                reply=TestConfig(
                    ToolCallEvent(
                        "search_memory", arguments='{"query":"original","user_id":123}'
                    ),
                    ToolCallEvent(
                        "chat_history", arguments='{"action":"list","user_id":123}'
                    ),
                    reply_json("Done"),
                ),
                decision=TestConfig("false"),
            ),
        )
        self.cleanup.push_async_callback(container.close)
        agent = await container.get(Agent[Reply])
        sessions = [(42, MemoryStream()), (99, MemoryStream())]
        await asyncio.gather(
            *(
                agent.ask(
                    "Read originals",
                    stream=stream,
                    variables={"user_id": user_id},
                    dependencies={"memory_access": MemoryAccess(42, True)},
                )
                for user_id, stream in sessions
            )
        )
        for user_id, stream in sessions:
            events = [
                event
                for event in await stream.history.get_events()
                if isinstance(event, ToolResultEvent)
            ]
            assert len(events) == 2
            for event in events:
                data = next(
                    part.data
                    for part in event.result.parts
                    if isinstance(part, DataInput)
                )
                assert isinstance(data, dict)
                blocks = data["message_blocks"]
                assert isinstance(blocks, list)
                block = blocks[0]
                assert isinstance(block, dict)
                header = block["text"]
                assert isinstance(header, str)
                assert f"Interlocutor user id: {user_id}" in header
                assert f"User {user_id}: Alice (@alice)" in header
                other = 99 if user_id == 42 else 42
                assert f"{other}: Alice (@alice)" in header

    async def test_injected_tools_share_formatter_and_forward_arguments(self) -> None:
        backend = AsyncMock(spec=MemoryBackend)
        backend.search.return_value = recall_result()
        channel = original(
            3, author_id=200, author="Channel", username="channel"
        ).render()
        backend.search.return_value["hits"][0]["sources"].append(channel)
        reader = AsyncMock(spec=HistoryReader)
        reader.query.return_value = {
            "messages": [original().render(), channel],
            "next_cursor": "opaque",
            "coverage": {"complete": False},
        }
        settings = Settings(
            bot_token="123:TEST",
            openai_api_key=TEST_API_KEY,
            owner_user_id=42,
            owner_author_ids=(200, -100200),
        )
        container = create_container(
            settings,
            MemoryOverride(backend, reader=reader),
            ModelOverrides(
                reply=TestConfig(
                    ToolCallEvent("search_memory", arguments='{"query":"original"}'),
                    ToolCallEvent(
                        "chat_history", arguments='{"action":"list","parent_hops":3}'
                    ),
                    reply_json("Done"),
                ),
                decision=TestConfig("false"),
            ),
        )
        self.cleanup.push_async_callback(container.close)
        formatter = await container.get(MessageFormatter)
        agent = await container.get(Agent[Reply])
        stream = MemoryStream()
        await agent.ask(
            "Read sources",
            stream=stream,
            variables={"user_id": 42},
            dependencies={"memory_access": MemoryAccess(42, True)},
        )
        events = [
            event
            for event in await stream.history.get_events()
            if isinstance(event, ToolResultEvent)
        ]
        assert len(events) == 2
        assert reader.query.await_args.kwargs["parent_hops"] == 3
        for event, raw in zip(
            events, [backend.search.return_value, reader.query.return_value], strict=True
        ):
            data = next(
                part.data for part in event.result.parts if isinstance(part, DataInput)
            )
            assert data == formatter.result(raw, interlocutor_id=42)
            assert "author_user=42" in str(data)
            assert "Interlocutor user id: 42" in str(data)
        backend.search.assert_awaited_once_with(
            "original",
            memory_id=None,
            chat=None,
            author_id=None,
            after=None,
            before=None,
            limit=8,
        )
        reader.query.assert_awaited_once()
