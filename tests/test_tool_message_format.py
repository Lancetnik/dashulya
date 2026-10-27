import asyncio
import copy
import json
import unittest
from dataclasses import replace
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, patch

from ag2 import Agent, MemoryStream
from ag2.events import DataInput, ModelRequest, TextInput, ToolCallEvent, ToolResultEvent
from ag2.testing import TestConfig

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
from dashulya.domain.messages import Reply
from dashulya.main import create_container
from dashulya.main.config import Settings
from tests.support import (
    TEST_API_KEY,
    MemoryOverride,
    ModelOverrides,
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


class ToolMessageFormatTests(unittest.TestCase):
    def setUp(self) -> None:
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
        self.assertIn("42: Alice (@alice, interlocutor)", first["text"])
        self.assertIn("99: Alice (@alice)", first["text"])
        self.assertIn("42: Alicia (@new_alice, interlocutor)", second["text"])
        self.assertNotIn("interlocutor", str(self.formatter.result(raw)))

    def test_search_deduplicates_rows_without_losing_hit_roles_or_source_metadata(
        self,
    ) -> None:
        raw = recall_result()
        baseline = copy.deepcopy(raw)
        result = self.formatter.result(raw)
        self.assertEqual(raw, baseline)
        self.assertEqual(result["coverage"], raw["coverage"])
        self.assertEqual(result["hits"][0]["sources"], [original(2).source_id])
        self.assertEqual(result["hits"][0]["context"], [original(1).source_id])
        self.assertEqual(result["hits"][0]["score"], 0.5)
        block = result["message_blocks"][0]
        self.assertEqual(block["chat"], "telegram:-100123")
        self.assertEqual(
            group_rows(block["text"]),
            [
                [1, 42, "00:01:00", "Original 1"],
                [2, 123, "00:02:00", "Original 2", 1],
            ],
        )
        self.assertIn("42: Alice (@alice)", block["text"])
        self.assertIn("123: Dashulya (@dashulya_bot, bot, self)", block["text"])
        self.assertEqual(block["text"].count("Europe/Moscow"), 1)
        self.assertIn("26 October 2026 (Monday)", block["text"])
        self.assertNotIn("DUPLICATE RETRIEVAL PAYLOAD", json.dumps(result))
        self.assertEqual(
            block["sources"][original(1).source_id],
            {
                "message_id": 1,
                "url": "https://t.me/group/1",
                "provenance": "observed",
            },
        )

    def test_all_history_actions_format_primary_context_and_nested_examples(self) -> None:
        cases: list[tuple[str, dict[str, Any]]] = [
            (
                "list",
                {
                    "messages": [original().render()],
                    "next_cursor": "opaque",
                    "matching_count": 9,
                },
            ),
            (
                "read",
                {"messages": [original().render()], "context": [original(2).render()]},
            ),
            ("count", {"count": 9, "examples": [original().render()]}),
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
            ),
            (
                "chats",
                {
                    "chats": [
                        {"chat": "telegram:-100123", "examples": [original().render()]}
                    ]
                },
            ),
        ]
        for action, body in cases:
            with self.subTest(action=action):
                raw = {"action": action, "filters": {"author_id": 42}, **body}
                result = self.formatter.result(raw)
                block = result.pop("message_blocks")[0]
                self.assertTrue(group_rows(block["text"]))
                self.assertNotIn('"text":', json.dumps(result))
                self.assertEqual(result["filters"], raw["filters"])
                self.assertIn(original().source_id, str(result))
                for key in ["count", "next_cursor", "next_offset", "matching_count"]:
                    if key in raw:
                        self.assertEqual(result[key], raw[key])

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
        self.assertEqual(len(set(result["messages"])), 3)
        self.assertEqual(len(result["message_blocks"]), 3)
        for block, body, template in zip(
            result["message_blocks"],
            ["Original 1", "Foreign", "Private"],
            ["https://t.me/group/{message_id}", "https://t.me/c/456/{message_id}", None],
            strict=True,
        ):
            self.assertEqual(group_rows(block["text"])[0][3], body)
            header = next(
                line for line in block["text"].splitlines() if line.startswith("chat: ")
            )
            chat = json.loads(header.removeprefix("chat: "))
            self.assertEqual(chat["id"], block["chat_id"])
            self.assertEqual(chat["message_url_template"], template)
        self.assertNotIn("url", result["message_blocks"][2]["sources"][private.source_id])

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
        self.assertIsNone(chat["message_url_template"])
        self.assertEqual(block["sources"][source.source_id]["url"], source.url)

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
        self.assertEqual(row, [1, 42, "00:01:00", source.text[2:7], 9, "photo"])
        metadata = block["sources"][source.source_id]
        self.assertEqual(metadata["offset"], 2)
        self.assertEqual(metadata["text_length"], len(source.text))
        self.assertEqual(metadata["next_offset"], 7)
        self.assertTrue(metadata["truncated"])
        self.assertEqual(metadata["topic_id"], 7)
        self.assertEqual(metadata["edited_at"], source.edited_at)
        self.assertEqual(metadata["archive_key"], source.archive_key)

    def test_longest_overlapping_preview_is_used_even_when_primary_text_budget_is_zero(
        self,
    ) -> None:
        source = original(text="Full original")
        raw = {"sources": [source.render(max_chars=0)], "context": [source.render()]}
        result = self.formatter.result(raw)
        self.assertEqual(result["sources"], result["context"])
        block = result["message_blocks"][0]
        self.assertEqual(group_rows(block["text"])[0][3], source.text)
        self.assertNotIn("truncated", block["sources"][source.source_id])

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
        self.assertEqual([row[0] for row in rows], [1, 3, 2])
        self.assertEqual(rows[0][1:3], ["unknown_1", None])
        self.assertIn("Unknown send date", block["text"])
        self.assertIn("25 October 2026 (Sunday)", block["text"])
        self.assertIn("26 October 2026 (Monday)", block["text"])

    def test_empty_results_and_derived_memories_keep_their_meaning(self) -> None:
        empty = {"hits": [], "messages": [], "count": 0, "coverage": {"complete": False}}
        self.assertEqual(self.formatter.result(empty), empty)
        result = self.formatter.result({
            "hits": [
                {
                    "kind": "summary",
                    "memory": "A derived conclusion",
                    "sources": [original().render()],
                }
            ]
        })
        self.assertEqual(result["hits"][0]["memory"], "A derived conclusion")


class ToolMessageRoundtripTests(unittest.IsolatedAsyncioTestCase):
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
                (-100123, None),
                current.text,
                user_id=99,
                user_name="Alice",
                sent_at=current.sent_at,
                group_context={"message_id": 3},
            )
        stream = ask.call_args.kwargs["stream"]
        rendered = "\n".join(
            part.content
            for event in await stream.history.get_events()
            if isinstance(event, ModelRequest)
            for part in event.parts
            if isinstance(part, TextInput)
        )
        self.assertIn("99: Alice (interlocutor)", rendered)
        self.assertIn("42: Alice", rendered)
        self.assertNotIn("42: Alice (interlocutor)", rendered)
        recent = await dialogue.recent_context((-100123, None), through_message_id=3)
        self.assertIn("99: Alice (interlocutor)", str(recent))
        self.assertNotIn("42: Alice (interlocutor)", str(recent))

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
                    ToolCallEvent("search_memory", arguments='{"query":"original"}'),
                    ToolCallEvent("chat_history", arguments='{"action":"list"}'),
                    reply_json("Done"),
                ),
                decision=TestConfig("false"),
            ),
        )
        self.addAsyncCleanup(container.close)
        agent = await container.get(Agent[Reply])
        sessions = [(42, MemoryStream()), (99, MemoryStream())]
        await asyncio.gather(
            *(
                agent.ask(
                    "Read originals",
                    stream=stream,
                    variables={"user_id": user_id},
                    dependencies={MemoryAccess: MemoryAccess(42, True)},
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
            self.assertEqual(len(events), 2)
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
                self.assertIn(f"{user_id}: Alice (@alice, interlocutor)", header)
                other = 99 if user_id == 42 else 42
                self.assertIn(f"{other}: Alice (@alice)", header)

    async def test_injected_tools_share_formatter_without_automatic_recall(self) -> None:
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
        self.addAsyncCleanup(container.close)
        formatter = await container.get(MessageFormatter)
        agent = await container.get(Agent[Reply])
        stream = MemoryStream()
        await agent.ask(
            "Read sources",
            stream=stream,
            variables={"user_id": 42},
            dependencies={MemoryAccess: MemoryAccess(42, True)},
        )
        events = [
            event
            for event in await stream.history.get_events()
            if isinstance(event, ToolResultEvent)
        ]
        self.assertEqual(len(events), 2)
        self.assertEqual(reader.query.await_args.kwargs["parent_hops"], 3)
        for event, raw in zip(
            events, [backend.search.return_value, reader.query.return_value], strict=True
        ):
            data = next(
                part.data for part in event.result.parts if isinstance(part, DataInput)
            )
            self.assertEqual(data, formatter.result(raw, interlocutor_id=42))
            self.assertIn("author_user=42, interlocutor", str(data))
        dialogue = await container.get(Dialogue)
        with patch.object(agent, "ask", wraps=agent.ask) as ask:
            answer = await dialogue.reply(
                (42, None),
                "original",
                user_id=42,
                user_name="Owner",
                sent_at=datetime(2026, 10, 27, tzinfo=UTC),
            )
        self.assertEqual(answer.text, "Done")
        message = ask.call_args.args[0]
        self.assertNotIn("Initial recall", message)
        self.assertEqual(backend.search.await_count, 2)
        self.assertIn("Telegram user ID: 42; interlocutor", message)
