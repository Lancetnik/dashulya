import asyncio
import unittest
from datetime import UTC, datetime, timedelta, timezone
from typing import Any, cast
from unittest.mock import AsyncMock, patch

from ag2 import Agent, MemoryStream, ResponseSchema
from ag2.agent import AgentReply
from ag2.events import ModelRequest, ModelResponse, TextInput
from ag2.testing import TestConfig, TrackingConfig

from dashulya.ai.agents.factory import create_reply_agent
from dashulya.ai.dialogue import Dialogue
from dashulya.domain.memory import ChatHistory, MemoryMessage
from dashulya.domain.messages import ConversationKey, Reply
from dashulya.main import create_container
from dashulya.main.config import Settings
from tests.support import TEST_API_KEY, MemoryOverride, ModelOverrides, reply_json

SENT_AT = datetime(2026, 10, 25, 12, 0, tzinfo=UTC)


class DialogueTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.config = TrackingConfig(TestConfig(reply_json("AI answer")))
        self.settings = Settings(
            bot_token="123:TEST",
            openai_api_key=TEST_API_KEY,
            owner_user_id=42,
        )
        self.history = AsyncMock(spec=ChatHistory)
        self.history.context.return_value = []
        self.container = create_container(
            self.settings,
            MemoryOverride(history=self.history),
            ModelOverrides(reply=self.config, decision=TestConfig("false")),
        )
        self.addAsyncCleanup(self.container.close)
        self.agent = await self.container.get(Agent[Reply])
        self.real_ask = self.agent.ask
        self.ask = AsyncMock(wraps=self.real_ask)
        ask_patch = patch.object(self.agent, "ask", self.ask)
        ask_patch.start()
        self.addCleanup(ask_patch.stop)
        self.dialogue = await self.container.get(Dialogue)

    async def reply(
        self,
        key: ConversationKey,
        text: str = "Hello",
        user_id: int = 42,
        sent_at: datetime = SENT_AT,
    ) -> str:
        return (
            await self.dialogue.reply(
                key,
                text,
                user_id=user_id,
                user_name="Anna",
                sent_at=sent_at,
            )
        ).text

    def last_stream(self) -> MemoryStream:
        return cast("MemoryStream", self.ask.call_args.kwargs["stream"])

    async def requests(self, stream: MemoryStream) -> list[str]:
        return [
            part.content
            for event in await stream.history.get_events()
            if isinstance(event, ModelRequest)
            for part in event.parts
            if isinstance(part, TextInput)
        ]

    async def test_real_ag2_turns_keep_history_and_sender_identity(self) -> None:
        self.assertEqual(await self.reply((42, None), "Remember Anna"), "AI answer")
        stream = self.last_stream()
        await self.reply((42, None), "Who am I?")
        self.assertIs(self.last_stream(), stream)
        requests = await self.requests(stream)
        self.assertEqual(len(requests), 2)
        self.assertIn("Remember Anna", requests[0])
        self.assertIn("Who am I?", requests[1])
        self.assertIn("Anna (Telegram user ID: 42)", requests[0])
        self.assertEqual(self.config.calls[-1].variables, {"user_id": 42, "chat_id": 42})
        responses = [
            e for e in await stream.history.get_events() if isinstance(e, ModelResponse)
        ]
        self.assertEqual([e.content for e in responses], [reply_json("AI answer")] * 2)

    async def test_each_message_keeps_its_send_time_in_ag2_history(self) -> None:
        first = datetime(2026, 10, 24, 20, 59, tzinfo=UTC)
        second = datetime(2026, 10, 25, 3, 5, tzinfo=timezone(timedelta(hours=3)))
        await self.reply((42, None), "Yesterday", sent_at=first)
        await self.reply((42, None), "Today", sent_at=second)
        requests = await self.requests(self.last_stream())
        self.assertTrue(all("Initial recall:" in request for request in requests))
        self.assertEqual(
            [
                request.split("\n\nCurrent request (original):\n", 1)[1]
                for request in requests
            ],
            [
                "Sent at: 24 October 2026 (Saturday) 23:59:00\nTimezone: Europe/Moscow (UTC+03:00)\nAnna (Telegram user ID: 42):\nYesterday",
                "Sent at: 25 October 2026 (Sunday) 03:05:00\nTimezone: Europe/Moscow (UTC+03:00)\nAnna (Telegram user ID: 42):\nToday",
            ],
        )

    async def test_reported_sunday_is_explicit_in_local_message_context(self) -> None:
        await self.reply(
            (42, None), "Hello", sent_at=datetime(2026, 10, 25, 9, 16, 50, tzinfo=UTC)
        )
        self.assertIn(
            "Sent at: 25 October 2026 (Sunday) 12:16:50\nTimezone: Europe/Moscow (UTC+03:00)",
            (await self.requests(self.last_stream()))[0],
        )

    async def test_local_date_can_differ_from_utc_date(self) -> None:
        await self.reply((42, None), sent_at=datetime(2026, 10, 24, 22, 0, tzinfo=UTC))
        self.assertIn(
            "25 October 2026 (Sunday) 01:00:00",
            (await self.requests(self.last_stream()))[0],
        )

    async def test_configured_timezone_distinguishes_repeated_dst_hour(self) -> None:
        self.dialogue = Dialogue(self.agent, timezone="America/New_York")
        await self.reply((42, None), sent_at=datetime(2026, 11, 1, 5, 30, tzinfo=UTC))
        await self.reply((42, None), sent_at=datetime(2026, 11, 1, 6, 30, tzinfo=UTC))
        requests = await self.requests(self.last_stream())
        for request in requests:
            self.assertIn("01 November 2026 (Sunday) 01:30:00", request)
        self.assertIn("America/New_York (UTC-04:00)", requests[0])
        self.assertIn("America/New_York (UTC-05:00)", requests[1])

    async def test_send_time_requires_explicit_timezone(self) -> None:
        with self.assertRaisesRegex(ValueError, "timezone"):
            await self.reply(
                (42, None),
                sent_at=datetime(2026, 10, 25, 12, 0, tzinfo=None),  # ruff: ignore[call-datetime-without-tzinfo] -- Exercise rejection of naive timestamps.
            )
        self.ask.assert_not_awaited()

    async def test_chats_and_topics_have_isolated_histories(self) -> None:
        streams = []
        for key in ((42, None), (43, None), (-123, 1), (-123, 2)):
            await self.reply(key, str(key))
            streams.append(self.last_stream())
        self.assertEqual(len({stream.id for stream in streams}), 4)
        for stream in streams:
            self.assertEqual(len(await self.requests(stream)), 1)

    async def test_clear_resets_only_current_conversation(self) -> None:
        await self.reply((42, None), "Old secret")
        old = self.last_stream()
        await self.reply((43, None), "Other chat")
        other = self.last_stream()
        await self.dialogue.clear((42, None))
        await self.reply((42, None), "Fresh")
        self.assertIsNot(self.last_stream(), old)
        self.assertEqual(len(await self.requests(self.last_stream())), 1)
        await self.reply((43, None))
        self.assertIs(self.last_stream(), other)
        await self.dialogue.clear((999, None))

    async def test_clear_waits_for_active_reply_and_other_chats_run(self) -> None:
        started, release = asyncio.Event(), asyncio.Event()

        async def blocked_ask(text: str, **kwargs: Any) -> AgentReply[str, Reply]:
            if "Blocked" in text:
                started.set()
                await release.wait()
            return cast("AgentReply[str, Reply]", await self.real_ask(text, **kwargs))

        self.ask.side_effect = blocked_ask
        task = asyncio.create_task(self.reply((42, None), "Blocked"))
        await started.wait()
        clearing = asyncio.create_task(self.dialogue.clear((42, None)))
        try:
            await asyncio.wait_for(self.reply((43, None)), timeout=2)
            self.assertFalse(clearing.done())
        finally:
            release.set()
            await task
            await clearing
        await self.reply((42, None), "Fresh")
        fresh = self.last_stream()
        self.assertEqual(
            [
                request.split("\n\nCurrent request (original):\n", 1)[1]
                for request in await self.requests(fresh)
            ],
            [
                "Sent at: 25 October 2026 (Sunday) 15:00:00\nTimezone: Europe/Moscow (UTC+03:00)\nAnna (Telegram user ID: 42):\nFresh"
            ],
        )

    async def test_failed_turn_can_be_retried_and_empty_reply_has_fallback(self) -> None:
        self.ask.side_effect = TimeoutError("secret provider details")
        with self.assertRaises(TimeoutError):
            await self.reply((42, None))
        self.ask.side_effect = None
        self.assertEqual(await self.reply((42, None)), "AI answer")
        empty = Dialogue(
            Agent(
                "empty",
                config=TestConfig(ModelResponse()),
                response_schema=ResponseSchema(Reply),
            )
        )
        self.assertIn(
            "couldn't generate",
            (
                await empty.reply(
                    (42, None),
                    "Hi",
                    user_id=42,
                    user_name="Anna",
                    sent_at=SENT_AT,
                )
            ).text,
        )

    async def test_recent_context_is_isolated_and_does_not_modify_history(self) -> None:
        await self.reply((-123, 7), "Remember this discussion")
        stream = self.last_stream()
        recent = await self.dialogue.recent_context((-123, 7))
        self.assertIn("Remember this discussion", recent[0]["text"])
        self.assertEqual(
            recent[1], {"role": "assistant", "text": reply_json("AI answer")}
        )
        self.assertEqual(len(await self.requests(stream)), 1)
        self.assertEqual(await self.dialogue.recent_context((-123, 8)), [])

    async def test_recent_context_is_bounded_and_clear_removes_it(self) -> None:
        for number in range(7):
            await self.reply((-123, None), f"{number}:" + "x" * 2000)
        recent = await self.dialogue.recent_context((-123, None))
        self.assertEqual(len(recent), 10)
        for entry in recent:
            self.assertLessEqual(len(entry["text"]), 1000)
        await self.dialogue.clear((-123, None))
        self.assertEqual(await self.dialogue.recent_context((-123, None)), [])

    async def test_recent_context_waits_for_reply_while_other_conversations_progress(
        self,
    ) -> None:
        started, release = asyncio.Event(), asyncio.Event()

        async def blocked_ask(text: str, **kwargs: Any) -> AgentReply[str, Reply]:
            if "Blocked" in text:
                started.set()
                await release.wait()
            return cast("AgentReply[str, Reply]", await self.real_ask(text, **kwargs))

        self.ask.side_effect = blocked_ask
        task = asyncio.create_task(self.reply((-123, 7), "Blocked"))
        await started.wait()
        reading = asyncio.create_task(self.dialogue.recent_context((-123, 7)))
        try:
            await asyncio.wait_for(self.reply((-123, 8)), timeout=2)
            self.assertFalse(reading.done())
        finally:
            release.set()
            await task
            await reading

    async def test_group_reply_preserves_send_time_and_quotes_reply_target(self) -> None:
        self.history.context.return_value = [
            MemoryMessage(
                chat_id=-123,
                message_id=1,
                text="Earlier message",
                sent_at=SENT_AT,
                author_id=123,
                author="Dashulya",
                chat_title="Group",
                thread_id=7,
            )
        ]
        context: dict[str, object] = {
            "message_id": 2,
            "text": "What do you mean?",
            "reply_to": {"message_id": 1, "text": "Earlier message", "sender_id": 123},
        }
        await self.dialogue.reply(
            (-123, 7),
            "What do you mean?",
            user_id=42,
            user_name="Anna",
            sent_at=SENT_AT,
            group_context=context,
        )
        stream = self.last_stream()
        request = (await self.requests(stream))[-1]
        self.assertIn("25 October 2026 (Sunday) 15:00:00", request)
        self.assertIn("Reply to message ID: 1", request)
        events = list(await stream.history.get_events())
        assert isinstance(events[0], ModelResponse)
        self.assertEqual(events[0].to_api()["role"], "assistant")
        self.assertIn("Earlier message", events[0].content or "")
        self.history.context.assert_awaited_once_with(
            (-123, 7), 2, limit=20, parent_hops=5
        )

    async def test_group_turns_rebuild_history_and_keep_actual_bot_role(self) -> None:
        streams = []
        for number in (10, 11):
            self.history.context.return_value = [
                MemoryMessage(
                    chat_id=-123,
                    message_id=1,
                    text="Earlier bot reply",
                    sent_at=SENT_AT,
                    author_id=123,
                    author="Dashulya",
                    chat_title="Group",
                ),
                MemoryMessage(
                    chat_id=-123,
                    message_id=2,
                    text="Passive participant",
                    sent_at=SENT_AT,
                    author_id=99,
                    author="Boris",
                    chat_title="Group",
                    reply_to_id=1,
                ),
                MemoryMessage(
                    chat_id=-123,
                    message_id=3,
                    text="",
                    sent_at=SENT_AT,
                    author_id=456,
                    author="Another bot",
                    chat_title="Group",
                    media_type="animation",
                ),
                MemoryMessage(
                    chat_id=-123,
                    message_id=number,
                    text=f"Request {number}",
                    sent_at=SENT_AT,
                    author_id=42,
                    author="Anna",
                    chat_title="Group",
                ),
            ]
            await self.dialogue.reply(
                (-123, None),
                f"Request {number}",
                user_id=42,
                user_name="Anna",
                sent_at=SENT_AT,
                group_context={"message_id": number},
            )
            stream = self.last_stream()
            streams.append(stream)
            events = list(await stream.history.get_events())
            self.assertIsInstance(events[0], ModelResponse)
            self.assertIsInstance(events[1], ModelRequest)
            self.assertIsInstance(events[2], ModelRequest)
            requests = await self.requests(stream)
            self.assertIn("Passive participant", requests[0])
            self.assertIn("Reply to message ID: 1", requests[0])
            self.assertIn("Attachment: animation; contents unavailable", requests[1])
            self.assertEqual(len(requests), 3)
            self.assertEqual(requests[-1].count(f"Request {number}"), 1)
            if number == 11:
                self.assertIsNot(stream, streams[0])
                self.assertNotIn("Request 10", "\n".join(requests))
            self.assertEqual(await self.dialogue.recent_context((-123, None)), [])

    async def test_group_classifier_uses_observed_history_instead_of_ai_turns(
        self,
    ) -> None:
        self.history.context.return_value = [
            MemoryMessage(
                chat_id=-123,
                message_id=1,
                text="Earlier delivered answer",
                sent_at=SENT_AT,
                author_id=123,
                author="Dashulya",
                chat_title="Group",
            ),
            MemoryMessage(
                chat_id=-123,
                message_id=2,
                text="Current request",
                sent_at=SENT_AT,
                author_id=42,
                author="Anna",
                chat_title="Group",
            ),
        ]
        recent = await self.dialogue.recent_context((-123, None), through_message_id=2)
        self.assertEqual(len(recent), 1)
        self.assertEqual(recent[0]["role"], "assistant")
        self.assertIn("Earlier delivered answer", recent[0]["text"])
        self.assertNotIn("Current request", recent[0]["text"])

    async def test_reply_target_must_be_in_the_selected_group_context(self) -> None:
        self.history.context.return_value = [
            MemoryMessage(
                chat_id=-123,
                message_id=1,
                text="A selected message",
                sent_at=SENT_AT,
                author_id=99,
                author="Boris",
                chat_title="Group",
                reply_to_id=999,
            ),
        ]
        for target, expected in (
            (None, None),
            (1, 1),
            (5, 5),
            (999, None),
            (0, None),
            (-1, None),
        ):
            with self.subTest(target=target):
                agent = create_reply_agent(
                    TestConfig(reply_json("  Answer  ", target)), tools=[]
                )
                dialogue = Dialogue(agent, history=self.history, bot_id=123)
                with self.assertLogs("dashulya.ai.dialogue", level="INFO") as logs:
                    result = await dialogue.reply(
                        (-123, None),
                        "Current request",
                        user_id=42,
                        user_name="Anna",
                        sent_at=SENT_AT,
                        group_context={"message_id": 5},
                    )
                self.assertEqual(result, Reply("Answer", expected))
                event = logs.records[-1].__dict__
                self.assertEqual(event["requested_reply_to_message_id"], target)
                self.assertEqual(event["reply_to_message_id"], expected or 5)
                self.assertEqual(event["context_message_ids"], [1, 5])
                reason = (
                    "default_current_request"
                    if target is None
                    else "selected"
                    if expected is not None
                    else "outside_context"
                )
                self.assertEqual(event["reply_target_reason"], reason)

    async def test_private_reply_ignores_model_selected_message_id(self) -> None:
        agent = create_reply_agent(
            TestConfig(reply_json("Private answer", 999)), tools=[]
        )
        result = await Dialogue(agent).reply(
            (42, None),
            "Hi",
            user_id=42,
            user_name="Anna",
            sent_at=SENT_AT,
        )
        self.assertEqual(result, Reply("Private answer"))
