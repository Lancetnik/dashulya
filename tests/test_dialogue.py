import asyncio
import unittest
from datetime import UTC, datetime, timedelta, timezone
from typing import Any, cast
from unittest.mock import AsyncMock, patch

from ag2 import Agent, MemoryStream
from ag2.agent import AgentReply
from ag2.events import ModelRequest, ModelResponse, TextInput
from ag2.testing import TestConfig, TrackingConfig

from dashulya.ai.dialogue import Dialogue
from dashulya.domain.messages import ConversationKey
from dashulya.main import create_container
from dashulya.main.config import Settings
from tests.support import TEST_API_KEY, MemoryOverride, ModelOverrides

SENT_AT = datetime(2026, 10, 25, 12, 0, tzinfo=UTC)


class DialogueTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.config = TrackingConfig(TestConfig("AI answer"))
        self.settings = Settings(
            bot_token="123:TEST",
            openai_api_key=TEST_API_KEY,
            owner_user_id=42,
        )
        self.container = create_container(
            self.settings,
            MemoryOverride(),
            ModelOverrides(reply=self.config, decision=TestConfig("false")),
        )
        self.addAsyncCleanup(self.container.close)
        self.agent = await self.container.get(Agent[str])
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
        return await self.dialogue.reply(
            key,
            text,
            user_id=user_id,
            user_name="Anna",
            sent_at=sent_at,
        )

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
        self.assertEqual([e.content for e in responses], ["AI answer", "AI answer"])

    async def test_each_message_keeps_its_send_time_in_ag2_history(self) -> None:
        first = datetime(2026, 10, 24, 20, 59, tzinfo=UTC)
        second = datetime(2026, 10, 25, 3, 5, tzinfo=timezone(timedelta(hours=3)))
        await self.reply((42, None), "Yesterday", sent_at=first)
        await self.reply((42, None), "Today", sent_at=second)
        requests = await self.requests(self.last_stream())
        self.assertEqual(
            requests,
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
                sent_at=datetime(2026, 10, 25, 12, 0, tzinfo=None),  # noqa: DTZ001 -- Exercise rejection of naive timestamps.
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

        async def blocked_ask(text: str, **kwargs: Any) -> AgentReply[str, str]:
            if "Blocked" in text:
                started.set()
                await release.wait()
            return await self.real_ask(text, **kwargs)

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
            await self.requests(fresh),
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
        empty = Dialogue(Agent("empty", config=TestConfig(ModelResponse())))
        self.assertIn(
            "couldn't generate",
            await empty.reply(
                (42, None),
                "Hi",
                user_id=42,
                user_name="Anna",
                sent_at=SENT_AT,
            ),
        )

    async def test_recent_context_is_isolated_and_does_not_modify_history(self) -> None:
        await self.reply((-123, 7), "Remember this discussion")
        stream = self.last_stream()
        recent = await self.dialogue.recent_context((-123, 7))
        self.assertIn("Remember this discussion", recent[0]["text"])
        self.assertEqual(recent[1], {"role": "assistant", "text": "AI answer"})
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

        async def blocked_ask(text: str, **kwargs: Any) -> AgentReply[str, str]:
            if "Blocked" in text:
                started.set()
                await release.wait()
            return await self.real_ask(text, **kwargs)

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
        context: dict[str, object] = {
            "text": "What do you mean?",
            "reply_to": {"text": "Earlier message", "sender_id": 123},
        }
        await self.dialogue.reply(
            (-123, 7),
            "What do you mean?",
            user_id=42,
            user_name="Anna",
            sent_at=SENT_AT,
            group_context=context,
        )
        request = (await self.requests(self.last_stream()))[0]
        self.assertIn("25 October 2026 (Sunday) 15:00:00", request)
        self.assertIn("Earlier message", request)
        self.assertIn("untrusted quoted data", request)
