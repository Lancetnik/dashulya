import asyncio
import json
import unittest
from datetime import UTC, datetime, timedelta, timezone
from typing import Any, cast
from unittest.mock import AsyncMock, patch

from ag2 import Agent, MemoryStream, ResponseSchema
from ag2.agent import AgentReply
from ag2.events import BinaryInput, ModelRequest, ModelResponse, TextInput
from ag2.testing import TestConfig, TrackingConfig
from aiogram import Bot
from aiogram.types import Message

from dashulya.ai.agents.factory import create_reply_agent
from dashulya.ai.dialogue import Dialogue
from dashulya.bot.images import TelegramImages
from dashulya.bot.messages import capture_message
from dashulya.bot.replies import ReplyDelivery
from dashulya.domain.memory import ChatHistory, MemoryBackend, MemoryMessage, ReplyContext
from dashulya.domain.messages import (
    ConversationKey,
    ImageAttachment,
    ImageOmission,
    Reply,
)
from dashulya.main import create_container
from dashulya.main.config import Settings
from tests.support import (
    TEST_API_KEY,
    MemoryOverride,
    ModelOverrides,
    group_rows,
    reply_json,
)

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
        self.memory = AsyncMock(spec=MemoryBackend)
        self.memory.search.return_value = {"coverage": {}, "hits": []}
        self.history.context.return_value = ReplyContext()
        self.container = create_container(
            self.settings,
            MemoryOverride(self.memory, history=self.history),
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
        self.assertIn("Anna (Telegram user ID: 42; interlocutor)", requests[0])
        self.assertEqual(self.config.calls[-1].variables, {"user_id": 42, "chat_id": 42})
        responses = [
            e for e in await stream.history.get_events() if isinstance(e, ModelResponse)
        ]
        self.assertEqual([e.content for e in responses], [reply_json("AI answer")] * 2)

    async def test_casual_reply_does_not_search_memory_automatically(self) -> None:
        await self.reply((42, None), "Hello")
        self.memory.search.assert_not_awaited()
        requests = await self.requests(self.last_stream())
        self.assertNotIn("Initial recall", requests[0])

    async def test_group_citation_template_reaches_agent_without_a_history_tool_call(
        self,
    ) -> None:
        self.history.context.return_value = ReplyContext((
            MemoryMessage(
                chat_id=-1002391257796,
                message_id=19715,
                text="Earlier statement",
                sent_at=SENT_AT,
                author_id=42,
                author="Anna",
                chat_title="Group",
            ),
        ))
        await self.dialogue.reply(
            (-1002391257796, None),
            "Link?",
            user_id=42,
            user_name="Anna",
            sent_at=SENT_AT,
            message_id=19735,
            group_context={"message_id": 19735},
        )
        requests = await self.requests(self.last_stream())
        line = next(
            (line for line in requests[0].splitlines() if line.startswith("chat: ")), None
        )
        self.assertIsNotNone(
            line, "The reply agent has IDs but no way to cite their Telegram messages"
        )
        assert line is not None
        chat = json.loads(line.removeprefix("chat: "))
        self.assertEqual(
            chat["message_url_template"].format(message_id=19715),
            "https://t.me/c/2391257796/19715",
        )
        self.assertTrue(any("Earlier statement" in request for request in requests))
        self.memory.search.assert_not_awaited()

    async def test_each_message_keeps_its_send_time_in_ag2_history(self) -> None:
        first = datetime(2026, 10, 24, 20, 59, tzinfo=UTC)
        second = datetime(2026, 10, 25, 3, 5, tzinfo=timezone(timedelta(hours=3)))
        await self.reply((42, None), "Yesterday", sent_at=first)
        await self.reply((42, None), "Today", sent_at=second)
        requests = await self.requests(self.last_stream())
        self.assertEqual(
            requests,
            [
                "Sent at: 24 October 2026 (Saturday) 23:59:00\nTimezone: Europe/Moscow (UTC+03:00)\nAnna (Telegram user ID: 42; interlocutor):\nYesterday",
                "Sent at: 25 October 2026 (Sunday) 03:05:00\nTimezone: Europe/Moscow (UTC+03:00)\nAnna (Telegram user ID: 42; interlocutor):\nToday",
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
            await self.requests(fresh),
            [
                "Sent at: 25 October 2026 (Sunday) 15:00:00\nTimezone: Europe/Moscow (UTC+03:00)\nAnna (Telegram user ID: 42; interlocutor):\nFresh"
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
        self.history.context.return_value = ReplyContext((
            MemoryMessage(
                chat_id=-123,
                message_id=1,
                text="Earlier message",
                sent_at=SENT_AT,
                author_id=123,
                author="Dashulya",
                chat_title="Group",
                thread_id=7,
            ),
        ))
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
        self.assertIn("25 October 2026 (Sunday)", "\n".join(await self.requests(stream)))
        self.assertEqual(group_rows(request)[-1][2], "15:00:00")
        self.assertEqual(group_rows(request)[-1][4], 1)
        events = list(await stream.history.get_events())
        assert isinstance(events[1], ModelResponse)
        self.assertEqual(events[1].to_api()["role"], "assistant")
        self.assertIn("Earlier message", events[1].content or "")
        self.history.context.assert_awaited_once_with(
            (-123, 7), 2, limit=20, parent_hops=5
        )

    async def test_group_turns_rebuild_history_and_keep_actual_bot_role(self) -> None:
        streams = []
        for number in (10, 11):
            self.history.context.return_value = ReplyContext((
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
            ))
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
            self.assertIsInstance(events[0], ModelRequest)
            self.assertIsInstance(events[1], ModelResponse)
            self.assertIsInstance(events[2], ModelRequest)
            self.assertIsInstance(events[3], ModelRequest)
            requests = await self.requests(stream)
            self.assertIn("Passive participant", requests[1])
            self.assertEqual(group_rows(requests[1])[0][4], 1)
            self.assertEqual(group_rows(requests[2])[0][5], "animation")
            self.assertEqual(len(requests), 4)
            self.assertEqual(requests[-1].count(f"Request {number}"), 1)
            if number == 11:
                self.assertIsNot(stream, streams[0])
                self.assertNotIn("Request 10", "\n".join(requests))
            self.assertEqual(await self.dialogue.recent_context((-123, None)), [])

    async def test_group_classifier_uses_observed_history_instead_of_ai_turns(
        self,
    ) -> None:
        self.history.context.return_value = ReplyContext((
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
        ))
        recent = await self.dialogue.recent_context((-123, None), through_message_id=2)
        self.assertEqual(len(recent), 2)
        self.assertEqual(recent[1]["role"], "assistant")
        self.assertIn("Earlier delivered answer", recent[1]["text"])
        self.assertNotIn("Current request", recent[0]["text"])

    async def test_reply_target_must_be_in_the_selected_group_context(self) -> None:
        self.history.context.return_value = ReplyContext((
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
        ))
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

    async def test_private_images_survive_followup_and_oldest_are_evicted_at_budget(
        self,
    ) -> None:
        dialogue = Dialogue(self.agent, max_image_bytes=5)
        for message_id, data in ((1, b"123"), (2, b"45"), (3, b"678")):
            await dialogue.reply(
                (42, None),
                "Inspect",
                user_id=42,
                user_name="Anna",
                sent_at=SENT_AT,
                images=(ImageAttachment(data, "image/png", message_id),),
            )
        await dialogue.reply(
            (42, None), "Compare", user_id=42, user_name="Anna", sent_at=SENT_AT
        )
        events = list(await self.last_stream().history.get_events())
        binary = [
            part
            for event in events
            if isinstance(event, ModelRequest)
            for part in event.parts
            if isinstance(part, BinaryInput)
        ]
        self.assertEqual([part.data for part in binary], [b"45", b"678"])
        self.assertEqual(
            [part.vendor_metadata["detail"] for part in binary], ["high", "high"]
        )
        self.assertIn(
            "Earlier image omitted", "\n".join(await self.requests(self.last_stream()))
        )
        await dialogue.clear((42, None))
        self.assertEqual(await dialogue.recent_context((42, None)), [])

    async def test_group_images_are_limited_to_selected_message_ids(self) -> None:
        self.history.context.return_value = ReplyContext((
            MemoryMessage(
                chat_id=-123,
                message_id=1,
                text="Photo caption",
                sent_at=SENT_AT,
                author_id=99,
                author="Boris",
                chat_title="Group",
                media_type="photo",
            ),
        ))
        await self.dialogue.reply(
            (-123, None),
            "Inspect",
            user_id=42,
            user_name="Anna",
            sent_at=SENT_AT,
            group_context={"message_id": 2},
            images=(
                ImageAttachment(b"selected", "image/png", 1),
                ImageAttachment(b"excluded", "image/png", 999),
            ),
        )
        parts = [
            part
            for event in await self.last_stream().history.get_events()
            if isinstance(event, ModelRequest)
            for part in event.parts
        ]
        self.assertEqual(
            [part.data for part in parts if isinstance(part, BinaryInput)], [b"selected"]
        )
        self.assertIn(
            "Attached image from Telegram message ID: 1",
            "\n".join(await self.requests(self.last_stream())),
        )

    async def test_passively_captured_photo_reaches_later_group_request(self) -> None:
        await self.assert_captured_media_reaches_followup({
            "photo": [
                {"file_id": "photo", "file_unique_id": "unique", "width": 2, "height": 2}
            ],
        })

    async def test_passively_captured_stickers_and_animations_reach_later_group_request(
        self,
    ) -> None:
        thumbnail = {
            "file_id": "thumbnail",
            "file_unique_id": "thumb",
            "width": 2,
            "height": 2,
        }
        sticker = {
            "file_id": "sticker",
            "file_unique_id": "unique",
            "type": "regular",
            "width": 2,
            "height": 2,
            "is_animated": False,
            "is_video": False,
        }
        for fields, expected_id, animation in (
            ({"sticker": sticker}, "sticker", False),
            (
                {"sticker": {**sticker, "is_animated": True, "thumbnail": thumbnail}},
                "thumbnail",
                False,
            ),
            (
                {"sticker": {**sticker, "is_video": True, "thumbnail": thumbnail}},
                "thumbnail",
                False,
            ),
            (
                {
                    "animation": {
                        "file_id": "animation",
                        "file_unique_id": "unique",
                        "width": 2,
                        "height": 2,
                        "duration": 1,
                        "thumbnail": thumbnail,
                    }
                },
                "animation",
                True,
            ),
            (
                {
                    "document": {
                        "file_id": "gif",
                        "file_unique_id": "unique",
                        "mime_type": "image/gif",
                        "thumbnail": thumbnail,
                    }
                },
                "gif",
                True,
            ),
        ):
            with self.subTest(fields=fields):
                await self.assert_captured_media_reaches_followup(
                    fields, expected_id=expected_id, animation=animation
                )

    async def assert_captured_media_reaches_followup(
        self,
        fields: dict[str, Any],
        *,
        expected_id: str = "photo",
        animation: bool = False,
    ) -> None:
        photo = Message.model_validate({
            "message_id": 1,
            "date": SENT_AT,
            "chat": {"id": -123, "type": "supergroup", "title": "Group"},
            "from": {"id": 99, "is_bot": False, "first_name": "Boris"},
            **fields,
        })
        request = Message.model_validate({
            "message_id": 2,
            "date": SENT_AT,
            "chat": photo.chat,
            "from": {"id": 42, "is_bot": False, "first_name": "Anna"},
            "text": "@dashulya_bot what is in the photo above?",
        })
        recorder = AsyncMock()
        await capture_message(recorder, photo)
        await capture_message(recorder, request)
        self.history.context.return_value = ReplyContext(
            tuple(call.args[0] for call in recorder.record.call_args_list)
        )
        bot = AsyncMock(spec=Bot)
        images = TelegramImages(bot, max_bytes=20_000_000)
        png = b"\x89PNG\r\n\x1a\nimage-data"
        with (
            patch.object(images, "_read", AsyncMock(return_value=png)) as download,
            patch.object(Message, "answer", AsyncMock()),
            patch("dashulya.bot.images.first_frame", return_value=b"frame"),
        ):
            await ReplyDelivery(bot, self.dialogue, images).reply(
                request, group_context={"message_id": 2}
            )
        parts = [
            part
            for event in await self.last_stream().history.get_events()
            if isinstance(event, ModelRequest)
            for part in event.parts
        ]
        self.assertEqual(
            [part.data for part in parts if isinstance(part, BinaryInput)],
            [b"frame" if animation else png],
        )
        download.assert_awaited_once()
        assert download.await_args is not None
        self.assertEqual(download.await_args.args[0].file_id, expected_id)
        if animation:
            self.assertIn(
                "first frame", "\n".join(await self.requests(self.last_stream()))
            )

    async def test_oversized_direct_image_input_never_invokes_model(self) -> None:
        dialogue = Dialogue(self.agent, max_image_bytes=1)
        with self.assertRaisesRegex(ValueError, "byte budget"):
            await dialogue.reply(
                (42, None),
                "Inspect",
                user_id=42,
                user_name="Anna",
                sent_at=SENT_AT,
                images=(ImageAttachment(b"12", "image/png", 1),),
            )
        self.ask.assert_not_awaited()

    async def test_omitted_image_is_an_explicit_model_note_with_no_binary_input(
        self,
    ) -> None:
        await self.dialogue.reply(
            (42, None),
            "Read this",
            user_id=42,
            user_name="Anna",
            sent_at=SENT_AT,
            images=(ImageOmission(1),),
        )
        text = "\n".join(await self.requests(self.last_stream()))
        self.assertIn("Telegram message ID: 1 omitted", text)
        self.assertIn("too large for the 20 MB", text)
        self.assertFalse(
            any(
                isinstance(part, BinaryInput)
                for event in await self.last_stream().history.get_events()
                if isinstance(event, ModelRequest)
                for part in event.parts
            )
        )

    async def test_sticker_preview_and_missing_thumbnail_are_explicit_model_inputs(
        self,
    ) -> None:
        await self.dialogue.reply(
            (42, None),
            "What is this sticker?",
            user_id=42,
            user_name="Anna",
            sent_at=SENT_AT,
            images=(
                ImageAttachment(b"preview", "image/webp", 1, preview_kind="sticker"),
                ImageOmission(2, "preview_unavailable"),
                ImageAttachment(
                    b"gif-preview", "image/jpeg", 3, preview_kind="animation"
                ),
            ),
        )
        text = "\n".join(await self.requests(self.last_stream()))
        self.assertIn("only a static sticker thumbnail", text)
        self.assertIn("only a static animation thumbnail", text)
        self.assertIn("animation and motion are unavailable", text)
        self.assertIn("Telegram message ID: 2 omitted", text)
        self.assertIn("Telegram did not provide a thumbnail", text)
        self.assertNotIn("too large", text)
        parts = [
            part
            for event in await self.last_stream().history.get_events()
            if isinstance(event, ModelRequest)
            for part in event.parts
            if isinstance(part, BinaryInput)
        ]
        self.assertEqual([part.data for part in parts], [b"preview", b"gif-preview"])

    async def test_animation_reaches_model_as_only_its_first_frame(
        self,
    ) -> None:
        await self.dialogue.reply(
            (42, None),
            "Please respond to this media in the context of our conversation.",
            user_id=42,
            user_name="Anna",
            sent_at=SENT_AT,
            images=(
                ImageAttachment(
                    b"first", "image/jpeg", 1, preview_kind="animation", first_frame=True
                ),
                ImageOmission(2, "decode_failed"),
            ),
        )
        events = await self.last_stream().history.get_events()
        parts = [
            part
            for event in events
            if isinstance(event, ModelRequest)
            for part in event.parts
        ]
        text = "\n".join(part.content for part in parts if isinstance(part, TextInput))
        self.assertIn("only the first frame from the original animation", text)
        self.assertIn("Later frames and motion are unavailable", text)
        self.assertNotIn("thumbnail", text.split("original animation.")[0])
        self.assertIn("original animation could not be decoded", text)
        self.assertEqual(
            [part.data for part in parts if isinstance(part, BinaryInput)],
            [b"first"],
        )
