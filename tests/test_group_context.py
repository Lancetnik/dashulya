import asyncio
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

from ag2 import Agent, MemoryStream
from ag2.events import ModelRequest, ModelResponse, TextInput
from ag2.testing import TestConfig
from aiogram import Bot, Dispatcher
from aiogram.exceptions import TelegramBadRequest, TelegramNetworkError
from aiogram.methods import SendMessage, SendRichMessage, TelegramMethod
from aiogram.types import Message, Update, User

from dashulya.ai.agents.factory import create_reply_agent
from dashulya.ai.dialogue import Dialogue
from dashulya.domain.messages import Reply
from dashulya.main import create_container
from dashulya.main.config import Settings
from dashulya.memory.store import MessageStore
from tests.postgres import PostgresTestCase
from tests.support import (
    TEST_API_KEY,
    BotOverride,
    MemoryOverride,
    ModelOverrides,
    RecorderOverride,
    reply_json,
)

NOW = datetime(2026, 10, 25, 12, tzinfo=UTC)
BOT: dict[str, Any] = {
    "id": 123,
    "is_bot": True,
    "first_name": "Dashulya",
    "username": "dashulya_bot",
}
OWNER = {"id": 42, "is_bot": False, "first_name": "Anna"}
CHAT = {"id": -100123, "type": "supergroup", "title": "Group"}


class GroupContextTests(PostgresTestCase):
    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        Path(directory.name)
        self.store = MessageStore(self.database)
        await self.store.initialize()
        self.bot = Bot("123:TEST")
        self.addAsyncCleanup(self.bot.session.close)
        self.next_id = 0
        self.sent: list[Message] = []
        self.requests = AsyncMock(side_effect=self.request)
        for target, name, replacement in (
            (self.bot, "me", AsyncMock(return_value=User(**BOT))),
            (self.bot.session, "make_request", self.requests),
        ):
            patcher = patch.object(target, name, replacement)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.container = create_container(
            Settings(
                bot_token=self.bot.token,
                openai_api_key=TEST_API_KEY,
                owner_user_id=42,
            ),
            BotOverride(self.bot),
            RecorderOverride(self.store),
            MemoryOverride(history=self.store),
            ModelOverrides(
                reply=TestConfig(reply_json("AI answer")), decision=TestConfig("false")
            ),
        )
        self.addAsyncCleanup(self.container.close)
        self.dispatcher = await self.container.get(Dispatcher)
        self.agent = await self.container.get(Agent[Reply])
        self.ask = AsyncMock(wraps=self.agent.ask)
        patcher = patch.object(self.agent, "ask", self.ask)
        patcher.start()
        self.addCleanup(patcher.stop)

    async def request(
        self, bot: Bot, method: TelegramMethod[Any], **kwargs: Any
    ) -> Message | bool:
        if not isinstance(method, (SendMessage, SendRichMessage)):
            return True
        self.next_id += 1
        text = method.text if isinstance(method, SendMessage) else None
        result = Message.model_validate({
            "message_id": self.next_id,
            "date": int(NOW.timestamp()),
            "chat": CHAT,
            "from": BOT,
            "text": text,
        })
        self.sent.append(result)
        return result

    async def feed(self, text: str, *, owner: bool = True, **fields: Any) -> Message:
        self.next_id += 1
        if fields.get("message_thread_id") is not None:
            fields.setdefault("is_topic_message", True)
        payload = {
            "message_id": self.next_id,
            "date": int(NOW.timestamp()),
            "chat": CHAT,
            "from": OWNER if owner else {**OWNER, "id": 99, "first_name": "Boris"},
            "text": text,
            **fields,
        }
        message = Message.model_validate(payload)
        await self.dispatcher.feed_update(
            self.bot,
            Update.model_validate({"update_id": self.next_id, "message": payload}),
        )
        return message

    def stream(self) -> MemoryStream:
        stream = self.ask.call_args.kwargs["stream"]
        assert isinstance(stream, MemoryStream)
        return stream

    async def test_passive_messages_and_delivered_bot_reply_reach_next_fresh_turn(
        self,
    ) -> None:
        await self.feed("We were discussing cats", owner=False, message_thread_id=7)
        first = await self.feed(
            "Dashulya",
            message_thread_id=7,
            entities=[
                {
                    "type": "text_mention",
                    "offset": 0,
                    "length": 8,
                    "user": BOT,
                }
            ],
        )
        first_stream = self.stream()
        response = self.sent[-1]
        second = await self.feed(
            "What about that?",
            message_thread_id=7,
            reply_to_message=response.model_dump(mode="json", exclude_none=True),
        )
        second_stream = self.stream()
        self.assertIsNot(first_stream, second_stream)
        events = list(await second_stream.history.get_events())
        self.assertIsInstance(events[0], ModelRequest)
        requests = [
            part.content
            for event in events
            if isinstance(event, ModelRequest)
            for part in event.parts
            if isinstance(part, TextInput)
        ]
        self.assertIn("We were discussing cats", requests[0])
        self.assertIn("Boris (Telegram user ID: 99)", requests[0])
        self.assertIn("Reply to message ID: 3", requests[-1])
        earlier_bot = [
            event
            for event in events
            if isinstance(event, ModelResponse)
            and "Telegram message ID: 3" in (event.content or "")
        ]
        self.assertEqual(len(earlier_bot), 1)
        self.assertEqual(earlier_bot[0].to_api()["role"], "assistant")
        saved = await self.store.context(
            (-100123, 7), self.next_id, limit=20, parent_hops=5
        )
        self.assertEqual([m.message_id for m in saved], [1, 2, 3, 4, 5])
        self.assertEqual(saved[2].reply_to_id, first.message_id)
        self.assertEqual(saved[-1].reply_to_id, second.message_id)
        self.assertEqual(saved[2].author_id, BOT["id"])
        self.assertEqual((await self.store.coverage())["indexed_count"], 0)

    async def test_rich_reply_records_markdown_with_actual_id(self) -> None:
        self.ask.return_value = SimpleNamespace(
            content=AsyncMock(return_value=Reply("**Hello**"))
        )
        self.ask.side_effect = None
        await self.feed(
            "Hi",
            reply_to_message={
                "message_id": 0,
                "date": int(NOW.timestamp()),
                "chat": CHAT,
                "from": BOT,
                "text": "Previous answer",
            },
        )
        saved = await self.store.context(
            (-100123, None), self.next_id, limit=20, parent_hops=5
        )
        self.assertEqual(saved[-1].message_id, self.sent[-1].message_id)
        self.assertEqual(saved[-1].text, "**Hello**")
        self.assertEqual(saved[-1].reply_to_id, 1)

    async def test_ambiguous_send_failure_does_not_record_or_retry(self) -> None:
        async def fail(bot: Bot, method: TelegramMethod[Any], **kwargs: Any) -> bool:
            if isinstance(method, (SendMessage, SendRichMessage)):
                raise TelegramNetworkError(method, "Ambiguous send failure")
            return True

        self.requests.side_effect = fail
        with self.assertRaises(TelegramNetworkError):
            await self.feed(
                "Dashulya",
                entities=[
                    {
                        "type": "text_mention",
                        "offset": 0,
                        "length": 8,
                        "user": BOT,
                    }
                ],
            )
        saved = await self.store.context(
            (-100123, None), self.next_id, limit=20, parent_hops=5
        )
        self.assertEqual([m.author_id for m in saved], [42])
        sends = [
            call
            for call in self.requests.call_args_list
            if isinstance(call.args[1], (SendMessage, SendRichMessage))
        ]
        self.assertEqual(len(sends), 1)

    async def test_rejected_rich_reply_records_only_successful_text_chunks(self) -> None:
        self.ask.return_value = SimpleNamespace(
            content=AsyncMock(return_value=Reply("**Hello**"))
        )
        self.ask.side_effect = None

        async def reject_rich(
            bot: Bot, method: TelegramMethod[Any], **kwargs: Any
        ) -> Message | bool:
            if isinstance(method, SendRichMessage):
                raise TelegramBadRequest(method, "Rich message rejected")
            return await self.request(bot, method, **kwargs)

        self.requests.side_effect = reject_rich
        with self.assertLogs("dashulya.bot.replies", level="WARNING"):
            await self.feed(
                "Dashulya",
                entities=[
                    {
                        "type": "text_mention",
                        "offset": 0,
                        "length": 8,
                        "user": BOT,
                    }
                ],
            )
        saved = await self.store.context(
            (-100123, None), self.next_id, limit=20, parent_hops=5
        )
        self.assertEqual(
            [(m.author_id, m.text) for m in saved], [(42, "Dashulya"), (123, "Hello")]
        )

    async def test_partial_chunk_delivery_records_only_the_delivered_chunk(self) -> None:
        self.ask.return_value = SimpleNamespace(
            content=AsyncMock(return_value=Reply("x" * 9000))
        )
        self.ask.side_effect = None
        sends = 0

        async def partial(
            bot: Bot, method: TelegramMethod[Any], **kwargs: Any
        ) -> Message | bool:
            nonlocal sends
            if isinstance(method, SendMessage):
                sends += 1
                if sends == 2:
                    raise TelegramNetworkError(method, "Connection lost")
            return await self.request(bot, method, **kwargs)

        self.requests.side_effect = partial
        with self.assertRaises(TelegramNetworkError):
            await self.feed(
                "Dashulya",
                entities=[
                    {
                        "type": "text_mention",
                        "offset": 0,
                        "length": 8,
                        "user": BOT,
                    }
                ],
            )
        saved = await self.store.context(
            (-100123, None), self.next_id, limit=20, parent_hops=5
        )
        self.assertEqual(len(saved), 2)
        self.assertEqual(saved[-1].text, self.sent[-1].text)
        self.assertTrue(0 < len(saved[-1].text) <= 4096)
        self.assertEqual(saved[-1].reply_to_id, 1)
        self.assertEqual(sends, 2)

    async def test_command_response_is_also_recorded_as_this_bot(self) -> None:
        await self.feed(
            "/help",
            entities=[
                {
                    "type": "bot_command",
                    "offset": 0,
                    "length": 5,
                }
            ],
        )
        saved = await self.store.context(
            (-100123, None), self.next_id, limit=20, parent_hops=5
        )
        self.assertEqual([m.author_id for m in saved], [42, 123])
        self.assertIn("Here's what I can do", saved[-1].text)
        self.ask.assert_not_awaited()

    async def test_delivery_boundary_serializes_same_conversation_only(self) -> None:
        dialogue = await self.container.get(Dialogue)
        entered = asyncio.Event()

        async def enter_same() -> None:
            async with dialogue.delivery((-100123, 7)):
                entered.set()

        async with dialogue.delivery((-100123, 7)):
            waiting = asyncio.create_task(enter_same())
            async with dialogue.delivery((-100123, 8)):
                await asyncio.sleep(0)
                self.assertFalse(entered.is_set())
        await asyncio.wait_for(waiting, timeout=2)
        self.assertTrue(entered.is_set())

    async def test_selected_reply_target_is_sent_and_persisted_for_each_chunk(
        self,
    ) -> None:
        await self.feed("The message to answer", owner=False, message_thread_id=7)
        text = "x" * 9000
        self.ask.return_value = SimpleNamespace(
            content=AsyncMock(return_value=Reply(text, 1))
        )
        self.ask.side_effect = None
        await self.feed(
            "Dashulya",
            message_thread_id=7,
            entities=[
                {
                    "type": "text_mention",
                    "offset": 0,
                    "length": 8,
                    "user": BOT,
                }
            ],
        )
        sends = [
            call.args[1]
            for call in self.requests.call_args_list
            if isinstance(call.args[1], SendMessage)
        ]
        self.assertGreater(len(sends), 1)
        self.assertEqual("".join(send.text for send in sends), text)
        for send in sends:
            assert send.reply_parameters is not None
            self.assertEqual(send.reply_parameters.message_id, 1)
        saved = await self.store.context(
            (-100123, 7), self.next_id, limit=20, parent_hops=5
        )
        self.assertEqual("".join(m.text for m in saved if m.author_id == 123), text)
        self.assertTrue(all(m.reply_to_id == 1 for m in saved if m.author_id == 123))

    async def test_selected_reply_target_survives_rich_rejection(self) -> None:
        await self.feed("The message to answer", owner=False)
        self.ask.return_value = SimpleNamespace(
            content=AsyncMock(return_value=Reply("**Answer**", 1))
        )
        self.ask.side_effect = None

        async def reject_rich(
            bot: Bot, method: TelegramMethod[Any], **kwargs: Any
        ) -> Message | bool:
            if isinstance(method, SendRichMessage):
                raise TelegramBadRequest(method, "Rich message rejected")
            return await self.request(bot, method, **kwargs)

        self.requests.side_effect = reject_rich
        with self.assertLogs("dashulya.bot.replies", level="WARNING"):
            await self.feed(
                "Dashulya",
                entities=[
                    {
                        "type": "text_mention",
                        "offset": 0,
                        "length": 8,
                        "user": BOT,
                    }
                ],
            )
        sends = [
            call.args[1]
            for call in self.requests.call_args_list
            if isinstance(call.args[1], (SendMessage, SendRichMessage))
        ]
        self.assertEqual(len(sends), 2)
        for send in sends:
            assert send.reply_parameters is not None
            self.assertEqual(send.reply_parameters.message_id, 1)
        saved = await self.store.context(
            (-100123, None), self.next_id, limit=20, parent_hops=5
        )
        self.assertEqual(saved[-1].text, "Answer")
        self.assertEqual(saved[-1].reply_to_id, 1)

    async def test_target_outside_current_topic_defaults_to_triggering_message(
        self,
    ) -> None:
        await self.feed("Other topic", owner=False, message_thread_id=8)
        self.ask.return_value = SimpleNamespace(
            content=AsyncMock(return_value=Reply("Answer", 1))
        )
        self.ask.side_effect = None
        trigger = await self.feed(
            "Dashulya",
            message_thread_id=7,
            entities=[
                {
                    "type": "text_mention",
                    "offset": 0,
                    "length": 8,
                    "user": BOT,
                }
            ],
        )
        sends = [
            call.args[1]
            for call in self.requests.call_args_list
            if isinstance(call.args[1], SendMessage)
        ]
        assert sends[-1].reply_parameters is not None
        self.assertEqual(sends[-1].reply_parameters.message_id, trigger.message_id)
        saved = await self.store.context(
            (-100123, 7), self.next_id, limit=20, parent_hops=5
        )
        self.assertEqual(saved[-1].reply_to_id, trigger.message_id)

    async def test_invalid_structured_output_uses_redacted_fallback(self) -> None:
        agent = create_reply_agent(
            TestConfig('{"reply_to_message_id":1,"detail":"sk-secret provider data"}'),
            tools=[],
        )
        self.ask.side_effect = agent.ask
        with self.assertLogs("dashulya.bot.replies", level="WARNING") as logs:
            trigger = await self.feed(
                "Dashulya",
                entities=[
                    {
                        "type": "text_mention",
                        "offset": 0,
                        "length": 8,
                        "user": BOT,
                    }
                ],
            )
        self.assertNotIn("sk-secret", str(logs.output))
        sends = [
            call.args[1]
            for call in self.requests.call_args_list
            if isinstance(call.args[1], SendMessage)
        ]
        self.assertEqual(len(sends), 1)
        self.assertIn("couldn't get an AI reply", sends[0].text)
        self.assertNotIn("sk-secret", sends[0].text)
        assert sends[0].reply_parameters is not None
        self.assertEqual(sends[0].reply_parameters.message_id, trigger.message_id)
