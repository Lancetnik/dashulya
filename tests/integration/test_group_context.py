import asyncio
from contextlib import AsyncExitStack
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from ag2 import Agent, MemoryStream
from ag2.events import ModelRequest, ModelResponse, TextInput
from ag2.testing import TestConfig
from aiogram import Bot, Dispatcher
from aiogram.exceptions import TelegramNetworkError
from aiogram.methods import SendMessage, SendRichMessage, TelegramMethod
from aiogram.types import Message, Update, User

from dashulya.ai.agents.factory import create_reply_agent
from dashulya.ai.dialogue import Dialogue
from dashulya.bot.messages import conversation_key
from dashulya.database import Database
from dashulya.domain.groups import Group, GroupRegistry
from dashulya.domain.memory import MemoryAccess, MemoryMessage
from dashulya.domain.messages import Reply
from dashulya.main import create_container
from dashulya.main.config import Settings
from dashulya.memory.store import MessageStore
from tests.support import (
    TEST_API_KEY,
    BotOverride,
    MemoryOverride,
    ModelOverrides,
    RecorderOverride,
    group_rows,
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
CHAT: dict[str, Any] = {"id": -100123, "type": "supergroup", "title": "Group"}


@pytest.mark.usefixtures("_setup")
class TestGroupContext:
    @pytest.fixture
    async def _setup(self, database: Database, cleanup: AsyncExitStack) -> None:
        self.database = database
        self.cleanup = cleanup
        directory = TemporaryDirectory()
        self.cleanup.callback(directory.cleanup)
        Path(directory.name)
        self.store = MessageStore(self.database)
        await self.store.initialize()
        self.bot = Bot("123:TEST")
        self.cleanup.push_async_callback(self.bot.session.close)
        self.next_id = 0
        self.chats: dict[int, dict[str, Any]] = {CHAT["id"]: CHAT}
        self.sent: list[Message] = []
        self.requests = AsyncMock(side_effect=self.request)
        for target, name, replacement in (
            (self.bot, "me", AsyncMock(return_value=User(**BOT))),
            (self.bot.session, "make_request", self.requests),
        ):
            patcher = patch.object(target, name, replacement)
            patcher.start()
            self.cleanup.callback(patcher.stop)
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
        self.cleanup.push_async_callback(self.container.close)
        groups = await self.container.get(GroupRegistry)
        await groups.observe(Group(CHAT["id"], CHAT["title"]))
        await groups.activate(CHAT["id"])
        self.dispatcher = await self.container.get(Dispatcher)
        self.agent = await self.container.get(Agent[Reply])
        self.ask = AsyncMock(wraps=self.agent.ask)
        patcher = patch.object(self.agent, "ask", self.ask)
        patcher.start()
        self.cleanup.callback(patcher.stop)

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
            "chat": self.chats.get(method.chat_id, CHAT)
            if isinstance(method.chat_id, int)
            else CHAT,
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
        self.chats[message.chat.id] = message.chat.model_dump(exclude_none=True)
        await self.dispatcher.feed_update(
            self.bot,
            Update.model_validate({"update_id": self.next_id, "message": payload}),
        )
        return message

    def stream(self) -> MemoryStream:
        stream = self.ask.call_args.kwargs["stream"]
        assert isinstance(stream, MemoryStream)
        return stream

    async def test_ordinary_group_reply_thread_keeps_twenty_chat_messages_and_parents(
        self,
    ) -> None:
        # Existing rows retain raw thread IDs from the former capture implementation.
        for number in range(1, 27):
            await self.store.record(
                MemoryMessage(
                    CHAT["id"],
                    number,
                    f"Original {number}",
                    NOW,
                    99,
                    "Boris",
                    "Group",
                    thread_id=20 if number >= 21 else None,
                    reply_to_id=1 if number == 26 else None,
                )
            )
        for chat, number, text in [
            (CHAT["id"], 99, "Future secret"),
            (-100456, 25, "Foreign secret"),
        ]:
            await self.store.record(
                MemoryMessage(chat, number, text, NOW, 99, "Boris", "Other")
            )
        self.next_id = 26
        trigger = await self.feed(
            "Dashulya",
            message_thread_id=20,
            is_topic_message=False,
            reply_to_message={
                "message_id": 26,
                "date": int(NOW.timestamp()),
                "chat": CHAT,
                "from": OWNER,
                "text": "Original 26",
                "message_thread_id": 20,
            },
            entities=[{"type": "text_mention", "offset": 0, "length": 8, "user": BOT}],
        )
        events = await self.stream().history.get_events()
        inputs = [
            part.content
            for event in events
            if isinstance(event, ModelRequest)
            for part in event.parts
            if isinstance(part, TextInput)
        ]
        rendered = "\n".join(inputs)
        assert [row[0] for row in group_rows(rendered)] == [1, *range(8, 28)]
        assert "Future secret" not in rendered
        assert "Foreign secret" not in rendered
        rows = await self.store.originals(MemoryAccess(CHAT["id"], False))
        assert (
            next(row for row in rows if row.message_id == trigger.message_id).topic_id
            is None
        )

    async def test_name_addressing_reads_the_same_whole_group_context(self) -> None:
        await self.feed("Earlier independent message", owner=False)
        await self.feed(
            "Earlier reply branch",
            owner=False,
            message_thread_id=1,
            is_topic_message=False,
        )
        dialogue = await self.container.get(Dialogue)
        with patch.object(
            dialogue, "recent_context", wraps=dialogue.recent_context
        ) as recent:
            await self.feed(
                "Dashulya, what do you think?",
                message_thread_id=1,
                is_topic_message=False,
            )
        assert recent.call_args.args[0] == (CHAT["id"], None)
        context = await dialogue.recent_context((CHAT["id"], None), through_message_id=3)
        assert len(context) == 3
        assert "Earlier independent message" in context[1]["text"]
        assert "Earlier reply branch" in context[2]["text"]
        self.ask.assert_not_awaited()

    async def test_forum_general_topic_does_not_read_other_topics(self) -> None:
        forum = {**CHAT, "is_forum": True}
        await self.feed(
            "Other topic secret", owner=False, chat=forum, message_thread_id=7
        )
        await self.feed("General discussion", owner=False, chat=forum)
        trigger = await self.feed(
            "Dashulya",
            chat=forum,
            entities=[{"type": "text_mention", "offset": 0, "length": 8, "user": BOT}],
        )
        assert conversation_key(trigger) == (CHAT["id"], 1)
        events = await self.stream().history.get_events()
        rendered = "\n".join(
            part.content
            for event in events
            if isinstance(event, ModelRequest)
            for part in event.parts
            if isinstance(part, TextInput)
        )
        assert "General discussion" in rendered
        assert "Other topic secret" not in rendered
        saved = (
            await self.store.context(
                (CHAT["id"], 1), self.next_id, limit=20, parent_hops=5
            )
        ).messages
        assert saved[-1].message_id == self.sent[-1].message_id

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
        assert first_stream is not second_stream
        events = list(await second_stream.history.get_events())
        assert isinstance(events[0], ModelRequest)
        requests = [
            part.content
            for event in events
            if isinstance(event, ModelRequest)
            for part in event.parts
            if isinstance(part, TextInput)
        ]
        assert "We were discussing cats" in "\n".join(requests)
        assert "User 99: Boris" in requests[1]
        assert group_rows(requests[-1])[-1][4] == 3
        earlier_bot = [
            event
            for event in events
            if isinstance(event, ModelResponse)
            and any(row[0] == 3 for row in group_rows(event.content or ""))
        ]
        assert len(earlier_bot) == 1
        assert earlier_bot[0].to_api()["role"] == "assistant"
        saved = (
            await self.store.context((-100123, 7), self.next_id, limit=20, parent_hops=5)
        ).messages
        assert [m.message_id for m in saved] == [1, 2, 3, 4, 5]
        assert saved[2].reply_to_id == first.message_id
        assert saved[-1].reply_to_id == second.message_id
        assert saved[2].author_id == BOT["id"]
        assert (await self.store.coverage())["indexed_count"] == 0

    async def test_formatted_reply_records_delivered_text_with_actual_id(self) -> None:
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
        saved = (
            await self.store.context(
                (-100123, None), self.next_id, limit=20, parent_hops=5
            )
        ).messages
        assert saved[-1].message_id == self.sent[-1].message_id
        assert saved[-1].text == "Hello"
        delivered = [
            call.args[1]
            for call in self.requests.call_args_list
            if isinstance(call.args[1], SendMessage)
        ]
        assert delivered[-1].text == saved[-1].text
        assert [entity.type for entity in delivered[-1].entities or []] == ["bold"]
        assert saved[-1].reply_to_id == 1

    async def test_ambiguous_send_failure_does_not_record_or_retry(self) -> None:
        async def fail(bot: Bot, method: TelegramMethod[Any], **kwargs: Any) -> bool:
            if isinstance(method, (SendMessage, SendRichMessage)):
                raise TelegramNetworkError(method, "Ambiguous send failure")
            return True

        self.requests.side_effect = fail
        with pytest.raises(TelegramNetworkError):
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
        saved = (
            await self.store.context(
                (-100123, None), self.next_id, limit=20, parent_hops=5
            )
        ).messages
        assert [m.author_id for m in saved] == [42]
        sends = [
            call
            for call in self.requests.call_args_list
            if isinstance(call.args[1], (SendMessage, SendRichMessage))
        ]
        assert len(sends) == 1

    async def test_entity_reply_records_only_the_delivered_message(self) -> None:
        self.ask.return_value = SimpleNamespace(
            content=AsyncMock(return_value=Reply("**Hello**"))
        )
        self.ask.side_effect = None

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
        saved = (
            await self.store.context(
                (-100123, None), self.next_id, limit=20, parent_hops=5
            )
        ).messages
        assert [(m.author_id, m.text) for m in saved] == [
            (42, "Dashulya"),
            (123, "Hello"),
        ]

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
        with pytest.raises(TelegramNetworkError):
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
        saved = (
            await self.store.context(
                (-100123, None), self.next_id, limit=20, parent_hops=5
            )
        ).messages
        assert len(saved) == 2
        assert saved[-1].text == self.sent[-1].text
        assert 0 < len(saved[-1].text) <= 4096
        assert saved[-1].reply_to_id == 1
        assert sends == 2

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
        saved = (
            await self.store.context(
                (-100123, None), self.next_id, limit=20, parent_hops=5
            )
        ).messages
        assert [m.author_id for m in saved] == [42, 123]
        assert "Here's what I can do" in saved[-1].text
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
                assert not entered.is_set()
        await asyncio.wait_for(waiting, timeout=2)
        assert entered.is_set()

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
        assert len(sends) > 1
        assert "".join(send.text for send in sends) == text
        for send in sends:
            assert send.reply_parameters is not None
            assert send.reply_parameters.message_id == 1
        saved = (
            await self.store.context((-100123, 7), self.next_id, limit=20, parent_hops=5)
        ).messages
        assert "".join(m.text for m in saved if m.author_id == 123) == text
        assert all(m.reply_to_id == 1 for m in saved if m.author_id == 123)

    async def test_selected_reply_target_is_preserved_in_entity_delivery(self) -> None:
        await self.feed("The message to answer", owner=False)
        self.ask.return_value = SimpleNamespace(
            content=AsyncMock(return_value=Reply("**Answer**", 1))
        )
        self.ask.side_effect = None

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
        assert len(sends) == 1
        for send in sends:
            assert send.reply_parameters is not None
            assert send.reply_parameters.message_id == 1
        saved = (
            await self.store.context(
                (-100123, None), self.next_id, limit=20, parent_hops=5
            )
        ).messages
        assert saved[-1].text == "Answer"
        assert saved[-1].reply_to_id == 1

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
        assert sends[-1].reply_parameters.message_id == trigger.message_id
        saved = (
            await self.store.context((-100123, 7), self.next_id, limit=20, parent_hops=5)
        ).messages
        assert saved[-1].reply_to_id == trigger.message_id

    async def test_invalid_structured_output_uses_redacted_fallback(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        agent = create_reply_agent(
            TestConfig('{"reply_to_message_id":1,"detail":"sk-secret provider data"}'),
            tools=[],
        )
        self.ask.side_effect = agent.ask
        caplog.clear()
        with caplog.at_level("WARNING", logger="dashulya.bot.replies"):
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
        assert any(
            record.name == "dashulya.bot.replies"
            or record.name.startswith("dashulya.bot.replies.")
            for record in caplog.records
        )
        assert "sk-secret" not in str(caplog.messages)
        sends = [
            call.args[1]
            for call in self.requests.call_args_list
            if isinstance(call.args[1], SendMessage)
        ]
        assert len(sends) == 1
        assert "couldn't get an AI reply" in sends[0].text
        assert "sk-secret" not in sends[0].text
        assert sends[0].reply_parameters is not None
        assert sends[0].reply_parameters.message_id == trigger.message_id
