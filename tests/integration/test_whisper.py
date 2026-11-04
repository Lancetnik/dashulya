"""Private group turns through the real dispatcher, dialogue and PostgreSQL."""

import asyncio
import json
from collections import deque
from contextlib import AsyncExitStack
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import patch

import httpx2
import pytest
from ag2.config import OpenAIResponsesConfig
from ag2.testing import TestConfig
from aiogram import Bot, Dispatcher
from aiogram.exceptions import TelegramNetworkError
from aiogram.methods import (
    DeleteMessage,
    GetChatMember,
    GetMe,
    GetStickerSet,
    SendChatAction,
    SendMessage,
    SendSticker,
    SetMessageReaction,
    SetMyCommands,
    TelegramMethod,
)
from aiogram.types import ChatMemberMember, Message, Update, User
from dishka import Provider, provide
from dishka_ag2 import AG2Scope

from dashulya.ai.dialogue import Dialogue
from dashulya.database import Database
from dashulya.domain.groups import Group, GroupRegistry
from dashulya.domain.memory import ChatMemoryService, MemoryMessage
from dashulya.domain.messages import ReplyRequest
from dashulya.main import create_container
from dashulya.main.config import Settings
from dashulya.memory.service import HybridChatMemory
from dashulya.memory.store import MessageStore
from dashulya.memory.vectors import EmbeddingCache
from dashulya.observability import JsonLogFormatter
from tests.integration.test_ai_handlers import administrator
from tests.support import (
    TEST_API_KEY,
    BotOverride,
    ModelOverrides,
    RecorderOverride,
    dashulya_sticker_set,
    reply_json,
)

NOW = datetime(2026, 11, 4, 12, tzinfo=UTC)
CHAT: dict[str, Any] = {
    "id": -100123,
    "type": "supergroup",
    "title": "Group",
    "is_forum": True,
}
OWNER: dict[str, Any] = {"id": 42, "is_bot": False, "first_name": "Anna"}
PARTICIPANT: dict[str, Any] = {"id": 99, "is_bot": False, "first_name": "Boris"}
BOT: dict[str, Any] = {
    "id": 123,
    "is_bot": True,
    "first_name": "Dashulya",
    "username": "dashulya_bot",
}


async def no_embeddings(texts: list[str]) -> list[list[float]]:
    error_message = "Whisper tests must not request embeddings"
    raise AssertionError(error_message)


class WhisperMemory(Provider):
    scope = AG2Scope.APP

    def __init__(self, memory: HybridChatMemory) -> None:
        super().__init__()
        self._memory = memory

    @provide(override=True)
    def memory(self) -> ChatMemoryService:
        return self._memory


@pytest.mark.usefixtures("_setup")
class TestWhisper:
    @pytest.fixture
    async def _setup(self, database: Database, cleanup: AsyncExitStack) -> None:
        self.clock = NOW
        self.admin = True
        self.fail_send = False
        self.fail_send_after = 0
        self.block_model = False
        self.model_cancelled = asyncio.Event()
        self.api: list[TelegramMethod[Any]] = []
        self.inputs: list[list[dict[str, Any]]] = []
        self.outputs: deque[list[dict[str, Any]]] = deque()
        self.answer = "Private answer"
        self.update_id = 0
        self.bot = Bot("123:TEST")
        cleanup.push_async_callback(self.bot.session.close)
        cleanup.enter_context(
            patch.object(self.bot.session, "make_request", self.telegram)
        )
        clock = cleanup.enter_context(
            patch("dashulya.bot.replies.datetime", wraps=datetime)
        )
        clock.now.side_effect = lambda *args: self.clock
        http = await cleanup.enter_async_context(
            httpx2.AsyncClient(transport=httpx2.MockTransport(self.model))
        )
        self.store = MessageStore(database)
        memory = HybridChatMemory(
            self.store, EmbeddingCache(database, no_embeddings), owner_user_id=42
        )
        cleanup.push_async_callback(memory.close)
        self.container = create_container(
            Settings(
                bot_token=self.bot.token, openai_api_key=TEST_API_KEY, owner_user_id=42
            ),
            BotOverride(self.bot),
            RecorderOverride(self.store),
            WhisperMemory(memory),
            ModelOverrides(
                reply=OpenAIResponsesConfig(
                    model="test-model", api_key=TEST_API_KEY, http_client=http
                ),
                decision=TestConfig("false"),
            ),
        )
        cleanup.push_async_callback(self.container.close)
        self.groups = await self.container.get(GroupRegistry)
        await self.groups.observe(Group(CHAT["id"], CHAT["title"]))
        await self.groups.activate(CHAT["id"])
        self.dispatcher = await self.container.get(Dispatcher)
        self.dialogue = await self.container.get(Dialogue)

    async def model(self, request: httpx2.Request) -> httpx2.Response:
        self.inputs.append(json.loads(request.content)["input"])
        if self.block_model:
            try:
                await asyncio.Future[None]()
            finally:
                self.model_cancelled.set()
        output = (
            self.outputs.popleft()
            if self.outputs
            else [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "output_text",
                            "text": reply_json(self.answer),
                            "annotations": [],
                        }
                    ],
                }
            ]
        )
        return httpx2.Response(
            200,
            json={
                "id": f"resp_{len(self.inputs)}",
                "object": "response",
                "created_at": 1700000000,
                "status": "completed",
                "model": "test-model",
                "output": output,
            },
        )

    async def telegram(self, bot: Bot, method: TelegramMethod[Any], **kwargs: Any) -> Any:
        self.api.append(method)
        if isinstance(method, GetMe):
            return User(**BOT)
        if isinstance(method, GetChatMember):
            return (
                administrator(bot.id)
                if self.admin
                else ChatMemberMember(user=User(**BOT))
            )
        if isinstance(method, GetStickerSet):
            return dashulya_sticker_set()
        if isinstance(method, (SendMessage, SendSticker)):
            if self.fail_send and len(self.sent()) > self.fail_send_after:
                raise TelegramNetworkError(method, "Synthetic ambiguous send failure")
            return Message.model_validate({
                "message_id": 0
                if method.ephemeral_message_parameters
                else 1000 + len(self.api),
                "ephemeral_message_id": len(self.api)
                if method.ephemeral_message_parameters
                else None,
                "receiver_user": PARTICIPANT
                if method.ephemeral_message_parameters
                else None,
                "date": NOW,
                "chat": CHAT,
                "from": BOT,
                "text": method.text if isinstance(method, SendMessage) else None,
            })
        return True

    async def feed(
        self,
        text: str,
        *,
        hidden: bool = True,
        sender: dict[str, Any] | None = None,
        **fields: Any,
    ) -> None:
        self.update_id += 1
        payload = {
            "message_id": 0 if hidden else self.update_id,
            "ephemeral_message_id": 700 if hidden else None,
            "date": NOW,
            "chat": CHAT,
            "from": sender or PARTICIPANT,
            "text": text,
            "message_thread_id": 7,
            "is_topic_message": True,
            "entities": [{"type": "mention", "offset": 0, "length": 13}]
            if text.startswith("@dashulya_bot")
            else [],
            **fields,
        }
        await self.dispatcher.feed_update(
            self.bot,
            Update.model_validate({
                "update_id": self.update_id,
                "message": payload,
            }),
        )

    def sent(self) -> list[SendMessage | SendSticker]:
        return [item for item in self.api if isinstance(item, (SendMessage, SendSticker))]

    async def test_whisper_logs_the_normal_request_lifecycle_and_correlated_tools(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        self.tool("chat_history", {"sql": ["SELECT count(*) FROM messages"]})
        with caplog.at_level("INFO", logger="dashulya"):
            await self.feed("/whisper Logged request")
        formatter = JsonLogFormatter()
        events = [json.loads(formatter.format(record)) for record in caplog.records]
        lifecycle = [
            event for event in events if event["logger"] == "dashulya.bot.replies"
        ]
        assert [event["message"] for event in lifecycle] == [
            "AI request started",
            "AI reply generated",
            "AI reply delivered",
        ]
        model_and_tools = [
            event
            for event in events
            if event["message"].startswith(("AI model call", "AI tool call"))
            or event["message"] == "Group reply target selected"
        ]
        assert any(
            event["message"] == "AI tool call completed" for event in model_and_tools
        )
        for event in [*lifecycle, *model_and_tools]:
            assert event["chat_id"] == -100123
            assert event["message_id"] is None
            assert event["user_id"] == 99
            assert event["topic_id"] == 7
            assert event["update_id"] == 1
            assert event["ephemeral_message_id"] == 700
            assert event["is_whisper"] is True
        assert lifecycle[-1]["message_count"] == 1
        assert lifecycle[-1]["chunk_count"] == 1
        assert lifecycle[-1]["duration_ms"] >= 0
        assert any(
            event.get("agent_output") == reply_json("Private answer") for event in events
        )

    async def test_whisper_logs_partial_delivery_failure_as_a_normal_request(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        self.answer = "x" * 5000
        self.fail_send = True
        self.fail_send_after = 1
        with caplog.at_level("INFO", logger="dashulya"):
            await self.feed("/whisper Long reply")
        formatter = JsonLogFormatter()
        events = [
            json.loads(formatter.format(record))
            for record in caplog.records
            if record.name == "dashulya.bot.replies"
        ]
        assert [event["message"] for event in events] == [
            "AI request started",
            "AI reply generated",
            "AI reply delivery failed",
        ]
        failure = events[-1]
        assert failure["chat_id"] == -100123
        assert failure["user_id"] == 99
        assert failure["update_id"] == 1
        assert failure["ephemeral_message_id"] == 700
        assert failure["is_whisper"] is True
        assert failure["error_type"] == "TelegramNetworkError"
        assert failure["delivery_stage"] == "send"
        assert failure["message_count"] == 1
        assert failure["chunk_count"] == 1
        assert failure["duration_ms"] >= 0
        assert len(self.sent()) == 2

    def tool(self, name: str, arguments: dict[str, Any]) -> None:
        self.outputs.append([
            {
                "type": "function_call",
                "id": "fc_1",
                "call_id": "call_1",
                "name": name,
                "arguments": json.dumps(arguments),
            }
        ])

    @staticmethod
    def request(text: str, *, user_id: int = 42) -> ReplyRequest:
        return ReplyRequest(
            (CHAT["id"], 7),
            text,
            user_id=user_id,
            user_name="Sender",
            sent_at=NOW,
            is_whisper=True,
        )

    async def test_hidden_question_and_answer_do_not_enter_later_public_context(
        self,
    ) -> None:
        self.clock = NOW + timedelta(seconds=30)
        await self.feed("/whisper@dashulya_bot Секретный вопрос")
        (answer,) = self.sent()
        assert isinstance(answer, SendMessage)
        assert answer.text == "Private answer"
        assert answer.chat_id == -100123
        assert answer.message_thread_id == 7
        assert answer.ephemeral_message_parameters is not None
        assert answer.ephemeral_message_parameters.receiver_user_id == 99
        assert answer.reply_parameters is None
        assert not any(
            isinstance(item, (SendChatAction, SetMessageReaction)) for item in self.api
        )
        await self.feed("@dashulya_bot Public follow-up", hidden=False)
        assert len(self.inputs) == 2
        public_input = json.dumps(self.inputs[-1], ensure_ascii=False)
        assert "Секретный вопрос" not in public_input
        assert "Private answer" not in public_input
        assert "Public follow-up" in public_input

    async def test_public_whisper_is_deleted_and_reply_snapshots_cannot_restore_it(
        self,
    ) -> None:
        await self.feed("/whisper PUBLIC QUESTION", hidden=False)
        sent = self.sent()
        assert len(sent) == 1
        assert sent[0].ephemeral_message_parameters is not None
        deletion = next(item for item in self.api if isinstance(item, DeleteMessage))
        assert (deletion.chat_id, deletion.message_id) == (-100123, 1)
        assert self.api.index(deletion) < self.api.index(sent[0])
        await self.feed(
            "@dashulya_bot Public follow-up",
            hidden=False,
            reply_to_message={
                "message_id": 1,
                "date": NOW,
                "chat": CHAT,
                "from": PARTICIPANT,
                "text": "/whisper PUBLIC QUESTION",
                "message_thread_id": 7,
                "is_topic_message": True,
            },
        )
        assert len(self.inputs) == 2
        public_input = json.dumps(self.inputs[-1])
        assert "PUBLIC QUESTION" not in public_input
        assert "Private answer" not in public_input

    async def test_public_question_without_administrator_rights_explains_required_setup(
        self,
    ) -> None:
        self.admin = False
        await self.feed("/whisper PUBLIC SECRET", hidden=False)
        sent = self.sent()
        assert len(sent) == 1, (
            "A public whisper must explain missing permissions instead of silently disappearing"
        )
        assert isinstance(sent[0], SendMessage)
        assert sent[0].text == (
            "Чтобы ответить на публичную команду /whisper скрытым сообщением в этой группе, мне нужны права администратора. "
            "Если нужно ещё и удалить вопрос, разреши мне удалять сообщения. "
            "Без этих прав выбери скрытую команду /whisper из меню в поддерживающем её клиенте Telegram."
        )
        assert "PUBLIC SECRET" not in sent[0].text
        assert self.inputs == []
        assert not any(isinstance(item, DeleteMessage) for item in self.api)

    @pytest.mark.parametrize("blocked", ["paused", "banned", "bot", "anonymous"])
    async def test_whisper_respects_group_access(self, blocked: str) -> None:
        fields: dict[str, Any] = {}
        if blocked == "paused":
            await self.groups.deactivate(CHAT["id"])
        elif blocked == "banned":
            await self.groups.ban(CHAT["id"], PARTICIPANT["id"])
        elif blocked == "bot":
            fields["from"] = {**PARTICIPANT, "is_bot": True}
        else:
            fields["sender_chat"] = CHAT
        await self.feed("/whisper Restricted question", **fields)
        assert self.sent() == []
        assert self.inputs == []

    @pytest.mark.parametrize("timing", ["fast", "slow", "expired"])
    async def test_nonadministrator_uses_hidden_reply_window(self, timing: str) -> None:
        self.admin = False
        if timing == "slow":
            self.clock = NOW + timedelta(seconds=11.5)
            self.block_model = True
        elif timing == "expired":
            self.clock = NOW + timedelta(seconds=16)
        await self.feed("/whisper Question")
        if timing == "expired":
            assert self.sent() == []
            assert self.inputs == []
            return
        sent = self.sent()
        assert len(sent) == 1
        answer = sent[0]
        assert isinstance(answer, SendMessage)
        assert answer.ephemeral_message_parameters is not None
        assert answer.ephemeral_message_parameters.receiver_user_id == 99
        assert answer.reply_parameters is not None
        assert answer.reply_parameters.ephemeral_message_id == 700
        assert answer.reply_parameters.message_id is None
        if timing == "slow":
            assert self.model_cancelled.is_set()
            assert (
                answer.text
                == "Для этого ответа нужно больше времени. Выдай мне права администратора группы, чтобы я могла отправлять долгие скрытые ответы."
            )
        else:
            assert answer.text == "Private answer"

    async def test_send_failure_does_not_retry_or_publish_the_private_answer(
        self,
    ) -> None:
        self.fail_send = True
        await self.feed("/whisper Secret")
        sent = self.sent()
        assert len(sent) == 1
        assert sent[0].ephemeral_message_parameters is not None
        assert len(self.inputs) == 1

    async def test_startup_registers_whisper_as_a_hidden_group_command(self) -> None:
        await self.dispatcher.emit_startup(bot=self.bot)
        registrations = [item for item in self.api if isinstance(item, SetMyCommands)]
        assert len(registrations) == 1
        registration = registrations[0]
        command = next(
            item for item in registration.commands if item.command == "whisper"
        )
        assert command.is_ephemeral is True
        assert command.description == "Задать Дашуле скрытый вопрос в группе"
        assert registration.scope is not None
        assert registration.scope.type == "all_group_chats"

    async def test_every_text_chunk_and_sticker_remains_private(self) -> None:
        self.outputs.append([
            {
                "type": "message",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "annotations": [],
                        "text": json.dumps({
                            "messages": [
                                {"text": "x" * 5000, "reply_to_message_id": 777},
                                {"sticker": "love", "reply_to_message_id": 888},
                            ]
                        }),
                    }
                ],
            }
        ])
        await self.feed("/whisper Long answer")
        sent = self.sent()
        assert len(sent) == 3
        assert (
            "".join(item.text for item in sent if isinstance(item, SendMessage))
            == "x" * 5000
        )
        assert isinstance(sent[-1], SendSticker)
        assert sent[-1].sticker == "love-file"
        for item in sent:
            assert item.ephemeral_message_parameters is not None
            assert item.ephemeral_message_parameters.receiver_user_id == 99
            assert item.reply_parameters is None
            assert item.chat_id == -100123
            assert item.message_thread_id == 7

    async def test_owner_whisper_context_and_history_stay_in_group_before_request(
        self,
    ) -> None:
        original = MemoryMessage(
            CHAT["id"],
            1,
            "Visible original",
            NOW - timedelta(minutes=5),
            99,
            "Boris",
            "Group",
            thread_id=7,
        )
        for message in (
            original,
            replace(original, chat_id=-100456, text="FOREIGN SECRET"),
            replace(original, chat_id=42, text="OWNER PRIVATE SECRET"),
            replace(
                original,
                message_id=2,
                text="FUTURE SECRET",
                sent_at=NOW + timedelta(seconds=1),
            ),
            replace(
                original,
                message_id=3,
                text="EDITED SECRET",
                edited_at=NOW + timedelta(seconds=1),
            ),
            replace(original, message_id=4, text="OTHER TOPIC", thread_id=8),
        ):
            await self.store.record(message)
        self.tool(
            "chat_history",
            {"sql": ["SELECT chat_id,text FROM messages ORDER BY message_id LIMIT 50"]},
        )
        result = await self.dialogue.reply(self.request("Read the saved discussion"))
        assert result.messages[0].text == "Private answer"
        assert len(self.inputs) == 2
        initial_context = json.dumps(self.inputs[0])
        assert "Visible original" in initial_context
        assert "OTHER TOPIC" not in initial_context
        tool_input = json.dumps(self.inputs[-1])
        assert any(item.get("type") == "function_call_output" for item in self.inputs[-1])
        assert "Visible original" in tool_input
        for secret in (
            "FOREIGN SECRET",
            "OWNER PRIVATE SECRET",
            "FUTURE SECRET",
            "EDITED SECRET",
        ):
            assert secret not in initial_context
            assert secret not in tool_input
        assert "OTHER TOPIC" in tool_input

    async def test_whisper_turns_do_not_retain_other_private_questions_or_answers(
        self,
    ) -> None:
        self.answer = "FIRST PRIVATE ANSWER"
        await self.dialogue.reply(self.request("FIRST PRIVATE QUESTION"))
        self.answer = "Second answer"
        result = await self.dialogue.reply(
            self.request("SECOND PRIVATE QUESTION", user_id=99)
        )
        assert result.messages[0].text == "Second answer"
        second_input = json.dumps(self.inputs[-1])
        assert "SECOND PRIVATE QUESTION" in second_input
        assert "FIRST PRIVATE QUESTION" not in second_input
        assert "FIRST PRIVATE ANSWER" not in second_input

    async def test_whisper_reaction_tool_cannot_publish_a_public_reaction(self) -> None:
        self.tool("set_reaction", {"message_id": 20, "emoji": "🔥"})
        result = await self.dialogue.reply(self.request("React privately"))
        assert result.messages[0].text == "Private answer"
        assert not any(isinstance(item, SetMessageReaction) for item in self.api)
        assert "missing_chat_context" in json.dumps(self.inputs[-1])

    @pytest.mark.parametrize("text", ["/whisper", "/stop"])
    async def test_empty_or_administrative_hidden_command_only_shows_private_usage(
        self, text: str
    ) -> None:
        await self.feed(text, sender=OWNER)
        assert await self.groups.is_active(CHAT["id"])
        assert self.inputs == []
        sent = self.sent()
        assert len(sent) == 1
        assert isinstance(sent[0], SendMessage)
        assert sent[0].text.startswith("Выбери /whisper")
        assert sent[0].ephemeral_message_parameters is not None
        assert sent[0].ephemeral_message_parameters.receiver_user_id == 42
