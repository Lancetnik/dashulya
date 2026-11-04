import asyncio
import json
from contextlib import AsyncExitStack
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, patch

import httpx2
import pytest
from ag2.config import OpenAIResponsesConfig
from ag2.testing import TestConfig
from aiogram import Bot
from aiogram.methods import SendMessage, TelegramMethod
from aiogram.types import Message

from dashulya.bot.middlewares.memory import CaptureSentMessagesMiddleware
from dashulya.bot.replies import ReplyDelivery
from dashulya.domain.memory import MemoryMessage, MessageRecorder
from dashulya.main import create_container
from dashulya.main.config import Settings
from tests.support import (
    TEST_API_KEY,
    BotOverride,
    MemoryOverride,
    ModelOverrides,
    activated_group_registry,
)


@pytest.mark.usefixtures("_setup")
class TestConcurrentReplies:
    @pytest.fixture
    async def _setup(self, cleanup: AsyncExitStack) -> None:
        self.slow_started = asyncio.Event()
        self.fast_started = asyncio.Event()
        self.release_model = asyncio.Event()
        self.release_model.set()
        self.inputs: dict[str, list[dict[str, Any]]] = {}
        self.sent: list[SendMessage] = []
        self.recorded: list[str] = []
        self.recording_started = asyncio.Event()
        self.release_recording = asyncio.Event()
        self.release_recording.set()
        self.tasks: list[asyncio.Task[None]] = []

        async def respond(request: httpx2.Request) -> httpx2.Response:
            body = json.loads(request.content)
            label = "Slow" if "Slow request" in str(body["input"][-1]) else "Fast"
            self.inputs[label] = body["input"]
            if label == "Slow":
                self.slow_started.set()
                await self.release_model.wait()
            else:
                self.fast_started.set()
            return httpx2.Response(
                200,
                json={
                    "id": f"resp_{label}",
                    "object": "response",
                    "created_at": 1700000000,
                    "status": "completed",
                    "model": "test-model",
                    "output": [
                        {
                            "type": "message",
                            "role": "assistant",
                            "content": [
                                {
                                    "type": "output_text",
                                    "text": json.dumps({
                                        "messages": [
                                            {"text": f"{label} one"},
                                            {"text": f"{label} two"},
                                        ]
                                    }),
                                    "annotations": [],
                                }
                            ],
                        }
                    ],
                },
            )

        async def send(bot: Bot, method: TelegramMethod[Any], **kwargs: Any) -> Any:
            if not isinstance(method, SendMessage):
                return True
            self.sent.append(method)
            return Message.model_validate({
                "message_id": 100 + len(self.sent),
                "date": datetime(2026, 11, 1, tzinfo=UTC),
                "chat": self.incoming(1, int(method.chat_id)).chat,
                "from": {"id": bot.id, "is_bot": True, "first_name": "Dashulya"},
                "text": method.text,
            })

        async def record(message: MemoryMessage, **kwargs: Any) -> None:
            if message.text == "Slow one":
                self.recording_started.set()
                await self.release_recording.wait()
            self.recorded.append(message.text)

        bot = Bot("123:TEST")
        cleanup.push_async_callback(bot.session.close)
        self.bot = bot
        cleanup.enter_context(patch.object(bot.session, "make_request", side_effect=send))
        recorder = AsyncMock(spec=MessageRecorder)
        recorder.record.side_effect = record
        bot.session.middleware(
            CaptureSentMessagesMiddleware(
                recorder, activated_group_registry(), owner_user_id=42
            )
        )
        http = await cleanup.enter_async_context(
            httpx2.AsyncClient(transport=httpx2.MockTransport(respond))
        )
        container = create_container(
            Settings(bot_token="123:TEST", openai_api_key=TEST_API_KEY, owner_user_id=42),
            BotOverride(bot),
            MemoryOverride(),
            ModelOverrides(
                reply=OpenAIResponsesConfig(
                    model="test-model", api_key=TEST_API_KEY, http_client=http
                ),
                decision=TestConfig("false"),
            ),
        )
        cleanup.push_async_callback(container.close)
        self.delivery = await container.get(ReplyDelivery)
        cleanup.push_async_callback(self.finish_tasks)

    async def finish_tasks(self) -> None:
        self.release_model.set()
        self.release_recording.set()
        await asyncio.gather(*self.tasks, return_exceptions=True)

    def incoming(self, message_id: int, chat_id: int) -> Message:
        return Message.model_validate({
            "message_id": message_id,
            "date": datetime(2026, 11, 1, tzinfo=UTC),
            "chat": {"id": chat_id, "type": "supergroup", "title": "Group"}
            if chat_id < 0
            else {"id": chat_id, "type": "private", "first_name": "Owner"},
            "from": {
                "id": message_id + 40 if chat_id < 0 else 42,
                "is_bot": False,
                "first_name": f"User {message_id}",
            },
            "text": "Slow request" if message_id == 1 else "Fast request",
        }).as_(self.bot)

    def start_reply(self, message_id: int, chat_id: int) -> asyncio.Task[None]:
        task = asyncio.create_task(
            self.delivery.reply(self.incoming(message_id, chat_id))
        )
        self.tasks.append(task)
        return task

    @pytest.mark.parametrize("chat_id", [-123, 42])
    async def test_group_generation_overlaps_while_private_history_stays_serial(
        self, chat_id: int
    ) -> None:
        self.release_model.clear()
        slow = self.start_reply(1, chat_id)
        await asyncio.wait_for(self.slow_started.wait(), 1)
        fast = self.start_reply(2, chat_id)
        try:
            await asyncio.wait({fast}, timeout=0.3)
            if chat_id < 0:
                assert fast.done(), "A group request waited for another generation"
                assert [item.text for item in self.sent] == ["Fast one", "Fast two"]
                assert "Slow request" not in str(self.inputs["Fast"])
                assert "Interlocutor user id: 42" in str(self.inputs["Fast"])
            else:
                assert not self.fast_started.is_set(), "Private history ran concurrently"
                assert self.sent == []
        finally:
            self.release_model.set()
            await asyncio.gather(slow, fast)
        expected = (
            ["Fast one", "Fast two", "Slow one", "Slow two"]
            if chat_id < 0
            else ["Slow one", "Slow two", "Fast one", "Fast two"]
        )
        assert [item.text for item in self.sent] == expected
        assert self.recorded == expected
        assert [
            item.reply_parameters.message_id if item.reply_parameters else None
            for item in self.sent
        ] == ([2, None, 1, None] if chat_id < 0 else [1, None, 2, None])
        if chat_id > 0:
            assert "Slow request" in str(self.inputs["Fast"])
            assert "Slow two" in str(self.inputs["Fast"])

    @pytest.mark.parametrize("chat_id", [-123, 42])
    async def test_reply_batches_stay_together_until_originals_are_recorded(
        self, chat_id: int
    ) -> None:
        self.release_recording.clear()
        slow = self.start_reply(1, chat_id)
        await asyncio.wait_for(self.recording_started.wait(), 1)
        fast = self.start_reply(2, chat_id)
        try:
            await asyncio.wait({fast}, timeout=0.3)
            assert self.fast_started.is_set() == (chat_id < 0), (
                "Group generation must overlap recording; private history must wait"
            )
            assert [item.text for item in self.sent] == ["Slow one"], (
                "Another reply interrupted a batch awaiting persistence"
            )
            assert self.recorded == []
            assert not fast.done()
        finally:
            self.release_recording.set()
            await asyncio.gather(slow, fast)
        expected = ["Slow one", "Slow two", "Fast one", "Fast two"]
        assert [item.text for item in self.sent] == expected
        assert self.recorded == expected
