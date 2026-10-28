import asyncio
import unittest
from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

from ag2 import Agent
from ag2.events import ToolCallEvent
from ag2.testing import TestConfig

from dashulya.ai.dialogue import Dialogue
from dashulya.domain.memory import (
    ChatMemoryService,
    HistoryReader,
    MemoryAccess,
    MemoryBackend,
)
from dashulya.domain.messages import Reply
from dashulya.main import create_container
from dashulya.main.config import Settings
from tests.support import TEST_API_KEY, MemoryOverride, ModelOverrides, reply_json

NOW = datetime(2026, 10, 27, tzinfo=UTC)


class DialogueRecallTests(unittest.IsolatedAsyncioTestCase):
    async def dialogue(
        self, *responses: str | ToolCallEvent
    ) -> tuple[Dialogue, Agent[Reply], ChatMemoryService]:
        self.backend = AsyncMock(spec=MemoryBackend)
        self.backend.search.return_value = {"coverage": {}, "hits": []}
        self.reader = AsyncMock(spec=HistoryReader)
        self.reader.query.return_value = {"count": 1, "coverage": {}}
        container = create_container(
            Settings(bot_token="123:TEST", openai_api_key=TEST_API_KEY, owner_user_id=42),
            MemoryOverride(self.backend, reader=self.reader),
            ModelOverrides(reply=TestConfig(*responses), decision=TestConfig("false")),
        )
        self.addAsyncCleanup(container.close)
        return (
            await container.get(Dialogue),
            await container.get(Agent[Reply]),
            await container.get(ChatMemoryService),
        )

    @staticmethod
    async def reply(dialogue: Dialogue, *, chat_id: int = 42) -> Reply:
        return await dialogue.reply(
            (chat_id, None),
            "Current request",
            user_id=42,
            user_name="Owner",
            sent_at=NOW,
            message_id=2,
            group_context={"message_id": 2} if chat_id < 0 else None,
        )

    async def test_private_and_group_answers_do_not_bind_or_search_memory_without_tool_calls(
        self,
    ) -> None:
        dialogue, _, service = await self.dialogue(
            reply_json("Answer"), reply_json("Answer")
        )
        with patch.object(service, "recall", wraps=service.recall) as recall:
            for chat_id in (42, -123):
                self.assertEqual(
                    await self.reply(dialogue, chat_id=chat_id), Reply("Answer")
                )
        recall.assert_not_called()
        self.backend.search.assert_not_awaited()
        self.reader.query.assert_not_awaited()

    async def test_model_selected_memory_and_history_share_current_trusted_scope(
        self,
    ) -> None:
        dialogue, agent, service = await self.dialogue(
            ToolCallEvent(
                "search_memory", arguments='{"query":"Что мы решили про RTK?"}'
            ),
            ToolCallEvent("chat_history", arguments='{"action":"count"}'),
            reply_json("Answer", 999),
        )
        with (
            patch.object(service, "recall", wraps=service.recall) as recall,
            patch.object(service, "history", wraps=service.history) as history,
            patch.object(agent, "ask", wraps=agent.ask) as ask,
            self.assertLogs("dashulya.ai.agents.tracing", level="INFO") as logs,
        ):
            self.assertEqual(await self.reply(dialogue, chat_id=-123), Reply("Answer"))
        access = ask.call_args.kwargs["dependencies"]["memory_access"]
        self.assertEqual(access, MemoryAccess(-123, False, before=NOW, before_id=2))
        recall.assert_called_once_with(access)
        history.assert_called_once_with(access)
        self.assertEqual(self.backend.search.await_count, 1)
        self.assertEqual(self.backend.search.await_args.args[0], "Что мы решили про RTK?")
        self.assertEqual(self.reader.query.await_count, 1)
        self.assertNotIn("Initial recall", ask.call_args.args[0])
        tools = [
            vars(record)
            for record in logs.records
            if record.message == "AI tool call completed"
        ]
        self.assertEqual(
            [record["tool_name"] for record in tools], ["search_memory", "chat_history"]
        )
        self.assertEqual([record["outcome"] for record in tools], ["ok", "ok"])
        self.assertTrue(
            all(
                record["chat_id"] == -123 and record["message_id"] == 2
                for record in tools
            )
        )

    async def test_history_only_request_does_not_invoke_semantic_memory(self) -> None:
        dialogue, _, service = await self.dialogue(
            ToolCallEvent("chat_history", arguments='{"action":"count"}'),
            reply_json("Counted"),
        )
        with patch.object(
            service, "recall", side_effect=AssertionError("Recall was not requested")
        ):
            self.assertEqual((await self.reply(dialogue)).text, "Counted")
        self.reader.query.assert_awaited_once()
        self.backend.search.assert_not_awaited()

    async def test_casual_followups_do_not_accumulate_automatic_recall(self) -> None:
        dialogue, agent, _ = await self.dialogue(
            reply_json("First"), reply_json("Second")
        )
        with patch.object(agent, "ask", wraps=agent.ask) as ask:
            await self.reply(dialogue)
            stream = ask.call_args.kwargs["stream"]
            await self.reply(dialogue)
            self.assertIs(ask.call_args.kwargs["stream"], stream)
            self.assertNotIn("Initial recall", ask.call_args.args[0])
        self.backend.search.assert_not_awaited()

    async def test_turn_deadline_cancels_a_selected_memory_tool(self) -> None:
        dialogue, _, _ = await self.dialogue(
            ToolCallEvent("search_memory", arguments='{"query":"Earlier discussion"}'),
            reply_json("Unreachable"),
        )
        cancelled = asyncio.Event()

        async def stalled(query: str, **kwargs: object) -> dict[str, object]:
            try:
                await asyncio.Future[None]()
            finally:
                cancelled.set()
            return {}

        self.backend.search.side_effect = stalled
        timeout = asyncio.timeout
        with (
            patch(
                "dashulya.ai.dialogue.asyncio.timeout",
                side_effect=lambda seconds: timeout(0.1),
            ),
            self.assertRaises(TimeoutError),
        ):
            await self.reply(dialogue)
        self.assertTrue(cancelled.is_set())
