import asyncio
import json
import unittest
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, patch

from ag2 import Agent, MemoryStream
from ag2.events import DataInput, ToolCallEvent, ToolResultEvent
from ag2.testing import TestConfig, TrackingConfig, Turn
from ag2.tools.final.function_tool import FunctionToolSchema
from aiogram import Bot
from aiogram.exceptions import TelegramBadRequest, TelegramNetworkError
from aiogram.methods import (
    SendChatAction,
    SendMessage,
    SendRichMessage,
    SetMessageReaction,
)
from aiogram.types import Message, ReactionTypeEmoji

from dashulya.ai.dialogue import Dialogue
from dashulya.ai.tools.reactions import ReactionTurn
from dashulya.bot.chat_actions import typing
from dashulya.bot.reactions import TelegramReactions
from dashulya.bot.replies import ReplyDelivery
from dashulya.domain.messages import Reply
from dashulya.main import create_container
from dashulya.main.config import Settings
from dashulya.observability import JsonLogFormatter
from tests.support import (
    TEST_API_KEY,
    BotOverride,
    MemoryOverride,
    ModelOverrides,
    reply_json,
)

SENT_AT = datetime(2026, 10, 27, 12, tzinfo=UTC)


def react(message_id: int, emoji: str = "🔥", **extra: object) -> ToolCallEvent:
    return ToolCallEvent(
        "set_reaction",
        arguments=json.dumps({"message_id": message_id, "emoji": emoji, **extra}),
    )


class ReactionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.bot = Bot("123:TEST")
        self.addAsyncCleanup(self.bot.session.close)
        self.requests = AsyncMock(return_value=True)
        request_patch = patch.object(self.bot.session, "make_request", self.requests)
        request_patch.start()
        self.addCleanup(request_patch.stop)

    async def dialogue(
        self, *turns: Turn
    ) -> tuple[Dialogue, TrackingConfig, Agent[Reply]]:
        config = TrackingConfig(TestConfig(*turns))
        container = create_container(
            Settings(bot_token="123:TEST", openai_api_key=TEST_API_KEY, owner_user_id=42),
            BotOverride(self.bot),
            MemoryOverride(),
            ModelOverrides(reply=config, decision=TestConfig("false")),
        )
        self.addAsyncCleanup(container.close)
        return await container.get(Dialogue), config, await container.get(Agent[Reply])

    async def reply(self, dialogue: Dialogue, chat_id: int = -123) -> Reply:
        return await dialogue.reply(
            (chat_id, None),
            "Please react",
            user_id=42,
            user_name="Anna",
            sent_at=SENT_AT,
            message_id=10,
            group_context={"message_id": 10} if chat_id < 0 else None,
        )

    async def deliver(self, *turns: Turn, chat_id: int = -123) -> None:
        dialogue, _, _ = await self.dialogue(*turns)
        images = AsyncMock()
        images.load.return_value = ()
        delivery = ReplyDelivery(self.bot, dialogue, images)
        message = Message.model_validate({
            "message_id": 10,
            "date": SENT_AT,
            "chat": {
                "id": chat_id,
                "type": "supergroup" if chat_id < 0 else "private",
                "title": "Test",
            },
            "from": {"id": 42, "is_bot": False, "first_name": "Anna"},
            "text": "Please react",
        }).as_(self.bot)
        await delivery.reply(
            message, group_context={"message_id": 10} if chat_id < 0 else None
        )

    def reactions(self) -> list[SetMessageReaction]:
        return [
            call.args[1]
            for call in self.requests.call_args_list
            if isinstance(call.args[1], SetMessageReaction)
        ]

    def messages(self) -> list[SendMessage | SendRichMessage]:
        return [
            call.args[1]
            for call in self.requests.call_args_list
            if isinstance(call.args[1], (SendMessage, SendRichMessage))
        ]

    async def test_tool_reacts_to_current_request_without_text(self) -> None:
        for chat_id in (-123, 42):
            with self.subTest(chat_id=chat_id):
                self.requests.reset_mock()
                await self.deliver(react(10, "❤️"), reply_json(""), chat_id=chat_id)
                self.assertEqual(len(self.reactions()), 1)
                reaction = self.reactions()[0]
                self.assertEqual((reaction.chat_id, reaction.message_id), (chat_id, 10))
                assert reaction.reaction is not None
                assert isinstance(reaction.reaction[0], ReactionTypeEmoji)
                self.assertEqual(reaction.reaction[0].emoji, "❤")
                self.assertEqual(self.messages(), [])

    async def test_confirmed_reactions_log_delivery(self) -> None:
        for turns, target in (
            ((react(10, "❤️"), reply_json("")), 10),
            ((react(1, "❤️"), reply_json("")), 1),
        ):
            with (
                self.subTest(target=target),
                self.assertLogs("dashulya.bot.reactions", level="INFO") as logs,
            ):
                await self.deliver(*turns)
            records = [
                json.loads(JsonLogFormatter().format(record)) for record in logs.records
            ]
            delivered = [
                record for record in records if record["message"] == "Reaction delivered"
            ]
            self.assertEqual(len(delivered), 1)
            self.assertEqual(delivered[0]["chat_id"], -123)
            self.assertEqual(delivered[0]["message_id"], target)
            self.assertEqual(delivered[0]["emoji"], "❤")
            self.assertEqual(delivered[0]["outcome"], "sent")
            self.assertGreaterEqual(delivered[0]["duration_ms"], 0)

    async def test_typing_network_error_does_not_leak_task_exception_or_block_reply(
        self,
    ) -> None:
        failed = asyncio.Event()
        loop = asyncio.get_running_loop()
        errors: list[str] = []
        previous_handler = loop.get_exception_handler()
        loop.set_exception_handler(
            lambda event_loop, context: errors.append(str(context.get("message")))
        )
        self.addCleanup(loop.set_exception_handler, previous_handler)
        real_ask = Agent.ask

        async def request(bot: Bot, method: Any, **kwargs: Any) -> bool:
            if isinstance(method, SendChatAction):
                failed.set()
                raise TelegramNetworkError(method, "PRIVATE TOKEN")
            return True

        async def answer_after_failure(
            agent: Agent[Any], *args: Any, **kwargs: Any
        ) -> Any:
            await failed.wait()
            return await real_ask(agent, *args, **kwargs)

        self.requests.side_effect = request
        with (
            patch.object(Agent, "ask", answer_after_failure),
            self.assertLogs("dashulya.bot", level="INFO") as logs,
        ):
            await self.deliver(reply_json("Answer survived"))
            await asyncio.sleep(0)
        self.assertEqual(errors, [])
        self.assertEqual(len(self.messages()), 1)
        self.assertNotIn("PRIVATE TOKEN", str(logs.output))

    async def test_reply_cancels_stalled_typing_before_delivering_text(self) -> None:
        started, cancelled = asyncio.Event(), asyncio.Event()
        real_ask = Agent.ask

        async def request(bot: Bot, method: Any, **kwargs: Any) -> bool:
            if isinstance(method, SendChatAction):
                started.set()
                try:
                    await asyncio.Future[None]()
                finally:
                    cancelled.set()
            if isinstance(method, SendMessage):
                self.assertTrue(cancelled.is_set())
            return True

        async def answer_after_typing_started(
            agent: Agent[Any], *args: Any, **kwargs: Any
        ) -> Any:
            await started.wait()
            return await real_ask(agent, *args, **kwargs)

        self.requests.side_effect = request
        with patch.object(Agent, "ask", answer_after_typing_started):
            await asyncio.wait_for(self.deliver(reply_json("Ready")), timeout=1)
        self.assertTrue(cancelled.is_set())
        self.assertEqual(len(self.messages()), 1)

    async def test_reply_cancellation_also_joins_the_typing_task(self) -> None:
        started, cancelled = asyncio.Event(), asyncio.Event()

        async def request(bot: Bot, method: Any, **kwargs: Any) -> bool:
            started.set()
            try:
                await asyncio.Future[None]()
            finally:
                cancelled.set()
            return True

        self.requests.side_effect = request
        message = Message.model_validate({
            "message_id": 10,
            "date": SENT_AT,
            "chat": {"id": 42, "type": "private"},
        })

        async def interrupted_reply() -> None:
            async with typing(self.bot, message):
                await asyncio.Future[None]()

        task = asyncio.create_task(interrupted_reply())
        await asyncio.wait_for(started.wait(), timeout=1)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertTrue(cancelled.is_set())

    async def test_negative_telegram_ack_is_logged_as_rejected_not_delivered(
        self,
    ) -> None:
        self.requests.return_value = False
        with self.assertLogs("dashulya.bot.reactions", level="WARNING") as logs:
            result = await TelegramReactions(self.bot).set(-123, 10, "👍")
        event = json.loads(JsonLogFormatter().format(logs.records[-1]))
        self.assertEqual(result.status, "rejected")
        self.assertEqual(event["outcome"], "rejected")
        self.assertEqual(event["message"], "Reaction rejected")
        self.assertNotIn("Reaction delivered", str(logs.output))

    async def test_text_response_does_not_set_a_reaction(self) -> None:
        await self.deliver(reply_json("A substantive answer"))
        self.assertEqual(self.reactions(), [])
        self.assertEqual(len(self.messages()), 1)

    async def test_agent_can_explain_rejection_in_its_own_words(self) -> None:
        async def request(bot: Bot, method: Any, **kwargs: Any) -> bool:
            if isinstance(method, SetMessageReaction):
                raise TelegramBadRequest(method, "PRIVATE TOKEN: REACTION_INVALID")
            return True

        self.requests.side_effect = request
        explanation = "I can't use reactions in this chat, but thank you!"
        with self.assertLogs("dashulya.bot.reactions", level="WARNING") as logs:
            await self.deliver(react(10, "😁"), reply_json(explanation))
        self.assertEqual(len(self.reactions()), 1)
        self.assertEqual(len(self.messages()), 1)
        self.assertNotIn("PRIVATE TOKEN", str(logs.output) + str(self.messages()))
        response = self.messages()[0]
        assert isinstance(response, SendMessage)
        self.assertEqual(response.text, explanation)

    async def test_tool_reaction_network_failure_never_sends_a_canned_response(
        self,
    ) -> None:
        async def request(bot: Bot, method: Any, **kwargs: Any) -> bool:
            if isinstance(method, SetMessageReaction):
                raise TelegramNetworkError(method, "PRIVATE TOKEN")
            return True

        self.requests.side_effect = request
        with self.assertLogs("dashulya.bot.reactions", level="WARNING"):
            await self.deliver(react(10, "👍"), reply_json(""))
        self.assertEqual(len(self.reactions()), 1)
        self.assertEqual(self.messages(), [])

    async def test_tool_reacts_to_multiple_ids_outside_reply_context_without_text(
        self,
    ) -> None:
        await self.deliver([react(1), react(765432, "🎉")], reply_json(""))
        self.assertEqual({method.message_id for method in self.reactions()}, {1, 765432})
        self.assertTrue(all(method.chat_id == -123 for method in self.reactions()))
        self.assertEqual(self.messages(), [])

    async def test_tool_reaction_can_be_followed_by_a_text_reply(self) -> None:
        await self.deliver(react(1), reply_json("That was useful."))
        self.assertEqual(len(self.reactions()), 1)
        self.assertEqual(len(self.messages()), 1)

    async def test_rejected_tool_can_be_corrected_without_a_canned_reply(self) -> None:
        async def request(bot: Bot, method: Any, **kwargs: Any) -> bool:
            if isinstance(method, SetMessageReaction):
                assert method.reaction is not None
                assert isinstance(method.reaction[0], ReactionTypeEmoji)
                if method.reaction[0].emoji == "😅":
                    raise TelegramBadRequest(method, "PRIVATE TOKEN: REACTION_INVALID")
            return True

        self.requests.side_effect = request
        with self.assertLogs("dashulya.bot.reactions", level="WARNING"):
            await self.deliver(react(10, "😅"), react(10, "😁"), reply_json(""))
        self.assertEqual(len(self.reactions()), 2)
        self.assertEqual(self.messages(), [])

    async def test_rejected_tool_does_not_create_a_canned_reply_on_empty_output(
        self,
    ) -> None:
        async def request(bot: Bot, method: Any, **kwargs: Any) -> bool:
            if isinstance(method, SetMessageReaction):
                raise TelegramBadRequest(method, "PRIVATE TOKEN: REACTION_INVALID")
            return True

        self.requests.side_effect = request
        with self.assertLogs("dashulya.bot.reactions", level="WARNING"):
            await self.deliver(react(10, "😅"), reply_json(""))
        self.assertEqual(self.messages(), [])

    async def test_tool_description_lists_supported_reactions_and_recovery(self) -> None:
        dialogue, config, _ = await self.dialogue(react(1), reply_json(""))
        await self.reply(dialogue)
        schema = next(
            item
            for item in config.calls[0].tools
            if isinstance(item, FunctionToolSchema)
            and item.function.name == "set_reaction"
        )
        description = schema.function.description or ""
        for emoji in ("😁", "❤", "👏", "🤝", "😡"):
            self.assertIn(emoji, description)
        self.assertIn("another allowed emoji", description)

    async def test_tool_schema_exposes_only_message_id_and_emoji(self) -> None:
        dialogue, config, _ = await self.dialogue(react(1), reply_json(""))
        await self.reply(dialogue)
        schema = next(
            item
            for item in config.calls[0].tools
            if isinstance(item, FunctionToolSchema)
            and item.function.name == "set_reaction"
        )
        self.assertEqual(
            set(schema.function.parameters["properties"]), {"message_id", "emoji"}
        )
        self.assertEqual(config.calls[0].variables["chat_id"], -123)

    async def test_concurrent_turns_bind_reactions_to_their_context_variable(
        self,
    ) -> None:
        dialogue, _, _ = await self.dialogue(react(765432), reply_json(""))
        results = await asyncio.gather(
            self.reply(dialogue, -123), self.reply(dialogue, -456)
        )
        self.assertEqual(results, [Reply(""), Reply("")])
        self.assertEqual(
            {(method.chat_id, method.message_id) for method in self.reactions()},
            {(-123, 765432), (-456, 765432)},
        )

    async def test_model_cannot_override_chat_identity_in_tool_arguments(self) -> None:
        dialogue, _, _ = await self.dialogue(react(765432, chat_id=-456), reply_json(""))
        await self.reply(dialogue)
        self.assertEqual(
            [(method.chat_id, method.message_id) for method in self.reactions()],
            [(-123, 765432)],
        )

    async def test_missing_or_invalid_context_variable_never_calls_telegram(self) -> None:
        for variables in ({}, {"chat_id": 0}, {"chat_id": "-123"}, {"chat_id": True}):
            with self.subTest(variables=variables):
                _, _, agent = await self.dialogue(react(1), reply_json("Unavailable"))
                stream = MemoryStream()
                await agent.ask(
                    "Please react",
                    stream=stream,
                    variables=variables,
                    dependencies={ReactionTurn: ReactionTurn()},
                )
                results = [
                    event
                    for event in await stream.history.get_events()
                    if isinstance(event, ToolResultEvent)
                ]
                part = results[0].result.parts[0]
                assert isinstance(part, DataInput)
                assert isinstance(part.data, dict)
                self.assertEqual(part.data["reason"], "missing_chat_context")
        self.assertEqual(self.reactions(), [])

    async def test_tool_failure_returns_redacted_error_without_a_canned_reply(
        self,
    ) -> None:
        method = SetMessageReaction(chat_id=-123, message_id=1)
        self.requests.side_effect = TelegramBadRequest(method, "PRIVATE TOKEN")
        dialogue, _, agent = await self.dialogue(react(1), reply_json(""))
        ask = AsyncMock(wraps=agent.ask)
        with (
            patch.object(agent, "ask", ask),
            self.assertLogs("dashulya.bot.reactions", level="WARNING") as logs,
        ):
            result = await self.reply(dialogue)
        self.assertEqual(result, Reply(""))
        stream = ask.call_args.kwargs["stream"]
        results = [
            event
            for event in await stream.history.get_events()
            if isinstance(event, ToolResultEvent)
        ]
        part = results[0].result.parts[0]
        assert isinstance(part, DataInput)
        assert isinstance(part.data, dict)
        self.assertEqual(part.data["status"], "rejected")
        self.assertEqual(part.data["reason"], "telegram_rejected")
        self.assertNotIn("PRIVATE TOKEN", str(logs.output) + str(part.data))

    async def test_unknown_tool_delivery_is_not_retried_with_another_emoji(self) -> None:
        method = SetMessageReaction(chat_id=-123, message_id=10)

        async def request(bot: Bot, call: Any, **kwargs: Any) -> bool:
            if isinstance(call, SetMessageReaction):
                raise TelegramNetworkError(method, "PRIVATE TOKEN")
            return True

        self.requests.side_effect = request
        with self.assertLogs("dashulya.bot.reactions", level="WARNING"):
            await self.deliver(react(10), react(10, "👍"), reply_json(""))
        self.assertEqual(len(self.reactions()), 1)
        self.assertEqual(self.messages(), [])

    async def test_turn_state_does_not_leak_to_a_later_empty_response(self) -> None:
        dialogue, _, agent = await self.dialogue(react(1), reply_json(""))
        self.assertEqual(await self.reply(dialogue, 42), Reply(""))
        with patch.object(agent, "ask", AsyncMock()) as ask:
            ask.return_value.content.return_value = Reply("")
            result = await self.reply(dialogue, 42)
        self.assertIn("couldn't generate", result.text)

    async def test_invalid_ids_and_empty_emoji_are_rejected_without_a_request(
        self,
    ) -> None:
        service = TelegramReactions(self.bot)
        for message_id, emoji in ((0, "👍"), (-1, "👍"), (1, ""), (1, " ")):
            with self.subTest(message_id=message_id, emoji=emoji):
                self.assertEqual(
                    (await service.set(-123, message_id, emoji)).status, "rejected"
                )
        self.requests.assert_not_awaited()

    async def test_cancelled_delivery_propagates_and_does_not_generate_fallback(
        self,
    ) -> None:
        self.requests.side_effect = asyncio.CancelledError()
        with self.assertRaises(asyncio.CancelledError):
            await TelegramReactions(self.bot).set(-123, 1, "👍")
        self.assertEqual(len(self.reactions()), 1)
        self.assertEqual(self.messages(), [])
