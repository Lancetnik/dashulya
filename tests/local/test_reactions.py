import asyncio
import json
from contextlib import AsyncExitStack
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from ag2 import Agent, MemoryStream
from ag2.events import DataInput, ToolCallEvent, ToolResultEvent
from ag2.testing import TestConfig, TrackingConfig, Turn
from ag2.tools.final.function_tool import FunctionToolSchema
from aiogram import Bot
from aiogram.exceptions import (
    TelegramAPIError,
    TelegramBadRequest,
    TelegramForbiddenError,
    TelegramNetworkError,
    TelegramRetryAfter,
    TelegramServerError,
    TelegramUnauthorizedError,
)
from aiogram.methods import (
    SendChatAction,
    SendMessage,
    SendRichMessage,
    SetMessageReaction,
)
from aiogram.types import Message, ReactionTypeEmoji
from dirty_equals import IsStr

from dashulya.ai.dialogue import Dialogue
from dashulya.ai.tools.reactions import ReactionTurn
from dashulya.bot.chat_actions import typing
from dashulya.bot.reactions import TelegramReactions
from dashulya.bot.replies import ReplyDelivery
from dashulya.domain.messages import Reply, ReplyRequest
from dashulya.domain.reactions import ReactionResult
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


@pytest.mark.usefixtures("_setup")
class TestReaction:
    @pytest.fixture
    async def _setup(self, cleanup: AsyncExitStack) -> None:
        self.cleanup = cleanup
        self.bot = Bot("123:TEST")
        self.cleanup.push_async_callback(self.bot.session.close)
        self.requests = AsyncMock(return_value=True)
        request_patch = patch.object(self.bot.session, "make_request", self.requests)
        request_patch.start()
        self.cleanup.callback(request_patch.stop)

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
        self.cleanup.push_async_callback(container.close)
        return await container.get(Dialogue), config, await container.get(Agent[Reply])

    async def reply(self, dialogue: Dialogue, chat_id: int = -123) -> Reply:
        return await dialogue.reply(
            ReplyRequest(
                (chat_id, None),
                "Please react",
                user_id=42,
                user_name="Anna",
                sent_at=SENT_AT,
                message_id=10,
            )
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
        await delivery.reply(message)

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

    async def test_tool_reacts_to_current_request_without_text(
        self, subtests: pytest.Subtests
    ) -> None:
        for chat_id in (-123, 42):
            with subtests.test(chat_id=chat_id):
                self.requests.reset_mock()
                await self.deliver(react(10, "❤️"), reply_json(""), chat_id=chat_id)
                assert len(self.reactions()) == 1
                reaction = self.reactions()[0]
                assert (reaction.chat_id, reaction.message_id) == (chat_id, 10)
                assert reaction.reaction is not None
                assert isinstance(reaction.reaction[0], ReactionTypeEmoji)
                assert reaction.reaction[0].emoji == "❤"
                assert self.messages() == []

    async def test_confirmed_reactions_log_delivery(
        self, caplog: pytest.LogCaptureFixture, subtests: pytest.Subtests
    ) -> None:
        for turns, target in (
            ((react(10, "❤️"), reply_json("")), 10),
            ((react(1, "❤️"), reply_json("")), 1),
        ):
            caplog.clear()
            with (
                subtests.test(target=target),
                caplog.at_level("INFO", logger="dashulya.bot.reactions"),
            ):
                await self.deliver(*turns)
            assert any(
                record.name == "dashulya.bot.reactions"
                or record.name.startswith("dashulya.bot.reactions.")
                for record in caplog.records
            )
            records = [
                json.loads(JsonLogFormatter().format(record)) for record in caplog.records
            ]
            delivered = [
                record for record in records if record["message"] == "Reaction delivered"
            ]
            assert len(delivered) == 1
            assert delivered[0]["chat_id"] == -123
            assert delivered[0]["message_id"] == target
            assert delivered[0]["emoji"] == "❤"
            assert delivered[0]["outcome"] == "sent"
            assert delivered[0]["duration_ms"] >= 0

    async def test_typing_network_error_does_not_leak_task_exception_or_block_reply(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        failed = asyncio.Event()
        loop = asyncio.get_running_loop()
        errors: list[str] = []
        previous_handler = loop.get_exception_handler()
        loop.set_exception_handler(
            lambda event_loop, context: errors.append(str(context.get("message")))
        )
        self.cleanup.callback(loop.set_exception_handler, previous_handler)
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
        caplog.clear()
        with (
            patch.object(Agent, "ask", answer_after_failure),
            caplog.at_level("INFO", logger="dashulya.bot"),
        ):
            await asyncio.wait_for(self.deliver(reply_json("Answer survived")), timeout=2)
            await asyncio.sleep(0)
        assert any(
            record.name == "dashulya.bot" or record.name.startswith("dashulya.bot.")
            for record in caplog.records
        )
        assert errors == []
        (message,) = self.messages()
        assert isinstance(message, SendMessage)
        assert message.text == "Answer survived"
        assert "PRIVATE TOKEN" not in str(caplog.messages)

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
                assert cancelled.is_set()
            return True

        async def answer_after_typing_started(
            agent: Agent[Any], *args: Any, **kwargs: Any
        ) -> Any:
            await started.wait()
            return await real_ask(agent, *args, **kwargs)

        self.requests.side_effect = request
        with patch.object(Agent, "ask", answer_after_typing_started):
            await asyncio.wait_for(self.deliver(reply_json("Ready")), timeout=1)
        assert cancelled.is_set()
        assert len(self.messages()) == 1

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
        with pytest.raises(asyncio.CancelledError):
            await task
        assert cancelled.is_set()

    async def test_negative_telegram_ack_is_logged_as_rejected_not_delivered(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        self.requests.return_value = False
        caplog.clear()
        with caplog.at_level("WARNING", logger="dashulya.bot.reactions"):
            result = await TelegramReactions(self.bot).set(-123, 10, "👍")
        assert any(
            record.name == "dashulya.bot.reactions"
            or record.name.startswith("dashulya.bot.reactions.")
            for record in caplog.records
        )
        event = json.loads(JsonLogFormatter().format(caplog.records[-1]))
        assert result.status == "rejected"
        assert event["outcome"] == "rejected"
        assert event["message"] == "Reaction rejected"
        assert "Reaction delivered" not in str(caplog.messages)

    async def test_text_response_does_not_set_a_reaction(self) -> None:
        await self.deliver(reply_json("A substantive answer"))
        assert self.reactions() == []
        assert len(self.messages()) == 1

    async def test_agent_can_explain_rejection_in_its_own_words(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        async def request(bot: Bot, method: Any, **kwargs: Any) -> bool:
            if isinstance(method, SetMessageReaction):
                raise TelegramBadRequest(method, "PRIVATE TOKEN: REACTION_INVALID")
            return True

        self.requests.side_effect = request
        explanation = "I can't use reactions in this chat, but thank you!"
        caplog.clear()
        with caplog.at_level("WARNING", logger="dashulya.bot.reactions"):
            await self.deliver(react(10, "😁"), reply_json(explanation))
        assert any(
            record.name == "dashulya.bot.reactions"
            or record.name.startswith("dashulya.bot.reactions.")
            for record in caplog.records
        )
        assert len(self.reactions()) == 1
        assert len(self.messages()) == 1
        assert "PRIVATE TOKEN" not in str(caplog.messages) + str(self.messages())
        response = self.messages()[0]
        assert isinstance(response, SendMessage)
        assert response.text == explanation

    async def test_tool_reaction_network_failure_never_sends_a_canned_response(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        async def request(bot: Bot, method: Any, **kwargs: Any) -> bool:
            if isinstance(method, SetMessageReaction):
                raise TelegramNetworkError(method, "PRIVATE TOKEN")
            return True

        self.requests.side_effect = request
        caplog.clear()
        with caplog.at_level("WARNING", logger="dashulya.bot.reactions"):
            await self.deliver(react(10, "👍"), reply_json(""))
        assert any(
            record.name == "dashulya.bot.reactions"
            or record.name.startswith("dashulya.bot.reactions.")
            for record in caplog.records
        )
        assert len(self.reactions()) == 1
        assert self.messages() == []

    async def test_tool_reacts_to_multiple_ids_outside_reply_context_without_text(
        self,
    ) -> None:
        await self.deliver([react(1), react(765432, "🎉")], reply_json(""))
        assert {method.message_id for method in self.reactions()} == {1, 765432}
        assert all(method.chat_id == -123 for method in self.reactions())
        assert self.messages() == []

    async def test_tool_reaction_can_be_followed_by_a_text_reply(self) -> None:
        await self.deliver(react(1), reply_json("That was useful."))
        assert len(self.reactions()) == 1
        assert len(self.messages()) == 1

    async def test_rejected_tool_can_be_corrected_without_a_canned_reply(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        async def request(bot: Bot, method: Any, **kwargs: Any) -> bool:
            if isinstance(method, SetMessageReaction):
                assert method.reaction is not None
                assert isinstance(method.reaction[0], ReactionTypeEmoji)
                if method.reaction[0].emoji == "😅":
                    raise TelegramBadRequest(method, "PRIVATE TOKEN: REACTION_INVALID")
            return True

        self.requests.side_effect = request
        caplog.clear()
        with caplog.at_level("WARNING", logger="dashulya.bot.reactions"):
            await self.deliver(react(10, "😅"), react(10, "😁"), reply_json(""))
        assert any(
            record.name == "dashulya.bot.reactions"
            or record.name.startswith("dashulya.bot.reactions.")
            for record in caplog.records
        )
        assert len(self.reactions()) == 2
        assert self.messages() == []

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
            assert emoji in description
        assert "another allowed emoji" in description

    async def test_tool_schema_exposes_only_message_id_and_emoji(self) -> None:
        dialogue, config, _ = await self.dialogue(react(1), reply_json(""))
        await self.reply(dialogue)
        schema = next(
            item
            for item in config.calls[0].tools
            if isinstance(item, FunctionToolSchema)
            and item.function.name == "set_reaction"
        )
        assert set(schema.function.parameters["properties"]) == {"message_id", "emoji"}
        assert config.calls[0].variables["chat_id"] == -123

    async def test_concurrent_turns_bind_reactions_to_their_context_variable(
        self,
    ) -> None:
        dialogue, _, _ = await self.dialogue(react(765432), reply_json(""))
        results = await asyncio.gather(
            self.reply(dialogue, -123), self.reply(dialogue, -456)
        )
        assert list(results) == [Reply(""), Reply("")]
        assert {(method.chat_id, method.message_id) for method in self.reactions()} == {
            (-123, 765432),
            (-456, 765432),
        }

    async def test_model_cannot_override_chat_identity_in_tool_arguments(self) -> None:
        dialogue, _, _ = await self.dialogue(react(765432, chat_id=-456), reply_json(""))
        await self.reply(dialogue)
        assert [(method.chat_id, method.message_id) for method in self.reactions()] == [
            (-123, 765432)
        ]

    async def test_missing_or_invalid_context_variable_never_calls_telegram(
        self, subtests: pytest.Subtests
    ) -> None:
        for variables in ({}, {"chat_id": 0}, {"chat_id": "-123"}, {"chat_id": True}):
            with subtests.test(variables=variables):
                _, _, agent = await self.dialogue(react(1), reply_json("Unavailable"))
                stream = MemoryStream()
                await agent.ask(
                    "Please react",
                    stream=stream,
                    variables=variables,
                    dependencies={"reaction_turn": ReactionTurn()},
                )
                results = [
                    event
                    for event in await stream.history.get_events()
                    if isinstance(event, ToolResultEvent)
                ]
                part = results[0].result.parts[0]
                assert isinstance(part, DataInput)
                assert isinstance(part.data, ReactionResult)
                assert part.data.reason == "missing_chat_context"
        assert self.reactions() == []

    async def test_missing_reaction_turn_never_calls_telegram(self) -> None:
        _, _, agent = await self.dialogue(react(1), reply_json("Unavailable"))
        stream = MemoryStream()
        await agent.ask("React", stream=stream, variables={"chat_id": -123})
        results = [
            event
            for event in await stream.history.get_events()
            if isinstance(event, ToolResultEvent)
        ]
        part = results[0].result.parts[0]
        assert isinstance(part, DataInput)
        assert part.data == ReactionResult("rejected", 1, "🔥", "missing_chat_context")
        assert self.reactions() == []

    async def test_tool_failure_returns_redacted_error_without_a_canned_reply(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        method = SetMessageReaction(chat_id=-123, message_id=1)
        self.requests.side_effect = TelegramBadRequest(method, "PRIVATE TOKEN")
        dialogue, _, agent = await self.dialogue(react(1, " ❤️ "), reply_json(""))
        ask = AsyncMock(wraps=agent.ask)
        caplog.clear()
        with (
            patch.object(agent, "ask", ask),
            caplog.at_level("WARNING", logger="dashulya.bot.reactions"),
        ):
            result = await self.reply(dialogue)
        assert any(
            record.name == "dashulya.bot.reactions"
            or record.name.startswith("dashulya.bot.reactions.")
            for record in caplog.records
        )
        assert result == Reply("")
        stream = ask.call_args.kwargs["stream"]
        results = [
            event
            for event in await stream.history.get_events()
            if isinstance(event, ToolResultEvent)
        ]
        part = results[0].result.parts[0]
        assert isinstance(part, DataInput)
        assert isinstance(part.data, ReactionResult)
        assert part.data.status == "rejected"
        assert part.data.reason == "telegram_rejected"
        assert json.loads(agent.serializer.encode(part.data)) == {
            "status": "rejected",
            "message_id": 1,
            "emoji": "❤",
            "reason": "telegram_rejected",
            "retry_after": None,
            "message": IsStr(min_length=1),
        }
        assert "unrecognized reason" in (part.data.message or "")
        assert "PRIVATE TOKEN" not in str(caplog.messages) + str(part.data)

    async def test_concrete_telegram_rejections_reach_the_model_without_raw_errors(
        self,
        caplog: pytest.LogCaptureFixture,
        subtests: pytest.Subtests,
    ) -> None:
        cases = (
            ("Bad Request: REACTION_INVALID", "reaction_invalid", "not allowed"),
            ("Bad Request: REACTION_EMPTY", "invalid_emoji", "empty"),
            ("Bad Request: MESSAGE_ID_INVALID", "message_not_found", "message"),
            ("Bad Request: MSG_ID_INVALID", "message_not_found", "message"),
            ("Bad Request: CHAT_ADMIN_REQUIRED", "permission_denied", "permission"),
            ("Bad Request: XREACTION_INVALIDX", "telegram_rejected", "unrecognized"),
            ("Forbidden: PRIVATE TOKEN", "permission_denied", "access"),
            ("Unauthorized: PRIVATE TOKEN", "bot_unauthorized", "authenticate"),
            ("Bad Request: REACTIONS_TOO_MANY", "reactions_too_many", "limit"),
            ("Bad Request: message to react not found", "message_not_found", "message"),
            ("Bad Request: chat not found", "chat_not_found", "chat"),
            ("Bad Request: not enough rights", "permission_denied", "permission"),
            (
                "Forbidden: bot was kicked from the supergroup chat",
                "bot_removed",
                "removed",
            ),
            (
                "Forbidden: bot is not a member of the channel chat",
                "bot_not_member",
                "member",
            ),
            ("Forbidden: bot was blocked by the user", "bot_blocked", "blocked"),
        )
        for description, reason, detail in cases:
            with subtests.test(description=description):
                method = SetMessageReaction(chat_id=-123, message_id=10)
                raw_error = description + " PRIVATE TOKEN"
                error: TelegramAPIError
                if description.startswith("Forbidden:"):
                    error = TelegramForbiddenError(method, raw_error)
                elif description.startswith("Unauthorized:"):
                    error = TelegramUnauthorizedError(method, raw_error)
                else:
                    error = TelegramBadRequest(method, raw_error)
                self.requests.side_effect = error
                self.requests.reset_mock()
                _, _, agent = await self.dialogue(react(10, " ❤️ "), reply_json(""))
                stream = MemoryStream()
                caplog.clear()
                with caplog.at_level("WARNING", logger="dashulya.bot.reactions"):
                    await agent.ask(
                        "React",
                        stream=stream,
                        variables={"chat_id": -123},
                        dependencies={"reaction_turn": ReactionTurn()},
                    )
                assert any(
                    record.name == "dashulya.bot.reactions"
                    or record.name.startswith("dashulya.bot.reactions.")
                    for record in caplog.records
                )
                results = [
                    event
                    for event in await stream.history.get_events()
                    if isinstance(event, ToolResultEvent)
                ]
                part = results[0].result.parts[0]
                assert isinstance(part, DataInput)
                payload = json.loads(agent.serializer.encode(part.data))
                assert payload == {
                    "status": "rejected",
                    "message_id": 10,
                    "emoji": "❤",
                    "reason": reason,
                    "retry_after": None,
                    "message": IsStr(min_length=1),
                }
                assert detail in payload["message"]
                assert "PRIVATE TOKEN" not in str(payload) + str(caplog.messages)
                assert len(self.reactions()) == 1

    async def test_rate_limit_returns_retry_delay_and_safe_instructions(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        method = SetMessageReaction(chat_id=-123, message_id=1)
        self.requests.side_effect = TelegramRetryAfter(method, "PRIVATE TOKEN", 37)
        dialogue, config, agent = await self.dialogue(react(1, " ❤️ "), reply_json(""))
        ask = AsyncMock(wraps=agent.ask)
        caplog.clear()
        with (
            patch.object(agent, "ask", ask),
            caplog.at_level("WARNING", logger="dashulya.bot.reactions"),
        ):
            result = await self.reply(dialogue)
        assert any(
            record.name == "dashulya.bot.reactions"
            or record.name.startswith("dashulya.bot.reactions.")
            for record in caplog.records
        )
        assert result == Reply("")
        assert len(self.reactions()) == 1
        stream = ask.call_args.kwargs["stream"]
        results = [
            event
            for event in await stream.history.get_events()
            if isinstance(event, ToolResultEvent)
        ]
        part = results[0].result.parts[0]
        assert isinstance(part, DataInput)
        assert isinstance(part.data, ReactionResult)
        assert part.data.status == "rate_limited"
        assert part.data.reason == "telegram_rate_limited"
        assert part.data.retry_after == 37
        assert json.loads(agent.serializer.encode(part.data)) == {
            "status": "rate_limited",
            "message_id": 1,
            "emoji": "❤",
            "reason": "telegram_rate_limited",
            "retry_after": 37,
            "message": IsStr(min_length=1),
        }
        assert (
            part.data.message
            == "Telegram is rate limiting reactions. Wait at least 37 seconds before trying again. Do not try a different emoji or another message to bypass this limit."
        )
        event = json.loads(JsonLogFormatter().format(caplog.records[-1]))
        assert event["outcome"] == "rate_limited"
        assert event["error_type"] == "TelegramRetryAfter"
        assert event["retry_after"] == 37
        assert "PRIVATE TOKEN" not in str(caplog.messages) + str(part.data)
        schema = next(
            item
            for item in config.calls[0].tools
            if isinstance(item, FunctionToolSchema)
            and item.function.name == "set_reaction"
        )
        assert "retry_after" in (schema.function.description or "")

    async def test_unknown_delivery_serializes_the_normalized_target(
        self, caplog: pytest.LogCaptureFixture, subtests: pytest.Subtests
    ) -> None:
        method = SetMessageReaction(chat_id=-123, message_id=765432)
        for error in (
            TelegramNetworkError(method, "PRIVATE TOKEN"),
            TelegramServerError(method, "PRIVATE TOKEN"),
            RuntimeError("PRIVATE TOKEN"),
        ):
            with subtests.test(error_type=type(error).__name__):
                self.requests.reset_mock()
                self.requests.side_effect = error
                _, _, agent = await self.dialogue(react(765432, " ❤️ "), reply_json(""))
                stream = MemoryStream()
                caplog.clear()
                with caplog.at_level("WARNING", logger="dashulya.bot.reactions"):
                    await agent.ask(
                        "React",
                        stream=stream,
                        variables={"chat_id": -123},
                        dependencies={"reaction_turn": ReactionTurn()},
                    )
                assert any(
                    record.name == "dashulya.bot.reactions"
                    or record.name.startswith("dashulya.bot.reactions.")
                    for record in caplog.records
                )
                results = [
                    event
                    for event in await stream.history.get_events()
                    if isinstance(event, ToolResultEvent)
                ]
                assert len(results) == 1
                part = results[0].result.parts[0]
                assert isinstance(part, DataInput)
                payload = json.loads(agent.serializer.encode(part.data))
                assert payload == {
                    "status": "unknown",
                    "message_id": 765432,
                    "emoji": "❤",
                    "reason": "delivery_unconfirmed",
                    "retry_after": None,
                    "message": None,
                }
                assert len(self.reactions()) == 1
                assert "PRIVATE TOKEN" not in str(payload) + str(caplog.messages)

    async def test_unknown_tool_delivery_is_not_retried_with_another_emoji(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        method = SetMessageReaction(chat_id=-123, message_id=10)

        async def request(bot: Bot, call: Any, **kwargs: Any) -> bool:
            if isinstance(call, SetMessageReaction):
                raise TelegramNetworkError(method, "PRIVATE TOKEN")
            return True

        self.requests.side_effect = request
        caplog.clear()
        with caplog.at_level("WARNING", logger="dashulya.bot.reactions"):
            await self.deliver(react(10), react(10, "👍"), reply_json(""))
        assert any(
            record.name == "dashulya.bot.reactions"
            or record.name.startswith("dashulya.bot.reactions.")
            for record in caplog.records
        )
        assert len(self.reactions()) == 1
        assert self.messages() == []

    async def test_parallel_calls_do_not_repeat_an_unconfirmed_target(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        async def request(bot: Bot, call: Any, **kwargs: Any) -> bool:
            if isinstance(call, SetMessageReaction):
                await asyncio.sleep(0)
                raise TelegramNetworkError(call, "PRIVATE TOKEN")
            return True

        self.requests.side_effect = request
        caplog.clear()
        with caplog.at_level("WARNING", logger="dashulya.bot.reactions"):
            await self.deliver([react(10), react(10, "👍")], reply_json(""))
        assert any(
            record.name == "dashulya.bot.reactions"
            or record.name.startswith("dashulya.bot.reactions.")
            for record in caplog.records
        )
        assert len(self.reactions()) == 1
        assert self.messages() == []

    async def test_turn_state_does_not_leak_to_a_later_empty_response(self) -> None:
        dialogue, _, agent = await self.dialogue(react(1), reply_json(""))
        assert await self.reply(dialogue, 42) == Reply("")
        with patch.object(agent, "ask", AsyncMock()) as ask:
            ask.return_value.content.return_value = Reply("")
            result = await self.reply(dialogue, 42)
        assert "couldn't generate" in result.text

    async def test_invalid_ids_and_empty_emoji_are_rejected_without_a_request(
        self,
        subtests: pytest.Subtests,
    ) -> None:
        service = TelegramReactions(self.bot)
        for message_id, emoji in ((0, "👍"), (-1, "👍"), (1, ""), (1, " ")):
            with subtests.test(message_id=message_id, emoji=emoji):
                assert (await service.set(-123, message_id, emoji)).status == "rejected"
        self.requests.assert_not_awaited()

    async def test_cancelled_delivery_propagates_and_does_not_generate_fallback(
        self,
    ) -> None:
        self.requests.side_effect = asyncio.CancelledError()
        with pytest.raises(asyncio.CancelledError):
            await TelegramReactions(self.bot).set(-123, 1, "👍")
        assert len(self.reactions()) == 1
        assert self.messages() == []
