import asyncio
import os
import sqlite3
import unittest
from collections.abc import AsyncIterator
from dataclasses import dataclass, field, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import AsyncMock, patch

from ag2 import Agent
from ag2.events import ToolCallEvent
from ag2.testing import TestConfig
from ag2.tools import tool
from aiogram import Bot, Dispatcher
from aiogram.types import Message, TelegramObject, Update
from dishka import Provider, provide
from dishka.integrations.aiogram import inject as inject_aiogram
from dishka_ag2 import AG2Scope, FromDishka, inject

from dashulya.ai.agents.models import AgentModels
from dashulya.ai.dialogue import Dialogue
from dashulya.domain.memory import MemoryBackend
from dashulya.domain.messages import Reply
from dashulya.main import create_container, run
from dashulya.main.config import Settings
from dashulya.storage.groups import GroupStore
from tests.support import (
    TEST_API_KEY,
    MemoryOverride,
    ModelOverrides,
    RecorderOverride,
    reply_json,
)


@dataclass
class TurnState:
    pass


@dataclass
class CallState:
    pass


@dataclass
class ScopeTrace:
    uses: list[tuple[TurnState, CallState]] = field(default_factory=list)
    closed_turns: list[TurnState] = field(default_factory=list)
    closed_calls: list[CallState] = field(default_factory=list)
    started: asyncio.Event = field(default_factory=asyncio.Event)
    release: asyncio.Event = field(default_factory=asyncio.Event)
    telegram_calls: list[tuple[Message, TelegramObject, CallState]] = field(
        default_factory=list
    )


class ScopeProvider(Provider):
    @provide(scope=AG2Scope.APP)
    def trace(self) -> ScopeTrace:
        return ScopeTrace()

    @provide(scope=AG2Scope.SESSION)
    async def turn(self, trace: ScopeTrace) -> AsyncIterator[TurnState]:
        state = TurnState()
        try:
            yield state
        finally:
            trace.closed_turns.append(state)

    @provide(scope=AG2Scope.REQUEST)
    async def call(self, trace: ScopeTrace) -> AsyncIterator[CallState]:
        state = CallState()
        try:
            yield state
        finally:
            trace.closed_calls.append(state)


@tool
@inject
async def inspect_scope(
    turn: FromDishka[TurnState],
    call: FromDishka[CallState],
    trace: FromDishka[ScopeTrace],
    *,
    block: bool = False,
) -> str:
    trace.uses.append((turn, call))
    if block:
        trace.started.set()
        await trace.release.wait()
    return "Inspected"


@inject_aiogram
async def record_update(
    message: Message,
    event: FromDishka[TelegramObject],
    call: FromDishka[CallState],
    trace: FromDishka[ScopeTrace],
) -> None:
    trace.telegram_calls.append((message, event, call))


class DependencyTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.settings = Settings(
            bot_token="123456789:TEST_ONLY_NOT_A_REAL_BOT_TOKEN",
            openai_api_key=TEST_API_KEY,
            owner_user_id=42,
            database_path=Path(self.directory.name) / "groups.sqlite3",
            memory_path=Path(self.directory.name) / "missing.sqlite3",
            live_memory_path=Path(self.directory.name) / "live.sqlite3",
        )

    async def test_agents_and_dialogue_are_shared_but_agent_roles_are_distinct(
        self,
    ) -> None:
        container = create_container(
            self.settings,
            MemoryOverride(),
            ModelOverrides(
                reply=TestConfig(reply_json("Hello")),
                decision=TestConfig('{"data":true}'),
            ),
        )
        self.addAsyncCleanup(container.close)
        reply = await container.get(Agent[Reply])
        decision = await container.get(Agent[bool])
        dialogue = await container.get(Dialogue)
        self.assertIsNot(reply, decision)
        for _ in range(2):
            async with container() as event:
                self.assertIs(event.scope, AG2Scope.REQUEST)
                self.assertIs(await event.get(Agent[Reply]), reply)
                self.assertIs(await event.get(Agent[bool]), decision)
                self.assertIs(await event.get(Dialogue), dialogue)
        memory = await container.get(MemoryBackend)
        async with container() as event:
            self.assertIs(await event.get(MemoryBackend), memory)
        self.assertEqual([tool.name for tool in reply.tools], ["search_memory"])
        self.assertEqual(await (await reply.ask("Hello")).content(), Reply("Hello"))
        self.assertTrue(await (await decision.ask("Addressed?")).content())

    async def test_feed_update_injects_event_context_and_finalizes_each_request(
        self,
    ) -> None:
        container = create_container(
            self.settings, ScopeProvider(), MemoryOverride(), RecorderOverride()
        )
        self.addAsyncCleanup(container.close)
        dispatcher = await container.get(Dispatcher)
        bot = await container.get(Bot)
        dispatcher.message.register(record_update)
        for update_id in (1, 2):
            await dispatcher.feed_update(
                bot,
                Update.model_validate({
                    "update_id": update_id,
                    "message": {
                        "message_id": update_id,
                        "date": 1700000000,
                        "chat": {"id": 42, "type": "private", "first_name": "Anna"},
                        "from": {"id": 42, "is_bot": False, "first_name": "Anna"},
                        "text": "Hello",
                    },
                }),
            )
        trace = await container.get(ScopeTrace)
        self.assertEqual(len(trace.telegram_calls), 2)
        self.assertIsNot(trace.telegram_calls[0][2], trace.telegram_calls[1][2])
        for message, event, _ in trace.telegram_calls:
            self.assertIs(message, event)
        self.assertEqual(len(trace.closed_calls), 2)

    async def test_model_settings_are_independent_and_hide_credentials(self) -> None:
        container = create_container(
            replace(
                self.settings,
                openai_model="reply-model",
                openai_decision_model="decision-model",
            )
        )
        self.addAsyncCleanup(container.close)
        models = await container.get(AgentModels)
        self.assertEqual(models.reply.model, "reply-model")
        self.assertEqual(models.decision.model, "decision-model")
        self.assertNotIn(self.settings.openai_api_key, repr(models))

    async def test_concurrent_ag2_turns_share_app_state_and_isolate_tool_scopes(
        self,
    ) -> None:
        container = create_container(
            self.settings,
            ScopeProvider(),
            MemoryOverride(),
            ModelOverrides(
                reply=TestConfig(
                    ToolCallEvent("inspect_scope", arguments="{}"),
                    ToolCallEvent("inspect_scope", arguments="{}"),
                    reply_json("Done"),
                ),
                decision=TestConfig("false"),
            ),
        )
        self.addAsyncCleanup(container.close)
        agent = await container.get(Agent[Reply])
        await asyncio.gather(
            agent.ask("First", tools=[inspect_scope]),
            agent.ask("Second", tools=[inspect_scope]),
        )
        trace = await container.get(ScopeTrace)
        self.assertEqual(len(trace.uses), 4)
        self.assertEqual(len({id(turn) for turn, _ in trace.uses}), 2)
        self.assertEqual(len({id(call) for _, call in trace.uses}), 4)
        self.assertEqual(len(trace.closed_turns), 2)
        self.assertEqual(len(trace.closed_calls), 4)

    async def test_ag2_cancellation_finalizes_turn_and_tool_dependencies(self) -> None:
        container = create_container(
            self.settings,
            ScopeProvider(),
            MemoryOverride(),
            ModelOverrides(
                reply=TestConfig(
                    ToolCallEvent("inspect_scope", arguments='{"block":true}'),
                    reply_json("Done"),
                ),
                decision=TestConfig("false"),
            ),
        )
        self.addAsyncCleanup(container.close)
        agent = await container.get(Agent[Reply])
        trace = await container.get(ScopeTrace)
        task = asyncio.create_task(agent.ask("Wait", tools=[inspect_scope]))
        try:
            await asyncio.wait_for(trace.started.wait(), timeout=2)
        finally:
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        self.assertEqual(len(trace.closed_turns), 1)
        self.assertEqual(len(trace.closed_calls), 1)

    async def test_runtime_closes_production_resources_on_exit_and_polling_failure(
        self,
    ) -> None:
        for failure in (None, RuntimeError("Polling failed"), asyncio.CancelledError()):
            with self.subTest(failure=failure):
                container = create_container(
                    self.settings, MemoryOverride(), RecorderOverride()
                )
                bot = await container.get(Bot)
                groups = await container.get(GroupStore)
                close = AsyncMock(wraps=bot.session.close)
                with (
                    patch(
                        "dashulya.main.runtime.create_container", return_value=container
                    ),
                    patch.object(bot.session, "close", close),
                    patch.object(
                        Dispatcher, "start_polling", AsyncMock(side_effect=failure)
                    ) as polling,
                ):
                    if failure is None:
                        await run(self.settings)
                    else:
                        with self.assertRaises(type(failure)):
                            await run(self.settings)
                polling.assert_awaited_once_with(bot, close_bot_session=False)
                close.assert_awaited_once()
                with self.assertRaises(sqlite3.ProgrammingError):
                    groups.list_groups()

    async def test_partial_startup_failure_closes_the_bot_session(self) -> None:
        settings = replace(self.settings, database_path=Path(self.directory.name))
        container = create_container(settings, MemoryOverride(), RecorderOverride())
        bot = await container.get(Bot)
        close = AsyncMock(wraps=bot.session.close)
        with (
            patch("dashulya.main.runtime.create_container", return_value=container),
            patch.object(bot.session, "close", close),
            self.assertRaises(sqlite3.OperationalError),
        ):
            await run(settings)
        close.assert_awaited_once()

    async def test_missing_memory_stops_startup_before_telegram_polling(self) -> None:
        marker = Path(self.directory.name) / "ready"
        marker.write_text("stale")
        settings = replace(self.settings, healthcheck_path=marker)
        container = create_container(settings)
        with (
            patch("dashulya.main.runtime.create_container", return_value=container),
            patch.object(Dispatcher, "start_polling", AsyncMock()) as polling,
            self.assertRaisesRegex(ValueError, "MEMORY_PATH"),
        ):
            await run(settings)
        polling.assert_not_awaited()
        self.assertFalse(marker.exists())

    async def test_container_readiness_requires_identity_and_is_cleared_on_shutdown(
        self,
    ) -> None:
        marker = Path(self.directory.name) / "ready"
        settings = replace(self.settings, healthcheck_path=marker)
        container = create_container(settings, MemoryOverride(), RecorderOverride())
        bot = await container.get(Bot)

        async def poll(*args: object, **kwargs: object) -> None:
            self.assertEqual(marker.read_text(), str(os.getpid()))

        with (
            patch("dashulya.main.runtime.create_container", return_value=container),
            patch.object(bot, "me", AsyncMock()) as identity,
            patch.object(Dispatcher, "start_polling", AsyncMock(side_effect=poll)),
        ):
            await run(settings)
        identity.assert_awaited_once()
        self.assertFalse(marker.exists())
