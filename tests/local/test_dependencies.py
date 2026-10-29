import asyncio
import json
from contextlib import AsyncExitStack
from dataclasses import replace
from unittest.mock import AsyncMock, patch

import httpx2
import pytest
from ag2 import Agent
from ag2.config import OpenAIDecisionsConfig, OpenAIResponsesConfig
from ag2.events import ToolCallEvent
from ag2.testing import TestConfig
from aiogram import Bot
from dirty_equals import IsPartialDict
from dishka_ag2 import AG2Scope
from openai import OpenAIError

from dashulya.ai.agents.models import AgentModels
from dashulya.ai.dialogue import Dialogue
from dashulya.database import Database
from dashulya.domain.memory import ChatMemoryService
from dashulya.domain.messages import Reply
from dashulya.main import create_container, run
from dashulya.main.config import Settings
from tests.scopes import ScopeProvider, ScopeTrace, inspect_scope
from tests.support import (
    TEST_API_KEY,
    MemoryOverride,
    ModelOverrides,
    RecorderOverride,
    reply_json,
)


@pytest.mark.usefixtures("_setup")
class TestDependency:
    @pytest.fixture
    def _setup(self, cleanup: AsyncExitStack) -> None:
        self.cleanup = cleanup
        self.settings = Settings(
            bot_token="123456789:TEST_ONLY_NOT_A_REAL_BOT_TOKEN",
            openai_api_key=TEST_API_KEY,
            owner_user_id=42,
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
        self.cleanup.push_async_callback(container.close)
        reply = await container.get(Agent[Reply])
        decision = await container.get(Agent[bool])
        dialogue = await container.get(Dialogue)
        assert id(reply) != id(decision)
        for _ in range(2):
            async with container() as event:
                assert event.scope is AG2Scope.REQUEST
                assert await event.get(Agent[Reply]) is reply
                assert await event.get(Agent[bool]) is decision
                assert await event.get(Dialogue) is dialogue
        memory = await container.get(ChatMemoryService)
        async with container() as event:
            assert await event.get(ChatMemoryService) is memory
        assert [tool.name for tool in reply.tools] == [
            "search_memory",
            "chat_history",
            "set_reaction",
        ]
        assert await (await reply.ask("Hello")).content() == Reply("Hello")
        assert await (await decision.ask("Addressed?")).content()

    async def test_model_policy_reaches_sdk_requests_and_bounds_retries(
        self, subtests: pytest.Subtests
    ) -> None:
        requests: list[httpx2.Request] = []

        def respond(request: httpx2.Request) -> httpx2.Response:
            requests.append(request)
            return httpx2.Response(
                500,
                request=request,
                headers={"retry-after-ms": "1"},
                json={"error": {"message": "Synthetic provider failure"}},
            )

        container = create_container(
            replace(
                self.settings,
                openai_model="reply-model",
                openai_decision_model="decision-model",
            ),
            MemoryOverride(),
        )
        self.cleanup.push_async_callback(container.close)
        models = await container.get(AgentModels)
        assert self.settings.openai_api_key not in repr(models)
        assert isinstance(models.reply, OpenAIResponsesConfig)
        assert isinstance(models.decision, OpenAIDecisionsConfig)
        async with httpx2.AsyncClient(transport=httpx2.MockTransport(respond)) as http:
            await self.check_model_requests(subtests, models, http, requests)

    async def check_model_requests(
        self,
        subtests: pytest.Subtests,
        models: AgentModels,
        http: httpx2.AsyncClient,
        requests: list[httpx2.Request],
    ) -> None:
        assert isinstance(models.reply, OpenAIResponsesConfig)
        assert isinstance(models.decision, OpenAIDecisionsConfig)
        container = create_container(
            self.settings,
            MemoryOverride(),
            ModelOverrides(
                reply=models.reply.copy(http_client=http),
                decision=models.decision.copy(http_client=http),
            ),
        )
        self.cleanup.push_async_callback(container.close)
        # A missing supplied key must not silently use an ambient credential.
        with (
            patch.dict(
                "os.environ",
                {
                    "OPENAI_API_KEY": "ambient-test-key",  # pragma: allowlist secret -- Fake fallback credential.
                },
            ),
        ):
            cases: tuple[tuple[Agent[Reply] | Agent[bool], str, str, int, int], ...] = (
                (
                    await container.get(Agent[Reply]),
                    "/v1/responses",
                    "reply-model",
                    2,
                    60,
                ),
                (
                    await container.get(Agent[bool]),
                    "/v1/decisions",
                    "decision-model",
                    1,
                    10,
                ),
            )
            for agent, path, model, attempts, deadline in cases:
                with subtests.test(role=model):
                    requests.clear()
                    with pytest.raises(OpenAIError):
                        await agent.ask("Synthetic request")
                    assert len(requests) == attempts
                    assert [r.headers["x-stainless-retry-count"] for r in requests] == [
                        str(i) for i in range(attempts)
                    ]
                    for request in requests:
                        assert request.url.path == path
                        assert (
                            request.headers["authorization"]
                            == f"Bearer {self.settings.openai_api_key}"
                        )
                        assert request.extensions["timeout"] == dict.fromkeys(
                            ("connect", "read", "write", "pool"), deadline
                        )
                        body = json.loads(request.content)
                        assert body["model"] == model
                        if path == "/v1/responses":
                            assert body == IsPartialDict(reasoning={"effort": "medium"})

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
        self.cleanup.push_async_callback(container.close)
        agent = await container.get(Agent[Reply])
        await asyncio.gather(
            agent.ask("First", tools=[inspect_scope]),
            agent.ask("Second", tools=[inspect_scope]),
        )
        trace = await container.get(ScopeTrace)
        assert len(trace.uses) == 4
        assert len({id(turn) for turn, _ in trace.uses}) == 2
        assert len({id(call) for _, call in trace.uses}) == 4
        assert len(trace.closed_turns) == 2
        assert len(trace.closed_calls) == 4

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
        self.cleanup.push_async_callback(container.close)
        agent = await container.get(Agent[Reply])
        trace = await container.get(ScopeTrace)
        task = asyncio.create_task(agent.ask("Wait", tools=[inspect_scope]))
        try:
            await asyncio.wait_for(trace.started.wait(), timeout=2)
        finally:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        assert len(trace.closed_turns) == 1
        assert len(trace.closed_calls) == 1

    async def test_partial_startup_failure_closes_the_bot_session(self) -> None:
        settings = self.settings
        container = create_container(settings, MemoryOverride(), RecorderOverride())
        bot = await container.get(Bot)
        close = AsyncMock(wraps=bot.session.close)
        with (
            patch("dashulya.main.runtime.create_container", return_value=container),
            patch.object(bot.session, "close", close),
            pytest.raises(RuntimeError),
            patch.object(
                Database,
                "open",
                AsyncMock(side_effect=RuntimeError("Database unavailable")),
            ),
        ):
            await run(settings)
        close.assert_awaited_once()
