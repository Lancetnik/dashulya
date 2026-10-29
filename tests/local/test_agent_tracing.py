import asyncio
import json
from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

import pytest
from ag2 import MemoryStream
from ag2.context import ConversationContext
from ag2.events import (
    ModelRequest,
    ModelResponse,
    TextInput,
    ToolCallEvent,
    ToolErrorEvent,
    ToolResultEvent,
    Usage,
)
from dirty_equals import IsInt, IsPartialDict

from dashulya.ai.agents.tracing import ReplyTrace
from dashulya.domain.memory import MemoryAccess
from dashulya.observability import JsonLogFormatter


@pytest.mark.usefixtures("_setup")
class TestAgentTracing:
    @pytest.fixture
    def _setup(self) -> None:
        self.context = ConversationContext(
            MemoryStream(),
            dependencies={
                "memory_access": MemoryAccess(
                    -123, False, before=datetime(2026, 10, 27, tzinfo=UTC), before_id=7
                )
            },
        )
        self.event = ModelRequest(parts=[TextInput("PRIVATE REQUEST")])
        self.trace = ReplyTrace(self.event, self.context)

    async def test_model_timing_and_usage_are_logged_without_payload(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        response = ModelResponse(
            usage=Usage(prompt_tokens=100, completion_tokens=25, thinking_tokens=15),
            model="test-model",
        )
        call = AsyncMock(return_value=response)
        caplog.clear()
        with (
            caplog.at_level("INFO", logger="dashulya.ai.agents.tracing"),
            patch("dashulya.ai.agents.tracing.perf_counter", side_effect=[10, 10.125]),
        ):
            result = await self.trace.on_llm_call(call, [self.event], self.context)
        assert any(
            record.name == "dashulya.ai.agents.tracing"
            or record.name.startswith("dashulya.ai.agents.tracing.")
            for record in caplog.records
        )
        assert result is response
        event = json.loads(JsonLogFormatter().format(caplog.records[-1]))
        assert event == IsPartialDict(
            duration_ms=125,
            prompt_tokens=100,
            thinking_tokens=15,
            message_id=7,
            outcome="ok",
        )
        assert "PRIVATE REQUEST" not in str(event)

    async def test_success_and_tool_error_keep_same_call_id_without_arguments_or_result(
        self,
        caplog: pytest.LogCaptureFixture,
        subtests: pytest.Subtests,
    ) -> None:
        call = ToolCallEvent(
            "search_memory", id="call-1", arguments='{"query":"PRIVATE QUERY"}'
        )
        for result, outcome in (
            (ToolResultEvent.from_call(call, "PRIVATE RESULT"), "ok"),
            (
                ToolErrorEvent.from_call(call, ValueError("PRIVATE PROVIDER ERROR")),
                "error",
            ),
        ):
            caplog.clear()
            with (
                subtests.test(outcome=outcome),
                caplog.at_level("INFO", logger="dashulya.ai.agents.tracing"),
            ):
                assert (
                    await self.trace.on_tool_execution(
                        AsyncMock(return_value=result), call, self.context
                    )
                    is result
                )
            assert any(
                record.name == "dashulya.ai.agents.tracing"
                or record.name.startswith("dashulya.ai.agents.tracing.")
                for record in caplog.records
            )
            events = [
                json.loads(JsonLogFormatter().format(record)) for record in caplog.records
            ]
            assert events == [
                IsPartialDict(tool_call_id="call-1", chat_id=-123),
                IsPartialDict(tool_call_id="call-1", chat_id=-123, outcome=outcome),
            ]
            assert "PRIVATE" not in json.dumps(events)

    async def test_tool_exception_and_cancellation_are_traced_and_propagated(
        self,
        caplog: pytest.LogCaptureFixture,
        subtests: pytest.Subtests,
    ) -> None:
        for error, outcome in (
            (RuntimeError("PRIVATE ERROR"), "error"),
            (asyncio.CancelledError(), "cancelled"),
        ):
            caplog.clear()
            with (
                subtests.test(outcome=outcome),
                caplog.at_level("INFO", logger="dashulya.ai.agents.tracing"),
                pytest.raises(type(error)),
            ):
                await self.trace.on_tool_execution(
                    AsyncMock(side_effect=error),
                    ToolCallEvent("search_memory"),
                    self.context,
                )
            assert any(
                record.name == "dashulya.ai.agents.tracing"
                or record.name.startswith("dashulya.ai.agents.tracing.")
                for record in caplog.records
            )
            event = json.loads(JsonLogFormatter().format(caplog.records[-1]))
            assert event == IsPartialDict(outcome=outcome, duration_ms=IsInt(ge=0))
            assert "PRIVATE ERROR" not in str(event)
