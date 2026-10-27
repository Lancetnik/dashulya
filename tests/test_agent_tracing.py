import asyncio
import json
import unittest
from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

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

from dashulya.ai.agents.tracing import ReplyTrace
from dashulya.domain.memory import MemoryAccess
from dashulya.observability import JsonLogFormatter


class AgentTracingTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.context = ConversationContext(
            MemoryStream(),
            dependencies={
                MemoryAccess: MemoryAccess(
                    -123, False, before=datetime(2026, 10, 27, tzinfo=UTC), before_id=7
                )
            },
        )
        self.event = ModelRequest(parts=[TextInput("PRIVATE REQUEST")])
        self.trace = ReplyTrace(self.event, self.context)

    async def test_model_timing_and_usage_are_logged_without_payload(self) -> None:
        response = ModelResponse(
            usage=Usage(prompt_tokens=100, completion_tokens=25, thinking_tokens=15),
            model="test-model",
        )
        call = AsyncMock(return_value=response)
        with (
            self.assertLogs("dashulya.ai.agents.tracing", level="INFO") as logs,
            patch("dashulya.ai.agents.tracing.perf_counter", side_effect=[10, 10.125]),
        ):
            result = await self.trace.on_llm_call(call, [self.event], self.context)
        self.assertIs(result, response)
        event = json.loads(JsonLogFormatter().format(logs.records[-1]))
        self.assertEqual(event["duration_ms"], 125)
        self.assertEqual(event["prompt_tokens"], 100)
        self.assertEqual(event["thinking_tokens"], 15)
        self.assertEqual(event["message_id"], 7)
        self.assertEqual(event["outcome"], "ok")
        self.assertNotIn("PRIVATE REQUEST", str(event))

    async def test_success_and_tool_error_keep_same_call_id_without_arguments_or_result(
        self,
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
            with (
                self.subTest(outcome=outcome),
                self.assertLogs("dashulya.ai.agents.tracing", level="INFO") as logs,
            ):
                self.assertIs(
                    await self.trace.on_tool_execution(
                        AsyncMock(return_value=result), call, self.context
                    ),
                    result,
                )
            events = [
                json.loads(JsonLogFormatter().format(record)) for record in logs.records
            ]
            self.assertEqual(
                [event["tool_call_id"] for event in events], ["call-1", "call-1"]
            )
            self.assertEqual(events[-1]["outcome"], outcome)
            self.assertTrue(all(event["chat_id"] == -123 for event in events))
            self.assertNotIn("PRIVATE", json.dumps(events))

    async def test_tool_exception_and_cancellation_are_traced_and_propagated(
        self,
    ) -> None:
        for error, outcome in (
            (RuntimeError("PRIVATE ERROR"), "error"),
            (asyncio.CancelledError(), "cancelled"),
        ):
            with (
                self.subTest(outcome=outcome),
                self.assertLogs("dashulya.ai.agents.tracing", level="INFO") as logs,
                self.assertRaises(type(error)),
            ):
                await self.trace.on_tool_execution(
                    AsyncMock(side_effect=error),
                    ToolCallEvent("search_memory"),
                    self.context,
                )
            event = json.loads(JsonLogFormatter().format(logs.records[-1]))
            self.assertEqual(event["outcome"], outcome)
            self.assertIn("duration_ms", event)
            self.assertNotIn("PRIVATE ERROR", str(event))
