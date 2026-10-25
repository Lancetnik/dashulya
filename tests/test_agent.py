import json
import unittest
from datetime import UTC, datetime
from typing import Any, cast
from unittest.mock import Mock

import httpx2

from dashulya.ai.agent import create_agent
from dashulya.ai.dialogue import Dialogue
from dashulya.memory.search import MemorySearch


class AgentAPITests(unittest.IsolatedAsyncioTestCase):
    async def test_reasoning_with_memory_uses_responses_and_executes_tool(self) -> None:
        requests: list[dict[str, Any]] = []
        memory = Mock(spec=MemorySearch)
        memory.search.return_value = {"hits": [], "coverage": {}}

        def respond(request: httpx2.Request) -> httpx2.Response:
            body = json.loads(request.content)
            requests.append(body)
            if request.url.path != "/v1/responses":
                return httpx2.Response(
                    400,
                    json={
                        "error": {
                            "message": "Function tools with reasoning_effort are not supported for gpt-5.6-luna in /v1/chat/completions.",
                            "type": "invalid_request_error",
                            "param": "reasoning_effort",
                            "code": None,
                        }
                    },
                )
            output: list[dict[str, Any]] = (
                [
                    {
                        "type": "function_call",
                        "id": "fc_test",
                        "call_id": "call_test",
                        "name": "search_memory",
                        "arguments": '{"query":"redis"}',
                        "status": "completed",
                    }
                ]
                if len(requests) == 1
                else [
                    {
                        "type": "message",
                        "id": "msg_test",
                        "role": "assistant",
                        "status": "completed",
                        "content": [
                            {
                                "type": "output_text",
                                "text": "No matches in the snapshot.",
                                "annotations": [],
                            }
                        ],
                    }
                ]
            )
            return httpx2.Response(
                200,
                json={
                    "id": f"resp_{len(requests)}",
                    "object": "response",
                    "created_at": 1700000000,
                    "status": "completed",
                    "model": "gpt-5.6-luna",
                    "output": output,
                    "usage": {
                        "input_tokens": 10,
                        "output_tokens": 4,
                        "total_tokens": 14,
                        "input_tokens_details": {"cached_tokens": 0},
                        "output_tokens_details": {"reasoning_tokens": 0},
                    },
                },
            )

        async with httpx2.AsyncClient(transport=httpx2.MockTransport(respond)) as client:
            agent = create_agent(
                model="gpt-5.6-luna",
                api_key="test-key",  # pragma: allowlist secret -- Mock transport credential.
                memory=cast("MemorySearch", memory),
            )
            agent.config = cast("Any", agent.config).copy(http_client=client)
            reply = await Dialogue(agent).reply(
                (42, None),
                "Search redis",
                user_id=42,
                user_name="Anna",
                sent_at=datetime(2026, 10, 25, tzinfo=UTC),
            )

        self.assertEqual(reply, "No matches in the snapshot.")
        self.assertEqual(len(requests), 2)
        self.assertEqual(requests[0]["model"], "gpt-5.6-luna")
        self.assertEqual(requests[0]["reasoning"], {"effort": "low"})
        self.assertEqual(requests[0]["tools"][0]["name"], "search_memory")
        self.assertTrue(
            any(
                item.get("type") == "function_call_output"
                for item in requests[1]["input"]
            )
        )
        memory.search.assert_called_once()
