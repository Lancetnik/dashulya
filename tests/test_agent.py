import json
import unittest
from datetime import UTC, datetime
from typing import Any
from unittest.mock import Mock

import httpx2
from ag2 import Agent
from ag2.config import OpenAIDecisionsConfig, OpenAIResponsesConfig

from dashulya.ai.addressing import GroupAddressing
from dashulya.ai.dialogue import Dialogue
from dashulya.domain.messages import GroupMessage
from dashulya.main import create_container
from dashulya.main.config import Settings
from dashulya.memory.search import MemorySearch
from tests.support import TEST_API_KEY, MemoryOverride, ModelOverrides


class AgentAPITests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.settings = Settings(
            bot_token="123456789:TEST_ONLY_NOT_A_REAL_BOT_TOKEN",
            openai_api_key=TEST_API_KEY,
            owner_user_id=42,
        )

    async def test_addressing_uses_decisions_predicate_and_typed_answers(self) -> None:
        requests: list[dict[str, Any]] = []

        def respond(request: httpx2.Request) -> httpx2.Response:
            self.assertEqual(request.url.path, "/v1/decisions")
            body = json.loads(request.content)
            requests.append(body)
            return httpx2.Response(
                200,
                json={
                    "model": "gpt-6-luna",
                    "answers": [
                        {
                            "type": "predicate",
                            "name": body["questions"][0]["name"],
                            "probability": 0.9 if len(requests) == 1 else 0.1,
                        }
                    ],
                    "usage": {
                        "input_tokens": 10,
                        "output_tokens": 0,
                        "total_tokens": 10,
                        "input_tokens_details": {
                            "cached_tokens": 0,
                            "cache_write_tokens": 0,
                        },
                        "output_tokens_details": {"reasoning_tokens": 0},
                    },
                },
            )

        async with httpx2.AsyncClient(transport=httpx2.MockTransport(respond)) as client:
            container = create_container(
                self.settings,
                ModelOverrides(
                    reply=OpenAIResponsesConfig(model="unused", api_key=TEST_API_KEY),
                    decision=OpenAIDecisionsConfig(
                        model="gpt-6-luna",
                        api_key=TEST_API_KEY,
                        http_client=client,
                    ),
                ),
            )
            self.addAsyncCleanup(container.close)
            classifier = await container.get(Agent[bool])
            addressing = GroupAddressing(
                classifier, bot_id=123, bot_username="dashulya_bot"
            )
            self.assertTrue(
                await addressing.should_reply(
                    GroupMessage(
                        1,
                        "Дашуля, помоги",
                        datetime(2026, 10, 25, tzinfo=UTC),
                        42,
                        "Anna",
                    ),
                    recent_conversation=[],
                )
            )
            self.assertFalse(
                await addressing.should_reply(
                    GroupMessage(
                        2,
                        "Дашуля вчера сказала",
                        datetime(2026, 10, 25, tzinfo=UTC),
                        42,
                        "Anna",
                    ),
                    recent_conversation=[],
                )
            )

        self.assertEqual(len(requests), 2)
        for request in requests:
            self.assertEqual(request["model"], "gpt-6-luna")
            self.assertEqual(request["questions"][0]["type"], "predicate")
            self.assertNotIn("tools", request)
            self.assertNotIn("reasoning", request)
            self.assertEqual(len(request["input"]), 1)

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
            container = create_container(
                self.settings,
                MemoryOverride(memory),
                ModelOverrides(
                    reply=OpenAIResponsesConfig(
                        model="gpt-5.6-luna",
                        api_key=TEST_API_KEY,
                        http_client=client,
                        reasoning={"effort": "low"},
                    ),
                    decision=OpenAIDecisionsConfig(model="unused", api_key=TEST_API_KEY),
                ),
            )
            self.addAsyncCleanup(container.close)
            dialogue = await container.get(Dialogue)
            reply = await dialogue.reply(
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
        tool = requests[0]["tools"][0]
        self.assertEqual(tool["name"], "search_memory")
        self.assertNotIn("memory", tool["parameters"]["properties"])
        self.assertNotIn("context", tool["parameters"]["properties"])
        self.assertTrue(
            any(
                item.get("type") == "function_call_output"
                for item in requests[1]["input"]
            )
        )
        memory.search.assert_awaited_once()
