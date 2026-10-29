import base64
import json
from contextlib import AsyncExitStack
from dataclasses import replace
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, Mock

import httpx2
import pytest
from ag2 import Agent
from ag2.config import OpenAIDecisionsConfig, OpenAIResponsesConfig

from dashulya.ai.addressing import GroupAddressing
from dashulya.ai.dialogue import Dialogue
from dashulya.domain.memory import ChatHistory, MemoryMessage, ReplyContext
from dashulya.domain.messages import GroupMessage, ImageAttachment, Reply, ReplyRequest
from dashulya.main import create_container
from dashulya.main.config import Settings
from dashulya.memory.search import MemorySearch
from tests.support import TEST_API_KEY, MemoryOverride, ModelOverrides, reply_json


@pytest.mark.usefixtures("_setup")
class TestAgentAPI:
    @pytest.fixture
    def _setup(self, cleanup: AsyncExitStack) -> None:
        self.cleanup = cleanup
        self.settings = Settings(
            bot_token="123456789:TEST_ONLY_NOT_A_REAL_BOT_TOKEN",
            openai_api_key=TEST_API_KEY,
            owner_user_id=42,
        )

    async def test_group_journal_messages_have_native_roles_in_responses_api(
        self,
    ) -> None:
        self.settings = replace(self.settings, bot_timezone="Asia/Tokyo")
        requests: list[dict[str, Any]] = []
        history = AsyncMock(spec=ChatHistory)
        history.context.return_value = ReplyContext((
            MemoryMessage(
                chat_id=-123,
                message_id=1,
                text="Participant discussion",
                sent_at=datetime(2026, 10, 25, tzinfo=UTC),
                author_id=99,
                author="Boris",
                chat_title="Group",
            ),
            MemoryMessage(
                chat_id=-123,
                message_id=2,
                text="Previously delivered answer",
                sent_at=datetime(2026, 10, 25, tzinfo=UTC),
                author_id=123456789,
                author="Dashulya",
                chat_title="Group",
                reply_to_id=1,
            ),
        ))

        def respond(request: httpx2.Request) -> httpx2.Response:
            requests.append(json.loads(request.content))
            return httpx2.Response(
                200,
                json={
                    "id": f"resp_{len(requests)}",
                    "object": "response",
                    "created_at": 1700000000,
                    "status": "completed",
                    "model": "test",
                    "output": [
                        {
                            "id": "msg_1",
                            "type": "message",
                            "role": "assistant",
                            "status": "completed",
                            "content": [
                                {
                                    "type": "output_text",
                                    "text": reply_json("Fresh answer", 1),
                                    "annotations": [],
                                }
                            ],
                        }
                    ],
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
                MemoryOverride(history=history),
                ModelOverrides(
                    reply=OpenAIResponsesConfig(
                        model="test",
                        api_key=TEST_API_KEY,
                        http_client=client,
                    ),
                    decision=OpenAIDecisionsConfig(model="unused", api_key=TEST_API_KEY),
                ),
            )
            self.cleanup.push_async_callback(container.close)
            dialogue = await container.get(Dialogue)
            for message_id in (3, 4):
                result = await dialogue.reply(
                    ReplyRequest(
                        (-123, None),
                        f"Current request {message_id}",
                        user_id=42,
                        user_name="Anna",
                        sent_at=datetime(2026, 10, 25, tzinfo=UTC),
                        images=(
                            ImageAttachment(b"native-image", "image/png", message_id),
                        ),
                        message_id=message_id,
                    )
                )
                assert result == Reply("Fresh answer", 1)
        assert len(requests) == 2
        for request in requests:
            output_format = request["text"]["format"]
            assert output_format["type"] == "json_schema"
            assert output_format["strict"]
            assert output_format["schema"]["required"] == ["text", "reply_to_message_id"]
            assert not output_format["schema"]["additionalProperties"]
            instructions = request["instructions"]
            assert json.dumps(instructions).count("Message timezone: Asia/Tokyo.") == 1
            columns = "The metadata keys are id, user_id"
            assert json.dumps(instructions).count(columns) == 1
            messages = [entry for entry in request["input"] if entry["role"] != "system"]
            assert "Asia/Tokyo" not in json.dumps(messages)
            assert "Timezone:" not in json.dumps(messages)
            assert columns not in json.dumps(messages)
            assert [entry["role"] for entry in messages] == [
                "user",
                "user",
                "assistant",
                "user",
                "user",
                "user",
            ]
            image = messages[-1]["content"][0]
            assert image["type"] == "input_image"
            assert image["detail"] == "high"
            assert (
                image["image_url"]
                == "data:image/png;base64," + base64.b64encode(b"native-image").decode()
            )
            assert self.settings.bot_token not in json.dumps(request)
            assert "Previously delivered answer" in json.dumps(messages[2])
            assert "Fresh answer" not in json.dumps(messages)
        assert "Current request 3" not in json.dumps(requests[1]["input"])

    async def test_addressing_uses_decisions_predicate_and_typed_answers(self) -> None:
        self.settings = replace(self.settings, bot_timezone="Asia/Tokyo")
        requests: list[dict[str, Any]] = []

        def respond(request: httpx2.Request) -> httpx2.Response:
            assert request.url.path == "/v1/decisions"
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
            self.cleanup.push_async_callback(container.close)
            classifier = await container.get(Agent[bool])
            addressing = GroupAddressing(
                classifier, bot_id=123, bot_username="dashulya_bot"
            )
            assert await addressing.should_reply(
                GroupMessage(
                    1, "Дашуля, помоги", datetime(2026, 10, 25, tzinfo=UTC), 42, "Anna"
                ),
                recent_conversation=[],
            )
            assert not await addressing.should_reply(
                GroupMessage(
                    2,
                    "Дашуля вчера сказала",
                    datetime(2026, 10, 25, tzinfo=UTC),
                    42,
                    "Anna",
                ),
                recent_conversation=[],
            )

        assert len(requests) == 2
        for request in requests:
            assert request["model"] == "gpt-6-luna"
            assert request["questions"][0]["type"] == "predicate"
            assert "tools" not in request
            assert "reasoning" not in request
            assert len(request["input"]) == 1
            columns = "The metadata keys are id, user_id"
            assert json.dumps(request["questions"]).count(columns) == 1
            assert columns not in json.dumps(request["input"])
            assert (
                json.dumps(request["questions"]).count("Message timezone: Asia/Tokyo.")
                == 1
            )
            assert "Asia/Tokyo" not in json.dumps(request["input"])

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
                                "text": reply_json("No matches in the snapshot."),
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
            self.cleanup.push_async_callback(container.close)
            dialogue = await container.get(Dialogue)
            reply = await dialogue.reply(
                ReplyRequest(
                    (42, None),
                    "Search redis",
                    user_id=42,
                    user_name="Anna",
                    sent_at=datetime(2026, 10, 25, tzinfo=UTC),
                )
            )

        assert reply == Reply("No matches in the snapshot.")
        assert len(requests) == 2
        assert requests[0]["model"] == "gpt-5.6-luna"
        assert requests[0]["reasoning"] == {"effort": "low"}
        tool = requests[0]["tools"][0]
        assert tool["name"] == "search_memory"
        assert "memory" not in tool["parameters"]["properties"]
        assert "context" not in tool["parameters"]["properties"]
        assert any(
            item.get("type") == "function_call_output" for item in requests[1]["input"]
        )
        memory.search.assert_awaited_once()
        assert memory.search.await_args_list[0].args == ("redis",)
