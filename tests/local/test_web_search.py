import asyncio
import json
from collections.abc import Callable, Coroutine
from contextlib import AsyncExitStack
from dataclasses import replace
from typing import Any
from unittest.mock import patch

import httpx
import pytest
from ag2 import Agent, MemoryStream
from ag2.events import DataInput, ToolCallEvent, ToolErrorEvent, ToolResultEvent
from ag2.testing import TestConfig, TrackingConfig
from ag2.tools.final.function_tool import FunctionToolSchema
from dirty_equals import IsPartialDict

from dashulya.ai.tools.web import PAGE_TEXT_CHARACTERS, SEARCH_TEXT_CHARACTERS
from dashulya.domain.messages import Reply
from dashulya.main import create_container
from dashulya.main.config import Settings
from tests.support import TEST_API_KEY, MemoryOverride, ModelOverrides, reply_json


@pytest.mark.usefixtures("_setup")
class TestWebSearch:
    @pytest.fixture
    def _setup(self, cleanup: AsyncExitStack) -> None:
        self.cleanup = cleanup
        self.settings = Settings(
            bot_token="123456789:TEST_ONLY_NOT_A_REAL_BOT_TOKEN",
            openai_api_key=TEST_API_KEY,
            owner_user_id=42,
            exa_api_key="exa-test-only",  # pragma: allowlist secret -- Fake test key.
        )
        self.requests: list[httpx.Request] = []
        self.clients: list[httpx.AsyncClient] = []
        self.http_client_type = httpx.AsyncClient

    async def agent(
        self, config: TestConfig | TrackingConfig, *, enabled: bool = True
    ) -> Agent[Reply]:
        container = create_container(
            self.settings if enabled else replace(self.settings, exa_api_key=""),
            MemoryOverride(),
            ModelOverrides(reply=config, decision=TestConfig("false")),
        )
        self.cleanup.push_async_callback(container.close)
        return await container.get(Agent[Reply])

    def transport(
        self,
        handler: Callable[[httpx.Request], httpx.Response]
        | Callable[[httpx.Request], Coroutine[Any, Any, httpx.Response]],
    ) -> Any:
        def create(**kwargs: Any) -> httpx.AsyncClient:
            client = self.http_client_type(
                transport=httpx.MockTransport(handler), **kwargs
            )
            self.clients.append(client)
            return client

        return patch("exa_py.api.httpx.AsyncClient", side_effect=create)

    async def results(self, stream: MemoryStream) -> list[ToolResultEvent]:
        return [
            e for e in await stream.history.get_events() if isinstance(e, ToolResultEvent)
        ]

    async def test_search_and_page_read_work_through_the_production_agent(self) -> None:
        url = "https://docs.ag2.ai/docs/user-guide/extensions/tools/search/exa/"

        def respond(request: httpx.Request) -> httpx.Response:
            self.requests.append(request)
            text = "Public page text " + "x" * PAGE_TEXT_CHARACTERS
            return httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "id": url,
                            "url": url,
                            "title": "AG2 Exa tools",
                            "text": text,
                            "publishedDate": "2026-10-21T00:00:00Z",
                        }
                    ]
                },
            )

        agent = await self.agent(
            TestConfig(
                ToolCallEvent(
                    "exa_search", arguments='{"query":"AG2 Exa documentation"}'
                ),
                ToolCallEvent("exa_get_contents", arguments=json.dumps({"urls": [url]})),
                reply_json(f"AG2 supports Exa. [Source]({url})"),
            )
        )
        stream = MemoryStream()
        with self.transport(respond):
            response = await agent.ask(
                "Find the docs. PRIVATE CHAT CONTEXT", stream=stream
            )
        reply = await response.content()
        assert reply is not None
        assert url in reply.text
        assert [r.url.path for r in self.requests] == ["/search", "/contents"]
        body = json.loads(self.requests[0].content)
        assert body["query"] == "AG2 Exa documentation"
        assert body["numResults"] == 5
        assert body["type"] == "auto"
        assert body["contents"]["text"]["maxCharacters"] == SEARCH_TEXT_CHARACTERS
        assert json.loads(self.requests[1].content)["urls"] == [url]
        for request in self.requests:
            assert request.headers["x-api-key"] == self.settings.exa_api_key
            assert b"PRIVATE CHAT CONTEXT" not in request.content
        results = await self.results(stream)
        for result, limit in zip(
            results, (SEARCH_TEXT_CHARACTERS, PAGE_TEXT_CHARACTERS), strict=True
        ):
            part = result.result.parts[0]
            assert isinstance(part, DataInput)
            assert isinstance(part.data, dict)
            pages = part.data["results"]
            assert isinstance(pages, list)
            page = pages[0]
            assert isinstance(page, dict)
            assert page["url"] == url
            assert page["title"] == "AG2 Exa tools"
            assert page["published_date"] == "2026-10-21T00:00:00Z"
            assert len(str(page["text"])) == limit
            assert page["truncated"]
        assert len(self.clients) == 2
        assert all(client.is_closed for client in self.clients)

    async def test_only_reply_agent_receives_web_tools_and_key_stays_out_of_schema(
        self,
        subtests: pytest.Subtests,
    ) -> None:
        for enabled in (True, False):
            with subtests.test(enabled=enabled):
                config = TrackingConfig(TestConfig(reply_json("Done")))
                agent = await self.agent(config, enabled=enabled)
                await agent.ask("Hello")
                tools = [
                    schema.function.name
                    for schema in config.calls[0].tools
                    if isinstance(schema, FunctionToolSchema)
                ]
                assert "search_memory" in tools
                assert "chat_history" in tools
                assert ("exa_search" in tools) == enabled
                assert ("exa_get_contents" in tools) == enabled
                assert "exa_answer" not in tools
                assert "exa_find_similar" not in tools
                assert self.settings.exa_api_key not in repr(config.calls[0])

        decision = TrackingConfig(TestConfig("false"))
        container = create_container(
            self.settings,
            ModelOverrides(reply=TestConfig(reply_json("Done")), decision=decision),
        )
        self.cleanup.push_async_callback(container.close)
        classifier = await container.get(Agent[bool])
        await classifier.ask("Does this address Dashulya?")
        assert decision.calls[0].tools == ()

    async def test_provider_errors_are_redacted_and_not_retried(
        self, caplog: pytest.LogCaptureFixture, subtests: pytest.Subtests
    ) -> None:
        for status in (401, 429, 500):
            with subtests.test(status=status):
                self.requests.clear()

                def respond(
                    request: httpx.Request, status: int = status
                ) -> httpx.Response:
                    self.requests.append(request)
                    return httpx.Response(
                        status, json={"error": "PRIVATE PROVIDER BODY exa-test-only"}
                    )

                agent = await self.agent(
                    TestConfig(
                        ToolCallEvent("exa_search", arguments='{"query":"public facts"}'),
                        reply_json("Unable to verify."),
                        raise_tool_errors=False,
                    )
                )
                stream = MemoryStream()
                caplog.clear()
                with (
                    self.transport(respond),
                    caplog.at_level("WARNING", logger="dashulya.ai.tools.web"),
                ):
                    await agent.ask("Find public facts", stream=stream)
                assert any(
                    record.name == "dashulya.ai.tools.web"
                    or record.name.startswith("dashulya.ai.tools.web.")
                    for record in caplog.records
                )
                results = await self.results(stream)
                assert isinstance(results[0], ToolErrorEvent)
                for output in (repr(results), " ".join(caplog.messages)):
                    assert "PRIVATE PROVIDER BODY" not in output
                    assert self.settings.exa_api_key not in output
                assert "Do not retry automatically" in repr(results)
                assert len(self.requests) == 1
                assert self.clients[-1].is_closed

    async def test_empty_results_remain_empty(self) -> None:
        agent = await self.agent(
            TestConfig(
                ToolCallEvent("exa_search", arguments='{"query":"public facts"}'),
                reply_json("No matches."),
            )
        )
        stream = MemoryStream()
        with self.transport(lambda _: httpx.Response(200, json={"results": []})):
            await agent.ask("Find", stream=stream)
        part = (await self.results(stream))[0].result.parts[0]
        assert isinstance(part, DataInput)
        assert isinstance(part.data, dict)
        assert part.data["results"] == []

    async def test_request_limits_reject_inputs_before_calling_exa(
        self, subtests: pytest.Subtests
    ) -> None:
        calls = [
            ToolCallEvent("exa_search", arguments='{"query":" "}'),
            ToolCallEvent("exa_search", arguments=json.dumps({"query": "x" * 2001})),
            ToolCallEvent("exa_get_contents", arguments='{"urls":[]}'),
            ToolCallEvent(
                "exa_get_contents",
                arguments=json.dumps({"urls": ["https://example.com"] * 4}),
            ),
            ToolCallEvent(
                "exa_get_contents",
                arguments=json.dumps({"urls": ["https://example.com/" + "x" * 2029]}),
            ),
            ToolCallEvent(
                "exa_get_contents", arguments='{"urls":["file:///etc/passwd"]}'
            ),
            ToolCallEvent(
                "exa_get_contents",
                arguments='{"urls":["https://user:password@example.com"]}',  # pragma: allowlist secret -- Synthetic URL credentials.
            ),
        ]
        for call in calls:
            with subtests.test(call=call):
                agent = await self.agent(
                    TestConfig(
                        call, reply_json("Invalid input."), raise_tool_errors=False
                    )
                )
                stream = MemoryStream()
                with patch("ag2.extensions.tools.search.exa.AsyncExa") as sdk:
                    await agent.ask("Find", stream=stream)
                sdk.assert_not_called()
                assert isinstance((await self.results(stream))[0], ToolErrorEvent)

    async def test_legal_input_boundaries_reach_the_exa_transport(
        self, subtests: pytest.Subtests
    ) -> None:
        url = "https://example.com/"
        cases = (
            ("exa_search", {"query": "x"}),
            ("exa_search", {"query": "x" * 2000}),
            ("exa_get_contents", {"urls": [url]}),
            ("exa_get_contents", {"urls": [url + str(i) for i in range(3)]}),
            ("exa_get_contents", {"urls": [url + "x" * 2028]}),
        )
        for name, arguments in cases:
            with subtests.test(name=name, arguments=arguments):
                requests: list[httpx.Request] = []

                def respond(
                    request: httpx.Request, requests: list[httpx.Request] = requests
                ) -> httpx.Response:
                    requests.append(request)
                    return httpx.Response(200, json={"results": []})

                agent = await self.agent(
                    TestConfig(
                        ToolCallEvent(name, arguments=json.dumps(arguments)),
                        reply_json("No matches."),
                        raise_tool_errors=False,
                    )
                )
                stream = MemoryStream()
                with self.transport(respond):
                    await agent.ask("Find", stream=stream)
                assert len(requests) == 1
                assert json.loads(requests[0].content) == IsPartialDict(**arguments)
                results = await self.results(stream)
                assert len(results) == 1
                assert not isinstance(results[0], ToolErrorEvent)

    async def test_timeout_and_cancellation_close_the_sdk_client(
        self, subtests: pytest.Subtests
    ) -> None:
        for cancel in (False, True):
            with subtests.test(cancel=cancel):
                started = asyncio.Event()
                interrupted = asyncio.Event()

                async def respond(
                    request: httpx.Request,
                    started: asyncio.Event = started,
                    interrupted: asyncio.Event = interrupted,
                ) -> httpx.Response:
                    started.set()
                    try:
                        await asyncio.Event().wait()
                    finally:
                        interrupted.set()
                    return httpx.Response(200, json={"results": []})

                agent = await self.agent(
                    TestConfig(
                        ToolCallEvent("exa_search", arguments='{"query":"public facts"}'),
                        reply_json("Unavailable."),
                        raise_tool_errors=False,
                    )
                )
                stream = MemoryStream()
                with (
                    self.transport(respond),
                    patch(
                        "dashulya.ai.tools.web.WEB_TIMEOUT_SECONDS",
                        10 if cancel else 0.05,
                    ),
                ):
                    task = asyncio.create_task(agent.ask("Find", stream=stream))
                    await asyncio.wait_for(started.wait(), timeout=2)
                    if cancel:
                        task.cancel()
                        with pytest.raises(asyncio.CancelledError):
                            await task
                    else:
                        await asyncio.wait_for(task, timeout=2)
                        results = await self.results(stream)
                        assert "timed out" in repr(results)
                assert interrupted.is_set()
                assert self.clients[-1].is_closed
