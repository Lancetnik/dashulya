import asyncio
import json
import unittest
from collections.abc import Callable, Coroutine
from dataclasses import replace
from typing import Any
from unittest.mock import patch

import httpx
from ag2 import Agent, MemoryStream
from ag2.events import DataInput, ToolCallEvent, ToolErrorEvent, ToolResultEvent
from ag2.testing import TestConfig, TrackingConfig
from ag2.tools.final.function_tool import FunctionToolSchema

from dashulya.ai.tools.web import PAGE_TEXT_CHARACTERS, SEARCH_TEXT_CHARACTERS
from dashulya.domain.messages import Reply
from dashulya.main import create_container
from dashulya.main.config import Settings
from tests.support import TEST_API_KEY, MemoryOverride, ModelOverrides, reply_json


class WebSearchTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
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
        self.addAsyncCleanup(container.close)
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
        self.assertIn(url, reply.text)
        self.assertEqual([r.url.path for r in self.requests], ["/search", "/contents"])
        body = json.loads(self.requests[0].content)
        self.assertEqual(body["query"], "AG2 Exa documentation")
        self.assertEqual(body["numResults"], 5)
        self.assertEqual(body["type"], "auto")
        self.assertEqual(
            body["contents"]["text"]["maxCharacters"], SEARCH_TEXT_CHARACTERS
        )
        self.assertEqual(json.loads(self.requests[1].content)["urls"], [url])
        for request in self.requests:
            self.assertEqual(request.headers["x-api-key"], self.settings.exa_api_key)
            self.assertNotIn(b"PRIVATE CHAT CONTEXT", request.content)
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
            self.assertEqual(page["url"], url)
            self.assertEqual(page["title"], "AG2 Exa tools")
            self.assertEqual(page["published_date"], "2026-10-21T00:00:00Z")
            self.assertEqual(len(str(page["text"])), limit)
            self.assertTrue(page["truncated"])
        self.assertEqual(len(self.clients), 2)
        self.assertTrue(all(client.is_closed for client in self.clients))

    async def test_only_reply_agent_receives_web_tools_and_key_stays_out_of_schema(
        self,
    ) -> None:
        for enabled in (True, False):
            with self.subTest(enabled=enabled):
                config = TrackingConfig(TestConfig(reply_json("Done")))
                agent = await self.agent(config, enabled=enabled)
                await agent.ask("Hello")
                tools = [
                    schema.function.name
                    for schema in config.calls[0].tools
                    if isinstance(schema, FunctionToolSchema)
                ]
                self.assertIn("search_memory", tools)
                self.assertIn("chat_history", tools)
                self.assertEqual("exa_search" in tools, enabled)
                self.assertEqual("exa_get_contents" in tools, enabled)
                self.assertNotIn("exa_answer", tools)
                self.assertNotIn("exa_find_similar", tools)
                self.assertNotIn(self.settings.exa_api_key, repr(config.calls[0]))

        decision = TrackingConfig(TestConfig("false"))
        container = create_container(
            self.settings,
            ModelOverrides(reply=TestConfig(reply_json("Done")), decision=decision),
        )
        self.addAsyncCleanup(container.close)
        classifier = await container.get(Agent[bool])
        await classifier.ask("Does this address Dashulya?")
        self.assertEqual(decision.calls[0].tools, ())

    async def test_provider_errors_are_redacted_and_not_retried(self) -> None:
        for status in (401, 429, 500):
            with self.subTest(status=status):
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
                with (
                    self.transport(respond),
                    self.assertLogs("dashulya.ai.tools.web", level="WARNING") as logs,
                ):
                    await agent.ask("Find public facts", stream=stream)
                results = await self.results(stream)
                self.assertIsInstance(results[0], ToolErrorEvent)
                for output in (repr(results), " ".join(logs.output)):
                    self.assertNotIn("PRIVATE PROVIDER BODY", output)
                    self.assertNotIn(self.settings.exa_api_key, output)
                self.assertIn("Do not retry automatically", repr(results))
                self.assertEqual(len(self.requests), 1)
                self.assertTrue(self.clients[-1].is_closed)

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
        self.assertEqual(part.data["results"], [])

    async def test_request_limits_reject_inputs_before_calling_exa(self) -> None:
        calls = [
            ToolCallEvent("exa_search", arguments='{"query":" "}'),
            ToolCallEvent("exa_search", arguments=json.dumps({"query": "x" * 2001})),
            ToolCallEvent("exa_get_contents", arguments='{"urls":[]}'),
            ToolCallEvent(
                "exa_get_contents",
                arguments=json.dumps({"urls": ["https://example.com"] * 4}),
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
            with self.subTest(call=call):
                agent = await self.agent(
                    TestConfig(
                        call, reply_json("Invalid input."), raise_tool_errors=False
                    )
                )
                stream = MemoryStream()
                with patch("ag2.extensions.tools.search.exa.AsyncExa") as sdk:
                    await agent.ask("Find", stream=stream)
                sdk.assert_not_called()
                self.assertIsInstance((await self.results(stream))[0], ToolErrorEvent)

    async def test_timeout_and_cancellation_close_the_sdk_client(self) -> None:
        for cancel in (False, True):
            with self.subTest(cancel=cancel):
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
                        with self.assertRaises(asyncio.CancelledError):
                            await task
                    else:
                        await asyncio.wait_for(task, timeout=2)
                        results = await self.results(stream)
                        self.assertIn("timed out", repr(results))
                self.assertTrue(interrupted.is_set())
                self.assertTrue(self.clients[-1].is_closed)
