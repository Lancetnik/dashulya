import json
import traceback

import httpx2
import pytest
from openai import AsyncOpenAI

from dashulya.ai.embeddings import MemoryEmbeddings
from tests.support import TEST_API_KEY


class TestMemoryEmbeddings:
    async def test_request_policy_and_response_order_through_real_sdk(
        self, subtests: pytest.Subtests
    ) -> None:
        for model in (None, "custom-embedding-model"):
            with subtests.test(model=model):
                requests: list[httpx2.Request] = []

                def respond(
                    request: httpx2.Request, requests: list[httpx2.Request] = requests
                ) -> httpx2.Response:
                    requests.append(request)
                    return httpx2.Response(
                        200,
                        json={
                            "data": [
                                {"index": 2, "embedding": [0.5, 0.6]},
                                {"index": 0, "embedding": [0.1, 0.2]},
                                {"index": 1, "embedding": [0.3, 0.4]},
                            ],
                            "model": "test",
                            "usage": {"prompt_tokens": 3, "total_tokens": 3},
                        },
                    )

                async with (
                    httpx2.AsyncClient(transport=httpx2.MockTransport(respond)) as http,
                    AsyncOpenAI(api_key=TEST_API_KEY, http_client=http) as client,
                ):
                    embed = (
                        MemoryEmbeddings(client)
                        if model is None
                        else MemoryEmbeddings(client, model=model)
                    )
                    result = await embed(["First original", "Второй оригинал", "Third"])
                assert result == [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
                assert len(requests) == 1
                request = requests[0]
                assert request.url.path == "/v1/embeddings"
                assert request.headers["authorization"] == f"Bearer {TEST_API_KEY}"
                assert json.loads(request.content) == {
                    "input": ["First original", "Второй оригинал", "Third"],
                    "model": model or "text-embedding-3-small",
                    "dimensions": 512,
                    "encoding_format": "float",
                }

    async def test_incomplete_duplicate_or_unexpected_indices_are_rejected(
        self, subtests: pytest.Subtests
    ) -> None:
        for indices in ([], [0], [0, 0], [-1, 0], [0, 2], [0, 1, 2]):
            with subtests.test(indices=indices):

                def respond(
                    request: httpx2.Request, indices: list[int] = indices
                ) -> httpx2.Response:
                    return httpx2.Response(
                        200,
                        json={
                            "data": [{"index": i, "embedding": [0.1]} for i in indices],
                            "model": "test",
                            "usage": {"prompt_tokens": 2, "total_tokens": 2},
                        },
                    )

                async with (
                    httpx2.AsyncClient(transport=httpx2.MockTransport(respond)) as http,
                    AsyncOpenAI(api_key=TEST_API_KEY, http_client=http) as client,
                ):
                    with pytest.raises(ValueError, match="response is incomplete"):
                        await MemoryEmbeddings(client)(["One", "Two"])

    async def test_provider_failures_expose_only_the_safe_error(
        self, subtests: pytest.Subtests
    ) -> None:
        for status in (401, 429, 500):
            with subtests.test(status=status):
                requests: list[httpx2.Request] = []

                def respond(
                    request: httpx2.Request,
                    status: int = status,
                    requests: list[httpx2.Request] = requests,
                ) -> httpx2.Response:
                    requests.append(request)
                    return httpx2.Response(
                        status,
                        json={
                            "error": {"message": f"PRIVATE PROVIDER BODY {TEST_API_KEY}"}
                        },
                    )

                async with (
                    httpx2.AsyncClient(transport=httpx2.MockTransport(respond)) as http,
                    AsyncOpenAI(
                        api_key=TEST_API_KEY, http_client=http, max_retries=0
                    ) as client,
                ):
                    with pytest.raises(RuntimeError) as raised:
                        await MemoryEmbeddings(client)(["One"])
                assert str(raised.value) == "Memory embedding request failed."
                exposed = "".join(traceback.format_exception(raised.value))
                assert "PRIVATE PROVIDER BODY" not in exposed
                assert TEST_API_KEY not in exposed
                assert len(requests) == 1
