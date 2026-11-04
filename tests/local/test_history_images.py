import base64
import json
from collections.abc import AsyncGenerator
from contextlib import AsyncExitStack
from dataclasses import replace
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, patch

import httpx2
import pytest
from ag2 import Agent
from ag2.config import ModelConfig, OpenAIResponsesConfig
from ag2.events import (
    BinaryInput,
    ModelRequest,
    TextInput,
    ToolCallEvent,
    ToolResultEvent,
)
from ag2.testing import TestConfig
from aiogram import Bot
from aiogram.types import File

from dashulya.ai.dialogue import Dialogue
from dashulya.domain.memory import (
    ChatHistory,
    ContextImages,
    HistoryReader,
    MemoryAccess,
    OriginalMessage,
)
from dashulya.domain.messages import ImageAttachment, ImageFile, ReplyBatch, ReplyRequest
from dashulya.main import create_container
from dashulya.main.config import Settings
from dashulya.memory.history import CanonicalHistory
from dashulya.memory.store import MessageStore
from tests.support import (
    TEST_API_KEY,
    BotOverride,
    MemoryOverride,
    ModelOverrides,
    input_payload,
    reply_json,
)


@pytest.mark.usefixtures("_setup")
class TestHistoryImages:
    @pytest.fixture
    def _setup(self, cleanup: AsyncExitStack) -> None:
        self.cleanup = cleanup
        self.now = datetime(2026, 11, 1, tzinfo=UTC)
        self.source = OriginalMessage(
            collection="telegram:-123",
            chat_id=-123,
            message_id=1,
            text="Old photo",
            sent_at=datetime(2026, 10, 1, tzinfo=UTC),
            author_id=42,
            author="Owner",
            chat_title="Group",
            chat_username=None,
            topic_id=None,
            reply_to_id=None,
            edited_at=None,
            media_type="photo",
            url=None,
            provenance="observed",
            image_file=ImageFile("saved-photo"),
        )
        self.store = AsyncMock(spec=MessageStore)
        self.store.originals.return_value = [self.source]
        self.bot = Bot("123:TEST")
        self.cleanup.push_async_callback(self.bot.session.close)
        self.image = b"\x89PNG\r\n\x1a\nsaved-image"
        self.metadata = AsyncMock(
            return_value=File(
                file_id="saved-photo", file_unique_id="unique", file_path="photo.png"
            )
        )

    def read(self, source_id: str | None = None) -> ToolCallEvent:
        return ToolCallEvent(
            "chat_history",
            arguments=json.dumps({
                "sql": [
                    f"SELECT * FROM messages WHERE source_id = '{source_id or self.source.source_id}' ORDER BY message_id LIMIT 1"
                ]
            }),
        )

    async def create(
        self,
        *turns: Any,
        max_bytes: int = 20_000_000,
        chat_id: int = -123,
        model: ModelConfig | None = None,
    ) -> Dialogue:
        canonical = CanonicalHistory(
            self.store, MemoryAccess(chat_id, chat_id > 0, before=self.now, before_id=99)
        )
        reader = AsyncMock(spec=HistoryReader)
        reader.original.side_effect = canonical.original

        async def query(sql: list[str]) -> dict[str, Any]:
            return {
                "query_count": len(sql),
                "results": [await one_query(item) for item in sql],
            }

        async def one_query(sql: str) -> dict[str, Any]:
            if "WHERE source_id =" in sql:
                originals = await self.store.originals()
                rows = [
                    source.render() for source in originals if source.source_id in sql
                ]
            else:
                rows = []
            return {
                "columns": list(rows[0]) if rows else [],
                "rows": rows,
                "coverage": {"complete": False},
            }

        reader.sql.side_effect = query
        container = create_container(
            Settings(bot_token="123:TEST", openai_api_key=TEST_API_KEY, owner_user_id=42),
            BotOverride(self.bot),
            MemoryOverride(reader=reader),
            ModelOverrides(
                reply=model or TestConfig(*turns), decision=TestConfig("false")
            ),
        )
        self.cleanup.push_async_callback(container.close)
        self.agent = await container.get(Agent[ReplyBatch])
        return Dialogue(
            self.agent,
            history=await container.get(ChatHistory),
            context_images=await container.get(ContextImages),
            max_image_bytes=max_bytes,
        )

    async def reply(
        self, dialogue: Dialogue, *, chat_id: int = -123, incoming: bool = False
    ) -> list[Any]:
        async def content(*args: Any, **kwargs: Any) -> AsyncGenerator[bytes]:
            yield self.image

        with (
            patch.object(self.agent, "ask", wraps=self.agent.ask) as ask,
            patch.object(self.bot, "get_file", self.metadata),
            patch.object(self.bot.session, "stream_content", content),
        ):
            await dialogue.reply(
                ReplyRequest(
                    (chat_id, None),
                    "Compare the old photo",
                    user_id=42,
                    user_name="Owner",
                    sent_at=self.now,
                    message_id=99,
                    images=(ImageAttachment(self.image, "image/png", 99),)
                    if incoming
                    else (),
                )
            )
        return list(await ask.call_args.kwargs["stream"].history.get_events())

    @staticmethod
    def results(events: list[Any]) -> list[ToolResultEvent]:
        return [event for event in events if isinstance(event, ToolResultEvent)]

    @staticmethod
    def data(result: ToolResultEvent) -> dict[str, Any]:
        part = result.result.parts[0]
        assert isinstance(part, TextInput)
        data = json.loads(input_payload(part.content))
        assert isinstance(data, dict)
        return data["results"][0] if len(data["results"]) == 1 else data

    async def test_read_supplies_saved_image_outside_reply_context_and_reuses_cache(
        self,
    ) -> None:
        dialogue = await self.create(
            self.read(), reply_json("Seen"), self.read(), reply_json("Seen again")
        )
        for _ in range(2):
            result = self.results(await self.reply(dialogue))[0]
            assert [
                part.data for part in result.result.parts if isinstance(part, BinaryInput)
            ] == [self.image]
            data = self.data(result)
            assert data["image"]["status"] == "loaded"
            assert data["image"]["untrusted"] is True
            assert "saved-photo" not in json.dumps(data)
        self.metadata.assert_awaited_once_with("saved-photo")

    async def test_repeated_read_in_one_turn_does_not_duplicate_image(self) -> None:
        dialogue = await self.create(self.read(), self.read(), reply_json("Seen"))
        first, second = self.results(await self.reply(dialogue))
        assert len(first.result.parts) == 2
        assert len(second.result.parts) == 1
        assert self.data(second)["image"]["status"] == "already_supplied"
        self.metadata.assert_awaited_once()

    async def test_history_image_shares_budget_with_current_context(self) -> None:
        dialogue = await self.create(
            self.read(), reply_json("Limited"), max_bytes=len(self.image)
        )
        (result,) = self.results(await self.reply(dialogue, incoming=True))
        assert len(result.result.parts) == 1
        assert self.data(result)["image"]["status"] == "too_large"
        self.metadata.assert_not_awaited()

    async def test_parallel_reads_share_one_turn_budget(self) -> None:
        other = replace(self.source, message_id=2, image_file=ImageFile("other-photo"))
        self.store.originals.return_value = [self.source, other]
        dialogue = await self.create(
            [self.read(), self.read(other.source_id)],
            reply_json("Limited"),
            max_bytes=len(self.image),
        )
        results = self.results(await self.reply(dialogue))
        assert (
            sum(
                isinstance(part, BinaryInput)
                for result in results
                for part in result.result.parts
            )
            == 1
        )
        assert {self.data(result)["image"]["status"] for result in results} == {
            "loaded",
            "too_large",
        }
        self.metadata.assert_awaited_once()

    @pytest.mark.parametrize(
        "sql",
        [
            "SELECT message_id FROM messages",
            "SELECT count(*) AS count FROM messages",
            "SELECT DISTINCT author_id FROM messages",
        ],
    )
    async def test_listing_and_aggregation_do_not_download_images(self, sql: str) -> None:
        dialogue = await self.create(
            ToolCallEvent("chat_history", arguments=json.dumps({"sql": [sql]})),
            reply_json("Listed"),
        )
        results = self.results(await self.reply(dialogue))
        assert len(results[0].result.parts) == 1
        self.metadata.assert_not_awaited()

    async def test_batched_images_share_budget_and_preserve_each_query_status(
        self,
    ) -> None:
        other = replace(self.source, message_id=2, image_file=ImageFile("other-photo"))
        self.store.originals.return_value = [self.source, other]
        queries = [
            self.read().serialized_arguments["sql"][0],
            self.read(other.source_id).serialized_arguments["sql"][0],
        ]
        dialogue = await self.create(
            ToolCallEvent("chat_history", arguments=json.dumps({"sql": queries})),
            reply_json("Seen"),
            max_bytes=len(self.image),
        )
        (event,) = self.results(await self.reply(dialogue))
        data = self.data(event)
        assert [item["image"]["status"] for item in data["results"]] == [
            "loaded",
            "too_large",
        ]
        assert len(event.result.parts) == 2
        self.metadata.assert_awaited_once()

    async def test_repeated_source_inside_batch_supplies_one_image(self) -> None:
        query = self.read().serialized_arguments["sql"][0]
        dialogue = await self.create(
            ToolCallEvent("chat_history", arguments=json.dumps({"sql": [query, query]})),
            reply_json("Seen"),
        )
        (event,) = self.results(await self.reply(dialogue))
        assert [item["image"]["status"] for item in self.data(event)["results"]] == [
            "loaded",
            "already_supplied",
        ]
        assert len(event.result.parts) == 2

    async def test_missing_file_reference_is_distinct_from_unavailable_download(
        self,
    ) -> None:
        self.store.originals.return_value = [replace(self.source, image_file=None)]
        dialogue = await self.create(self.read(), reply_json("Missing reference"))
        (result,) = self.results(await self.reply(dialogue))
        assert self.data(result)["image"]["status"] == "no_saved_reference"
        self.metadata.assert_not_awaited()

    async def test_unavailable_file_preserves_text_and_reports_omission(self) -> None:
        self.metadata.side_effect = TimeoutError
        dialogue = await self.create(self.read(), reply_json("Unavailable"))
        (result,) = self.results(await self.reply(dialogue))
        assert self.data(result)["image"]["status"] == "unavailable"
        assert "Old photo" in str(self.data(result))
        assert len(result.result.parts) == 1

    async def test_animated_sticker_read_supplies_only_its_labeled_preview(self) -> None:
        self.store.originals.return_value = [
            replace(
                self.source,
                media_type="sticker",
                image_file=ImageFile(
                    "animated-sticker",
                    preview_kind="sticker",
                    thumbnail_file_id="thumbnail",
                ),
            )
        ]
        dialogue = await self.create(self.read(), reply_json("Preview"))
        (result,) = self.results(await self.reply(dialogue))
        assert self.data(result)["image"]["status"] == "loaded"
        assert self.data(result)["image"]["preview_kind"] == "sticker"
        assert self.data(result)["image"]["first_frame"] is False
        self.metadata.assert_awaited_once_with("thumbnail")

    async def test_private_next_turn_evicts_old_tool_images_within_shared_budget(
        self,
    ) -> None:
        dialogue = await self.create(
            self.read(),
            reply_json("Seen"),
            reply_json("New picture"),
            max_bytes=len(self.image),
            chat_id=42,
        )
        await self.reply(dialogue, chat_id=42)
        events = await self.reply(dialogue, chat_id=42, incoming=True)
        parts = [
            part
            for event in events
            if isinstance(event, (ModelRequest, ToolResultEvent))
            for part in (
                event.parts if isinstance(event, ModelRequest) else event.result.parts
            )
        ]
        assert sum(
            len(part.data) for part in parts if isinstance(part, BinaryInput)
        ) == len(self.image)
        old = self.results(events)[0]
        assert not any(isinstance(part, BinaryInput) for part in old.result.parts)
        assert "Earlier image omitted" in str(old.result.parts)

    async def test_responses_api_receives_native_history_image_and_no_chat_selectors(
        self,
    ) -> None:
        requests: list[dict[str, Any]] = []

        def respond(request: httpx2.Request) -> httpx2.Response:
            requests.append(json.loads(request.content))
            output = (
                [
                    {
                        "type": "function_call",
                        "id": "fc_1",
                        "call_id": "history-image",
                        "name": "chat_history",
                        "arguments": self.read().arguments,
                        "status": "completed",
                    }
                ]
                if len(requests) == 1
                else [
                    {
                        "type": "message",
                        "id": "msg_1",
                        "role": "assistant",
                        "status": "completed",
                        "content": [
                            {
                                "type": "output_text",
                                "text": reply_json("Seen"),
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
                    "model": "test",
                    "output": output,
                },
            )

        async with httpx2.AsyncClient(transport=httpx2.MockTransport(respond)) as client:
            dialogue = await self.create(
                model=OpenAIResponsesConfig(
                    model="test", api_key=TEST_API_KEY, http_client=client
                )
            )
            await self.reply(dialogue)
        assert len(requests) == 2
        tool_output = next(
            item
            for item in requests[1]["input"]
            if item.get("type") == "function_call_output"
        )
        assert tool_output["call_id"] == "history-image"
        assert tool_output["output"][1] == {
            "type": "input_image",
            "image_url": "data:image/png;base64," + base64.b64encode(self.image).decode(),
        }
        for schema in requests[0]["tools"]:
            if schema.get("name") in {"chat_history", "search_memory"}:
                assert "chat" not in schema["parameters"]["properties"]
                assert "image_turn" not in schema["parameters"]["properties"]
                assert all(
                    field.get("description")
                    for field in schema["parameters"]["properties"].values()
                )
                for bound in (
                    ("after", "before") if schema["name"] == "search_memory" else ()
                ):
                    assert {"format": "date-time", "type": "string"} in schema[
                        "parameters"
                    ]["properties"][bound]["anyOf"]
            if schema.get("name") == "chat_history":
                assert set(schema["parameters"]["properties"]) == {"sql"}
                field = schema["parameters"]["properties"]["sql"]
                assert field["type"] == "array"
                assert field["items"] == {"type": "string"}
                assert (field["minItems"], field["maxItems"]) == (1, 5)
