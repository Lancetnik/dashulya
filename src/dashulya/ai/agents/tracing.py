"""Request-correlated model and tool timings without conversation payloads."""

import asyncio
import logging
from collections.abc import Sequence
from time import perf_counter

from ag2 import Context
from ag2.events import (
    BaseEvent,
    BinaryInput,
    BinaryType,
    ModelRequest,
    ModelResponse,
    TextInput,
    ToolCallEvent,
    ToolErrorEvent,
)
from ag2.middleware.base import BaseMiddleware, LLMCall, ToolExecution, ToolResultType

from dashulya.domain.memory import MemoryAccess

logger = logging.getLogger(__name__)


def _request(context: Context) -> dict[str, object]:
    access = context.dependencies.get(MemoryAccess)
    if isinstance(access, MemoryAccess):
        return {"chat_id": access.chat_id, "message_id": access.before_id}
    return {}


class ReplyTrace(BaseMiddleware):
    async def on_llm_call(
        self, call_next: LLMCall, events: Sequence[BaseEvent], context: Context
    ) -> ModelResponse:
        fields = _request(context)
        parts = [
            part
            for event in events
            if isinstance(event, ModelRequest)
            for part in event.parts
        ]
        fields.update(
            history_events=len(events),
            input_chars=sum(
                len(part.content) for part in parts if isinstance(part, TextInput)
            ),
            image_count=sum(
                isinstance(part, BinaryInput) and part.kind == BinaryType.IMAGE
                for part in parts
            ),
        )
        started = perf_counter()
        logger.info("AI model call started", extra=fields)
        try:
            response = await call_next(events, context)
        except asyncio.CancelledError:
            self._completed("AI model call completed", started, fields, "cancelled")
            raise
        except Exception as error:
            self._completed(
                "AI model call completed",
                started,
                {**fields, "error_type": type(error).__name__},
                "error",
            )
            raise
        self._completed(
            "AI model call completed",
            started,
            {
                **fields,
                "model": response.model,
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "thinking_tokens": response.usage.thinking_tokens,
                "cached_input_tokens": response.usage.cache_read_input_tokens,
            },
            "ok",
        )
        return response

    async def on_tool_execution(
        self, call_next: ToolExecution, event: ToolCallEvent, context: Context
    ) -> ToolResultType:
        fields = {**_request(context), "tool_name": event.name, "tool_call_id": event.id}
        started = perf_counter()
        logger.info("AI tool call started", extra=fields)
        try:
            result = await call_next(event, context)
        except asyncio.CancelledError:
            self._completed("AI tool call completed", started, fields, "cancelled")
            raise
        except Exception as error:
            self._completed(
                "AI tool call completed",
                started,
                {**fields, "error_type": type(error).__name__},
                "error",
            )
            raise
        if isinstance(result, ToolErrorEvent):
            fields["error_type"] = type(result.error).__name__
        self._completed(
            "AI tool call completed",
            started,
            fields,
            "error" if isinstance(result, ToolErrorEvent) else "ok",
        )
        return result

    @staticmethod
    def _completed(
        message: str, started: float, fields: dict[str, object], outcome: str
    ) -> None:
        logger.info(
            message,
            extra={
                **fields,
                "duration_ms": round((perf_counter() - started) * 1000),
                "outcome": outcome,
            },
        )
