"""Request-correlated model output and tool timings, including history SQL."""

import asyncio
import json
import logging
from collections.abc import Callable, Sequence
from time import perf_counter
from typing import Annotated, override

from ag2 import Context, Inject, observer
from ag2.events import (
    BaseEvent,
    BinaryInput,
    BinaryType,
    BuiltinToolCallEvent,
    BuiltinToolResultEvent,
    ModelRequest,
    ModelResponse,
    TextInput,
    ToolCallEvent,
    ToolErrorEvent,
    ToolResult,
    ToolResultEvent,
)
from ag2.middleware.base import BaseMiddleware, LLMCall, ToolExecution, ToolResultType
from fast_depends import inject
from pydantic import ValidationError

from dashulya.domain.memory import MemoryAccess, MemoryInputError
from dashulya.observability import exception_fields

logger = logging.getLogger(__name__)


def _memory_failure(event: ToolCallEvent, error: Exception) -> ToolErrorEvent:
    message = f"{event.name} failed ({type(error).__name__}): "
    if isinstance(error, MemoryInputError):
        message += (
            f"{error} Correct the arguments within the current memory access scope."
        )
    elif isinstance(error, ValidationError):
        details = "; ".join(
            f"{'.'.join(str(part) for part in item['loc'])}: {item['type']}"
            for item in error.errors(include_input=False, include_context=False)
        )
        message += f"Invalid arguments: {details}. Correct the arguments and try again."
    else:
        message += (
            "Memory lookup could not be completed. Do not treat this as an empty result "
            "or claim that no matching messages exist."
        )
        if event.name == "search_memory":
            message += " Use chat_history for canonical originals if available."
    return ToolErrorEvent(
        name=event.name,
        parent_id=event.id,
        error=error,
        result=ToolResult(message),
    )


def _request_fields(
    access: Annotated[MemoryAccess | None, Inject("memory_access", default=None)],
) -> dict[str, object]:
    if isinstance(access, MemoryAccess):
        return {"chat_id": access.chat_id, "message_id": access.before_id}
    return {}


# AG2 calls middleware hooks directly, outside its annotation resolver.
_request: Callable[..., dict[str, object]] = inject(_request_fields, cast=False)


@observer(BuiltinToolCallEvent | BuiltinToolResultEvent)
async def log_builtin_tool(event: BaseEvent, context: Context) -> None:
    fields = _request(__ctx__=context)
    if isinstance(event, BuiltinToolCallEvent):
        fields["tool_name"] = event.name
        fields["tool_call_id"] = event.id
        logger.info("AI builtin tool call observed", extra=fields)
    elif isinstance(event, BuiltinToolResultEvent):
        fields["tool_name"] = event.name
        fields["tool_call_id"] = event.parent_id
        status = event.result.metadata.get("status")
        if isinstance(status, str):
            fields["tool_status"] = status
        logger.log(
            logging.ERROR if status == "failed" else logging.INFO,
            "AI builtin tool result observed",
            extra=fields,
        )


def _outcome_fields(
    started: float, fields: dict[str, object], outcome: str
) -> dict[str, object]:
    return {
        **fields,
        "duration_ms": round((perf_counter() - started) * 1000),
        "outcome": outcome,
    }


def _output_fields(response: ModelResponse) -> dict[str, object]:
    output = response.content
    count = None
    if output is not None:
        try:
            value = json.loads(output)
        except ValueError:
            pass
        else:
            if isinstance(value, dict) and isinstance(value.get("messages"), list):
                count = len(value["messages"])
    return {
        "agent_output": output,
        "message_count": count,
        "tool_call_count": len(response.tool_calls),
    }


def _history_sql_fields(event: ToolCallEvent) -> dict[str, object]:
    if event.name != "chat_history":
        return {}
    try:
        query = event.serialized_arguments.get("sql")
    except (AttributeError, ValueError, TypeError):
        return {}
    if isinstance(query, list) and all(isinstance(item, str) for item in query):
        return {
            "sql": [item[:16_000] for item in query[:5]],
            "sql_truncated": len(query) > 5
            or any(len(item) > 16_000 for item in query[:5]),
            "query_count": len(query),
        }
    return {}


class ReplyTrace(BaseMiddleware):
    @override
    async def on_llm_call(
        self, call_next: LLMCall, events: Sequence[BaseEvent], context: Context
    ) -> ModelResponse:
        fields = _request(__ctx__=context)
        parts = [
            part
            for event in events
            if isinstance(event, (ModelRequest, ToolResultEvent))
            for part in (
                event.parts if isinstance(event, ModelRequest) else event.result.parts
            )
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
            logger.info(
                "AI model call completed",
                extra=_outcome_fields(started, fields, "cancelled"),
            )
            raise
        except Exception as error:
            logger.exception(
                "AI model call completed",
                extra={
                    **_outcome_fields(started, fields, "error"),
                    **exception_fields(error),
                },
            )
            raise
        logger.info(
            "AI model call completed",
            extra={
                **_outcome_fields(started, fields, "ok"),
                **_output_fields(response),
                "model": response.model,
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "thinking_tokens": response.usage.thinking_tokens,
                "cached_input_tokens": response.usage.cache_read_input_tokens,
            },
        )
        return response

    @override
    async def on_tool_execution(
        self, call_next: ToolExecution, event: ToolCallEvent, context: Context
    ) -> ToolResultType:
        fields = {
            **_request(__ctx__=context),
            "tool_name": event.name,
            "tool_call_id": event.id,
            **_history_sql_fields(event),
        }
        started = perf_counter()
        logger.info("AI tool call started", extra=fields)
        try:
            result = await call_next(event, context)
        except asyncio.CancelledError:
            logger.info(
                "AI tool call completed",
                extra=_outcome_fields(started, fields, "cancelled"),
            )
            raise
        except (MemoryInputError, ValidationError) as error:
            logger.warning(
                "AI tool call completed",
                extra={
                    **_outcome_fields(started, fields, "error"),
                    **exception_fields(error),
                },
            )
            raise
        except Exception as error:
            logger.exception(
                "AI tool call completed",
                extra={
                    **_outcome_fields(started, fields, "error"),
                    **exception_fields(error),
                },
            )
            raise
        if isinstance(result, ToolErrorEvent):
            extra = {
                **_outcome_fields(started, fields, "error"),
                **exception_fields(result.error),
            }
            if isinstance(result.error, (MemoryInputError, ValidationError)):
                logger.warning("AI tool call completed", extra=extra)
            else:
                logger.error("AI tool call completed", extra=extra)
            if event.name in {"search_memory", "chat_history"}:
                return _memory_failure(event, result.error)
        else:
            logger.info(
                "AI tool call completed", extra=_outcome_fields(started, fields, "ok")
            )
        return result
