import asyncio
import logging
from dataclasses import asdict
from typing import Any
from urllib.parse import urlsplit

from ag2 import Context
from ag2.events import DataInput, ToolCallEvent, ToolErrorEvent, ToolResultEvent
from ag2.extensions.tools.search.exa import (
    ExaContentResult,
    ExaSearchResponse,
    ExaSearchResult,
    ExaToolkit,
)
from ag2.middleware import ToolExecution, ToolResultType
from ag2.tools.tool import Tool

logger = logging.getLogger(__name__)

WEB_TIMEOUT_SECONDS = 30
SEARCH_RESULTS = 5
SEARCH_TEXT_CHARACTERS = 4_000
PAGE_TEXT_CHARACTERS = 12_000
MAX_URLS = 3


def _input_error(event: ToolCallEvent) -> str | None:
    arguments = event.serialized_arguments
    if event.name == "exa_search":
        query = arguments.get("query")
        if not isinstance(query, str) or not query.strip() or len(query) > 2_000:
            return "Use a nonempty public query of at most 2000 characters."
    else:
        urls = arguments.get("urls")
        if not isinstance(urls, list) or not 1 <= len(urls) <= MAX_URLS:
            return "Provide one to three public HTTP(S) URLs."
        for value in urls:
            if not isinstance(value, str) or len(value) > 2_048:
                return "Provide valid public HTTP(S) URLs without credentials."
            parsed = urlsplit(value)
            if (
                parsed.scheme not in {"http", "https"}
                or not parsed.hostname
                or parsed.username is not None
                or parsed.password is not None
            ):
                return "Provide valid public HTTP(S) URLs without credentials."
    return None


def _failure(event: ToolCallEvent, message: str) -> ToolErrorEvent:
    return ToolErrorEvent.from_call(event, RuntimeError(message))


def _page(page: ExaContentResult | ExaSearchResult, limit: int) -> dict[str, Any]:
    item = asdict(page)
    text = page.text or ""
    item["text"] = text[:limit]
    item["truncated"] = len(text) >= limit
    return item


def _bounded_output(event: ToolCallEvent, result: ToolResultType) -> ToolResultType:
    if not isinstance(result, ToolResultEvent):
        return result
    if len(result.result.parts) != 1 or not isinstance(result.result.parts[0], DataInput):
        return _failure(event, "Web lookup returned an unreadable result.")
    data = result.result.parts[0].data
    if isinstance(data, ExaSearchResponse):
        output = {
            "query": data.query,
            "results": [
                _page(page, SEARCH_TEXT_CHARACTERS)
                for page in data.results[:SEARCH_RESULTS]
            ],
        }
    elif isinstance(data, list) and all(
        isinstance(page, ExaContentResult) for page in data
    ):
        output = {
            "results": [_page(page, PAGE_TEXT_CHARACTERS) for page in data[:MAX_URLS]]
        }
    else:
        return _failure(event, "Web lookup returned an unreadable result.")
    return ToolResultEvent.from_call(event, output)


async def _web_call(
    call_next: ToolExecution, event: ToolCallEvent, context: Context
) -> ToolResultType:
    try:
        if error := _input_error(event):
            return _failure(event, error)
        async with asyncio.timeout(WEB_TIMEOUT_SECONDS):
            result = await call_next(event, context)
    except TimeoutError:
        return _failure(
            event,
            "Web lookup timed out. Do not retry automatically; state what could not be verified.",
        )
    except Exception as error:
        logger.warning("Web lookup failed", extra={"error_type": type(error).__name__})
        return _failure(
            event,
            "Web lookup failed. Do not retry automatically; state what could not be verified.",
        )
    else:
        if isinstance(result, ToolErrorEvent):
            # AG2 captures executor exceptions as events; replace their raw tracebacks.
            logger.warning(
                "Web lookup failed", extra={"error_type": type(result.error).__name__}
            )
            return _failure(
                event,
                "Web lookup failed. Do not retry automatically; state what could not be verified.",
            )
        return _bounded_output(event, result)


def create_web_tools(api_key: str) -> tuple[Tool, ...]:
    if not api_key:
        return ()
    toolkit = ExaToolkit(api_key=api_key)
    return (
        toolkit.search(
            num_results=SEARCH_RESULTS,
            max_characters=SEARCH_TEXT_CHARACTERS,
            search_type="auto",
            middleware=[_web_call],
        ),
        toolkit.get_contents(middleware=[_web_call]),
    )
