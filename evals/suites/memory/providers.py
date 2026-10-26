"""Structured model calls shared by local and explicitly configured cloud experiments."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Literal
from urllib.parse import urlparse

from pydantic import BaseModel

if TYPE_CHECKING:
    from openai import AsyncOpenAI


def configured_api() -> Literal["chat", "responses"]:
    value = os.getenv("MEMORY_EVAL_API", "responses")
    if value == "chat":
        return "chat"
    if value == "responses":
        return "responses"
    msg = "MEMORY_EVAL_API must be chat or responses."
    raise ValueError(msg)


def validate_local_endpoint(base_url: str | None, model: str) -> None:
    """Reject any remote destination when the local evaluation mode is requested."""
    if os.getenv("MEMORY_EVAL_LOCAL_ONLY", "0") != "1":
        return
    parsed = urlparse(base_url or "")
    if parsed.scheme != "http" or parsed.hostname not in {
        "ollama",
        "localhost",
        "127.0.0.1",
        "::1",
    }:
        msg = "Local evaluation requires an explicit local HTTP model endpoint."
        raise ValueError(msg)
    if "cloud" in model.casefold() or "/" in model or "://" in model:
        msg = "Local evaluation does not allow remote model routes."
        raise ValueError(msg)


async def parse_structured[Result: BaseModel](
    client: AsyncOpenAI,
    model: str,
    instructions: str,
    input: str,
    schema: type[Result],
    max_tokens: int = 8192,
) -> tuple[Result, dict[str, object]]:
    """Return a validated schema and token usage without changing evidence semantics."""
    validate_local_endpoint(str(client.base_url), model)
    if configured_api() == "chat":
        completion = await client.chat.completions.parse(
            model=model,
            messages=[
                {"role": "system", "content": instructions},
                {"role": "user", "content": input},
            ],
            response_format=schema,
            max_tokens=max_tokens,
            reasoning_effort="none",
            temperature=0,
        )
        parsed = completion.choices[0].message.parsed
        usage: dict[str, object] = {
            "input_tokens": completion.usage.prompt_tokens if completion.usage else 0,
            "output_tokens": completion.usage.completion_tokens
            if completion.usage
            else 0,
            "api": "chat",
            "model": model,
        }
    else:
        response = await client.responses.parse(
            model=model,
            instructions=instructions,
            input=input,
            text_format=schema,
            max_output_tokens=max_tokens,
            reasoning={"effort": "low"},
        )
        parsed = response.output_parsed
        usage = {
            "input_tokens": response.usage.input_tokens if response.usage else 0,
            "output_tokens": response.usage.output_tokens if response.usage else 0,
            "api": "responses",
            "model": model,
        }
    if parsed is None:
        msg = "Model returned no structured result."
        raise ValueError(msg)
    return parsed, usage
