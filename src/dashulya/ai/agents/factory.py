from collections.abc import Iterable
from zoneinfo import ZoneInfo

from ag2 import Agent, ResponseSchema
from ag2.config import ModelConfig
from ag2.middleware import Middleware
from ag2.middleware.base import MiddlewareFactory
from ag2.tools.tool import Tool

from dashulya.ai.agents.tracing import ReplyTrace
from dashulya.ai.prompts import (
    CHARACTER_PROMPT,
    GROUP_ADDRESSING_PROMPT,
    INPUT_FORMAT_PROMPT,
    MEMORY_TOOLS_PROMPT,
    MESSAGE_FORMAT_PROMPT,
    RESPONSE_RULES_PROMPT,
    TIMEZONE_PROMPT,
    WEB_SEARCH_PROMPT,
)
from dashulya.domain.messages import Reply


def create_decision_agent(
    config: ModelConfig,
    *,
    timezone: str = "Europe/Moscow",
    middleware: Iterable[MiddlewareFactory] = (),
) -> Agent[bool]:
    return Agent(
        "group_addressing",
        prompt=[
            GROUP_ADDRESSING_PROMPT,
            TIMEZONE_PROMPT.format(timezone=ZoneInfo(timezone).key),
            MESSAGE_FORMAT_PROMPT,
        ],
        response_schema=ResponseSchema(
            bool,
            description="Does the current message call for a reply from Dashulya?",
        ),
        config=config,
        middleware=middleware,
    )


def create_reply_agent(
    config: ModelConfig,
    *,
    tools: Iterable[Tool],
    timezone: str = "Europe/Moscow",
    middleware: Iterable[MiddlewareFactory] = (),
) -> Agent[Reply]:
    return Agent(
        "dashulya",
        prompt=[
            CHARACTER_PROMPT,
            RESPONSE_RULES_PROMPT,
            TIMEZONE_PROMPT.format(timezone=ZoneInfo(timezone).key),
            MESSAGE_FORMAT_PROMPT,
            INPUT_FORMAT_PROMPT,
            MEMORY_TOOLS_PROMPT,
            WEB_SEARCH_PROMPT,
        ],
        tools=tools,
        config=config,
        response_schema=ResponseSchema(
            Reply,
            description="Telegram reply text and optional target message ID.",
        ),
        middleware=[Middleware(ReplyTrace), *middleware],
    )
