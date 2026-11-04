from collections.abc import Iterable
from zoneinfo import ZoneInfo

from ag2 import Agent, ResponseSchema
from ag2.config import ModelConfig
from ag2.middleware import Middleware
from ag2.middleware.base import MiddlewareFactory
from ag2.tools.tool import Tool

from dashulya.ai.agents.tracing import ReplyTrace, log_builtin_tool
from dashulya.ai.prompts import (
    CHARACTER_PROMPT,
    GROUP_ADDRESSING_PROMPT,
    INPUT_FORMAT_PROMPT,
    INPUT_TRUST_PROMPT,
    MEMORY_TOOLS_PROMPT,
    MESSAGE_FORMAT_PROMPT,
    REPLY_WORKFLOW_PROMPT,
    RESPONSE_RULES_PROMPT,
    TIMEZONE_PROMPT,
    WEB_SEARCH_PROMPT,
)
from dashulya.domain.messages import ReplyBatch


def create_decision_agent(
    config: ModelConfig,
    *,
    timezone: str = "Europe/Moscow",
    middleware: Iterable[MiddlewareFactory] = (),
) -> Agent[bool]:
    return Agent(
        "group_addressing",
        prompt=[
            INPUT_TRUST_PROMPT,
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
) -> Agent[ReplyBatch]:
    return Agent(
        "dashulya",
        prompt=[
            CHARACTER_PROMPT,
            INPUT_TRUST_PROMPT,
            REPLY_WORKFLOW_PROMPT,
            TIMEZONE_PROMPT.format(timezone=ZoneInfo(timezone).key),
            MESSAGE_FORMAT_PROMPT,
            INPUT_FORMAT_PROMPT,
            MEMORY_TOOLS_PROMPT,
            WEB_SEARCH_PROMPT,
            RESPONSE_RULES_PROMPT,
        ],
        tools=tools,
        config=config,
        response_schema=ResponseSchema(
            ReplyBatch,
            description="Up to three ordered Telegram messages: text or a Dashulya sticker. For messages to the same person, set reply_to_message_id only on the first. Null targets the incoming request on the first output and sends without a reply on later outputs.",
        ),
        middleware=[Middleware(ReplyTrace), *middleware],
        observers=[log_builtin_tool],
    )
