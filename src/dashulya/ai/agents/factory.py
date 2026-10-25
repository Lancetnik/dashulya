from collections.abc import Iterable

from ag2 import Agent, ResponseSchema
from ag2.config import ModelConfig
from ag2.middleware.base import MiddlewareFactory
from ag2.tools.tool import Tool

from dashulya.ai.prompts import (
    CHARACTER_PROMPT,
    GROUP_ADDRESSING_PROMPT,
    INPUT_FORMAT_PROMPT,
    MEMORY_TOOLS_PROMPT,
    RESPONSE_RULES_PROMPT,
)


def create_decision_agent(
    config: ModelConfig, *, middleware: Iterable[MiddlewareFactory] = ()
) -> Agent[bool]:
    return Agent(
        "group_addressing",
        prompt=GROUP_ADDRESSING_PROMPT,
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
    middleware: Iterable[MiddlewareFactory] = (),
) -> Agent[str]:
    return Agent(
        "dashulya",
        prompt=[
            CHARACTER_PROMPT,
            RESPONSE_RULES_PROMPT,
            INPUT_FORMAT_PROMPT,
            MEMORY_TOOLS_PROMPT,
        ],
        tools=tools,
        config=config,
        middleware=middleware,
    )
