from ag2 import Agent
from ag2.config import OpenAIResponsesConfig

from dashulya.ai.prompts import (
    CHARACTER_PROMPT,
    INPUT_FORMAT_PROMPT,
    MEMORY_TOOLS_PROMPT,
    NO_TOOLS_PROMPT,
    RESPONSE_RULES_PROMPT,
)
from dashulya.ai.tools.memory import create_memory_tool
from dashulya.memory.search import MemorySearch


def create_agent(
    *,
    model: str,
    api_key: str,
    memory: MemorySearch | None = None,
) -> Agent:
    return Agent(
        "dashulya",
        prompt=[
            CHARACTER_PROMPT,
            RESPONSE_RULES_PROMPT,
            INPUT_FORMAT_PROMPT,
            MEMORY_TOOLS_PROMPT if memory is not None else NO_TOOLS_PROMPT,
        ],
        tools=[create_memory_tool(memory)] if memory is not None else [],
        config=OpenAIResponsesConfig(
            model=model,
            api_key=api_key,
            reasoning={"effort": "low"},
            timeout=60,
            max_retries=1,
        ),
    )
