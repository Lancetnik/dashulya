from dataclasses import dataclass

from ag2.config import ModelConfig, OpenAIDecisionsConfig, OpenAIResponsesConfig


@dataclass(frozen=True, repr=False)
class AgentModels:
    reply: ModelConfig
    decision: ModelConfig


def create_models(*, api_key: str, reply_model: str, decision_model: str) -> AgentModels:
    return AgentModels(
        reply=OpenAIResponsesConfig(
            model=reply_model,
            api_key=api_key,
            reasoning={"effort": "medium"},
            timeout=60,
            max_retries=1,
        ),
        decision=OpenAIDecisionsConfig(
            model=decision_model,
            api_key=api_key,
            timeout=10,
            max_retries=0,
        ),
    )
