from unittest.mock import AsyncMock

from ag2 import Agent
from ag2.config import ModelConfig
from aiogram import Bot
from dishka import Provider, provide
from dishka_ag2 import AG2Scope

from dashulya.ai.agents.models import AgentModels
from dashulya.ai.dialogue import Dialogue
from dashulya.domain.memory import MemoryBackend

TEST_API_KEY = "test-key"  # pragma: allowlist secret -- Fake test credential.


class BotOverride(Provider):
    scope = AG2Scope.APP

    def __init__(self, bot: Bot) -> None:
        super().__init__()
        self._bot = bot

    @provide(override=True)
    def bot(self) -> Bot:
        return self._bot


class DialogueOverride(Provider):
    scope = AG2Scope.APP

    def __init__(self, dialogue: Dialogue) -> None:
        super().__init__()
        self._dialogue = dialogue

    @provide(override=True)
    def dialogue(self) -> Dialogue:
        return self._dialogue


class DecisionOverride(Provider):
    scope = AG2Scope.APP

    def __init__(self, agent: Agent[bool]) -> None:
        super().__init__()
        self._agent = agent

    @provide(override=True)
    def agent(self) -> Agent[bool]:
        return self._agent


class MemoryOverride(Provider):
    scope = AG2Scope.APP

    def __init__(self, memory: MemoryBackend | None = None) -> None:
        super().__init__()
        if memory is None:
            memory = AsyncMock(spec=MemoryBackend)
            memory.search.return_value = {"coverage": {}, "hits": []}
        self._memory = memory

    @provide(override=True)
    def memory(self) -> MemoryBackend:
        return self._memory


class ModelOverrides(Provider):
    scope = AG2Scope.APP

    def __init__(self, *, reply: ModelConfig, decision: ModelConfig) -> None:
        super().__init__()
        self._reply = reply
        self._decision = decision

    @provide(override=True)
    def models(self) -> AgentModels:
        return AgentModels(reply=self._reply, decision=self._decision)
