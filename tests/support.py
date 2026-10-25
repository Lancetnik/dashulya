import json
from unittest.mock import AsyncMock

from ag2 import Agent
from ag2.config import ModelConfig
from aiogram import Bot
from dishka import Provider, provide
from dishka_ag2 import AG2Scope

from dashulya.ai.agents.models import AgentModels
from dashulya.ai.dialogue import Dialogue
from dashulya.domain.memory import ChatHistory, MemoryBackend, MessageRecorder


def reply_json(text: str, reply_to_message_id: int | None = None) -> str:
    return json.dumps({"text": text, "reply_to_message_id": reply_to_message_id})


class RecorderOverride(Provider):
    scope = AG2Scope.APP

    def __init__(self, recorder: MessageRecorder | None = None) -> None:
        super().__init__()
        self._recorder = (
            recorder if recorder is not None else AsyncMock(spec=MessageRecorder)
        )

    @provide(override=True)
    def recorder(self) -> MessageRecorder:
        return self._recorder


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

    def __init__(
        self, memory: MemoryBackend | None = None, *, history: ChatHistory | None = None
    ) -> None:
        super().__init__()
        if memory is None:
            memory = AsyncMock(spec=MemoryBackend)
            memory.search.return_value = {"coverage": {}, "hits": []}
        self._memory = memory
        if history is None:
            history = AsyncMock(spec=ChatHistory)
            history.context.return_value = []
        self._history = history

    @provide(override=True)
    def memory(self) -> MemoryBackend:
        return self._memory

    @provide(override=True)
    def history(self) -> ChatHistory:
        return self._history


class ModelOverrides(Provider):
    scope = AG2Scope.APP

    def __init__(self, *, reply: ModelConfig, decision: ModelConfig) -> None:
        super().__init__()
        self._reply = reply
        self._decision = decision

    @provide(override=True)
    def models(self) -> AgentModels:
        return AgentModels(reply=self._reply, decision=self._decision)
