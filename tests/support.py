import json
import re
from typing import Any
from unittest.mock import AsyncMock

from ag2 import Agent
from ag2.config import ModelConfig
from aiogram import Bot
from dishka import Provider, provide
from dishka_ag2 import AG2Scope

from dashulya.ai.agents.models import AgentModels
from dashulya.ai.dialogue import Dialogue
from dashulya.domain.memory import (
    ChatHistory,
    ChatMemoryService,
    HistoryReader,
    MemoryAccess,
    MemoryBackend,
    MessageRecorder,
    ReplyContext,
)


def reply_json(text: str, reply_to_message_id: int | None = None) -> str:
    return json.dumps({
        "text": text,
        "reply_to_message_id": reply_to_message_id,
    })


def group_records(text: str) -> list[dict[str, Any]]:
    """Read timestamp/JSON headers and source bodies from formatted evidence."""
    records = []
    pattern = (
        r'^(\d{2}:\d{2}:\d{2}(?: \(UTC[+-]\d{2}:\d{2}\))?|Unknown time) (\{"id":.*\})\n'
    )
    for block in text.split("\n===\n"):
        matches = list(re.finditer(pattern, block, re.MULTILINE))
        for index, match in enumerate(matches):
            end = matches[index + 1].start() if index + 1 < len(matches) else len(block)
            records.append({
                **json.loads(match[2]),
                "time": None if match[1] == "Unknown time" else match[1],
                "text": block[match.end() : end],
            })
    return records


def group_rows(text: str) -> list[list[Any]]:
    """Normalize formatted records for existing context-selection assertions."""
    rows = []
    for record in group_records(text):
        row = [record["id"], record["user_id"], record["time"], record["text"]]
        if "reply_to" in record or "attachment" in record:
            row.append(record.get("reply_to"))
        if "attachment" in record:
            row.append(record["attachment"])
        rows.append(row)
    return rows


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


class _MemoryService:
    def __init__(self, memory: MemoryBackend, reader: HistoryReader) -> None:
        self.memory = memory
        self.reader = reader

    def recall(self, access: MemoryAccess) -> MemoryBackend:
        return self.memory

    def history(self, access: MemoryAccess) -> HistoryReader:
        return self.reader


class MemoryOverride(Provider):
    scope = AG2Scope.APP

    def __init__(
        self,
        memory: MemoryBackend | None = None,
        *,
        history: ChatHistory | None = None,
        reader: HistoryReader | None = None,
    ) -> None:
        super().__init__()
        if memory is None:
            memory = AsyncMock(spec=MemoryBackend)
            memory.search.return_value = {"coverage": {}, "hits": []}
        self._memory = memory
        if history is None:
            history = AsyncMock(spec=ChatHistory)
            history.context.return_value = ReplyContext()
        self._history = history
        if reader is None:
            reader = AsyncMock(spec=HistoryReader)
            reader.query.return_value = {"coverage": {}, "messages": []}
        self._service = _MemoryService(self._memory, reader)

    @provide(override=True)
    def memory(self) -> ChatMemoryService:
        return self._service

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
