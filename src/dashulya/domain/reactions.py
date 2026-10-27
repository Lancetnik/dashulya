from dataclasses import dataclass
from typing import Literal, Protocol


@dataclass(frozen=True)
class ReactionResult:
    status: Literal["sent", "rejected", "unknown"]
    message_id: int
    emoji: str
    reason: str | None = None


class MessageReactions(Protocol):
    async def set(self, chat_id: int, message_id: int, emoji: str) -> ReactionResult: ...
