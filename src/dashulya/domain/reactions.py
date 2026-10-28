from dataclasses import dataclass
from typing import Literal, Protocol


@dataclass(frozen=True)
class ReactionResult:
    status: Literal["sent", "rejected", "unknown", "rate_limited"]
    message_id: int
    emoji: str
    reason: str | None = None
    retry_after: int | None = None
    message: str | None = None


class MessageReactions(Protocol):
    async def set(self, chat_id: int, message_id: int, emoji: str) -> ReactionResult: ...
