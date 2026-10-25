from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol

from dashulya.domain.messages import ConversationKey


@dataclass(frozen=True)
class MemoryMessage:
    chat_id: int
    message_id: int
    text: str
    sent_at: datetime
    author_id: int | None
    author: str
    chat_title: str
    chat_username: str | None = None
    thread_id: int | None = None
    reply_to_id: int | None = None
    edited_at: datetime | None = None
    media_type: str | None = None
    update_id: int = 0


class MessageRecorder(Protocol):
    async def record(
        self, message: MemoryMessage, *, if_missing: bool = False
    ) -> None: ...


class ChatHistory(Protocol):
    async def context(
        self,
        key: ConversationKey,
        through_message_id: int,
        *,
        limit: int,
        parent_hops: int,
    ) -> list[MemoryMessage]: ...


class MemoryBackend(Protocol):
    """The single asynchronous read interface exposed to the assistant."""

    async def search(
        self,
        query: str = "",
        *,
        memory_id: str | None = None,
        author_id: int | None = None,
        after: str | None = None,
        before: str | None = None,
        limit: int = 3,
    ) -> dict[str, Any]: ...
