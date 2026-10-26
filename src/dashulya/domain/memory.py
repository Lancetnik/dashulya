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
    username: str | None = None
    is_bot: bool | None = None


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


@dataclass(frozen=True)
class MemoryAccess:
    """Trusted conversation authority; model-provided filters can only narrow it."""

    chat_id: int
    is_private: bool
    before: datetime | None = None
    before_id: int | None = None

    def __post_init__(self) -> None:
        if self.before is not None and self.before.utcoffset() is None:
            error_message = "A fixed memory cutoff must include a timezone."
            raise ValueError(error_message)
        if self.before is None and self.before_id is not None:
            error_message = "A fixed message cutoff requires its original timestamp."
            raise ValueError(error_message)


class HistoryReader(Protocol):
    """Read canonical originals within hidden conversation authority."""

    async def query(
        self,
        action: str = "list",
        *,
        text_contains: str = "",
        case_sensitive: bool = False,
        author_id: int | None = None,
        after: str | None = None,
        before: str | None = None,
        chat: str | None = None,
        source_id: str | None = None,
        cursor: str | None = None,
        offset: int = 0,
        limit: int = 20,
        topic_id: int | None = None,
    ) -> dict[str, Any]: ...


class ChatMemoryService(Protocol):
    """Bind application-owned memory resources to trusted conversation authority."""

    def recall(self, access: MemoryAccess) -> MemoryBackend: ...

    def history(self, access: MemoryAccess) -> HistoryReader: ...
