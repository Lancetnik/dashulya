import hashlib
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol

from dashulya.domain.messages import ConversationKey


def message_source_id(collection: str, message_id: int) -> str:
    key = hashlib.sha256(collection.encode()).hexdigest()[:24]
    return f"source:{key}:{message_id}"


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


@dataclass(frozen=True)
class TruncatedThread:
    """A depth-cut boundary and the root established from saved reply links."""

    boundary_id: int
    root_id: int | None
    source_id: str | None = None


@dataclass(frozen=True)
class ReplyContext:
    """Bounded source messages and navigation metadata for omitted ancestors."""

    messages: tuple[MemoryMessage, ...] = ()
    truncated_threads: tuple[TruncatedThread, ...] = ()


@dataclass(frozen=True)
class OriginalMessage:
    collection: str
    chat_id: int | None
    message_id: int
    text: str
    sent_at: datetime | None
    author_id: int | None
    author: str
    chat_title: str
    chat_username: str | None
    topic_id: int | None
    reply_to_id: int | None
    edited_at: str | None
    media_type: str | None
    url: str | None
    provenance: str
    archive_key: str | None = None
    username: str | None = None
    is_bot: bool | None = None

    @property
    def source_id(self) -> str:
        return message_source_id(self.collection, self.message_id)

    @property
    def order(self) -> tuple[str, str, int]:
        instant = self.sent_at.isoformat(timespec="microseconds") if self.sent_at else ""
        return instant, self.collection, self.message_id

    def render(self, *, offset: int = 0, max_chars: int = 1200) -> dict[str, Any]:
        end = min(len(self.text), offset + max_chars)
        return {
            "source_id": self.source_id,
            "message_id": self.message_id,
            "chat": self.collection,
            "chat_id": self.chat_id,
            "chat_title": self.chat_title,
            "chat_username": self.chat_username,
            "topic_id": self.topic_id,
            "text": self.text[offset:end],
            "sent_at": self.sent_at.isoformat() if self.sent_at else None,
            "author_id": self.author_id,
            "author": self.author,
            "reply_to_message_id": self.reply_to_id,
            "edited_at": self.edited_at,
            "media_type": self.media_type,
            "is_bot": self.is_bot,
            "username": self.username,
            "url": self.url,
            "provenance": self.provenance,
            "archive_key": self.archive_key,
            "offset": offset,
            "text_length": len(self.text),
            "truncated": offset != 0 or end < len(self.text),
            "next_offset": end if end < len(self.text) else None,
        }


class MessageRecorder(Protocol):
    async def record(
        self, message: MemoryMessage, *, if_missing: bool = False
    ) -> None: ...


class ChatHistory(Protocol):
    """Immediate context: no topic selects a whole chat; a topic narrows it."""

    async def context(
        self,
        key: ConversationKey,
        through_message_id: int,
        *,
        limit: int,
        parent_hops: int,
    ) -> ReplyContext: ...


class MemoryBackend(Protocol):
    """The single asynchronous read interface exposed to the assistant."""

    async def search(
        self,
        query: str = "",
        *,
        memory_id: str | None = None,
        chat: str | None = None,
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
        name_contains: str | None = None,
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
