"""Shared input and output contract for memory experiments."""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class SourceMessage:
    id: int
    chat_key: str
    date: str
    text: str
    user_id: int | None
    author: str
    user_name: str
    reply_to_msg_id: int | None = None
    thread_post_id: int | None = None
    is_bot: bool = False


@dataclass(frozen=True)
class QueryContext:
    chat_key: str
    before: str
    before_id: int = 0
    author_id: int | None = None
    after: str | None = None


@dataclass(frozen=True)
class MemoryHit:
    text: str
    source_ids: tuple[int, ...]
    score: float = 0.0
    chat_key: str = ""


class MemoryVariant(Protocol):
    """Variants receive source messages and requests, never benchmark gold."""

    async def prepare(self, messages: Sequence[SourceMessage]) -> dict[str, object]: ...

    async def search(
        self, query: str, context: QueryContext, *, limit: int = 8
    ) -> list[MemoryHit]: ...

    async def close(self) -> None: ...


class VariantFactory(Protocol):
    def __call__(self, workspace: Path) -> MemoryVariant: ...
