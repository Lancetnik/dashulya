from typing import Any, Protocol


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
