"""Bounded recall and source reads over an authorized canonical snapshot."""

from datetime import UTC, datetime, timedelta
from typing import Any

from dashulya.memory._engines import EngineCache
from dashulya.memory.history import CanonicalHistory


def _history_before(value: str | None, after: str | None) -> str | None:
    """Translate recall's inclusive upper bound to canonical history's exclusive one."""
    if value is None:
        return None
    upper = datetime.fromisoformat(value)
    if upper.utcoffset() is None:
        error_message = "Search dates must include a timezone."
        raise ValueError(error_message)
    if after is not None:
        lower = datetime.fromisoformat(after)
        if lower.utcoffset() is None:
            error_message = "Search dates must include a timezone."
            raise ValueError(error_message)
        if lower > upper:
            error_message = "after must not be later than before."
            raise ValueError(error_message)
    try:
        upper = upper.astimezone(UTC)
    except OverflowError:
        error_message = "Search date is outside the supported UTC range."
        raise ValueError(error_message) from None
    if upper == datetime.max.replace(tzinfo=UTC):
        # Every representable canonical timestamp satisfies this upper bound.
        return None
    return (upper + timedelta(microseconds=1)).isoformat()


class ScopedRecall:
    def __init__(self, history: CanonicalHistory, engines: EngineCache) -> None:
        self.history = history
        self._engines = engines

    async def search(
        self,
        query: str = "",
        *,
        memory_id: str | None = None,
        chat: str | None = None,
        author_id: int | None = None,
        after: str | None = None,
        before: str | None = None,
        limit: int = 8,
    ) -> dict[str, Any]:
        if memory_id is not None:
            return await self.history.query(
                "read",
                source_id=memory_id,
                chat=chat,
                author_id=author_id,
                after=after,
                before=_history_before(before, after),
            )
        originals = await self.history.originals(chat=chat)
        if not query.strip():
            return {"hits": [], "coverage": self.history.coverage(originals)}
        async with self._engines.lease(originals) as engine:
            hits = await engine.search(
                query, author_id=author_id, after=after, before=before, limit=limit
            )
            coverage = await engine.coverage()
        output = []
        remaining = 48_000
        for hit in hits:
            sources = []
            for source in hit.sources:
                item = source.render(max_chars=min(12_000, max(0, remaining)))
                remaining -= len(item["text"])
                sources.append(item)
            context = self.history.related(hit.sources[-1], originals)
            output.append({
                "memory_id": hit.sources[-1].source_id,
                "kind": "message",
                "memory": hit.text[:6000],
                "score": hit.score,
                "sources": sources,
                "context": [item.render(max_chars=3000) for item in context],
            })
        return {
            "hits": output,
            "coverage": {
                **coverage,
                **self.history.coverage(originals),
                "complete_history": False,
                "notice": "Only observed and imported messages are available. Search is bounded; use chat_history for exact counts or original-message scans.",
            },
        }
