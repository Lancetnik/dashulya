"""Application-owned hybrid recall and independently available canonical history."""

import asyncio
import hashlib
import json
import logging
from collections import OrderedDict
from collections.abc import AsyncGenerator, Sequence
from contextlib import asynccontextmanager, suppress
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from dashulya.domain.memory import MemoryAccess, MemoryBackend, OriginalMessage
from dashulya.memory._workers import run_in_thread
from dashulya.memory.history import CanonicalHistory
from dashulya.memory.hybrid import HybridMemory, embedding_texts
from dashulya.memory.live import LiveMemory
from dashulya.memory.search import MemorySearch
from dashulya.memory.vectors import EmbeddingCache

logger = logging.getLogger(__name__)


@dataclass
class _CachedEngine:
    engine: HybridMemory
    users: int = 0
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


def _fingerprint(originals: Sequence[OriginalMessage]) -> str:
    return hashlib.sha256(
        json.dumps(
            [asdict(item) for item in originals], default=str, sort_keys=True
        ).encode()
    ).hexdigest()


class HybridChatMemory:
    def __init__(
        self,
        live: LiveMemory,
        vectors: EmbeddingCache,
        *,
        owner_user_id: int,
        imported: MemorySearch | None = None,
        import_chat_id: int | None = None,
    ) -> None:
        self.live = live
        self.vectors = vectors
        self.owner_user_id = owner_user_id
        self.imported = imported
        self.import_chat_id = import_chat_id
        self._worker: asyncio.Task[None] | None = None
        self._engines: OrderedDict[str, _CachedEngine] = OrderedDict()
        self._engine_lock = asyncio.Lock()
        self._index_generation = 0
        self._closed = False
        self._needs_refresh = True
        self._idle = asyncio.Event()
        self._idle.set()

    def history(self, access: MemoryAccess) -> CanonicalHistory:
        return CanonicalHistory(
            self.live,
            access,
            imported=self.imported,
            import_chat_id=self.import_chat_id,
        )

    def recall(self, access: MemoryAccess) -> MemoryBackend:
        return _Recall(self.history(access), self)

    async def _evict(self) -> None:
        for key in list(self._engines):
            if len(self._engines) <= 2:
                break
            entry = self._engines[key]
            if entry.users == 0:
                del self._engines[key]
                await asyncio.to_thread(entry.engine.close)

    @asynccontextmanager
    async def engine(
        self, originals: Sequence[OriginalMessage]
    ) -> AsyncGenerator[HybridMemory, None]:
        fingerprint = await asyncio.to_thread(_fingerprint, originals)
        key = f"{self._index_generation}:{fingerprint}"
        async with self._engine_lock:
            if self._closed:
                error_message = "Memory retrieval is closed."
                raise RuntimeError(error_message)
            entry = self._engines.get(key)
            if entry is None:
                texts = await asyncio.to_thread(embedding_texts, originals)
                vectors = await self.vectors.vectors(texts, create_missing=False)

                async def embed_query(text: str) -> list[float]:
                    return (await self.vectors.vectors([text]))[text]

                engine = await run_in_thread(
                    lambda: HybridMemory(originals, vectors, embed_query),
                    on_cancel=lambda result: result.close(),
                )
                entry = _CachedEngine(engine)
                self._engines[key] = entry
            self._engines.move_to_end(key)
            entry.users += 1
            self._idle.clear()
        try:
            async with self._engine_lock:
                await self._evict()
            async with entry.lock:
                yield entry.engine
        finally:
            async with self._engine_lock:
                entry.users -= 1
                if not any(item.users for item in self._engines.values()):
                    self._idle.set()
                await self._evict()

    async def refresh_once(self) -> None:
        pending = await self.live.pending(limit=None)
        originals = await self.history(
            MemoryAccess(self.owner_user_id, is_private=True)
        ).originals()
        texts = await asyncio.to_thread(embedding_texts, originals)
        await self.vectors.vectors(texts, load_vectors=False)
        async with self._engine_lock:
            self._index_generation += 1
            # Completed vectors can make a formerly pending block eligible for dense recall.
            for key, entry in list(self._engines.items()):
                if entry.users == 0:
                    del self._engines[key]
                    await asyncio.to_thread(entry.engine.close)
        # A newer edit cannot be acknowledged by an older embedding batch.
        await self.live.indexed(pending)
        self._needs_refresh = False

    async def start(self) -> None:
        await self.vectors.initialize()
        try:
            await self.refresh_once()
        except Exception as error:
            logger.warning("Memory indexing pending (%s)", type(error).__name__)
        self._worker = asyncio.create_task(self._run(), name="memory-embeddings")

    async def _run(self) -> None:
        while True:
            await asyncio.sleep(2)
            try:
                if self._needs_refresh or await self.live.pending(limit=1):
                    await self.refresh_once()
            except Exception as error:
                # Provider text may contain source data or credentials.
                logger.warning("Memory indexing failed (%s)", type(error).__name__)
                await asyncio.sleep(10)

    async def close(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            with suppress(asyncio.CancelledError):
                await self._worker
            self._worker = None
        async with self._engine_lock:
            self._closed = True
        await self._idle.wait()
        async with self._engine_lock:
            for entry in self._engines.values():
                await asyncio.to_thread(entry.engine.close)
            self._engines.clear()


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


class _Recall:
    def __init__(self, history: CanonicalHistory, service: HybridChatMemory) -> None:
        self.history = history
        self.service = service

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
        async with self.service.engine(originals) as engine:
            hits = await engine.search(
                query, author_id=author_id, after=after, before=before, limit=limit
            )
            coverage = dict(engine.coverage)
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
