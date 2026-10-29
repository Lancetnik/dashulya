"""Durable indexing revisions and the application-owned retry worker."""

import asyncio
import logging
from collections.abc import Awaitable, Callable
from contextlib import suppress

from dashulya.domain.memory import MemoryAccess
from dashulya.memory.history import CanonicalHistory
from dashulya.memory.hybrid import embedding_texts, lexical_texts
from dashulya.memory.lexical import LexicalIndex
from dashulya.memory.store import MessageStore
from dashulya.memory.vectors import EmbeddingCache

logger = logging.getLogger(__name__)


class IndexingWorker:
    def __init__(
        self,
        store: MessageStore,
        vectors: EmbeddingCache,
        lexical: LexicalIndex,
        *,
        owner_user_id: int,
        on_prepared: Callable[[], Awaitable[None]],
    ) -> None:
        self.store = store
        self.vectors = vectors
        self.lexical = lexical
        self._history = CanonicalHistory(
            store, MemoryAccess(owner_user_id, is_private=True)
        )
        self._on_prepared = on_prepared
        self._worker: asyncio.Task[None] | None = None
        self._needs_refresh = True

    async def refresh_once(self) -> None:
        pending = await self.store.pending(limit=None)
        originals = await self._history.originals()
        texts = await asyncio.to_thread(embedding_texts, originals)
        await self.lexical.prepare(await asyncio.to_thread(lexical_texts, originals))
        await self.vectors.vectors(texts, load_vectors=False)
        await self._on_prepared()
        # A newer edit cannot be acknowledged by an older embedding batch.
        await self.store.indexed(pending)
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
                if self._needs_refresh or await self.store.pending(limit=1):
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
