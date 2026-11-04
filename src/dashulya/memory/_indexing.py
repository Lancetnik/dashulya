"""Durable indexing revisions and the application-owned retry worker."""

import asyncio
import logging
from asyncio import sleep as _sleep
from contextlib import suppress

from dashulya.memory.search_index import SearchIndex
from dashulya.memory.store import MessageStore
from dashulya.memory.vectors import EmbeddingCache

logger = logging.getLogger(__name__)


class IndexingWorker:
    def __init__(
        self,
        store: MessageStore,
        vectors: EmbeddingCache,
        index: SearchIndex,
        *,
        owner_user_id: int,
    ) -> None:
        self.store = store
        self.vectors = vectors
        self.index = index
        self._worker: asyncio.Task[None] | None = None
        self._needs_refresh = True

    async def refresh_once(self) -> None:
        pending = await self.store.pending(limit=None)
        while await self.index.prepare():
            pass
        await self.index.finish()
        await self.index.embed_pending()
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
            await _sleep(2)
            try:
                if (
                    self._needs_refresh
                    or await self.store.pending(limit=1)
                    or await self.index.has_pending()
                ):
                    await self.refresh_once()
            except Exception as error:
                # Provider text may contain source data or credentials.
                logger.warning("Memory indexing failed (%s)", type(error).__name__)
                await _sleep(10)

    async def close(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            with suppress(asyncio.CancelledError):
                await self._worker
            self._worker = None
