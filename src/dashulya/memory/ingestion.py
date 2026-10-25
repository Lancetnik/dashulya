"""Retry persisted live-message batches independently of Telegram replies."""

import asyncio
import logging
from collections import defaultdict
from contextlib import suppress
from typing import Any

import openviking_sdk  # type: ignore[import-untyped]

from dashulya.memory.live import LiveMemory

logger = logging.getLogger(__name__)


class MemoryIngestion:
    def __init__(self, store: LiveMemory, *, url: str, api_key: str) -> None:
        self.store = store
        self.url = url
        self.api_key = api_key
        self._task: asyncio.Task[None] | None = None
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        if self._task is not None:
            return
        await self._ensure_root()
        self._task = asyncio.create_task(self._run(), name="live-memory-ingestion")

    async def close(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with suppress(asyncio.CancelledError):
                await self._task
            self._task = None

    async def _ensure_root(self) -> None:
        client = self._client()
        try:
            await client.initialize()
            for uri in (self.store.root.rpartition("/")[0], self.store.root):
                await self._mkdir(client, uri)
        finally:
            await client.close()

    def _client(self) -> Any:
        return openviking_sdk.AsyncHTTPClient(
            url=self.url,
            api_key=self.api_key,
            timeout=180,
            account="default",
            user="default",
        )

    @staticmethod
    async def _mkdir(client: Any, uri: str) -> None:
        try:
            await client.stat(uri)
        except openviking_sdk.OpenVikingError as error:
            if error.code != "NOT_FOUND":
                raise
        else:
            return
        try:
            await client.mkdir(uri)
        except openviking_sdk.OpenVikingError as error:
            if error.code != "ALREADY_EXISTS":
                raise

    async def _run(self) -> None:
        while True:
            try:
                await self.flush_once()
            except Exception as error:
                # Retry store/service outages; never log remote error bodies or source text.
                logger.warning(
                    "Live memory worker unavailable",
                    extra={"error_type": type(error).__name__},
                )
            await asyncio.sleep(2)

    async def flush_once(self) -> int:
        """Publish current revisions; failed or superseded revisions remain queued."""
        async with self._lock:
            rows = await self.store.pending()
            if not rows:
                return 0
            groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in rows:
                groups[self.store.block_uri(row["chat_id"], row["message_id"])].append(
                    row
                )
            published = 0
            client = self._client()
            try:
                await client.initialize()
                for root, batch in groups.items():
                    try:
                        await self._mkdir(client, root)
                        result = await client.batch_write(
                            root,
                            [
                                {
                                    "uri": self.store.uri(row),
                                    "content": self.store.document(row),
                                    "mode": "upsert",
                                }
                                for row in batch
                            ],
                            wait=True,
                            options={"processing_mode": "semantic_and_vectors"},
                        )
                        # A timed-out earlier attempt may have committed bytes but not indexing.
                        if (
                            result.get("semantic_status") == "skipped"
                            or result.get("vector_status") == "skipped"
                            or (
                                result.get("unchanged")
                                and not result.get("created")
                                and not result.get("updated")
                            )
                        ):
                            result = await client.reindex(
                                root, mode="semantic_and_vectors", force=True, wait=True
                            )
                        if not self._complete(result):
                            error_message = (
                                "OpenViking did not complete live-message indexing."
                            )
                            raise RuntimeError(error_message)
                        await self.store.indexed(batch)
                        published += len(batch)
                    except Exception as error:
                        await self.store.failed(batch)
                        logger.warning(
                            "Live memory batch deferred",
                            extra={
                                "chat_id": batch[0]["chat_id"],
                                "error_type": type(error).__name__,
                            },
                        )
            finally:
                await client.close()
            return published

    @classmethod
    def _complete(cls, result: dict[str, Any]) -> bool:
        return not cls._has_errors(result) and (
            all(
                result.get(name) in {"complete", "not_needed"}
                for name in ("semantic_status", "vector_status")
            )
            or (result.get("status") == "completed" and "failed_records" in result)
        )

    @classmethod
    def _has_errors(cls, value: Any) -> bool:
        if isinstance(value, dict):
            if any(
                value.get(name)
                for name in (
                    "error_count",
                    "failed_records",
                    "unsupported_records",
                    "errors",
                )
            ) or (
                isinstance(value.get("status"), str)
                and value["status"] in {"error", "failed"}
            ):
                return True
            return any(cls._has_errors(child) for child in value.values())
        if isinstance(value, list):
            return any(cls._has_errors(child) for child in value)
        return False
