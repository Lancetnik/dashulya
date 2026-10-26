"""Content-addressed, persistent vectors with bounded resumable preparation."""

import asyncio
import hashlib
import json
import math
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path

import aiosqlite

type Embedder = Callable[[list[str]], Awaitable[list[list[float]]]]


class EmbeddingCache:
    def __init__(
        self,
        path: Path,
        embedder: Embedder,
        *,
        model: str = "text-embedding-3-small",
        dimensions: int = 512,
    ) -> None:
        self.path = path
        self.embedder = embedder
        self.model = model
        self.dimensions = dimensions
        self._ready = False
        self._initialize_lock = asyncio.Lock()

    async def initialize(self) -> None:
        async with self._initialize_lock:
            if self._ready:
                return
            self.path.parent.mkdir(parents=True, exist_ok=True)
            async with aiosqlite.connect(self.path) as connection:
                await connection.execute("PRAGMA journal_mode=WAL")
                await connection.execute(
                    "CREATE TABLE IF NOT EXISTS embeddings "
                    "(cache_key TEXT PRIMARY KEY, vector TEXT NOT NULL)"
                )
                await connection.commit()
            self._ready = True

    def _key(self, text: str) -> str:
        return hashlib.sha256(
            f"{self.model}:{self.dimensions}:{text[:12000]}".encode()
        ).hexdigest()

    def _validated(self, vector: list[float]) -> list[float]:
        if len(vector) != self.dimensions or any(
            type(value) not in {int, float} or not math.isfinite(value)
            for value in vector
        ):
            error_message = "Embedding provider returned an invalid vector."
            raise ValueError(error_message)
        norm = math.hypot(*vector)
        if not math.isfinite(norm):
            error_message = "Embedding provider returned an invalid vector magnitude."
            raise ValueError(error_message)
        return [value / norm for value in vector] if norm else list(vector)

    async def vectors(
        self,
        texts: Sequence[str],
        *,
        create_missing: bool = True,
        load_vectors: bool = True,
    ) -> dict[str, list[float]]:
        """Prepare durable vectors; load_vectors=False returns no vector payloads."""
        await self.initialize()
        keyed = {self._key(text): text[:12000] for text in texts}
        cached: dict[str, list[float]] = {}
        keys = list(keyed)
        async with aiosqlite.connect(self.path) as connection:
            await connection.execute("PRAGMA busy_timeout=5000")
            for start in range(0, len(keys), 200):
                selected = keys[start : start + 200]
                placeholders = ",".join("?" for _ in selected)
                columns = "cache_key, vector" if load_vectors else "cache_key, NULL"
                rows = await connection.execute_fetchall(
                    f"SELECT {columns} FROM embeddings WHERE cache_key IN ({placeholders})",  # ruff: ignore[hardcoded-sql-expression] -- Only placeholder count is interpolated.
                    selected,
                )
                for key, payload in rows:
                    cached[key] = (
                        self._validated(json.loads(payload)) if load_vectors else []
                    )
        if create_missing:
            missing = [key for key in keys if key not in cached]
            semaphore = asyncio.Semaphore(4)

            async def batch(batch_keys: list[str]) -> None:
                async with semaphore:
                    vectors = await self.embedder([keyed[key] for key in batch_keys])
                    if len(vectors) != len(batch_keys):
                        error_message = "Embedding provider returned an incomplete batch."
                        raise ValueError(error_message)
                    ready = {
                        key: self._validated(vector)
                        for key, vector in zip(batch_keys, vectors, strict=True)
                    }
                    async with aiosqlite.connect(self.path) as connection:
                        await connection.execute("PRAGMA busy_timeout=5000")
                        await connection.executemany(
                            "INSERT OR REPLACE INTO embeddings VALUES (?, ?)",
                            [(key, json.dumps(vector)) for key, vector in ready.items()],
                        )
                        await connection.commit()
                    cached.update(ready)

            # Do not create unbounded tasks; completed batches survive cancellation.
            for start in range(0, len(missing), 512):
                outcomes = await asyncio.gather(
                    *(
                        batch(missing[offset : offset + 128])
                        for offset in range(start, min(start + 512, len(missing)), 128)
                    ),
                    return_exceptions=True,
                )
                for outcome in outcomes:
                    if isinstance(outcome, BaseException):
                        raise outcome
        return (
            {text: cached[self._key(text)] for text in texts if self._key(text) in cached}
            if load_vectors
            else {}
        )
