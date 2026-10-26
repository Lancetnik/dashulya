"""Persistent normalized pgvector values and native exact similarity search."""

import asyncio
import hashlib
import json
import math
from collections.abc import Awaitable, Callable, Sequence

from dashulya.database import Database

type Embedder = Callable[[list[str]], Awaitable[list[list[float]]]]


class EmbeddingCache:
    def __init__(
        self,
        database: Database,
        embedder: Embedder,
        *,
        model: str = "text-embedding-3-small",
        dimensions: int = 512,
    ) -> None:
        if dimensions != 512:
            error_message = "The PostgreSQL embedding column requires 512 dimensions."
            raise ValueError(error_message)
        self.database = database
        self.embedder = embedder
        self.model = model
        self.dimensions = dimensions

    async def initialize(self) -> None:
        await self.database.open()

    def key(self, text: str) -> str:
        return hashlib.sha256(
            f"{self.model}:{self.dimensions}:{text[:12000]}".encode()
        ).hexdigest()

    def _validated(self, vector: list[float]) -> list[float]:
        if len(vector) != self.dimensions or any(
            type(x) not in {int, float} or not math.isfinite(x) for x in vector
        ):
            error_message = "Embedding provider returned an invalid vector."
            raise ValueError(error_message)
        norm = math.hypot(*vector)
        if not math.isfinite(norm):
            error_message = "Embedding provider returned an invalid vector magnitude."
            raise ValueError(error_message)
        return [x / norm for x in vector] if norm else list(vector)

    async def saved_keys(self, keys: Sequence[str]) -> set[str]:
        if not keys:
            return set()
        async with self.database.connection() as connection:
            cursor = await connection.execute(
                "SELECT cache_key FROM embeddings WHERE cache_key=ANY(%s)", (list(keys),)
            )
            return {row["cache_key"] for row in await cursor.fetchall()}

    async def put(self, values: dict[str, list[float]]) -> None:
        async with (
            self.database.connection() as connection,
            connection.cursor() as cursor,
        ):
            await cursor.executemany(
                "INSERT INTO embeddings(cache_key,embedding) VALUES (%s,%s::vector) ON CONFLICT(cache_key) DO NOTHING",
                [
                    (key, json.dumps(self._validated(vector)))
                    for key, vector in values.items()
                ],
            )

    async def vectors(
        self,
        texts: Sequence[str],
        *,
        create_missing: bool = True,
        load_vectors: bool = True,
    ) -> dict[str, list[float]]:
        keyed = {self.key(text): text[:12000] for text in texts}
        saved = await self.saved_keys(list(keyed))
        if create_missing:
            missing = [key for key in keyed if key not in saved]
            semaphore = asyncio.Semaphore(4)

            async def batch(keys: list[str]) -> None:
                async with semaphore:
                    vectors = await self.embedder([keyed[k] for k in keys])
                    if len(vectors) != len(keys):
                        error_message = "Embedding provider returned an incomplete batch."
                        raise ValueError(error_message)
                    await self.put(dict(zip(keys, vectors, strict=True)))

            for start in range(0, len(missing), 512):
                results = await asyncio.gather(
                    *(
                        batch(missing[offset : offset + 128])
                        for offset in range(start, min(start + 512, len(missing)), 128)
                    ),
                    return_exceptions=True,
                )
                for result in results:
                    if isinstance(result, BaseException):
                        raise result
        if not load_vectors or not texts:
            return {}
        async with self.database.connection() as connection:
            cursor = await connection.execute(
                "SELECT cache_key,embedding::text AS payload FROM embeddings WHERE cache_key=ANY(%s)",
                (list(keyed),),
            )
            values = {
                row["cache_key"]: self._validated(json.loads(row["payload"]))
                for row in await cursor.fetchall()
            }
        return {
            text: values[self.key(text)] for text in texts if self.key(text) in values
        }

    async def nearest(
        self, keys: Sequence[str], query: list[float], *, limit: int
    ) -> list[str]:
        if not keys:
            return []
        # Unit vectors make native inner product equivalent to cosine ranking,
        # preserving the previous dot=0 behavior for synthetic zero vectors.
        vector = json.dumps(self._validated(query))
        async with self.database.connection() as connection:
            cursor = await connection.execute(
                """SELECT e.cache_key FROM embeddings e
                JOIN unnest(%s::text[]) WITH ORDINALITY AS allowed(cache_key,position)
                    ON e.cache_key=allowed.cache_key
                ORDER BY e.embedding <#> %s::vector,allowed.position LIMIT %s""",
                (list(keys), vector, limit),
            )
            return [row["cache_key"] for row in await cursor.fetchall()]
