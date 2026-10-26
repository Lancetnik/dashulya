import math
from unittest.mock import AsyncMock

from dashulya.memory.vectors import EmbeddingCache
from tests.postgres import PostgresTestCase


def vector(x: float, y: float = 0) -> list[float]:
    return [x, y] + [0.0] * 510


class EmbeddingCacheTests(PostgresTestCase):
    async def test_restart_reuses_cache_and_model_changes_do_not(self) -> None:
        embed = AsyncMock(return_value=[vector(3, 4)])
        cache = EmbeddingCache(self.database, embed)
        result = (await cache.vectors(["Original"]))["Original"]
        self.assertAlmostEqual(result[0], 0.6, places=6)
        self.assertAlmostEqual(result[1], 0.8, places=6)
        await EmbeddingCache(self.database, embed).vectors(["Original"])
        self.assertEqual(embed.await_count, 1)
        await EmbeddingCache(self.database, embed, model="different").vectors([
            "Original"
        ])
        self.assertEqual(embed.await_count, 2)

    async def test_lookup_does_not_embed_missing_or_expose_stale_text(self) -> None:
        embed = AsyncMock(return_value=[vector(1)])
        cache = EmbeddingCache(self.database, embed)
        await cache.vectors(["Mistaken fact"])
        self.assertEqual(
            await cache.vectors(["Corrected fact"], create_missing=False), {}
        )
        self.assertEqual(embed.await_count, 1)

    async def test_invalid_or_incomplete_provider_results_are_not_cached(self) -> None:
        embed = AsyncMock()
        cache = EmbeddingCache(self.database, embed)
        for result in ([], [[1.0]], [vector(math.nan)], [vector(math.inf)]):
            embed.return_value = result
            with self.assertRaises(ValueError):
                await cache.vectors(["Source"])
        self.assertEqual(await cache.vectors(["Source"], create_missing=False), {})

    async def test_bounded_input_and_cache_identity(self) -> None:
        embed = AsyncMock(return_value=[vector(1)])
        cache = EmbeddingCache(self.database, embed)
        await cache.vectors(["a" * 12001])
        embed.assert_awaited_once_with(["a" * 12000])
        await cache.vectors(["a" * 12000 + "b"])
        self.assertEqual(embed.await_count, 1)
        self.assertEqual(await cache.vectors(["a" * 12001], load_vectors=False), {})

    async def test_native_search_limits_to_allowed_keys_and_preserves_zero_vector_ties(
        self,
    ) -> None:
        cache = EmbeddingCache(self.database, AsyncMock())
        await cache.put({
            "best": vector(1),
            "opposite": vector(-1),
            "orthogonal": vector(0, 1),
            "zero": vector(0),
        })
        self.assertEqual(
            await cache.nearest(
                ["opposite", "orthogonal", "zero", "best"], vector(1), limit=4
            ),
            ["best", "orthogonal", "zero", "opposite"],
        )
        self.assertEqual(
            await cache.nearest(["opposite", "zero"], vector(1), limit=1), ["zero"]
        )
