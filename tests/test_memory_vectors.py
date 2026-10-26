import math
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import AsyncMock

from dashulya.memory.vectors import EmbeddingCache


class EmbeddingCacheTests(unittest.IsolatedAsyncioTestCase):
    async def test_restart_reuses_cache_and_model_changes_do_not(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "vectors.sqlite3"
            embed = AsyncMock(return_value=[[3.0, 4.0]])
            cache = EmbeddingCache(path, embed, dimensions=2)
            self.assertEqual(await cache.vectors(["Original"]), {"Original": [0.6, 0.8]})
            restarted = EmbeddingCache(path, embed, dimensions=2)
            self.assertEqual(
                await restarted.vectors(["Original"]), {"Original": [0.6, 0.8]}
            )
            self.assertEqual(embed.await_count, 1)
            changed = EmbeddingCache(path, embed, model="different", dimensions=2)
            await changed.vectors(["Original"])
            self.assertEqual(embed.await_count, 2)

    async def test_lookup_does_not_embed_missing_or_expose_stale_text(self) -> None:
        with TemporaryDirectory() as directory:
            embed = AsyncMock(return_value=[[1.0, 0.0]])
            cache = EmbeddingCache(
                Path(directory) / "vectors.sqlite3", embed, dimensions=2
            )
            await cache.vectors(["Mistaken fact"])
            found = await cache.vectors(["Corrected fact"], create_missing=False)
            self.assertEqual(found, {})
            self.assertEqual(embed.await_count, 1)

    async def test_invalid_or_incomplete_provider_results_are_not_cached(self) -> None:
        with TemporaryDirectory() as directory:
            embed = AsyncMock()
            cache = EmbeddingCache(
                Path(directory) / "vectors.sqlite3", embed, dimensions=2
            )
            for result in ([], [[1.0]], [[math.nan, 0.0]], [[math.inf, 0.0]]):
                embed.return_value = result
                with self.assertRaises(ValueError):
                    await cache.vectors(["Source"])
            self.assertEqual(await cache.vectors(["Source"], create_missing=False), {})

    async def test_bounded_text_is_the_actual_provider_input_and_cache_identity(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            embed = AsyncMock(return_value=[[1.0, 0.0]])
            cache = EmbeddingCache(
                Path(directory) / "vectors.sqlite3", embed, dimensions=2
            )
            original = "a" * 12001
            await cache.vectors([original])
            embed.assert_awaited_once_with([original[:12000]])
            await cache.vectors(["a" * 12000 + "b"])
            self.assertEqual(embed.await_count, 1)

    async def test_preparation_can_reuse_keys_without_loading_all_vector_payloads(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            embed = AsyncMock(return_value=[[1.0, 0.0]])
            cache = EmbeddingCache(
                Path(directory) / "vectors.sqlite3", embed, dimensions=2
            )
            self.assertEqual(await cache.vectors(["Source"], load_vectors=False), {})
            self.assertEqual(await cache.vectors(["Source"], load_vectors=False), {})
            self.assertEqual(await cache.vectors(["Source"]), {"Source": [1.0, 0.0]})
            self.assertEqual(embed.await_count, 1)
