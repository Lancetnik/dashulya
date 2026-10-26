import hashlib
import json
import sqlite3
import unittest
from collections.abc import Sequence
from contextlib import closing
from dataclasses import asdict, replace
from itertools import starmap
from pathlib import Path
from tempfile import TemporaryDirectory

from evals.suites.memory.combinations import (
    CombinationVariant,
    DenseClaimsVariant,
    RetrievalComponent,
    build_combination,
)
from evals.suites.memory.raw_hybrid import RawHybridVariant
from evals.suites.memory.types import MemoryHit, QueryContext, SourceMessage


class FixedMemory:
    def __init__(self, hits: list[MemoryHit]) -> None:
        self.hits = hits
        self.preparations = 0
        self.search_limits: list[int] = []
        self.closed = 0

    async def prepare(self, messages: Sequence[SourceMessage]) -> dict[str, object]:
        self.preparations += 1
        return {"reused_index": True, "source_count": len(messages)}

    async def search(
        self, _query: str, _context: QueryContext, *, limit: int = 8
    ) -> list[MemoryHit]:
        self.search_limits.append(limit)
        return self.hits[:limit]

    async def close(self) -> None:
        self.closed += 1


class PoolMemory(FixedMemory):
    async def candidate_pool(
        self, _query: str, _context: QueryContext, *, limit: int = 100
    ) -> list[MemoryHit]:
        return self.hits[:limit]


class CombinationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.context = QueryContext("fixture_chat", "2026-02-01T00:00:00+00:00")
        self.messages = [
            SourceMessage(
                i,
                "fixture_chat",
                "2026-01-01T10:00:00+00:00",
                f"Original report {i}",
                10 if i % 2 else 20,
                "alice" if i % 2 else "bob",
                "Alice" if i % 2 else "Bob",
            )
            for i in range(1, 151)
        ]

    def hit(self, mid: int, text: str = "source-backed record") -> MemoryHit:
        return MemoryHit(text, (mid,), chat_key="fixture_chat")

    def variant(self, **components: FixedMemory) -> CombinationVariant:
        return CombinationVariant(
            self.root,
            components=list(starmap(RetrievalComponent, components.items())),
        )

    async def test_rrf_merges_only_exact_source_sets_and_is_order_independent(
        self,
    ) -> None:
        raw = FixedMemory([self.hit(1, "raw one"), self.hit(2, "raw two")])
        claims = FixedMemory([self.hit(2, "qualified two"), self.hit(3, "claim three")])
        first = self.variant(raw=raw, claims=claims)
        second = self.variant(claims=claims, raw=raw)
        await first.prepare(self.messages)
        await second.prepare(self.messages)
        hits = await first.candidate_pool("storage", self.context)
        self.assertEqual(hits, await second.candidate_pool("storage", self.context))
        self.assertEqual(hits[0].source_ids, (2,))
        self.assertEqual(hits[0].score, 1 / 61 + 1 / 62)
        self.assertIn("raw two", hits[0].text)
        self.assertIn("qualified two", hits[0].text)
        self.assertNotIn("raw one", hits[0].text)

    async def test_duplicate_votes_cannot_crowd_out_independent_sources(self) -> None:
        repeated = FixedMemory([self.hit(1, f"claim {i}") for i in range(90)])
        distinct = FixedMemory([self.hit(i) for i in range(2, 12)])
        variant = self.variant(repeated=repeated, distinct=distinct)
        await variant.prepare(self.messages)
        hits = await variant.search("storage", self.context, limit=100)
        self.assertEqual(len(hits), 8)
        self.assertEqual(sum(h.source_ids == (1,) for h in hits), 1)
        self.assertEqual(hits[0].score, 1 / 61)
        self.assertTrue(all(len(h.text) <= 6000 for h in hits))
        self.assertEqual(repeated.search_limits, [100])

    async def test_scope_guards_reject_foreign_unknown_future_and_wrong_author(
        self,
    ) -> None:
        memory = FixedMemory([
            replace(self.hit(1), chat_key="foreign_chat"),
            self.hit(999),
            self.hit(2),
            self.hit(3),
            self.hit(1),
        ])
        variant = self.variant(memory=memory)
        await variant.prepare(self.messages)
        context = replace(
            self.context, before=self.messages[0].date, before_id=3, author_id=10
        )
        hits = await variant.search("storage", context)
        self.assertEqual([h.source_ids for h in hits], [(1,)])
        self.assertEqual(
            await variant.search(
                "storage", replace(context, after="2026-01-02T00:00:00+00:00")
            ),
            [],
        )

    async def test_candidate_pool_caps_at100_and_uses_explicit_pool_interface(
        self,
    ) -> None:
        pool = PoolMemory([self.hit(i) for i in range(1, 151)])
        variant = self.variant(pool=pool)
        await variant.prepare(self.messages)
        self.assertEqual(
            len(await variant.candidate_pool("storage", self.context, limit=1000)), 100
        )
        self.assertEqual(pool.search_limits, [])
        self.assertEqual(
            len(await variant.search("storage", self.context, limit=1000)), 8
        )

    async def test_overlapping_records_do_not_repeat_fully_covered_evidence(self) -> None:
        memory = FixedMemory([
            self.hit(1),
            MemoryHit("combined", (1, 2), chat_key="fixture_chat"),
            self.hit(2),
            self.hit(3),
        ])
        variant = self.variant(memory=memory)
        await variant.prepare(self.messages)
        hits = await variant.search("storage", self.context)
        self.assertEqual([h.source_ids for h in hits], [(1,), (2,), (3,)])

    async def test_components_are_prepared_once_without_duplicating_import_objects(
        self,
    ) -> None:
        memory = FixedMemory([self.hit(1)])
        with self.assertRaisesRegex(ValueError, "vote twice"):
            self.variant(first=memory, alias=memory)
        variant = self.variant(native=memory)
        metadata = await variant.prepare(self.messages)
        self.assertEqual(memory.preparations, 1)
        self.assertEqual(
            metadata["components"],
            {"native": {"reused_index": True, "source_count": 150}},
        )
        await variant.close()
        self.assertEqual(memory.closed, 1)


class DenseClaimsTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.claims = self.root / "published"
        self.claims.mkdir()
        self.embeddings = self.root / "existing-raw"
        self.context = QueryContext("fixture_chat", "2026-02-01T00:00:00+00:00")
        self.messages = [
            SourceMessage(
                i,
                "fixture_chat",
                f"2026-01-{i:02}T10:00:00+00:00",
                f"Original report {i}",
                10,
                "alice",
                "Alice",
            )
            for i in range(1, 13)
        ]
        self.calls: list[list[str]] = []
        self.records = [
            (
                1,
                "fixture_chat",
                "[statement; historical; change] Alice: legacy storage",
                [1],
            ),
            (
                2,
                "fixture_chat",
                "[statement; current; change] Alice: migrated storage",
                [2],
            ),
            (
                3,
                "fixture_chat",
                "[opinion; current; uncertain] Alice: prefers voice",
                [3],
            ),
            (
                4,
                "fixture_chat",
                "[statement; current; correct] Alice: corrected employer",
                [4],
            ),
        ]
        self.publish(self.records)

    def publish(self, records: list[tuple[int, str, str, list[int]]]) -> None:
        with (
            closing(sqlite3.connect(self.claims / "claims.sqlite3")) as connection,
            connection,
        ):
            connection.execute(
                "CREATE TABLE IF NOT EXISTS knowledge (id INTEGER PRIMARY KEY,chat_key TEXT,text TEXT,source_ids TEXT)"
            )
            connection.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS knowledge_fts USING fts5(text)"
            )
            connection.execute("DELETE FROM knowledge")
            connection.execute("DELETE FROM knowledge_fts")
            connection.executemany(
                "INSERT INTO knowledge VALUES(?,?,?,?)",
                [(i, chat, text, json.dumps(ids)) for i, chat, text, ids in records],
            )
            connection.executemany(
                "INSERT INTO knowledge_fts(rowid,text) VALUES(?,?)",
                [(i, text) for i, _chat, text, _ids in records],
            )
        (self.claims / "preparation.json").write_text(
            json.dumps({
                "scope": asdict(self.context),
                "source_count": len(self.messages),
                "result": {"claims": len(records)},
            })
        )

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        return [[1.0, 0.0] if "storage" in text else [0.0, 1.0] for text in texts]

    def variant(self, workspace: Path | None = None) -> DenseClaimsVariant:
        variant = DenseClaimsVariant(
            workspace or self.root / "dense",
            claims_workspace=self.claims,
            embedding_workspace=self.embeddings,
            dimensions=2,
            embedder=self.embed,
        )
        self.addAsyncCleanup(variant.close)
        return variant

    async def test_reuses_exact_qualified_rows_without_modifying_published_claims(
        self,
    ) -> None:
        before = hashlib.sha256((self.claims / "claims.sqlite3").read_bytes()).hexdigest()
        variant = self.variant()
        metadata = await variant.prepare(self.messages)
        hits = await variant.candidate_pool("storage", self.context)
        self.assertEqual([h.text for h in hits], [r[2] for r in self.records])
        self.assertEqual(metadata["extraction_calls"], 0)
        self.assertTrue(metadata["reused_published_claims"])
        self.assertEqual(
            before,
            hashlib.sha256((self.claims / "claims.sqlite3").read_bytes()).hexdigest(),
        )

    async def test_reuses_existing_embedding_cache_without_rebuilding_raw_index(
        self,
    ) -> None:
        raw = RawHybridVariant(self.embeddings, dimensions=2, embedder=self.embed)
        self.addAsyncCleanup(raw.close)
        await raw.prepare(self.messages)
        with closing(
            sqlite3.connect(self.embeddings / "raw-hybrid.sqlite3")
        ) as connection:
            before = connection.execute(
                "SELECT * FROM sources ORDER BY message_id"
            ).fetchall()
        first = self.variant()
        await first.prepare(self.messages)
        await first.search("storage", self.context)
        self.calls.clear()
        second = self.variant(self.root / "dense-second")
        await second.prepare(self.messages)
        await second.search("storage", self.context)
        self.assertEqual(self.calls, [])
        with closing(
            sqlite3.connect(self.embeddings / "raw-hybrid.sqlite3")
        ) as connection:
            self.assertEqual(
                before,
                connection.execute(
                    "SELECT * FROM sources ORDER BY message_id"
                ).fetchall(),
            )

    async def test_wrong_prefix_and_unknown_provenance_fail_before_embedding(
        self,
    ) -> None:
        variant = self.variant()
        with self.assertRaisesRegex(ValueError, "different source prefix"):
            await variant.prepare(self.messages[:-1])
        self.assertEqual(self.calls, [])
        self.publish([(1, "fixture_chat", "claim", [999])])
        with self.assertRaisesRegex(ValueError, "outside the supplied prefix"):
            await variant.prepare(self.messages)
        self.assertEqual(self.calls, [])

    async def test_all_sources_must_satisfy_chat_time_and_author_scope(self) -> None:
        self.publish([(1, "fixture_chat", "storage comparison", [1, 2])])
        variant = self.variant()
        await variant.prepare(self.messages)
        self.assertEqual(
            await variant.search(
                "storage",
                replace(self.context, before=self.messages[1].date, before_id=2),
            ),
            [],
        )
        self.assertEqual(
            await variant.search("storage", replace(self.context, author_id=99)), []
        )
        self.assertEqual(
            await variant.search(
                "storage", replace(self.context, chat_key="foreign_chat")
            ),
            [],
        )
        self.assertEqual(
            await variant.search(
                "storage", replace(self.context, after=self.messages[1].date)
            ),
            [],
        )

    async def test_many_claims_from_one_original_collapse_before_pool_limit(self) -> None:
        records = [
            (i, "fixture_chat", "storage " + str(i) + "x" * 500, [1])
            for i in range(1, 121)
        ]
        records.extend(
            (120 + i, "fixture_chat", f"storage report {i}", [i]) for i in range(2, 13)
        )
        self.publish(records)
        variant = self.variant()
        await variant.prepare(self.messages)
        hits = await variant.search("storage", self.context)
        self.assertEqual(len(hits), 8)
        self.assertEqual(sum(h.source_ids == (1,) for h in hits), 1)
        self.assertTrue(all(len(h.text) <= 6000 for h in hits))

    async def test_reusing_dense_workspace_with_changed_published_claims_fails(
        self,
    ) -> None:
        variant = self.variant()
        await variant.prepare(self.messages)
        self.publish([(1, "fixture_chat", "changed claim", [1])])
        with self.assertRaisesRegex(ValueError, "another published source snapshot"):
            await variant.prepare(self.messages)

    async def test_lexical_maintained_component_reuses_published_fts_without_embeddings(
        self,
    ) -> None:
        variant = DenseClaimsVariant(
            self.root / "lexical",
            claims_workspace=self.claims,
            embedding_workspace=self.embeddings,
            semantic=False,
            dimensions=2,
            embedder=self.embed,
        )
        self.addAsyncCleanup(variant.close)
        await variant.prepare(self.messages)
        hits = await variant.candidate_pool("storage", self.context)
        self.assertEqual({h.source_ids for h in hits}, {(1,), (2,)})
        self.assertEqual(self.calls, [])

    async def test_factory_defaults_reuse_sibling_source_prefix_indices(self) -> None:
        workspace = self.root / "indices" / "raw_claims_dense_openviking_rrf" / "prefix"
        variant = build_combination("raw_claims_dense_openviking_rrf", workspace)
        self.addAsyncCleanup(variant.close)
        assert isinstance(variant, CombinationVariant)
        by_name = {c.name: c.memory for c in variant.components}
        dense = by_name["claims_dense"]
        assert isinstance(dense, DenseClaimsVariant)
        self.assertEqual(
            dense.claims_workspace, self.root / "indices" / "claims" / "prefix"
        )
        self.assertEqual(
            dense.embedding_workspace, self.root / "indices" / "raw_hybrid" / "prefix"
        )


if __name__ == "__main__":
    unittest.main()
