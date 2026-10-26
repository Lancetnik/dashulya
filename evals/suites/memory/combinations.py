"""Dense published claims and source-scoped reciprocal-rank ensembles."""

import asyncio
import hashlib
import json
import math
import sqlite3
from collections.abc import Sequence
from contextlib import closing
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from evals.suites.memory.openviking import OpenVikingVariant
from evals.suites.memory.raw_hybrid import (
    Embedder,
    RawHybridVariant,
    VectorMatrix,
    query_terms,
)
from evals.suites.memory.types import (
    MemoryHit,
    MemoryVariant,
    QueryContext,
    SourceMessage,
)

ABLATION_COMPONENTS = {
    "claims_dense": ("claims_dense",),
    "raw_claims_dense_rrf": ("raw_hybrid", "claims_dense"),
    "raw_claims_rrf": ("raw_hybrid", "claims_lexical_reused"),
    "raw_openviking_rrf": ("raw_hybrid", "openviking"),
    "claims_dense_openviking_rrf": ("claims_dense", "openviking"),
    "raw_claims_dense_openviking_rrf": (
        "raw_hybrid",
        "claims_dense",
        "openviking",
    ),
}
_POOL = 100
_FINAL = 8
_TEXT = 6000


def _date(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        error = "Source and query times must include a timezone."
        raise ValueError(error)
    return parsed


def _allowed(message: SourceMessage, context: QueryContext) -> bool:
    sent = _date(message.date)
    return (
        message.chat_key == context.chat_key
        and (sent, message.id) < (_date(context.before), context.before_id)
        and (context.after is None or sent >= _date(context.after))
        and (context.author_id is None or message.user_id == context.author_id)
    )


def _source_map(
    messages: Sequence[SourceMessage],
) -> dict[tuple[str, int], SourceMessage]:
    result = {(m.chat_key, m.id): m for m in messages}
    if len(result) != len(messages):
        error = "Duplicate original source identity in combination preparation."
        raise ValueError(error)
    for message in messages:
        _date(message.date)
    return result


def _text(parts: Sequence[str]) -> str:
    unique = list(dict.fromkeys(parts))[:16]
    budget = max(1, (_TEXT - max(0, len(unique) - 1) * 2) // max(1, len(unique)))
    return "\n\n".join(part[:budget] for part in unique)[:_TEXT]


def _diverse(hits: Sequence[MemoryHit], limit: int) -> list[MemoryHit]:
    """Prefer independent evidence, then fill with records adding a new original."""
    selected: list[MemoryHit] = []
    covered: set[tuple[str, int]] = set()
    remaining = []
    for hit in hits:
        keys = {(hit.chat_key, i) for i in hit.source_ids}
        if keys.isdisjoint(covered):
            selected.append(hit)
            covered.update(keys)
            if len(selected) >= limit:
                return selected
        else:
            remaining.append(hit)
    for hit in remaining:
        keys = {(hit.chat_key, i) for i in hit.source_ids}
        if keys - covered:
            selected.append(hit)
            covered.update(keys)
            if len(selected) >= limit:
                break
    return selected


class _SharedVectors(RawHybridVariant):
    """Reuse the existing cache/provider implementation without rebuilding raw recall."""

    async def vectors(self, texts: Sequence[str]) -> VectorMatrix:
        return await self._vectors(texts)

    def usage(self) -> dict[str, int]:
        return {"calls": self._embedding_calls, "input_tokens": self._embedding_tokens}


@dataclass(frozen=True)
class _Claim:
    id: int
    chat_key: str
    text: str
    source_ids: tuple[int, ...]


class DenseClaimsVariant:
    """Embed exact published knowledge rows, preserving correction/change states."""

    def __init__(
        self,
        workspace: Path,
        *,
        claims_workspace: Path | None = None,
        embedding_workspace: Path | None = None,
        model: str | None = None,
        dimensions: int = 512,
        embedder: Embedder | None = None,
        semantic: bool = True,
    ) -> None:
        self.workspace = workspace
        self.claims_workspace = claims_workspace or _index_path(workspace, "claims")
        self.embedding_workspace = embedding_workspace or _index_path(
            workspace, "raw_hybrid"
        )
        self._semantic = semantic
        self._vectors = _SharedVectors(
            self.embedding_workspace,
            model=model,
            dimensions=dimensions,
            embedder=embedder,
        )
        self._claims: list[_Claim] = []
        self._sources: dict[tuple[str, int], SourceMessage] = {}
        self._matrix: VectorMatrix = np.empty((0, dimensions), dtype=np.float32)

    async def prepare(self, messages: Sequence[SourceMessage]) -> dict[str, object]:
        sources = _source_map(messages)
        metadata_path = self.claims_workspace / "preparation.json"
        metadata = json.loads(metadata_path.read_text())
        prepared_scope = QueryContext(**metadata["scope"])
        if metadata["source_count"] != len(messages) or not all(
            _allowed(message, prepared_scope) for message in messages
        ):
            error = "Published claims belong to a different source prefix."
            raise ValueError(error)
        path = self.claims_workspace / "claims.sqlite3"
        with closing(
            sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
        ) as connection:
            rows = connection.execute(
                "SELECT id, chat_key, text, source_ids FROM knowledge ORDER BY id"
            ).fetchall()
        claims = [
            _Claim(i, chat, text, tuple(sorted(set(json.loads(ids)))))
            for i, chat, text, ids in rows
        ]
        if len(claims) != metadata["result"]["claims"]:
            error = "Published claim count differs from completed preparation."
            raise ValueError(error)
        if any(
            not claim.source_ids
            or any((claim.chat_key, i) not in sources for i in claim.source_ids)
            for claim in claims
        ):
            error = "Published claim references an original outside the supplied prefix."
            raise ValueError(error)
        self.workspace.mkdir(parents=True, exist_ok=True)
        fingerprint = hashlib.sha256(
            json.dumps(
                [asdict(sources[k]) for k in sorted(sources)],
                sort_keys=True,
                ensure_ascii=False,
            ).encode()
        ).hexdigest()
        claim_digest = hashlib.sha256(
            json.dumps(rows, ensure_ascii=False).encode()
        ).hexdigest()
        checkpoint = {
            "source_fingerprint": fingerprint,
            "published_claims_fingerprint": claim_digest,
        }
        checkpoint_path = self.workspace / "dense-claims-checkpoint.json"
        if (
            checkpoint_path.exists()
            and json.loads(checkpoint_path.read_text()) != checkpoint
        ):
            error = "Dense claims workspace belongs to another published source snapshot."
            raise ValueError(error)
        before = self._vectors.usage()
        if self._semantic:
            self._matrix = await self._vectors.vectors([c.text for c in claims])
        after = self._vectors.usage()
        self._claims, self._sources = claims, sources
        checkpoint_path.write_text(json.dumps(checkpoint, indent=2) + "\n")
        return {
            "variant": "claims_dense" if self._semantic else "claims_lexical_reused",
            "claims": len(claims),
            "source_count": len(messages),
            "claims_workspace": str(self.claims_workspace),
            "embedding_workspace": str(self.embedding_workspace),
            "reused_published_claims": True,
            "extraction_calls": 0,
            "embedding_calls": after["calls"] - before["calls"],
            "embedding_input_tokens": after["input_tokens"] - before["input_tokens"],
            **checkpoint,
        }

    async def candidate_pool(
        self, query: str, context: QueryContext, *, limit: int = _POOL
    ) -> list[MemoryHit]:
        if not query.strip() or limit <= 0:
            return []
        eligible = [
            i
            for i, claim in enumerate(self._claims)
            if claim.chat_key == context.chat_key
            and all(
                _allowed(self._sources[claim.chat_key, mid], context)
                for mid in claim.source_ids
            )
        ]
        if not eligible:
            return []
        ranked: list[tuple[int, float]]
        if self._semantic:
            vector = (await self._vectors.vectors([query]))[0]
            scores = self._matrix[eligible] @ vector
            order = np.argsort(-scores, kind="stable")
            ranked = [(eligible[int(p)], float(scores[p])) for p in order]
        else:
            expression = " OR ".join(f'"{token[:24]}"*' for token in query_terms(query))
            if not expression:
                return []
            with closing(
                sqlite3.connect(
                    (self.claims_workspace / "claims.sqlite3").resolve().as_uri()
                    + "?mode=ro",
                    uri=True,
                )
            ) as connection:
                rows = connection.execute(
                    "SELECT k.id,bm25(knowledge_fts) FROM knowledge_fts JOIN knowledge k ON k.id=knowledge_fts.rowid WHERE knowledge_fts MATCH ? AND k.chat_key=? ORDER BY bm25(knowledge_fts),k.id",
                    (expression, context.chat_key),
                ).fetchall()
            allowed_ids = {self._claims[i].id: i for i in eligible}
            ranked = [
                (allowed_ids[i], -float(score)) for i, score in rows if i in allowed_ids
            ]
        grouped: dict[tuple[int, ...], list[str]] = {}
        best: dict[tuple[int, ...], float] = {}
        for index, score in ranked:
            claim = self._claims[index]
            key = claim.source_ids
            if key not in grouped:
                if len(grouped) >= min(limit, _POOL):
                    continue
                grouped[key] = []
                best[key] = score
            grouped[key].append(claim.text)
        return [
            MemoryHit(_text(texts), key, best[key], context.chat_key)
            for key, texts in grouped.items()
        ]

    async def search(
        self, query: str, context: QueryContext, *, limit: int = _FINAL
    ) -> list[MemoryHit]:
        if limit <= 0:
            return []
        return self.finalize_candidates(
            await self.candidate_pool(query, context), limit=limit
        )

    @staticmethod
    def finalize_candidates(hits: Sequence[MemoryHit], *, limit: int) -> list[MemoryHit]:
        return _diverse(hits, min(limit, _FINAL)) if limit > 0 else []

    async def close(self) -> None:
        await self._vectors.close()


@dataclass(frozen=True)
class RetrievalComponent:
    name: str
    memory: MemoryVariant


@runtime_checkable
class CandidateMemory(Protocol):
    async def candidate_pool(
        self, query: str, context: QueryContext, *, limit: int = _POOL
    ) -> list[MemoryHit]: ...


class CombinationVariant:
    """Gold-blind RRF over explicitly supplied, reusable source-prefix components."""

    def __init__(
        self,
        workspace: Path,
        *,
        components: Sequence[RetrievalComponent],
        rrf_constant: int = 60,
    ) -> None:
        if not components or rrf_constant < 1:
            error = "Supply components and a positive reciprocal-rank constant."
            raise ValueError(error)
        if len({c.name for c in components}) != len(components):
            error = "Retrieval component names must be unique."
            raise ValueError(error)
        if len({id(c.memory) for c in components}) != len(components):
            error = "A retrieval component cannot vote twice under different names."
            raise ValueError(error)
        self.workspace = workspace
        self.components = tuple(sorted(components, key=lambda c: c.name))
        self.rrf_constant = rrf_constant
        self._sources: dict[tuple[str, int], SourceMessage] = {}

    async def prepare(self, messages: Sequence[SourceMessage]) -> dict[str, object]:
        self._sources = _source_map(messages)
        prepared = {}
        for component in self.components:
            prepared[component.name] = await component.memory.prepare(messages)
        return {
            "variant": "rrf_combination",
            "source_count": len(messages),
            "components": prepared,
            "component_candidate_limit": _POOL,
            "fused_candidate_limit": _POOL,
            "final_hit_limit": _FINAL,
            "rrf_constant": self.rrf_constant,
            "index_strategy": "reuse explicitly supplied source-prefix components",
        }

    async def candidate_pool(
        self, query: str, context: QueryContext, *, limit: int = _POOL
    ) -> list[MemoryHit]:
        if not query.strip() or limit <= 0:
            return []
        rankings = await asyncio.gather(
            *(
                c.memory.candidate_pool(query, context, limit=_POOL)
                if isinstance(c.memory, CandidateMemory)
                else c.memory.search(query, context, limit=_POOL)
                for c in self.components
            )
        )
        scores: dict[tuple[int, ...], list[float]] = {}
        texts: dict[tuple[int, ...], list[str]] = {}
        for ranking in rankings:
            seen: set[tuple[int, ...]] = set()
            for rank, hit in enumerate(ranking[:_POOL], 1):
                key = tuple(sorted(set(hit.source_ids)))
                originals = [self._sources.get((context.chat_key, i)) for i in key]
                if (
                    hit.chat_key != context.chat_key
                    or not originals
                    or any(m is None or not _allowed(m, context) for m in originals)
                ):
                    continue
                texts.setdefault(key, []).append(hit.text)
                if key not in seen:
                    scores.setdefault(key, []).append(1 / (self.rrf_constant + rank))
                    seen.add(key)
        ordered = sorted(scores, key=lambda key: (-math.fsum(scores[key]), key))
        return [
            MemoryHit(_text(texts[key]), key, math.fsum(scores[key]), context.chat_key)
            for key in ordered[: min(limit, _POOL)]
        ]

    async def search(
        self, query: str, context: QueryContext, *, limit: int = _FINAL
    ) -> list[MemoryHit]:
        if limit <= 0:
            return []
        return self.finalize_candidates(
            await self.candidate_pool(query, context), limit=limit
        )

    @staticmethod
    def finalize_candidates(hits: Sequence[MemoryHit], *, limit: int) -> list[MemoryHit]:
        return _diverse(hits, min(limit, _FINAL)) if limit > 0 else []

    async def close(self) -> None:
        for component in self.components:
            await component.memory.close()


def _index_path(workspace: Path, variant: str) -> Path:
    return workspace.parent.parent / variant / workspace.name


def build_combination(name: str, workspace: Path) -> MemoryVariant:
    """Construct the finite ablation matrix using existing per-prefix index paths."""
    if name not in ABLATION_COMPONENTS:
        error = "Unknown memory combination."
        raise ValueError(error)
    components: list[RetrievalComponent] = []
    for component in ABLATION_COMPONENTS[name]:
        memory: MemoryVariant
        if component == "raw_hybrid":
            memory = RawHybridVariant(_index_path(workspace, component))
        elif component == "openviking":
            memory = OpenVikingVariant(_index_path(workspace, component))
        else:
            memory = DenseClaimsVariant(
                _index_path(workspace, component),
                semantic=component == "claims_dense",
            )
        components.append(RetrievalComponent(component, memory))
    if name == "claims_dense":
        return components[0].memory
    return CombinationVariant(workspace, components=components)


class RawClaimsDenseRRFVariant(CombinationVariant):
    def __init__(self, workspace: Path) -> None:
        built = build_combination("raw_claims_dense_rrf", workspace)
        assert isinstance(built, CombinationVariant)
        super().__init__(workspace, components=built.components)


class RawClaimsRRFVariant(CombinationVariant):
    def __init__(self, workspace: Path) -> None:
        built = build_combination("raw_claims_rrf", workspace)
        assert isinstance(built, CombinationVariant)
        super().__init__(workspace, components=built.components)


class RawOpenVikingRRFVariant(CombinationVariant):
    def __init__(self, workspace: Path) -> None:
        built = build_combination("raw_openviking_rrf", workspace)
        assert isinstance(built, CombinationVariant)
        super().__init__(workspace, components=built.components)


class RawClaimsDenseOpenVikingRRFVariant(CombinationVariant):
    def __init__(self, workspace: Path) -> None:
        built = build_combination("raw_claims_dense_openviking_rrf", workspace)
        assert isinstance(built, CombinationVariant)
        super().__init__(workspace, components=built.components)
