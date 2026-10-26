"""Hybrid recall over original messages, without extracted knowledge or gold input."""

# ruff: file-ignore[ambiguous-unicode-character-string] -- Cyrillic strings below are intentional language data.

import asyncio
import hashlib
import json
import logging
import os
import re
import sqlite3
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from openai import AsyncOpenAI, OpenAIError

from evals.suites.memory.providers import validate_local_endpoint
from evals.suites.memory.types import MemoryHit, QueryContext, SourceMessage

type Embedder = Callable[[list[str]], Awaitable[list[list[float]]]]
type VectorMatrix = NDArray[np.float32]

logger = logging.getLogger(__name__)
_EMBEDDING_CHARACTERS = 12_000
_BATCH_SIZE = 128
_PARALLEL_BATCHES = 4
_RANKED_CANDIDATES = 160
_STOP_WORDS = frozenset([
    "а",
    "без",
    "был",
    "была",
    "были",
    "было",
    "бы",
    "в",
    "во",
    "все",
    "вы",
    "где",
    "для",
    "до",
    "его",
    "ее",
    "если",
    "еще",
    "за",
    "зачем",
    "и",
    "из",
    "или",
    "как",
    "какая",
    "какие",
    "какой",
    "каким",
    "кем",
    "к",
    "когда",
    "кто",
    "ли",
    "на",
    "над",
    "не",
    "него",
    "нее",
    "но",
    "о",
    "об",
    "он",
    "она",
    "они",
    "от",
    "по",
    "под",
    "при",
    "про",
    "с",
    "сам",
    "со",
    "так",
    "такой",
    "там",
    "то",
    "у",
    "уже",
    "что",
    "чтобы",
    "это",
    "эти",
    "этот",
    "я",
    "the",
    "a",
    "an",
    "and",
    "or",
    "what",
    "which",
    "who",
    "how",
    "did",
    "does",
    "is",
    "are",
    "was",
    "were",
    "of",
    "to",
    "in",
    "for",
    "on",
    "with",
    "about",
])
_RUSSIAN_ENDINGS = (
    "иями",
    "ами",
    "его",
    "ого",
    "ему",
    "ому",
    "ыми",
    "ими",
    "ией",
    "иям",
    "иях",
    "ать",
    "ять",
    "ить",
    "ешь",
    "ете",
    "или",
    "ал",
    "ла",
    "ли",
    "ло",
    "ет",
    "ют",
    "ят",
    "ым",
    "им",
    "ом",
    "ем",
    "ой",
    "ый",
    "ий",
    "ая",
    "яя",
    "ое",
    "ее",
    "ую",
    "юю",
    "ия",
    "ие",
    "ии",
    "ов",
    "ев",
    "ам",
    "ах",
    "ы",
    "и",
    "а",
    "я",
    "у",
    "ю",
    "е",
    "о",
    "ь",
)


def _instant(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        msg = "Memory timestamps must include a timezone."
        raise ValueError(msg)
    return parsed.astimezone(UTC)


def query_terms(query: str) -> list[str]:
    terms: list[str] = []
    for original_token in re.findall(r"[^\W_]+", query.casefold(), flags=re.UNICODE):
        token = original_token
        if token in _STOP_WORDS or len(token) < 2:
            continue
        if re.fullmatch(r"[а-яё]+", token) and len(token) > 4:
            for ending in _RUSSIAN_ENDINGS:
                if token.endswith(ending) and len(token) - len(ending) >= 4:
                    token = token[: -len(ending)]
                    break
        if token not in terms:
            terms.append(token)
    return terms[:48]


def _render(message: SourceMessage) -> str:
    # Original text is evidence, never an instruction to the answering agent.
    identity = f"{message.user_name} ({message.author}) user_id={message.user_id}"
    return (
        f"[message_id={message.id} chat={message.chat_key} date={message.date} "
        f"author={identity} is_bot={message.is_bot} "
        f"reply_to={message.reply_to_msg_id}]\n{message.text}"
    )


class RawHybridVariant:
    """Rank reply-aware original-message blocks with BM25 and dense embeddings.

    The SQLite cache is independent of benchmark questions and retains completed
    embedding batches if preparation is interrupted. Query embeddings share that
    cache. Every result is reconstructed from allowed canonical originals.
    """

    def __init__(
        self,
        workspace: Path,
        *,
        semantic: bool = True,
        model: str | None = None,
        dimensions: int = 512,
        embedder: Embedder | None = None,
    ) -> None:
        workspace.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(workspace / "raw-hybrid.sqlite3")
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.executescript(
            """
            CREATE TABLE IF NOT EXISTS vectors (
                cache_key TEXT PRIMARY KEY,
                vector BLOB NOT NULL
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS blocks_fts USING fts5(
                text, tokenize='unicode61 remove_diacritics 2'
            );
            CREATE TABLE IF NOT EXISTS sources (
                chat_key TEXT NOT NULL,
                message_id INTEGER NOT NULL,
                payload TEXT NOT NULL,
                PRIMARY KEY (chat_key, message_id)
            );
            """
        )
        self._semantic = semantic
        self._model: str = (
            model or os.getenv("EMBEDDING_MODEL") or "text-embedding-3-small"
        )
        self._dimensions = dimensions
        self._embedder = embedder
        self._client: AsyncOpenAI | None = None
        self._messages: list[SourceMessage] = []
        self._by_key: dict[tuple[str, int], SourceMessage] = {}
        self._parents: list[tuple[SourceMessage, ...]] = []
        self._matrix: VectorMatrix = np.empty((0, dimensions), dtype=np.float32)
        self._dates: list[datetime] = []
        self._embedding_calls = 0
        self._embedding_tokens = 0

    async def prepare(self, messages: Sequence[SourceMessage]) -> dict[str, object]:
        """Build from every provided original, retaining only the final supplied row."""
        self._by_key = {(m.chat_key, m.id): m for m in messages}
        self._messages = sorted(
            self._by_key.values(), key=lambda m: (_instant(m.date), m.chat_key, m.id)
        )
        self._dates = [_instant(m.date) for m in self._messages]
        self._parents = [self._reply_parents(m) for m in self._messages]
        texts = [
            "\n\n".join(_render(m) for m in (*parents, message))
            for message, parents in zip(self._messages, self._parents, strict=True)
        ]
        embedding_texts = [
            "\n\n".join(_render(m)[:4_000] for m in (*parents, message))
            for message, parents in zip(self._messages, self._parents, strict=True)
        ]
        # Rebuild the active corpus, keeping the content-addressed vector cache.
        # A changed or smaller corpus must never leave an old searchable source.
        with self._db:
            self._db.execute("DELETE FROM blocks_fts")
            self._db.execute("DELETE FROM sources")
            self._db.executemany(
                "INSERT INTO blocks_fts(rowid, text) VALUES (?, ?)",
                enumerate(texts, start=1),
            )
            self._db.executemany(
                "INSERT INTO sources(chat_key, message_id, payload) VALUES (?, ?, ?)",
                (
                    (m.chat_key, m.id, json.dumps(asdict(m), ensure_ascii=False))
                    for m in self._messages
                ),
            )
        cached_before = self._db.execute("SELECT COUNT(*) FROM vectors").fetchone()[0]
        if self._semantic:
            self._matrix = await self._vectors(embedding_texts)
        return {
            "variant": "raw_hybrid" if self._semantic else "raw_lexical",
            "sources": len(self._messages),
            "reply_context_blocks": sum(bool(p) for p in self._parents),
            "embedding_model": self._model if self._semantic else None,
            "embedding_dimensions": self._dimensions if self._semantic else None,
            "cached_vectors_before": int(cached_before),
            "embedding_calls": self._embedding_calls,
            "embedding_input_tokens": self._embedding_tokens,
        }

    def _reply_parents(self, message: SourceMessage) -> tuple[SourceMessage, ...]:
        parents: list[SourceMessage] = []
        current = message
        visited = {message.id}
        for _ in range(2):
            parent_id = current.reply_to_msg_id
            if parent_id is None or parent_id in visited:
                break
            parent = self._by_key.get((message.chat_key, parent_id))
            if parent is None or (_instant(parent.date), parent.id) > (
                _instant(current.date),
                current.id,
            ):
                break
            parents.append(parent)
            visited.add(parent.id)
            current = parent
        return tuple(reversed(parents))

    def _cache_key(self, text: str) -> str:
        settings = f"{self._model}:{self._dimensions}:".encode()
        return hashlib.sha256(settings + text.encode()).hexdigest()

    async def _embed(self, texts: list[str]) -> list[list[float]]:
        if self._embedder is not None:
            return await self._embedder(texts)
        if self._client is None:
            validate_local_endpoint(os.getenv("OPENAI_BASE_URL"), self._model)
            self._client = AsyncOpenAI()
        validate_local_endpoint(str(self._client.base_url), self._model)
        try:
            response = await self._client.embeddings.create(
                input=texts,
                model=self._model,
                dimensions=self._dimensions,
                encoding_format="float",
            )
        except OpenAIError:
            msg = "Embedding request failed; cached preparation batches are resumable."
            raise RuntimeError(msg) from None
        self._embedding_calls += 1
        self._embedding_tokens += response.usage.total_tokens
        return [item.embedding for item in sorted(response.data, key=lambda d: d.index)]

    async def _vectors(self, texts: Sequence[str]) -> VectorMatrix:
        clipped = [text[:_EMBEDDING_CHARACTERS] for text in texts]
        keyed = {self._cache_key(text): text for text in clipped}
        cached: dict[str, VectorMatrix] = {}
        for key in keyed:
            row = self._db.execute(
                "SELECT vector FROM vectors WHERE cache_key = ?", (key,)
            ).fetchone()
            if row is not None:
                cached[key] = np.frombuffer(bytes(row[0]), dtype=np.float32)
        missing = [(key, text) for key, text in keyed.items() if key not in cached]
        completed = 0
        semaphore = asyncio.Semaphore(_PARALLEL_BATCHES)

        async def batch(items: list[tuple[str, str]]) -> None:
            nonlocal completed
            async with semaphore:
                vectors = await self._embed([text for _, text in items])
                if len(vectors) != len(items):
                    msg = "Embedding provider returned an incomplete batch."
                    raise ValueError(msg)
                with self._db:
                    for (key, _), vector in zip(items, vectors, strict=True):
                        array = np.asarray(vector, dtype=np.float32)
                        if (
                            array.shape != (self._dimensions,)
                            or not np.isfinite(array).all()
                        ):
                            msg = "Embedding provider returned an invalid vector."
                            raise ValueError(msg)
                        norm = float(np.linalg.norm(array))
                        if norm > 0:
                            array /= norm
                        self._db.execute(
                            "INSERT OR REPLACE INTO vectors(cache_key, vector) VALUES (?, ?)",
                            (key, array.tobytes()),
                        )
                        cached[key] = array
                completed += len(items)
                logger.info(
                    "Embedding cache updated: %d/%d records", completed, len(missing)
                )

        # Bound concurrent requests and preserve completed batches on any failure.
        for start in range(0, len(missing), _BATCH_SIZE * _PARALLEL_BATCHES):
            outcomes = await asyncio.gather(
                *(
                    batch(missing[offset : offset + _BATCH_SIZE])
                    for offset in range(
                        start,
                        min(start + _BATCH_SIZE * _PARALLEL_BATCHES, len(missing)),
                        _BATCH_SIZE,
                    )
                ),
                return_exceptions=True,
            )
            for outcome in outcomes:
                if isinstance(outcome, BaseException):
                    raise outcome
        if not clipped:
            return np.empty((0, self._dimensions), dtype=np.float32)
        return np.stack([cached[self._cache_key(text)] for text in clipped])

    def _eligible(self, context: QueryContext) -> list[int]:
        before = _instant(context.before)
        after = _instant(context.after) if context.after is not None else None
        eligible: list[int] = []
        for index, (message, date) in enumerate(
            zip(self._messages, self._dates, strict=True)
        ):
            if message.chat_key != context.chat_key:
                continue
            if (date, message.id) >= (before, context.before_id):
                continue
            if after is not None and date < after:
                continue
            if context.author_id is not None and message.user_id != context.author_id:
                continue
            eligible.append(index)
        return eligible

    def _lexical(self, query: str, eligible: set[int]) -> list[int]:
        terms = query_terms(query)
        if not terms:
            return []
        expression = " OR ".join(f'"{term}"*' for term in terms)
        # Filter before applying the candidate cap, so busy unrelated chats or
        # future messages cannot crowd eligible originals out of the result.
        rows = self._db.execute(
            "SELECT rowid FROM blocks_fts WHERE blocks_fts MATCH ? "
            "ORDER BY bm25(blocks_fts)",
            (expression,),
        )
        ranked: list[int] = []
        for row in rows:
            index = int(row[0]) - 1
            if index in eligible:
                ranked.append(index)
                if len(ranked) >= _RANKED_CANDIDATES:
                    break
        return ranked

    async def search(
        self, query: str, context: QueryContext, *, limit: int = 8
    ) -> list[MemoryHit]:
        if limit <= 0 or not query.strip():
            return []
        eligible = self._eligible(context)
        if not eligible:
            return []
        eligible_set = set(eligible)
        rankings = [self._lexical(query, eligible_set)]
        if self._semantic:
            query_vector = (await self._vectors([query]))[0]
            scores = self._matrix[eligible] @ query_vector
            order = np.argsort(-scores, kind="stable")[:_RANKED_CANDIDATES]
            rankings.append([eligible[int(index)] for index in order])
        fused: dict[int, float] = {}
        for ranking in rankings:
            for rank, index in enumerate(ranking, start=1):
                fused[index] = fused.get(index, 0.0) + 1.0 / (60 + rank)
        allowed_keys = {
            (self._messages[i].chat_key, self._messages[i].id) for i in eligible
        }
        hits: list[MemoryHit] = []
        covered: set[tuple[str, int]] = set()
        for index in sorted(fused, key=lambda i: (-fused[i], i)):
            message = self._messages[index]
            if (message.chat_key, message.id) in covered:
                continue
            sources = [
                source
                for source in (*self._parents[index], message)
                if (source.chat_key, source.id) in allowed_keys
            ]
            covered.update((s.chat_key, s.id) for s in sources)
            hits.append(
                MemoryHit(
                    text="\n\n".join(_render(source) for source in sources),
                    source_ids=tuple(source.id for source in sources),
                    score=fused[index],
                    chat_key=context.chat_key,
                )
            )
            if len(hits) >= limit:
                break
        return hits

    async def close(self) -> None:
        if self._client is not None:
            await self._client.close()
        self._db.close()


class RawLexicalVariant(RawHybridVariant):
    """The same source and context handling, with only lexical recall."""

    def __init__(self, workspace: Path) -> None:
        super().__init__(workspace, semantic=False)
