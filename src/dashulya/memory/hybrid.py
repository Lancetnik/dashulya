"""Reply-aware BM25 and dense recall over a caller-authorized source snapshot."""

# ruff: file-ignore[ambiguous-unicode-character-string] -- Russian tokenization rules are intentional language data.

import math
import re
import sqlite3
from array import array
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

from dashulya.domain.memory import OriginalMessage
from dashulya.memory._workers import run_in_thread
from dashulya.memory.ranking import reciprocal_rank_fusion, select_source_diverse

_QUERY_CANDIDATES = 160

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


def _time(message: OriginalMessage) -> datetime | None:
    if message.sent_at is None:
        return None
    if message.sent_at.utcoffset() is None:
        error = "Original message timestamps must include a timezone."
        raise ValueError(error)
    return message.sent_at.astimezone(UTC)


def _render(message: OriginalMessage) -> str:
    date = _time(message)
    return (
        f"[message_id={message.message_id} chat={message.collection} "
        f"date={date.isoformat() if date is not None else None} "
        f"author={message.author} ({message.username or message.author}) "
        f"user_id={message.author_id} is_bot={message.is_bot} "
        f"reply_to={message.reply_to_id}]\n{message.text}"
    )


def _blocks(originals: Sequence[OriginalMessage]) -> list[tuple[OriginalMessage, ...]]:
    by_key = {(m.collection, m.message_id): m for m in originals}
    if len(by_key) != len(originals):
        error = "Hybrid recall requires unique canonical collection/message identities."
        raise ValueError(error)
    ordered = sorted(
        originals,
        key=lambda m: (
            _time(m) or datetime.min.replace(tzinfo=UTC),
            m.collection,
            m.message_id,
        ),
    )
    blocks = []
    for message in ordered:
        current = message
        parents = []
        visited = {message.message_id}
        for _ in range(2):
            if current.reply_to_id is None or current.reply_to_id in visited:
                break
            parent = by_key.get((message.collection, current.reply_to_id))
            if parent is None:
                break
            parent_time, current_time = _time(parent), _time(current)
            if (
                parent_time is None
                or current_time is None
                or (parent_time, parent.message_id) > (current_time, current.message_id)
            ):
                break
            parents.append(parent)
            visited.add(parent.message_id)
            current = parent
        blocks.append((*reversed(parents), message))
    return blocks


def _embedding_text(block: Sequence[OriginalMessage]) -> str:
    return "\n\n".join(_render(m)[:4000] for m in block)[:12000]


def embedding_texts(originals: Sequence[OriginalMessage]) -> list[str]:
    """Exact current block payloads used as the provider cache's original-text keys."""
    return [_embedding_text(block) for block in _blocks(originals)]


def _date(value: str | None) -> datetime | None:
    if value is None:
        return None
    parsed = datetime.fromisoformat(value)
    if parsed.utcoffset() is None:
        error = "Search dates must include a timezone."
        raise ValueError(error)
    return parsed.astimezone(UTC)


def _vector(values: Sequence[float], dimensions: int) -> Sequence[float]:
    vector = array("f", values)
    if len(vector) != dimensions or not all(math.isfinite(value) for value in vector):
        error = "Embedding vector must be finite and match configured dimensions."
        raise ValueError(error)
    norm = math.hypot(*vector)
    if not math.isfinite(norm):
        error = "Embedding vector magnitude is invalid."
        raise ValueError(error)
    return array("f", (value / norm for value in vector)) if norm else vector


@dataclass(frozen=True)
class HybridHit:
    text: str
    sources: tuple[OriginalMessage, ...]
    score: float


class HybridMemory:
    """Rank canonical raw blocks; provider calls and durable indexing stay outside."""

    def __init__(
        self,
        originals: Sequence[OriginalMessage],
        source_vectors: Mapping[str, Sequence[float]],
        embed_query: Callable[[str], Awaitable[Sequence[float]]],
        *,
        dimensions: int = 512,
    ) -> None:
        if dimensions < 1:
            error = "Embedding dimensions must be positive."
            raise ValueError(error)
        self._blocks = _blocks(originals)
        self._dates = [_time(block[-1]) for block in self._blocks]
        self._embed_query = embed_query
        self._dimensions = dimensions
        self._vectors = {
            i: _vector(source_vectors[text], dimensions)
            for i, block in enumerate(self._blocks)
            if (text := _embedding_text(block)) in source_vectors
        }
        self._db = sqlite3.connect(":memory:", check_same_thread=False)
        self._db.execute(
            "CREATE VIRTUAL TABLE blocks_fts USING fts5(text, tokenize='unicode61 remove_diacritics 2')"
        )
        self._db.executemany(
            "INSERT INTO blocks_fts(rowid,text) VALUES(?,?)",
            (
                (i, "\n\n".join(_render(m) for m in block))
                for i, block in enumerate(self._blocks, 1)
            ),
        )

    @property
    def coverage(self) -> dict[str, int]:
        return {
            "source_count": len(self._blocks),
            "embedded_blocks": len(self._vectors),
            "pending_blocks": len(self._blocks) - len(self._vectors),
        }

    async def search(
        self,
        query: str,
        *,
        author_id: int | None = None,
        after: str | None = None,
        before: str | None = None,
        limit: int = 8,
    ) -> list[HybridHit]:
        if limit <= 0 or not query.strip():
            return []
        eligible = await run_in_thread(lambda: self._eligible(author_id, after, before))
        if not eligible:
            return []
        query_vector = None
        if eligible & self._vectors.keys():
            # The injected provider stays on its owning event loop.
            query_vector = _vector(await self._embed_query(query), self._dimensions)
        return await run_in_thread(
            lambda: self._search_sync(query, eligible, query_vector, limit)
        )

    def _eligible(
        self, author_id: int | None, after: str | None, before: str | None
    ) -> set[int]:
        lower, upper = _date(after), _date(before)
        if lower is not None and upper is not None and lower > upper:
            error = "after must not be later than before."
            raise ValueError(error)
        return {
            i
            for i, (block, sent) in enumerate(zip(self._blocks, self._dates, strict=True))
            if (author_id is None or block[-1].author_id == author_id)
            and (lower is None or (sent is not None and sent >= lower))
            and (upper is None or (sent is not None and sent <= upper))
        }

    def _search_sync(
        self,
        query: str,
        eligible: set[int],
        query_vector: Sequence[float] | None,
        limit: int,
    ) -> list[HybridHit]:
        lexical = []
        terms = query_terms(query)
        if terms:
            expression = " OR ".join(f'"{term}"*' for term in terms)
            for row in self._db.execute(
                "SELECT rowid FROM blocks_fts WHERE blocks_fts MATCH ? ORDER BY bm25(blocks_fts)",
                (expression,),
            ):
                i = int(row[0]) - 1
                if i in eligible:
                    lexical.append(i)
                    if len(lexical) >= _QUERY_CANDIDATES:
                        break
        rankings = [lexical]
        semantic = eligible & self._vectors.keys()
        if query_vector is not None:
            scores = {
                i: math.fsum(
                    a * b for a, b in zip(self._vectors[i], query_vector, strict=True)
                )
                for i in semantic
            }
            rankings.append(
                sorted(scores, key=lambda i: (-scores[i], i))[:_QUERY_CANDIDATES]
            )
        fused = reciprocal_rank_fusion(rankings, tie_order={i: i for i in eligible})
        allowed = {self._blocks[i][-1].source_id for i in eligible}
        candidates = [
            HybridHit(
                "\n\n".join(
                    _render(m) for m in self._blocks[i] if m.source_id in allowed
                ),
                tuple(m for m in self._blocks[i] if m.source_id in allowed),
                score,
            )
            for i, score in fused
        ]
        return select_source_diverse(
            candidates,
            anchor=lambda hit: hit.sources[-1].source_id,
            sources=lambda hit: (m.source_id for m in hit.sources),
            limit=min(limit, 8),
        )

    def close(self) -> None:
        self._db.close()
