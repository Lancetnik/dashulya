"""Reply-aware BM25 and dense recall over a caller-authorized source snapshot."""

# ruff: file-ignore[ambiguous-unicode-character-string] -- Russian tokenization rules are intentional language data.

import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

from dashulya.domain.memory import OriginalMessage
from dashulya.memory._workers import run_in_thread
from dashulya.memory.lexical import LexicalIndex
from dashulya.memory.ranking import reciprocal_rank_fusion, select_source_diverse
from dashulya.memory.vectors import EmbeddingCache

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


@dataclass(frozen=True)
class HybridHit:
    text: str
    sources: tuple[OriginalMessage, ...]
    score: float


def lexical_texts(originals: Sequence[OriginalMessage]) -> list[str]:
    return ["\n\n".join(_render(m) for m in block) for block in _blocks(originals)]


class HybridMemory:
    """Rank one authorized snapshot using PostgreSQL BM25 and pgvector."""

    def __init__(
        self,
        originals: Sequence[OriginalMessage],
        vectors: EmbeddingCache,
        lexical: LexicalIndex,
    ) -> None:
        self._blocks = _blocks(originals)
        self._dates = [_time(block[-1]) for block in self._blocks]
        self._vectors = vectors
        self._lexical = lexical
        self._texts = ["\n\n".join(_render(m) for m in block) for block in self._blocks]
        self._lexical_keys = [lexical.key(text) for text in self._texts]
        self._vector_keys = [
            vectors.key(_embedding_text(block)) for block in self._blocks
        ]

    async def coverage(self) -> dict[str, int]:
        embedded = len(await self._vectors.saved_keys(self._vector_keys))
        return {
            "source_count": len(self._blocks),
            "embedded_blocks": embedded,
            "pending_blocks": len(self._blocks) - embedded,
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
        await self._lexical.prepare(self._texts)
        lexical = await self._lexical.search(
            self._lexical_keys,
            [self._lexical_keys[i] for i in sorted(eligible)],
            query_terms(query),
            limit=_QUERY_CANDIDATES,
        )
        lex_positions = {key: i for i, key in enumerate(self._lexical_keys)}
        rankings = [[lex_positions[key] for key in lexical]]
        vector_keys = [self._vector_keys[i] for i in sorted(eligible)]
        if await self._vectors.saved_keys(vector_keys):
            query_vector = (await self._vectors.vectors([query]))[query]
            dense = await self._vectors.nearest(
                vector_keys, query_vector, limit=_QUERY_CANDIDATES
            )
            positions = {key: i for i, key in enumerate(self._vector_keys)}
            rankings.append([positions[key] for key in dense])
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

    def _eligible(
        self, author_id: int | None, after: str | None, before: str | None
    ) -> set[int]:
        lower, upper = _date(after), _date(before)
        if lower is not None and upper is not None and lower > upper:
            error_message = "after must not be later than before."
            raise ValueError(error_message)
        return {
            i
            for i, (block, sent) in enumerate(zip(self._blocks, self._dates, strict=True))
            if (author_id is None or block[-1].author_id == author_id)
            and (lower is None or (sent is not None and sent >= lower))
            and (upper is None or (sent is not None and sent <= upper))
        }
