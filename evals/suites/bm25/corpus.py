"""Authorized corpus and reporting helpers for native ParadeDB tuning."""

# ruff: file-ignore[import-private-name] -- Match production source blocks and normalization.

import bisect
import json
import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from dashulya.domain.memory import OriginalMessage
from dashulya.memory._unicode61 import tokens
from dashulya.memory.hybrid import _blocks, _render
from dashulya.memory.lexical import LexicalIndex


def instant(value: str) -> datetime:
    result = datetime.fromisoformat(value)
    if result.utcoffset() is None:
        msg = "Benchmark timestamps must include a timezone."
        raise ValueError(msg)
    return result


@dataclass(frozen=True)
class Case:
    case_id: str
    query: str
    cutoff: datetime
    collections: tuple[str, ...] = ()
    after: datetime | None = None
    before: datetime | None = None
    author_id: int | None = None
    requirements: tuple[tuple[tuple[str, ...], ...], ...] = ()

    @classmethod
    def load(cls, row: dict[str, Any]) -> Case:
        return cls(
            case_id=row["case_id"],
            query=row["query"],
            cutoff=instant(row["cutoff"]),
            collections=tuple(row.get("collections", [])),
            after=instant(row["after"]) if row.get("after") else None,
            before=instant(row["before"]) if row.get("before") else None,
            author_id=row.get("author_id"),
            requirements=tuple(
                tuple(tuple(support) for support in requirement)
                for requirement in row.get("requirements", [])
            ),
        )


@dataclass(frozen=True)
class Document:
    key: str
    text: str
    terms: tuple[str, ...]
    collection: str
    author_id: int | None
    sent_at: datetime | None
    sources: tuple[str, ...]
    edited_at: datetime | None


def documents(originals: Sequence[OriginalMessage], case: Case) -> list[Document]:
    """Select primary blocks after trusted scope/cutoff, retaining allowed parents."""
    visible = [
        message
        for message in originals
        if (not case.collections or message.collection in case.collections)
        and message.sent_at is not None
        and message.sent_at < case.cutoff
        and (message.edited_at is None or instant(message.edited_at) < case.cutoff)
    ]
    selected = []
    for block in _blocks(visible):
        anchor = block[-1]
        assert anchor.sent_at is not None
        wrong_author = case.author_id is not None and anchor.author_id != case.author_id
        too_early = case.after is not None and anchor.sent_at < case.after
        too_late = case.before is not None and anchor.sent_at >= case.before
        if wrong_author or too_early or too_late:
            continue
        text = "\n\n".join(_render(message) for message in block)
        selected.append(
            Document(
                LexicalIndex.key(text),
                text,
                tuple(tokens(text)),
                anchor.collection,
                anchor.author_id,
                anchor.sent_at,
                tuple(f"{message.collection}/{message.message_id}" for message in block),
                instant(anchor.edited_at) if anchor.edited_at else None,
            )
        )
    return selected


def expand(prefixes: Sequence[str], vocabulary: list[str]) -> list[str]:
    """Expand without a silent term cap; the dictionary belongs to this snapshot."""
    expanded = set()
    for prefix in prefixes:
        position = bisect.bisect_left(vocabulary, prefix)
        while position < len(vocabulary) and vocabulary[position].startswith(prefix):
            expanded.add(vocabulary[position])
            position += 1
    return sorted(expanded)


def distribution(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    ordered = sorted(values)
    return {
        "p50_ms": round(statistics.median(ordered), 3),
        "p95_ms": round(ordered[math.ceil(len(ordered) * 0.95) - 1], 3),
        "max_ms": round(ordered[-1], 3),
    }


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
