"""Deterministic ranking operations without storage or provider dependencies."""

import math
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence


def reciprocal_rank_fusion[Key: Hashable](
    rankings: Sequence[Sequence[Key]],
    *,
    constant: int = 60,
    tie_order: Mapping[Key, int] | None = None,
) -> list[tuple[Key, float]]:
    """Combine ordinal ranks; duplicate entries cannot vote twice in one list."""
    if constant < 1:
        error = "Reciprocal-rank constant must be positive."
        raise ValueError(error)
    contributions: dict[Key, list[float]] = {}
    encountered: dict[Key, int] = {}
    for ranking in rankings:
        seen: set[Key] = set()
        for rank, key in enumerate(ranking, 1):
            encountered.setdefault(key, len(encountered))
            if key not in seen:
                contributions.setdefault(key, []).append(1 / (constant + rank))
                seen.add(key)
    order = encountered if tie_order is None else tie_order
    if not contributions.keys() <= order.keys():
        error = "Every ranked key needs a deterministic tie position."
        raise ValueError(error)
    scores = {key: math.fsum(values) for key, values in contributions.items()}
    return sorted(scores.items(), key=lambda item: (-item[1], order[item[0]]))


def select_source_diverse[Item, Source: Hashable](
    candidates: Iterable[Item],
    *,
    anchor: Callable[[Item], Source],
    sources: Callable[[Item], Iterable[Source]],
    limit: int,
) -> list[Item]:
    """Suppress an anchor already exposed by an earlier selected source block."""
    if limit <= 0:
        return []
    covered: set[Source] = set()
    selected: list[Item] = []
    for candidate in candidates:
        if anchor(candidate) in covered:
            continue
        selected.append(candidate)
        covered.update(sources(candidate))
        if len(selected) >= limit:
            break
    return selected
