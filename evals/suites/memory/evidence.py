"""Bound serialized tool evidence without exposing omitted source records."""

import copy
import json
from collections.abc import Iterator
from typing import Any


def objects(value: object) -> Iterator[dict[str, Any]]:
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from objects(child)
    elif isinstance(value, list):
        for child in value:
            yield from objects(child)


def serialized_size(value: object) -> int:
    return len(json.dumps(value, ensure_ascii=False, separators=(",", ":")))


def bound_evidence(value: dict[str, Any], budget: int) -> dict[str, Any]:
    if serialized_size(value) <= budget:
        return value
    if budget < 300:
        return {
            "error": "Evidence budget exhausted. Answer using already collected sources."
        }
    result = copy.deepcopy(value)
    result["evidence_truncated"] = True
    while serialized_size(result) > budget:
        candidates = [
            (node, key)
            for node in objects(result)
            for key in ("text", "memory")
            if isinstance(node.get(key), str) and len(node[key]) > 160
        ]
        if candidates:
            node, key = max(candidates, key=lambda pair: len(pair[0][pair[1]]))
            target = max(160, len(node[key]) - serialized_size(result) + budget - 80)
            node[key] = node[key][:target]
            node["truncated" if key == "text" else "memory_truncated"] = True
            continue
        lists = [
            (key, node[key])
            for node in objects(result)
            for key in (
                "reply_context",
                "nearby_context",
                "hits",
                "messages",
                "sources",
                "example_sources",
            )
            if isinstance(node.get(key), list) and node[key]
        ]
        if not lists:
            return {
                "error": "Evidence budget exhausted. Narrow the search or answer from collected sources."
            }
        contexts = [
            items for key, items in lists if key in {"reply_context", "nearby_context"}
        ]
        records = [items for key, items in lists if key != "hits"]
        selected = max(
            contexts or records or [items for _, items in lists], key=serialized_size
        )
        selected.pop()
    return result
