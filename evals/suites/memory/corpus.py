"""Canonical source access, independent of experimental knowledge indices."""

import json
from bisect import bisect_left, bisect_right
from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from evals.suites.memory.types import QueryContext, SourceMessage

KNOWN_BOTS = frozenset({8748025221, 8911687928, 8441087938, 7963515450, 8905227848})


def aware_time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.utcoffset() is None:
        error = "Source and query timestamps must include a timezone."
        raise ValueError(error)
    return parsed


def allows(message: SourceMessage, context: QueryContext) -> bool:
    if message.chat_key != context.chat_key:
        return False
    if (aware_time(message.date), message.id) >= (
        aware_time(context.before),
        context.before_id,
    ):
        return False
    if context.after is not None and aware_time(message.date) < aware_time(context.after):
        return False
    return context.author_id is None or message.user_id == context.author_id


def load_messages(path: Path) -> list[SourceMessage]:
    messages = []
    for line in path.read_text(encoding="utf-8").splitlines():
        raw = json.loads(line)
        author = raw.get("author") or raw.get("user_username") or ""
        messages.append(
            SourceMessage(
                id=int(raw["id"]),
                chat_key=str(raw["chat_key"]),
                date=str(raw["date"]),
                text=str(raw.get("text") or ""),
                user_id=raw.get("user_id"),
                author=str(author),
                user_name=str(raw.get("user_name") or ""),
                reply_to_msg_id=raw.get("reply_to_msg_id"),
                thread_post_id=raw.get("thread_post_id"),
                is_bot=raw.get("user_id") in KNOWN_BOTS,
            )
        )
    return messages


class SourceArchive:
    def __init__(self, messages: Iterable[SourceMessage]) -> None:
        dated = [(aware_time(m.date), m) for m in messages]
        dated.sort(key=lambda pair: (pair[0], pair[1].id))
        self.messages = [message for _, message in dated]
        self.by_key = {(m.chat_key, m.id): m for m in self.messages}
        if len(self.by_key) != len(self.messages):
            error = "Canonical input contains duplicate chat/message identities."
            raise ValueError(error)
        self._dates = {(m.chat_key, m.id): date for date, m in dated}
        self._chat_messages: dict[str, list[SourceMessage]] = {}
        self._chat_keys: dict[str, list[tuple[datetime, int]]] = {}
        self._chat_dates: dict[str, list[datetime]] = {}
        for date, message in dated:
            self._chat_messages.setdefault(message.chat_key, []).append(message)
            self._chat_keys.setdefault(message.chat_key, []).append((date, message.id))
            self._chat_dates.setdefault(message.chat_key, []).append(date)

    def available(self, context: QueryContext) -> list[SourceMessage]:
        return [m for m in self.messages if allows(m, context)]

    def get(self, message_id: int, context: QueryContext) -> SourceMessage | None:
        message = self.by_key.get((context.chat_key, message_id))
        return message if message is not None and allows(message, context) else None

    def list_messages(
        self,
        context: QueryContext,
        *,
        text_contains: str = "",
        case_sensitive: bool = False,
        cursor: int | None = None,
        limit: int = 20,
    ) -> tuple[list[SourceMessage], int | None]:
        matching = self.matching(
            context, text_contains=text_contains, case_sensitive=case_sensitive
        )
        if cursor is not None:
            indices = [i for i, m in enumerate(matching) if m.id == cursor]
            if not indices:
                error = "History cursor is not in the matching authorized result set."
                raise ValueError(error)
            matching = matching[: indices[0]]
        page = matching[-max(1, min(limit, 50)) :]
        next_cursor = page[0].id if len(matching) > len(page) else None
        return page, next_cursor

    def matching(
        self,
        context: QueryContext,
        *,
        text_contains: str = "",
        case_sensitive: bool = False,
    ) -> list[SourceMessage]:
        needle = text_contains if case_sensitive else text_contains.casefold()
        return [
            m
            for m in self.available(context)
            if needle in (m.text if case_sensitive else m.text.casefold())
        ]

    def authors(self, name: str, context: QueryContext) -> list[dict[str, Any]]:
        groups: dict[int | None, list[SourceMessage]] = {}
        needle = name.lstrip("@").casefold()
        for message in self.available(context):
            if needle in f"{message.author} {message.user_name}".casefold():
                groups.setdefault(message.user_id, []).append(message)
        return [
            {
                "user_id": user_id,
                "aliases": sorted({m.author for m in records}),
                "names": sorted({m.user_name for m in records}),
                "available_messages": len(records),
            }
            for user_id, records in groups.items()
        ][:30]

    def context(
        self, message_id: int, scope: QueryContext, *, hops: int = 3
    ) -> list[SourceMessage]:
        found: dict[int, SourceMessage] = {}
        current = self.get(message_id, scope)
        for _ in range(hops):
            if current is None:
                break
            parent_id = current.reply_to_msg_id
            if parent_id is None:
                break
            child = current
            current = self.get(parent_id, scope)
            if current is not None and (aware_time(current.date), current.id) >= (
                aware_time(child.date),
                child.id,
            ):
                break
            if current is not None:
                found[current.id] = current
        return sorted(found.values(), key=lambda m: (aware_time(m.date), m.id))

    def coverage(self, context: QueryContext) -> dict[str, Any]:
        messages = self.available(context)
        return {
            "chat_key": context.chat_key,
            "available_messages": len(messages),
            "first_saved_at": messages[0].date if messages else None,
            "last_saved_at": messages[-1].date if messages else None,
            "complete": False,
            "note": "Saved snapshot has gaps; date endpoints do not imply continuous coverage.",
            "authors": len(Counter(m.user_id for m in messages)),
        }

    def neighbors(
        self,
        message_id: int,
        scope: QueryContext,
        *,
        radius: int = 2,
        max_seconds: int = 180,
    ) -> list[SourceMessage]:
        """Return bounded adjacent evidence, without asserting a discussion relation.

        Select the nearest authorized, thread-compatible messages on either side.
        A missing thread ID cannot establish a mismatch. Context remains limited
        to two messages per side and three minutes even if larger bounds are asked.
        """
        target = self.by_key.get((scope.chat_key, message_id))
        if target is None:
            return []
        instant = self._dates[scope.chat_key, message_id]
        cutoff = (aware_time(scope.before), scope.before_id)
        if (instant, message_id) >= cutoff:
            return []
        after = aware_time(scope.after) if scope.after is not None else None
        if (after is not None and instant < after) or (
            scope.author_id is not None and target.user_id != scope.author_id
        ):
            return []
        if radius <= 0 or max_seconds < 0:
            return []
        radius = min(radius, 2)
        span = timedelta(seconds=min(max_seconds, 180))
        dates = self._chat_dates[scope.chat_key]
        keys = self._chat_keys[scope.chat_key]
        start = bisect_left(dates, instant - span)
        if after is not None:
            start = max(start, bisect_left(dates, after))
        end = min(bisect_right(dates, instant + span), bisect_left(keys, cutoff))
        position = bisect_left(keys, (instant, message_id))

        def select(indices: range) -> list[SourceMessage]:
            selected = []
            for index in indices:
                candidate = self._chat_messages[scope.chat_key][index]
                if scope.author_id is not None and candidate.user_id != scope.author_id:
                    continue
                if (
                    target.thread_post_id is not None
                    and candidate.thread_post_id is not None
                    and target.thread_post_id != candidate.thread_post_id
                ):
                    continue
                selected.append(candidate)
                if len(selected) >= radius:
                    break
            return selected

        preceding = select(range(position - 1, start - 1, -1))
        return list(reversed(preceding)) + select(range(position + 1, end))


def source_json(message: SourceMessage, *, max_chars: int = 12000) -> dict[str, Any]:
    result: dict[str, Any] = asdict(message)
    result["url"] = f"https://t.me/{message.chat_key}/{message.id}"
    result["text"] = message.text[:max_chars]
    result["truncated"] = len(message.text) > max_chars
    result["total_chars"] = len(message.text)
    return result
