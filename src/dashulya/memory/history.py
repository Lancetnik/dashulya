"""Scope-bound original-message history over the unified PostgreSQL store."""

import base64
import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from typing import Any

from dashulya.domain.memory import (
    MemoryAccess,
    MemoryInputError,
    OriginalMessage,
    TruncatedThread,
)
from dashulya.memory.sql_history import SqlHistory
from dashulya.memory.store import MessageStore


def _identity(message: OriginalMessage) -> tuple[int | None, str | None, str | None]:
    # A missing numeric ID does not merge unrelated anonymous participants.
    return (
        (message.author_id, None, None)
        if message.author_id is not None
        else (None, message.author, message.username)
    )


def _name_matches(message: OriginalMessage, value: str | None) -> bool:
    if not value:
        return True
    needle = value.strip().lstrip("@").casefold()
    return (
        needle in message.author.casefold()
        or needle in (message.username or "").lstrip("@").casefold()
    )


def _date(value: str | None) -> datetime | None:
    if value is None:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        error_message = "History dates must use ISO 8601 with a timezone."
        raise MemoryInputError(error_message) from None
    if parsed.utcoffset() is None:
        error_message = "History dates must include a timezone."
        raise MemoryInputError(error_message)
    return parsed.astimezone(UTC)


def _authorized(message: OriginalMessage, access: MemoryAccess) -> bool:
    if not access.is_private and message.chat_id != access.chat_id:
        return False
    if access.before is None:
        return True
    if message.sent_at is None:
        return False
    cutoff = access.before.astimezone(UTC)
    edited_at = _date(message.edited_at)
    if edited_at is not None and edited_at >= cutoff:
        # No prior version is stored; a later edit cannot become historical evidence.
        return False
    if access.before_id is None or message.chat_id != access.chat_id:
        return message.sent_at < cutoff
    return (message.sent_at, message.message_id) < (cutoff, access.before_id)


def _fingerprint(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _cursor_encode(fingerprint: str, position: tuple[str, str, int]) -> str:
    body = json.dumps({"query": fingerprint, "before": position}).encode()
    return base64.urlsafe_b64encode(body).decode()


def _cursor_decode(value: str, fingerprint: str) -> tuple[str, str, int]:
    if len(value) > 2048:
        error_message = "History cursor is invalid for this access scope and query."
        raise MemoryInputError(error_message)
    try:
        decoded = json.loads(base64.b64decode(value, altchars=b"-_", validate=True))
        position = decoded["before"]
        if (
            decoded["query"] != fingerprint
            or len(position) != 3
            or not isinstance(position[0], str)
            or not isinstance(position[1], str)
            or type(position[2]) is not int
        ):
            raise ValueError
    except ValueError, KeyError, TypeError, IndexError:
        error_message = "History cursor is invalid for this access scope and query."
        raise MemoryInputError(error_message) from None
    return position[0], position[1], position[2]


def _reply_context(
    seeds: Sequence[OriginalMessage],
    originals: Sequence[OriginalMessage],
    parent_hops: int,
    *,
    topic_id: int | None = None,
) -> tuple[list[OriginalMessage], dict[str, dict[str, Any]]]:
    """Expand the seed union by minimum parent distance; trace omitted IDs only."""
    by_id = {
        (message.collection, message.message_id): message
        for message in originals
        if topic_id is None or message.topic_id == topic_id
    }

    def parent(child: OriginalMessage) -> OriginalMessage | None:
        if child.reply_to_id is None or child.reply_to_id >= child.message_id:
            return None
        return by_id.get((child.collection, child.reply_to_id))

    selected = {message.source_id: message for message in seeds}
    distances = dict.fromkeys(selected, 0)
    frontier = list(seeds)
    for depth in range(1, parent_hops + 1):
        following = []
        for child in frontier:
            ancestor = parent(child)
            if ancestor is not None and ancestor.source_id not in selected:
                selected[ancestor.source_id] = ancestor
                distances[ancestor.source_id] = depth
                following.append(ancestor)
        frontier = following
        if not frontier:
            break

    selected_ids = {(m.collection, m.message_id) for m in selected.values()}
    cuts = {}
    for reference, child in selected.items():
        if (
            distances[reference] != parent_hops
            or child.reply_to_id is None
            or child.reply_to_id >= child.message_id
            or (child.collection, child.reply_to_id) in selected_ids
        ):
            continue
        root: OriginalMessage | None = child
        remaining = 0
        while root is not None and root.reply_to_id is not None:
            root = parent(root)
            remaining += 1
        cuts[reference] = asdict(
            TruncatedThread(
                child.message_id,
                root.message_id if root else None,
                root.source_id if root else None,
                remaining if root else None,
            )
        )
    seed_ids = {message.source_id for message in seeds}
    return (
        sorted(
            (message for key, message in selected.items() if key not in seed_ids),
            key=lambda message: message.order,
        ),
        cuts,
    )


def _thread_messages(
    originals: Sequence[OriginalMessage], message_id: int, topic_id: int | None
) -> tuple[set[str], dict[str, Any]]:
    anchors = [
        m
        for m in originals
        if m.message_id == message_id and (topic_id is None or m.topic_id == topic_id)
    ]
    if len(anchors) > 1:
        error_message = "Thread anchor is ambiguous across saved collections."
        raise MemoryInputError(error_message)
    if not anchors:
        return set(), {
            "anchor_message_id": message_id,
            "root_message_id": None,
            "root_known": False,
            "available_messages": 0,
        }
    anchor = anchors[0]
    peers = {
        m.message_id: m
        for m in originals
        if m.collection == anchor.collection and m.topic_id == anchor.topic_id
    }
    root = anchor
    known = True
    while root.reply_to_id is not None:
        parent = peers.get(root.reply_to_id)
        if parent is None or parent.message_id >= root.message_id:
            known = False
            break
        root = parent
    selected = {root.message_id: root}
    for candidate in sorted(peers.values(), key=lambda m: m.message_id):
        if (
            candidate.reply_to_id is not None
            and candidate.reply_to_id in selected
            and candidate.reply_to_id < candidate.message_id
        ):
            selected[candidate.message_id] = candidate
    return {m.source_id for m in selected.values()}, {
        "anchor_message_id": message_id,
        "root_message_id": root.message_id,
        "source_id": root.source_id,
        "root_known": known,
        "topic_id": anchor.topic_id,
        "available_messages": len(selected),
    }


def _validate_selection(
    action: str,
    source_id: str | None,
    min_message_id: int | None,
    max_message_id: int | None,
    thread_message_id: int | None,
    text_offset: int,
    limit: int,
    cursor: str | None,
) -> None:
    for value in (min_message_id, max_message_id, thread_message_id):
        if value is not None and (type(value) is not int or value < 1):
            error_message = "Message ID selectors must be positive integers."
            raise MemoryInputError(error_message)
    if (
        min_message_id is not None
        and max_message_id is not None
        and min_message_id > max_message_id
    ):
        error_message = "min_message_id must not exceed max_message_id."
        raise MemoryInputError(error_message)
    if thread_message_id is not None and source_id is not None:
        error_message = "Use either source_id or thread_message_id, not both."
        raise MemoryInputError(error_message)
    valid_offset = type(text_offset) is int and text_offset >= 0
    exact_source = (
        action == "list" and source_id is not None and limit == 1 and cursor is None
    )
    if not valid_offset or (text_offset and not exact_source):
        error_message = "Use a nonnegative text_offset with list, source_id and limit=1 for character continuation."
        raise MemoryInputError(error_message)


def _selection_thread(
    originals: Sequence[OriginalMessage],
    source_id: str | None,
    min_message_id: int | None,
    max_message_id: int | None,
    thread_message_id: int | None,
    topic_id: int | None,
) -> tuple[set[str] | None, dict[str, Any] | None]:
    if source_id is None and any(
        value is not None for value in (min_message_id, max_message_id, thread_message_id)
    ):
        chats = {
            (m.chat_id, m.collection if m.chat_id is None else None) for m in originals
        }
        if len(chats) > 1:
            error_message = "Numeric message ID selection requires one authorized chat."
            raise MemoryInputError(error_message)
    return (
        _thread_messages(originals, thread_message_id, topic_id)
        if thread_message_id is not None
        else (None, None)
    )


class CanonicalHistory:
    """Bind canonical source access to one trusted conversation.

    Groups read their own numeric chat across topics. Private access federates
    saved chats and unbound archives; import binding is established at ingestion.
    """

    def __init__(self, store: MessageStore, access: MemoryAccess) -> None:
        self.store = store
        self.access = access

    async def sql(self, query: list[str]) -> dict[str, Any]:
        return await SqlHistory(self.store.database, self.access).query(query)

    async def originals(self, *, chat: str | None = None) -> list[OriginalMessage]:
        self._chat(chat)
        visible = await self.store.originals(self.access, chat=chat)
        if chat is not None and self.access.is_private and not visible:
            error_message = "Unknown saved chat collection within this access scope."
            raise MemoryInputError(error_message)
        return visible

    async def original(self, source_id: str) -> OriginalMessage | None:
        """Resolve file references through the same fixed authority and cutoff."""
        return next(
            (
                message
                for message in await self.originals()
                if message.source_id == source_id
            ),
            None,
        )

    def coverage(self, originals: Sequence[OriginalMessage]) -> dict[str, Any]:
        """Describe only supplied, authorized canonical rows, never global index bounds."""
        visible = [m for m in originals if _authorized(m, self.access)]
        dates = [m.sent_at for m in visible if m.sent_at is not None]
        periods: dict[tuple[str, str], list[OriginalMessage]] = {}
        for message in visible:
            periods.setdefault((message.collection, message.provenance), []).append(
                message
            )
        known = []
        for (collection, provenance), messages in sorted(periods.items()):
            saved = [m.sent_at for m in messages if m.sent_at is not None]
            known.append({
                "chat": collection,
                "provenance": provenance,
                "available_messages": len(messages),
                "first_saved_at": min(saved).isoformat() if saved else None,
                "last_saved_at": max(saved).isoformat() if saved else None,
            })
        return {
            "complete": False,
            "available_messages": len(visible),
            "first_saved_at": min(dates).isoformat() if dates else None,
            "last_saved_at": max(dates).isoformat() if dates else None,
            "unknown_timestamps": sum(m.sent_at is None for m in visible),
            "known_periods": known,
            "bounds_basis": "Authorized currently available canonical originals, including visible imported rows; endpoints do not establish continuous history.",
            "note": "Only saved originals are available; coverage may have gaps. No live Telegram fetch occurs.",
            "deletion_tracking": False,
            "edit_history": "Latest canonical live versions only; prior edit versions are unavailable.",
        }

    def _chat(self, selector: str | None) -> None:
        if (
            selector is not None
            and not self.access.is_private
            and selector != f"telegram:{self.access.chat_id}"
        ):
            error_message = "Chat selector is outside this conversation's memory access."
            raise MemoryInputError(error_message)

    @staticmethod
    def _matches(
        message: OriginalMessage,
        *,
        text: str,
        case_sensitive: bool,
        author_id: int | None,
        after: datetime | None,
        before: datetime | None,
        chat: str | None,
        topic_id: int | None,
    ) -> bool:
        haystack = message.text if case_sensitive else message.text.casefold()
        needle = text if case_sensitive else text.casefold()
        return (
            needle in haystack
            and (author_id is None or message.author_id == author_id)
            and (chat is None or message.collection == chat)
            and (topic_id is None or message.topic_id == topic_id)
            and (
                after is None
                or (message.sent_at is not None and message.sent_at >= after)
            )
            and (
                before is None
                or (message.sent_at is not None and message.sent_at < before)
            )
        )

    @staticmethod
    def related(
        target: OriginalMessage, originals: list[OriginalMessage]
    ) -> list[OriginalMessage]:
        peers = [
            m
            for m in originals
            if m.collection == target.collection and m.topic_id == target.topic_id
        ]
        by_id = {m.message_id: m for m in peers}
        found: dict[str, OriginalMessage] = {}
        child = target
        for _ in range(5):
            parent = by_id.get(child.reply_to_id or -1)
            if (
                parent is None
                or parent.order >= child.order
                or parent.message_id >= child.message_id
            ):
                break
            found[parent.source_id] = parent
            child = parent
        if target.sent_at is not None:
            nearby = [
                m
                for m in peers
                if m.source_id != target.source_id
                and m.sent_at is not None
                and abs((m.sent_at - target.sent_at).total_seconds()) <= 180
            ]
            before = [m for m in nearby if m.order < target.order][-2:]
            after = [m for m in nearby if m.order > target.order][:2]
            found.update({m.source_id: m for m in [*before, *after]})
        return sorted(found.values(), key=lambda m: m.order)

    async def query(
        self,
        action: str = "list",
        *,
        text_contains: str = "",
        name_contains: str | None = None,
        case_sensitive: bool = False,
        author_id: int | None = None,
        after: str | None = None,
        before: str | None = None,
        chat: str | None = None,
        source_id: str | None = None,
        min_message_id: int | None = None,
        max_message_id: int | None = None,
        thread_message_id: int | None = None,
        cursor: str | None = None,
        offset: int = 0,
        text_offset: int = 0,
        limit: int = 20,
        topic_id: int | None = None,
        parent_hops: int = 5,
    ) -> dict[str, Any]:
        """Page originals and full reply threads inside fixed authority/cutoff.

        Message-ID bounds are inclusive. Date after is inclusive; before exclusive.
        offset skips rows. A single-item list returns up to 12000 characters;
        text_offset continues an exact source. Other excerpts remain capped at1200.
        parent_hops expands ancestors separately from primary matches and counts.
        """
        if action not in {"list", "count", "authors"}:
            error_message = "Unknown history action."
            raise MemoryInputError(error_message)
        if type(parent_hops) is not int or parent_hops < 0:
            error_message = "parent_hops must be a nonnegative integer."
            raise MemoryInputError(error_message)
        if not 1 <= limit <= 50 or offset < 0 or (cursor is not None and offset):
            error_message = (
                "Use limit 1-50, a nonnegative offset, and no offset with cursor."
            )
            raise MemoryInputError(error_message)
        if action != "list" and source_id is not None:
            error_message = "source_id is supported by list only."
            raise MemoryInputError(error_message)
        _validate_selection(
            action,
            source_id,
            min_message_id,
            max_message_id,
            thread_message_id,
            text_offset,
            limit,
            cursor,
        )
        if action != "list" and cursor is not None:
            error_message = "History cursors are supported by list only."
            raise MemoryInputError(error_message)
        self._chat(chat)
        lower, upper = _date(after), _date(before)
        if lower is not None and upper is not None and lower > upper:
            error_message = "after must not be later than before."
            raise MemoryInputError(error_message)
        originals = await self.originals(chat=chat)
        thread_ids, thread = _selection_thread(
            originals,
            source_id,
            min_message_id,
            max_message_id,
            thread_message_id,
            topic_id,
        )
        identities = {_identity(m) for m in originals if _name_matches(m, name_contains)}
        matched = [
            m
            for m in originals
            if _identity(m) in identities
            and (source_id is None or m.source_id == source_id)
            and (min_message_id is None or m.message_id >= min_message_id)
            and (max_message_id is None or m.message_id <= max_message_id)
            and (thread_ids is None or m.source_id in thread_ids)
            and self._matches(
                m,
                text=text_contains,
                case_sensitive=case_sensitive,
                author_id=author_id,
                after=lower,
                before=upper,
                chat=chat,
                topic_id=topic_id,
            )
        ]
        result: dict[str, Any] = {
            "action": action,
            "coverage": self.coverage(originals),
            "filters": {
                "text_contains": text_contains,
                "name_contains": name_contains,
                "case_sensitive": case_sensitive,
                "author_id": author_id,
                "after": lower.isoformat() if lower else None,
                "before": upper.isoformat() if upper else None,
                "chat": chat,
                "topic_id": topic_id,
                "source_id": source_id,
                "min_message_id": min_message_id,
                "max_message_id": max_message_id,
                "thread_message_id": thread_message_id,
            },
        }
        if thread is not None:
            result["thread"] = thread
        if action == "count":
            result.update(
                count=len(matched),
                examples=[m.render(max_chars=500) for m in matched[-3:]],
            )
        elif action == "authors":
            result.update(self._authors(matched, offset, limit))
        else:
            fingerprint = _fingerprint({
                "access": {
                    **asdict(self.access),
                    "before": self.access.before.isoformat()
                    if self.access.before
                    else None,
                },
                "filters": result["filters"],
            })
            page = matched
            if cursor is not None:
                position = _cursor_decode(cursor, fingerprint)
                page = [m for m in page if m.order < position]
            if offset:
                page = page[:-offset] if offset < len(page) else []
            chosen = page[-limit:]
            context, cuts = _reply_context(
                chosen,
                originals
                if thread_ids is None
                else [m for m in originals if m.source_id in thread_ids],
                parent_hops,
                topic_id=topic_id,
            )
            result.update(
                messages=[
                    m.render(offset=text_offset, max_chars=12000 if limit == 1 else 1200)
                    for m in chosen
                ],
                context=[m.render() for m in context],
                parent_hops=parent_hops,
                truncated_threads=cuts,
                next_cursor=_cursor_encode(fingerprint, chosen[0].order)
                if len(page) > len(chosen)
                else None,
                matching_count=len(matched),
            )
        return result

    @staticmethod
    def _authors(
        matched: list[OriginalMessage], offset: int, limit: int
    ) -> dict[str, Any]:
        grouped: dict[
            tuple[int | None, str | None, str | None], list[OriginalMessage]
        ] = {}
        for message in matched:
            grouped.setdefault(_identity(message), []).append(message)
        rows: list[dict[str, Any]] = [
            {
                "author_id": messages[0].author_id,
                "names": sorted({m.author for m in messages}),
                "usernames": sorted({m.username for m in messages if m.username}),
                "message_count": len(messages),
                "examples": [m.render(max_chars=500) for m in messages[-3:]],
            }
            for messages in grouped.values()
        ]
        return {
            "authors": rows[offset : offset + limit],
            "total": len(rows),
            "next_offset": offset + limit if len(rows) > offset + limit else None,
        }
