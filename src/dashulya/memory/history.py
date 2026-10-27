"""Scope-bound original-message history over the unified PostgreSQL store."""

import base64
import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from typing import Any

from dashulya.domain.memory import MemoryAccess, OriginalMessage, TruncatedThread
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
    parsed = datetime.fromisoformat(value)
    if parsed.utcoffset() is None:
        error_message = "History dates must include a timezone."
        raise ValueError(error_message)
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
        raise ValueError(error_message)
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
    except (ValueError, KeyError, TypeError, IndexError):
        error_message = "History cursor is invalid for this access scope and query."
        raise ValueError(error_message) from None
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


class CanonicalHistory:
    """Bind canonical source access to one trusted conversation.

    Groups read their own numeric chat across topics. Private access federates
    saved chats and unbound archives; import binding is established at ingestion.
    """

    def __init__(self, store: MessageStore, access: MemoryAccess) -> None:
        self.store = store
        self.access = access

    async def originals(self, *, chat: str | None = None) -> list[OriginalMessage]:
        self._chat(chat)
        visible = await self.store.originals(self.access, chat=chat)
        if chat is not None and self.access.is_private and not visible:
            error_message = "Unknown saved chat collection within this access scope."
            raise ValueError(error_message)
        return visible

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
            raise ValueError(error_message)

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
        cursor: str | None = None,
        offset: int = 0,
        limit: int = 20,
        topic_id: int | None = None,
        parent_hops: int = 5,
    ) -> dict[str, Any]:
        """Read originals; after is inclusive and before exclusive, in UTC.

        list pages run backwards and render chronologically. offset skips rows for
        list, or characters for read. read returns up to 12000 original characters;
        next_offset reads the remainder. List/context excerpts are capped at1200.
        Primary filters do not constrain bounded context, which always remains
        within fixed authority/cutoff, the source chat and an explicit topic selector.
        list/read expand reply parents only; parent_hops0 leaves just the primaries.
        """
        if action not in {"list", "read", "count", "authors", "chats"}:
            error_message = "Unknown history action."
            raise ValueError(error_message)
        if type(parent_hops) is not int or parent_hops < 0:
            error_message = "parent_hops must be a nonnegative integer."
            raise ValueError(error_message)
        if not 1 <= limit <= 50 or offset < 0 or (cursor is not None and offset):
            error_message = (
                "Use limit1-50, a nonnegative offset, and no offset with cursor."
            )
            raise ValueError(error_message)
        if action == "read" and source_id is None:
            error_message = "Reading an original requires its source_id."
            raise ValueError(error_message)
        if action != "read" and source_id is not None:
            error_message = "source_id is supported by read only."
            raise ValueError(error_message)
        if action != "list" and cursor is not None:
            error_message = "History cursors are supported by list only."
            raise ValueError(error_message)
        self._chat(chat)
        lower, upper = _date(after), _date(before)
        if lower is not None and upper is not None and lower > upper:
            error_message = "after must not be later than before."
            raise ValueError(error_message)
        originals = await self.originals(chat=chat)
        identities = {_identity(m) for m in originals if _name_matches(m, name_contains)}
        matched = [
            m
            for m in originals
            if _identity(m) in identities
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
            },
        }
        if action == "count":
            result.update(
                count=len(matched),
                examples=[m.render(max_chars=500) for m in matched[-3:]],
            )
        elif action == "read":
            target = next((m for m in matched if m.source_id == source_id), None)
            context, cuts = _reply_context(
                [target] if target else [], originals, parent_hops, topic_id=topic_id
            )
            result.update(
                messages=[target.render(offset=offset, max_chars=12000)]
                if target
                else [],
                context=[m.render() for m in context],
                parent_hops=parent_hops,
                truncated_threads=cuts,
                context_note="Context may contain other authors/dates than primary filters; fixed access and snapshot cutoff always apply.",
            )
        elif action in {"authors", "chats"}:
            result.update(self._groups(action, matched, offset, limit))
        else:
            scope = asdict(self.access)
            scope["before"] = (
                self.access.before.isoformat() if self.access.before else None
            )
            fingerprint = _fingerprint({"access": scope, "filters": result["filters"]})
            page = matched
            if cursor is not None:
                position = _cursor_decode(cursor, fingerprint)
                page = [m for m in page if m.order < position]
            if offset:
                page = page[:-offset] if offset < len(page) else []
            chosen = page[-limit:]
            context, cuts = _reply_context(
                chosen, originals, parent_hops, topic_id=topic_id
            )
            result.update(
                messages=[m.render() for m in chosen],
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
    def _groups(
        action: str, matched: list[OriginalMessage], offset: int, limit: int
    ) -> dict[str, Any]:
        grouped: dict[
            str | tuple[int | None, str | None, str | None], list[OriginalMessage]
        ] = {}
        for message in matched:
            key = _identity(message) if action == "authors" else message.collection
            grouped.setdefault(key, []).append(message)
        rows: list[dict[str, Any]] = []
        for messages in grouped.values():
            if action == "authors":
                rows.append({
                    "author_id": messages[0].author_id,
                    "names": sorted({m.author for m in messages}),
                    "usernames": sorted({m.username for m in messages if m.username}),
                    "message_count": len(messages),
                    "examples": [m.render(max_chars=500) for m in messages[-3:]],
                })
            else:
                first = messages[0]
                rows.append({
                    "chat": first.collection,
                    "chat_id": first.chat_id,
                    "title": first.chat_title,
                    "username": first.chat_username,
                    "message_count": len(messages),
                    "examples": [m.render(max_chars=500) for m in messages[-3:]],
                })
        return {
            action: rows[offset : offset + limit],
            "total": len(rows),
            "next_offset": offset + limit if len(rows) > offset + limit else None,
        }
