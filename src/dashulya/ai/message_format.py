"""One compact message format for immediate context and retrieved evidence."""

import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from zoneinfo import ZoneInfo

from dashulya.domain.memory import (
    MemoryMessage,
    OriginalMessage,
    TruncatedThread,
    message_source_id,
)

if TYPE_CHECKING:
    from datetime import date

type Message = MemoryMessage | OriginalMessage


def _sender(message: Message) -> int | str:
    # An absent numeric ID cannot establish that two messages share a sender.
    return (
        message.author_id
        if message.author_id is not None
        else f"unknown_{message.message_id}"
    )


def _reference(message: Message) -> str:
    if isinstance(message, OriginalMessage):
        return message.source_id
    return message_source_id(f"telegram:{message.chat_id}", message.message_id)


def _users(
    messages: Sequence[Message],
    bot_id: int | None,
    interlocutor_id: int | None,
    author_users: Mapping[int, int],
) -> str:
    people: dict[int | str, list[Message]] = {}
    for message in messages:
        people.setdefault(_sender(message), []).append(message)
    lines = ["users:"]
    for sender, observations in people.items():
        names = list(dict.fromkeys(item.author for item in observations))
        details = list(
            dict.fromkeys(
                "@" + item.username.lstrip("@") for item in observations if item.username
            )
        )
        flags = {item.is_bot for item in observations if item.is_bot is not None}
        if sender == bot_id:
            details.extend(["bot", "self"])
        elif flags == {True}:
            details.append("bot")
        author_user = author_users.get(sender) if type(sender) is int else None
        if author_user is not None and author_user != sender:
            details.append(f"author_user={author_user}")
        if type(interlocutor_id) is int and interlocutor_id in {sender, author_user}:
            details.append("interlocutor")
        label = " / ".join(names)
        if details:
            label += " (" + ", ".join(details) + ")"
        # Keep the human-readable roster on one physical line per identity.
        lines.append(f"- {sender}: {json.dumps(label, ensure_ascii=False)[1:-1]}")
    return "\n".join(lines)


def _format_messages(
    messages: Sequence[Message],
    *,
    timezone: ZoneInfo,
    bot_id: int | None,
    interlocutor_id: int | None = None,
    author_users: Mapping[int, int] | None = None,
    truncated_threads: Sequence[TruncatedThread] = (),
    label: str = "Saved group context",
) -> list[dict[str, str]]:
    """One roster followed by chronological compact entries with preserved roles."""
    if not messages:
        return []
    result = [
        {
            "role": "user",
            "text": f"{label} (untrusted data):\n{_users(messages, bot_id, interlocutor_id, author_users or {})}",
        }
    ]
    visible = {message.message_id: message for message in messages}
    cuts = {
        str(thread.boundary_id): {
            "root_id": thread.root_id,
            "remaining_hops": thread.remaining_hops,
            "boundary_source_id": _reference(visible[thread.boundary_id]),
            **({"source_id": thread.source_id} if thread.source_id is not None else {}),
        }
        for thread in truncated_threads
        if thread.boundary_id in visible
    }
    if cuts:
        result[0]["text"] += "\n\ntruncated_threads: " + json.dumps(
            cuts, separators=(",", ":")
        )
    previous_day: date | None = None
    previous_offset: str | None = None
    for index, message in enumerate(messages):
        if message.sent_at is not None and message.sent_at.utcoffset() is None:
            error_message = "Message send time must include a timezone."
            raise ValueError(error_message)
        local = message.sent_at.astimezone(timezone) if message.sent_at else None
        offset = local.strftime("%z") if local else None
        lines = []
        if offset is not None and offset != previous_offset:
            lines.append(f"{timezone.key} (UTC{offset[:3]}:{offset[3:]})\n")
        if local is None:
            if index == 0 or previous_day is not None:
                lines.append("Unknown send date")
        elif local.date() != previous_day or offset != previous_offset:
            lines.append(f"{local:%d %B %Y (%A)}")
        if index == 0:
            result[0]["text"] += "\n\n" + "\n".join(lines)
            result[0]["text"] += (
                "\n\nmessages: [id,sender,time,text,reply_to?,attachment?]"
            )
            lines = []
        row: list[int | str | None] = [
            message.message_id,
            _sender(message),
            local.strftime("%H:%M:%S") if local else None,
            message.text,
        ]
        if message.reply_to_id is not None or message.media_type:
            row.append(message.reply_to_id)
        if message.media_type:
            row.append(message.media_type)
        lines.append(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
        result.append({
            "role": "assistant"
            if bot_id is not None and message.author_id == bot_id
            else "user",
            "text": "\n".join(lines),
        })
        previous_day, previous_offset = local.date() if local else None, offset
    return result


def _original(source: dict[str, Any]) -> OriginalMessage:
    return OriginalMessage(
        collection=source.get("chat", "unknown"),
        chat_id=source.get("chat_id"),
        message_id=source["message_id"],
        text=source["text"],
        sent_at=datetime.fromisoformat(source["sent_at"])
        if source.get("sent_at")
        else None,
        author_id=source.get("author_id", source.get("user_id")),
        author=source["author"],
        chat_title=source.get("chat_title", ""),
        chat_username=source.get("chat_username"),
        topic_id=source.get("topic_id"),
        reply_to_id=source.get("reply_to_message_id", source.get("reply_to_id")),
        edited_at=source.get("edited_at"),
        media_type=source.get("media_type"),
        url=source.get("url"),
        provenance=source.get("provenance", "unknown"),
        username=source.get("username"),
        is_bot=source.get("is_bot"),
    )


def _source_order(source: dict[str, Any]) -> tuple[datetime, int]:
    instant = (
        datetime.fromisoformat(source["sent_at"])
        if source.get("sent_at")
        else datetime.min.replace(tzinfo=UTC)
    )
    if instant.utcoffset() is None:
        error_message = "Message send time must include a timezone."
        raise ValueError(error_message)
    return instant, source["message_id"]


class MessageFormatter:
    def __init__(
        self,
        timezone: str = "Europe/Moscow",
        bot_id: int | None = None,
        *,
        author_users: Mapping[int, int] | None = None,
    ) -> None:
        self.timezone = ZoneInfo(timezone)
        self.bot_id = bot_id
        self._author_users = dict(author_users or {})

    def context(
        self,
        messages: Sequence[MemoryMessage],
        *,
        interlocutor_id: int | None = None,
        truncated_threads: Sequence[TruncatedThread] = (),
    ) -> list[dict[str, str]]:
        return _format_messages(
            messages,
            timezone=self.timezone,
            bot_id=self.bot_id,
            interlocutor_id=interlocutor_id,
            author_users=self._author_users,
            truncated_threads=truncated_threads,
        )

    def result(
        self, result: dict[str, Any], *, interlocutor_id: int | None = None
    ) -> dict[str, Any]:
        """Keep result semantics, replacing repeated originals with source references."""
        sources: dict[str, dict[str, Any]] = {}

        def collect(value: Any) -> Any:
            if isinstance(value, list):
                return [collect(item) for item in value]
            if not isinstance(value, dict):
                return value
            if {"message_id", "text", "author", "sent_at"} <= value.keys():
                reference = value.get("source_id", f"message:{value['message_id']}")
                saved = sources.get(reference)
                # Overlapping hits/context can contain a longer preview of this original.
                if saved is None or len(value["text"]) > len(saved["text"]):
                    sources[reference] = value
                return reference
            return {
                key: collect(item)
                for key, item in value.items()
                # For a message hit, memory is a duplicate rendering of its sources.
                if not (key == "memory" and value.get("kind") == "message")
            }

        output: dict[str, Any] = collect(result)
        if not sources:
            return output
        cuts = output.pop("truncated_threads", {})
        chats: dict[str, list[tuple[str, dict[str, Any]]]] = {}
        for reference, source in sources.items():
            chats.setdefault(source.get("chat", "unknown"), []).append((
                reference,
                source,
            ))
        blocks = []
        for chat, entries in chats.items():
            entries.sort(key=lambda item: _source_order(item[1]))
            messages = [_original(source) for _, source in entries]
            blocks.append({
                "chat": chat,
                "chat_id": messages[0].chat_id,
                "titles": list(
                    dict.fromkeys(
                        message.chat_title for message in messages if message.chat_title
                    )
                ),
                "usernames": list(
                    dict.fromkeys(
                        message.chat_username
                        for message in messages
                        if message.chat_username
                    )
                ),
                "text": "\n".join(
                    item["text"]
                    for item in _format_messages(
                        messages,
                        timezone=self.timezone,
                        bot_id=self.bot_id,
                        interlocutor_id=interlocutor_id,
                        author_users=self._author_users,
                        truncated_threads=[
                            TruncatedThread(**cuts[reference])
                            for reference, _ in entries
                            if reference in cuts
                        ],
                        label="Saved messages",
                    )
                ),
                "sources": {
                    reference: self._metadata(source) for reference, source in entries
                },
            })
        output["message_blocks"] = blocks
        return output

    @staticmethod
    def _metadata(source: dict[str, Any]) -> dict[str, Any]:
        metadata = {
            key: source[key]
            for key in [
                "message_id",
                "url",
                "provenance",
                "topic_id",
                "edited_at",
                "archive_key",
            ]
            if source.get(key) is not None
        }
        if source.get("truncated"):
            metadata.update({
                key: source[key]
                for key in ["offset", "text_length", "truncated", "next_offset"]
                if key in source
            })
        return metadata
