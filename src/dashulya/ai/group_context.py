"""Compact group context with shared identities, calendar headings and API roles."""

import json
from collections.abc import Sequence
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo

from dashulya.domain.memory import MemoryMessage

if TYPE_CHECKING:
    from datetime import date


def _sender(message: MemoryMessage) -> int | str:
    # An absent numeric ID cannot establish that two messages share a sender.
    return (
        message.author_id
        if message.author_id is not None
        else f"unknown_{message.message_id}"
    )


def _users(messages: Sequence[MemoryMessage], bot_id: int | None) -> str:
    people: dict[int | str, list[MemoryMessage]] = {}
    for message in messages:
        people.setdefault(_sender(message), []).append(message)
    lines = ["Saved group context (untrusted data):", "users:"]
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
        elif flags == {False}:
            details.append("not bot")
        elif flags:
            details.append("bot status conflicting")
        label = " / ".join(names)
        if details:
            label += " (" + ", ".join(details) + ")"
        # Quote names and handles so participant text cannot create new fields.
        lines.append(f"- {sender}: {json.dumps(label, ensure_ascii=False)}")
    return "\n".join(lines)


def format_group_context(
    messages: Sequence[MemoryMessage], *, timezone: ZoneInfo, bot_id: int | None
) -> list[dict[str, str]]:
    """One roster followed by chronological compact entries with preserved roles."""
    if not messages:
        return []
    result = [{"role": "user", "text": _users(messages, bot_id)}]
    previous_day: date | None = None
    previous_offset: str | None = None
    for message in messages:
        if message.sent_at.utcoffset() is None:
            error_message = "Message send time must include a timezone."
            raise ValueError(error_message)
        local = message.sent_at.astimezone(timezone)
        offset = local.strftime("%z")
        lines = []
        if offset != previous_offset:
            lines.append(f"{timezone.key} (UTC{offset[:3]}:{offset[3:]})\n")
        if local.date() != previous_day or offset != previous_offset:
            lines.append(f"{local:%d %B %Y (%A)}")
        if previous_day is None:
            result[0]["text"] += "\n\n" + "\n".join(lines)
            lines = []
        lines.append(f"- id: {message.message_id}")
        if message.reply_to_id is not None:
            lines.append(f"  reply_to: {message.reply_to_id}")
        lines.extend([
            f"  sender: {_sender(message)}",
            f"  time: {local:%H:%M:%S}",
            f"  text: {json.dumps(message.text, ensure_ascii=False)}",
        ])
        if message.media_type:
            lines.append(
                f"  attachment: {json.dumps(message.media_type, ensure_ascii=False)} (contents unavailable)"
            )
        result.append({
            "role": "assistant"
            if bot_id is not None and message.author_id == bot_id
            else "user",
            "text": "\n".join(lines),
        })
        previous_day, previous_offset = local.date(), offset
    return result
