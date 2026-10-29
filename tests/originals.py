from datetime import UTC, datetime

from dashulya.domain.memory import OriginalMessage


def original(
    mid: int,
    text: str,
    *,
    chat: int = -100123,
    author: int = 10,
    parent: int | None = None,
) -> OriginalMessage:
    return OriginalMessage(
        collection=f"telegram:{chat}",
        chat_id=chat,
        message_id=mid,
        text=text,
        sent_at=datetime(2026, 1, 1, 10, 0, mid, tzinfo=UTC),
        author_id=author,
        author="Alice",
        chat_title="Fixture",
        chat_username="fixture_chat",
        topic_id=None,
        reply_to_id=parent,
        edited_at=None,
        media_type=None,
        url=f"https://t.me/fixture_chat/{mid}",
        provenance="live",
        username="alice",
        is_bot=False,
    )
