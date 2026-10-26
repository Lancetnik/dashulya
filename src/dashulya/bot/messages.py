from datetime import UTC, datetime

from aiogram.enums import ChatType
from aiogram.types import Message

from dashulya.domain.memory import MemoryMessage, MessageRecorder


async def capture_message(
    recorder: MessageRecorder,
    message: Message,
    *,
    update_id: int = 0,
    if_missing: bool = False,
    text: str | None = None,
    reply_to_id: int | None = None,
    allow_private: bool = False,
) -> None:
    if message.chat.type not in {ChatType.GROUP, ChatType.SUPERGROUP} and not (
        allow_private and message.chat.type == ChatType.PRIVATE
    ):
        return
    body = (
        text
        if text is not None
        else (
            message.text
            or message.caption
            or (
                message.rich_message.model_dump_json(exclude_none=True)
                if message.rich_message
                else ""
            )
        )
    )
    if not (
        body
        or message.photo
        or message.document
        or message.video
        or message.audio
        or message.voice
        or message.sticker
        or message.animation
        or message.video_note
    ):
        return
    if reply_to_id is None and message.reply_to_message is not None:
        reply_to_id = message.reply_to_message.message_id
    sender = message.sender_chat or message.from_user
    author = (
        (message.sender_chat.title or "Unknown sender")
        if message.sender_chat
        else (message.from_user.full_name if message.from_user else "Unknown sender")
    )
    await recorder.record(
        MemoryMessage(
            chat_id=message.chat.id,
            message_id=message.message_id,
            text=body,
            sent_at=message.date,
            author_id=sender.id if sender else None,
            author=author,
            chat_title=message.chat.title or message.chat.full_name or "Untitled chat",
            chat_username=message.chat.username,
            thread_id=message.message_thread_id,
            reply_to_id=reply_to_id,
            edited_at=datetime.fromtimestamp(message.edit_date, UTC)
            if message.edit_date
            else None,
            media_type=message.content_type
            if message.content_type not in {"text", "rich_message"}
            else None,
            update_id=update_id,
            username=sender.username if sender else None,
            is_bot=message.from_user.is_bot
            if message.from_user is not None and message.sender_chat is None
            else None,
        ),
        if_missing=if_missing,
    )
