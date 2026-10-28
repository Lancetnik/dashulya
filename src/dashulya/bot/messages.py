from datetime import UTC, datetime
from pathlib import PurePosixPath
from typing import Literal

from aiogram.enums import ChatType
from aiogram.types import Animation, Document, Message, Sticker

from dashulya.domain.memory import MemoryMessage, MessageRecorder
from dashulya.domain.messages import ConversationKey, ImageFile


def _visual_file(
    file: Animation | Document | Sticker,
    *,
    preview_kind: Literal["sticker", "animation"] | None = None,
) -> ImageFile:
    thumbnail = file.thumbnail if preview_kind is not None else None
    return ImageFile(
        file.file_id,
        file.file_size,
        preview_kind=preview_kind,
        thumbnail_file_id=thumbnail.file_id if thumbnail else None,
        thumbnail_file_size=thumbnail.file_size if thumbnail else None,
    )


def image_file(message: Message) -> ImageFile | None:
    """Retain the same visual inputs for current messages and later reply context."""
    if message.animation is not None:
        return _visual_file(message.animation, preview_kind="animation")
    if message.sticker is not None:
        sticker = message.sticker
        return _visual_file(
            sticker,
            preview_kind="sticker" if sticker.is_animated or sticker.is_video else None,
        )
    if message.photo:
        photo = max(message.photo, key=lambda item: item.width * item.height)
        return ImageFile(photo.file_id, photo.file_size)
    document = message.document
    if document is not None:
        extension = PurePosixPath(document.file_name or "").suffix.lower()
        mime = (document.mime_type or "").lower()
        if mime == "image/gif" or extension == ".gif":
            return _visual_file(document, preview_kind="animation")
        if mime in {"image/jpeg", "image/png", "image/webp"} or extension in {
            ".jpg",
            ".jpeg",
            ".png",
            ".webp",
        }:
            return ImageFile(document.file_id, document.file_size)
    return None


def conversation_key(message: Message) -> ConversationKey:
    """Ordinary groups share one conversation; real forum topics stay separate."""
    topic = (
        message.message_thread_id
        if message.is_topic_message or message.chat.type == ChatType.PRIVATE
        else None
    )
    if topic is None and message.chat.is_forum:
        # General is topic 1 even when Telegram omits a thread ID.
        topic = 1
    return message.chat.id, topic


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
            thread_id=conversation_key(message)[1],
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
            image_file=image_file(message),
        ),
        if_missing=if_missing,
    )
