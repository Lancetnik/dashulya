import asyncio
from contextlib import aclosing
from pathlib import PurePosixPath
from typing import Literal

from aiogram import Bot
from aiogram.types import Document, Message, PhotoSize, Sticker

from dashulya.domain.messages import ImageAttachment, ImageOmission

ImageMediaType = Literal["image/jpeg", "image/png", "image/webp"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif"}


class ImageInputError(ValueError):
    """A safe explanation of an unsupported or oversized image input."""


class ImageTooLargeError(ImageInputError):
    """Image bytes cannot fit within this request's remaining budget."""


def _preview_kind(message: Message) -> Literal["sticker", "animation"] | None:
    if message.animation is not None:
        return "animation"
    if message.sticker and (message.sticker.is_animated or message.sticker.is_video):
        return "sticker"
    document = message.document
    if document and (
        (document.mime_type or "").lower() == "image/gif"
        or PurePosixPath(document.file_name or "").suffix.lower() == ".gif"
    ):
        return "animation"
    return None


def _image_file(message: Message) -> Document | PhotoSize | Sticker | None:
    # Telegram may also populate `document` for an animation; its thumbnail takes precedence.
    if message.animation is not None:
        return message.animation.thumbnail
    if message.sticker is not None:
        sticker = message.sticker
        return sticker.thumbnail if sticker.is_animated or sticker.is_video else sticker
    if message.photo:
        return max(message.photo, key=lambda photo: photo.width * photo.height)
    if message.document and _preview_kind(message) == "animation":
        return message.document.thumbnail
    return message.document


def _image_candidate(message: Message) -> bool:
    if message.photo or message.sticker or message.animation:
        return True
    document = message.document
    return document is not None and (
        (document.mime_type or "").startswith("image/")
        or PurePosixPath(document.file_name or "").suffix.lower() in IMAGE_EXTENSIONS
    )


def _media_type(data: bytes) -> ImageMediaType:
    # Telegram metadata is only a candidate; the downloaded signature determines the MIME.
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return "image/webp"
    error_message = "Please send an image in JPEG, PNG or WebP format."
    raise ImageInputError(error_message)


class TelegramImages:
    def __init__(self, bot: Bot, *, max_bytes: int) -> None:
        self._bot = bot
        self._max_bytes = max_bytes

    def _check_size(self, size: int) -> None:
        if size > self._max_bytes:
            raise ImageTooLargeError

    async def _read(
        self, file: Document | PhotoSize | Sticker, *, used_bytes: int
    ) -> bytes:
        self._check_size(used_bytes + 1)
        self._check_size((file.file_size or 0) + used_bytes)
        metadata = await self._bot.get_file(file.file_id)
        self._check_size((metadata.file_size or 0) + used_bytes)
        if metadata.file_path is None:
            error_message = "I couldn't download this image. Please send it again."
            raise ImageInputError(error_message)
        # The credential-bearing URL stays inside the Telegram adapter.
        url = self._bot.session.api.file_url(self._bot.token, metadata.file_path)
        data = bytearray()
        async with aclosing(
            self._bot.session.stream_content(url, chunk_size=65_536)
        ) as stream:
            async for chunk in stream:
                self._check_size(len(data) + len(chunk) + used_bytes)
                data.extend(chunk)
        return bytes(data)

    async def load(self, message: Message) -> tuple[ImageAttachment | ImageOmission, ...]:
        sources = (
            [message]
            if message.photo or message.document or message.sticker or message.animation
            else []
        )
        parent = message.reply_to_message
        if (
            parent is not None
            and parent.chat.id == message.chat.id
            and parent.message_thread_id == message.message_thread_id
            and _image_candidate(parent)
        ):
            sources.append(parent)
        images: list[ImageAttachment | ImageOmission] = []
        used_bytes = 0
        async with asyncio.timeout(30):
            for source in sources:
                file = _image_file(source)
                if file is None:
                    images.append(ImageOmission(source.message_id, "preview_unavailable"))
                    continue
                if not _image_candidate(source):
                    error_message = "Please send an image in JPEG, PNG or WebP format."
                    raise ImageInputError(error_message)
                try:
                    data = await self._read(file, used_bytes=used_bytes)
                except ImageTooLargeError:
                    images.append(ImageOmission(source.message_id))
                    continue
                images.append(
                    ImageAttachment(
                        data,
                        _media_type(data),
                        source.message_id,
                        preview_kind=_preview_kind(source),
                    )
                )
                used_bytes += len(data)
        return tuple(images)
