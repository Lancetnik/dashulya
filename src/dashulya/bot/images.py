import asyncio
from contextlib import aclosing
from typing import Literal

from aiogram import Bot
from aiogram.types import Message

from dashulya.bot.media_frames import FrameBudgetError, FrameDecodeError, first_frame
from dashulya.bot.messages import conversation_key, image_file
from dashulya.domain.memory import MemoryMessage
from dashulya.domain.messages import ImageAttachment, ImageFile, ImageOmission

ImageMediaType = Literal["image/jpeg", "image/png", "image/webp"]


class ImageInputError(ValueError):
    """A safe explanation of an unsupported or oversized image input."""


class ImageTooLargeError(ImageInputError):
    """Image bytes cannot fit within this request's remaining budget."""


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
        self,
        file: ImageFile,
        *,
        used_bytes: int,
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

    async def _animation(
        self, file: ImageFile, message_id: int, *, used_bytes: int
    ) -> ImageAttachment | ImageOmission:
        reason: Literal["too_large", "decode_failed"] = "decode_failed"
        try:
            data = await self._read(file, used_bytes=used_bytes)
            frame = await asyncio.to_thread(
                first_frame, data, max_bytes=self._max_bytes - used_bytes
            )
        except (ImageTooLargeError, FrameBudgetError):
            reason = "too_large"
        except (ImageInputError, FrameDecodeError):
            pass
        else:
            return ImageAttachment(
                frame,
                "image/jpeg",
                message_id,
                preview_kind="animation",
                first_frame=True,
            )
        thumbnail = self._thumbnail(file)
        if thumbnail is None:
            return ImageOmission(message_id, reason)
        try:
            preview = await self._read(thumbnail, used_bytes=used_bytes)
        except ImageTooLargeError:
            return ImageOmission(message_id)
        return ImageAttachment(
            preview, _media_type(preview), message_id, preview_kind="animation"
        )

    @staticmethod
    def _thumbnail(file: ImageFile) -> ImageFile | None:
        if file.thumbnail_file_id is None:
            return None
        return ImageFile(file.thumbnail_file_id, file.thumbnail_file_size)

    async def _load_image(
        self, file: ImageFile, message_id: int, *, used_bytes: int
    ) -> ImageAttachment | ImageOmission:
        if file.preview_kind == "animation":
            return await self._animation(file, message_id, used_bytes=used_bytes)
        source = file
        if file.preview_kind == "sticker":
            thumbnail = self._thumbnail(file)
            if thumbnail is None:
                return ImageOmission(message_id, "preview_unavailable")
            source = thumbnail
        try:
            data = await self._read(source, used_bytes=used_bytes)
        except ImageTooLargeError:
            return ImageOmission(message_id)
        return ImageAttachment(
            data, _media_type(data), message_id, preview_kind=file.preview_kind
        )

    async def load(self, message: Message) -> tuple[ImageAttachment | ImageOmission, ...]:
        sources = (
            [message]
            if message.photo or message.document or message.sticker or message.animation
            else []
        )
        parent = message.reply_to_message
        if (
            parent is not None
            and conversation_key(parent) == conversation_key(message)
            and image_file(parent) is not None
        ):
            sources.append(parent)
        images: list[ImageAttachment | ImageOmission] = []
        used_bytes = 0
        async with asyncio.timeout(30):
            for source in sources:
                file = image_file(source)
                if file is None:
                    error_message = "Please send an image in JPEG, PNG or WebP format."
                    raise ImageInputError(error_message)
                image = await self._load_image(
                    file, source.message_id, used_bytes=used_bytes
                )
                images.append(image)
                if isinstance(image, ImageAttachment):
                    used_bytes += len(image.data)
        return tuple(images)

    async def load_context(
        self,
        messages: tuple[MemoryMessage, ...],
        *,
        images: tuple[ImageAttachment | ImageOmission, ...] = (),
    ) -> tuple[ImageAttachment | ImageOmission, ...]:
        """Load saved image references only from the selected reply context."""
        allowed = {message.message_id for message in messages}
        result = [image for image in images if image.message_id in allowed]
        supplied = {image.message_id for image in result}
        used_bytes = sum(
            len(image.data) for image in result if isinstance(image, ImageAttachment)
        )
        # Prefer the most recent context images after the current/direct-reply inputs.
        pending = [
            message
            for message in reversed(messages)
            if message.image_file is not None and message.message_id not in supplied
        ]
        deadline = asyncio.get_running_loop().time() + 30
        for message in pending:
            image = await self._context_image(
                message, used_bytes=used_bytes, deadline=deadline
            )
            result.append(image)
            if isinstance(image, ImageAttachment):
                used_bytes += len(image.data)
        return tuple(result)

    async def _context_image(
        self, message: MemoryMessage, *, used_bytes: int, deadline: float
    ) -> ImageAttachment | ImageOmission:
        assert message.image_file is not None
        if asyncio.get_running_loop().time() >= deadline:
            return ImageOmission(message.message_id, "unavailable")
        try:
            async with asyncio.timeout_at(deadline):
                return await self._load_image(
                    message.image_file, message.message_id, used_bytes=used_bytes
                )
        except Exception:
            # A stale file must not suppress an otherwise answerable request.
            return ImageOmission(message.message_id, "unavailable")
