import asyncio
import logging
from collections.abc import Sequence
from contextlib import aclosing
from dataclasses import replace
from time import monotonic, perf_counter
from typing import Literal

from aiogram import Bot
from aiogram.types import Message
from cachetools import TTLCache

from dashulya.bot.media_frames import FrameBudgetError, FrameDecodeError, first_frame
from dashulya.bot.messages import conversation_key, image_file
from dashulya.domain.memory import MemoryMessage, OriginalMessage
from dashulya.domain.messages import ImageAttachment, ImageFile, ImageOmission
from dashulya.observability import exception_fields

logger = logging.getLogger(__name__)

ImageMediaType = Literal["image/jpeg", "image/png", "image/webp"]
MEDIA_TIMEOUT_SECONDS = 10
NETWORK_TIMEOUT_SECONDS = 5
CACHE_BYTES = 64 * 1024 * 1024
CACHE_TTL_SECONDS = 300
FAILURE_CACHE_SIZE = 256
FAILURE_TTL_SECONDS = 30


class ImageInputError(ValueError):
    """A safe explanation of an unsupported or oversized image input."""


class ImageTooLargeError(ImageInputError):
    """Image bytes cannot fit within this request's remaining budget."""


def _check_size(size: int, limit: int) -> None:
    if size > limit:
        raise ImageTooLargeError


def _media_type(data: bytes) -> ImageMediaType:
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return "image/webp"
    error_message = "Пришли изображение в формате JPEG, PNG или WebP."
    raise ImageInputError(error_message)


async def _decode(data: bytes, budget: "_ImageBudget") -> bytes:
    budget.stage = "decode"
    started = perf_counter()
    logger.info(
        "Media stage started", extra={**budget.fields, "media_stage": budget.stage}
    )
    assert budget.capacity is not None
    frame = await asyncio.to_thread(first_frame, data, max_bytes=budget.growth_capacity)
    await budget.check(len(frame))
    logger.info(
        "Media stage completed",
        extra={
            **budget.fields,
            "media_stage": budget.stage,
            "duration_ms": round((perf_counter() - started) * 1000),
            "image_bytes": len(frame),
        },
    )
    return frame


class _BatchBudget:
    """Admit ready inputs and reclaim lower-priority completed inputs when needed."""

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.available = limit
        self.items: dict[int, _ImageBudget] = {}
        self._active: set[int] = set()
        self._changed = asyncio.Condition()

    def _reclaim(self, index: int, required: int) -> None:
        for item in reversed(self.items.values()):
            if self.available >= required:
                break
            if (
                item.index > index
                and item.index not in self._active
                and isinstance(item.result, ImageAttachment)
            ):
                self.available += len(item.result.data)
                item.result = ImageOmission(item.result.message_id)
                logger.warning(
                    "Media item omitted for higher priority input",
                    extra={
                        **item.fields,
                        "media_stage": "budget",
                        "image_bytes": 0,
                        "outcome": "too_large",
                    },
                )

    async def reserve(self, index: int, expected: int | None) -> int:
        async with self._changed:
            required = 1 if expected is None else expected
            self._reclaim(index, required)
            while required > self.available and self._active:
                await self._changed.wait()
                self._reclaim(index, required)
            capacity = (
                self.available
                if expected is None
                else expected
                if expected <= self.available
                else 0
            )
            self.available -= capacity
            self._active.add(index)
            return capacity

    async def grow(self, index: int, size: int) -> None:
        async with self._changed:
            self._reclaim(index, size)
            while size > self.available:
                if any(other < index for other in self._active) or self._active == {
                    index
                }:
                    raise ImageTooLargeError
                await self._changed.wait()
                self._reclaim(index, size)
            self.available -= size

    async def release(
        self, item: "_ImageBudget", result: ImageAttachment | ImageOmission
    ) -> None:
        async with self._changed:
            assert item.capacity is not None
            retained = len(result.data) if isinstance(result, ImageAttachment) else 0
            self.available += item.capacity - retained
            self._active.remove(item.index)
            item.result = result
            self._changed.notify_all()


class _ImageBudget:
    def __init__(
        self, batch: _BatchBudget, index: int, fields: dict[str, object]
    ) -> None:
        self.batch = batch
        self.index = index
        self.fields = fields
        self.capacity: int | None = None
        self.result: ImageAttachment | ImageOmission | None = None
        self.stage = "budget"
        batch.items[index] = self

    async def reserve(self, expected: int | None) -> None:
        if self.capacity is None:
            self.stage = "budget"
            self.capacity = await self.batch.reserve(self.index, expected)

    @property
    def growth_capacity(self) -> int:
        assert self.capacity is not None
        return (
            self.capacity
            + self.batch.available
            + sum(
                len(item.result.data)
                for item in self.batch.items.values()
                if item.index > self.index and isinstance(item.result, ImageAttachment)
            )
        )

    async def check(self, size: int) -> None:
        assert self.capacity is not None
        if size > self.capacity:
            await self.batch.grow(self.index, size - self.capacity)
            self.capacity = size

    async def finish(self, result: ImageAttachment | ImageOmission) -> None:
        if self.capacity is None:
            self.capacity = await self.batch.reserve(self.index, 0)
        assert self.capacity is not None
        await self.batch.release(self, result)


class TelegramImages:
    def __init__(self, bot: Bot, *, max_bytes: int) -> None:
        self._bot = bot
        self._max_bytes = max_bytes
        self._cache: TTLCache[ImageFile, ImageAttachment] = TTLCache(
            maxsize=CACHE_BYTES,
            ttl=CACHE_TTL_SECONDS,
            timer=monotonic,
            getsizeof=lambda image: len(image.data),
        )
        self._failures: TTLCache[ImageFile, ImageOmission] = TTLCache(
            maxsize=FAILURE_CACHE_SIZE, ttl=FAILURE_TTL_SECONDS, timer=monotonic
        )

    async def _read(self, file: ImageFile, *, budget: _ImageBudget) -> bytes:
        _check_size(1, budget.batch.limit)
        _check_size(file.file_size or 0, budget.batch.limit)
        budget.stage = "metadata"
        started = perf_counter()
        logger.info(
            "Media stage started", extra={**budget.fields, "media_stage": budget.stage}
        )
        async with asyncio.timeout(NETWORK_TIMEOUT_SECONDS):
            metadata = await self._bot.get_file(file.file_id)
        logger.info(
            "Media stage completed",
            extra={
                **budget.fields,
                "media_stage": budget.stage,
                "duration_ms": round((perf_counter() - started) * 1000),
            },
        )
        _check_size(metadata.file_size or 0, budget.batch.limit)
        if metadata.file_path is None:
            error_message = "Не удалось скачать изображение. Пришли его ещё раз."  # ruff: ignore[ambiguous-unicode-character-string]
            raise FileNotFoundError(error_message)
        expected = (
            None
            if file.preview_kind == "animation"
            else metadata.file_size or file.file_size or None
        )
        await budget.reserve(expected)
        await budget.check(1)
        await budget.check(metadata.file_size or file.file_size or 0)
        # The credential-bearing URL stays inside the Telegram adapter.
        url = self._bot.session.api.file_url(self._bot.token, metadata.file_path)
        budget.stage = "download"
        started = perf_counter()
        logger.info(
            "Media stage started", extra={**budget.fields, "media_stage": budget.stage}
        )
        data = bytearray()
        async with (
            asyncio.timeout(NETWORK_TIMEOUT_SECONDS),
            aclosing(self._bot.session.stream_content(url, chunk_size=65_536)) as stream,
        ):
            async for chunk in stream:
                await budget.check(len(data) + len(chunk))
                data.extend(chunk)
        logger.info(
            "Media stage completed",
            extra={
                **budget.fields,
                "media_stage": budget.stage,
                "duration_ms": round((perf_counter() - started) * 1000),
                "image_bytes": len(data),
            },
        )
        return bytes(data)

    async def _animation(
        self, file: ImageFile, message_id: int, *, budget: _ImageBudget
    ) -> ImageAttachment | ImageOmission:
        reason: Literal["too_large", "decode_failed"] = "decode_failed"
        try:
            frame = await _decode(await self._read(file, budget=budget), budget)
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
        logger.warning(
            "Media preview fallback",
            extra={**budget.fields, "media_stage": budget.stage, "outcome": reason},
        )
        thumbnail = self._thumbnail(file)
        if thumbnail is None:
            return ImageOmission(message_id, reason)
        try:
            preview = await self._read(thumbnail, budget=budget)
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
        self, file: ImageFile, message_id: int, *, budget: _ImageBudget
    ) -> ImageAttachment | ImageOmission:
        if file.preview_kind == "animation":
            return await self._animation(file, message_id, budget=budget)
        source = file
        if file.preview_kind == "sticker":
            thumbnail = self._thumbnail(file)
            if thumbnail is None:
                return ImageOmission(message_id, "preview_unavailable")
            source = thumbnail
        data = await self._read(source, budget=budget)
        return ImageAttachment(
            data, _media_type(data), message_id, preview_kind=file.preview_kind
        )

    async def _resolve_image(
        self, file: ImageFile | None, message_id: int, budget: _ImageBudget
    ) -> tuple[ImageAttachment | ImageOmission, str]:
        if file is None:
            return ImageOmission(message_id, "unsupported"), "miss"
        cached = self._cache.get(file)
        if cached is not None:
            await budget.reserve(len(cached.data))
            await budget.check(len(cached.data))
            return replace(cached, message_id=message_id), "hit"
        failure = self._failures.get(file)
        if failure is not None:
            return replace(failure, message_id=message_id), "negative_hit"
        return await self._load_image(file, message_id, budget=budget), "miss"

    async def _load_one(
        self,
        file: ImageFile | None,
        message_id: int,
        *,
        budget: _ImageBudget,
        deadline: float,
    ) -> _ImageBudget:
        started = perf_counter()
        cache_status = "miss"
        result: ImageAttachment | ImageOmission = ImageOmission(message_id, "unavailable")
        logger.info("Media item started", extra=budget.fields)
        try:
            async with asyncio.timeout_at(deadline):
                result, cache_status = await self._resolve_image(file, message_id, budget)
        except ImageTooLargeError:
            result = ImageOmission(message_id)
        except ImageInputError as error:
            result = ImageOmission(message_id, "unsupported")
            logger.warning(
                "Media input rejected",
                extra={
                    **budget.fields,
                    **exception_fields(error),
                    "media_stage": budget.stage,
                },
            )
        except Exception as error:
            logger.exception(
                "Media item failed",
                extra={
                    **budget.fields,
                    **exception_fields(error),
                    "media_stage": budget.stage,
                    "duration_ms": round((perf_counter() - started) * 1000),
                },
            )
        except asyncio.CancelledError:
            logger.info("Media item cancelled", extra=budget.fields)
            raise
        finally:
            await budget.finish(result)
        if file is not None and cache_status == "miss":
            if isinstance(result, ImageAttachment):
                # A thumbnail chosen only because this batch was full must not
                # replace a future request's opportunity to decode the original.
                if (result.preview_kind != "animation" or result.first_frame) and len(
                    result.data
                ) <= self._cache.maxsize:
                    self._cache[file] = result
            elif result.reason != "too_large" and budget.stage != "budget":
                self._failures[file] = result
        extra = {
            **budget.fields,
            "cache_status": cache_status,
            "duration_ms": round((perf_counter() - started) * 1000),
            "image_bytes": len(result.data) if isinstance(result, ImageAttachment) else 0,
            "outcome": "loaded" if isinstance(result, ImageAttachment) else result.reason,
        }
        if isinstance(result, ImageAttachment):
            logger.info("Media item completed", extra=extra)
        else:
            logger.warning("Media item completed", extra=extra)
        return budget

    async def _load_many(
        self,
        sources: Sequence[tuple[int, ImageFile | None]],
        *,
        chat_id: int,
        request_message_id: int | None,
        phase: str,
        images: tuple[ImageAttachment | ImageOmission, ...] = (),
        max_bytes: int | None = None,
    ) -> tuple[ImageAttachment | ImageOmission, ...]:
        started = perf_counter()
        used = sum(
            len(image.data) for image in images if isinstance(image, ImageAttachment)
        )
        limit = self._max_bytes if max_bytes is None else min(self._max_bytes, max_bytes)
        _check_size(used, limit)
        batch = _BatchBudget(limit - used)
        fields: dict[str, object] = {
            "chat_id": chat_id,
            "request_message_id": request_message_id,
            "media_phase": phase,
            "media_count": len(sources),
        }
        logger.info("Media batch started", extra=fields)
        deadline = asyncio.get_running_loop().time() + MEDIA_TIMEOUT_SECONDS
        tasks = [
            asyncio.create_task(
                self._load_one(
                    file,
                    message_id,
                    budget=_ImageBudget(
                        batch,
                        index,
                        {
                            **fields,
                            "message_id": message_id,
                            "media_kind": file.preview_kind or "image"
                            if file is not None
                            else "unsupported",
                        },
                    ),
                    deadline=deadline,
                )
            )
            for index, (message_id, file) in enumerate(sources)
        ]
        try:
            loaded = await asyncio.gather(*tasks)
        except BaseException:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            logger.info("Media batch cancelled", extra=fields)
            raise
        finally:
            batch.items.clear()
        resolved = []
        for item in loaded:
            assert item.result is not None
            resolved.append(item.result)
        result = (*images, *resolved)
        logger.info(
            "Media batch completed",
            extra={
                **fields,
                "duration_ms": round((perf_counter() - started) * 1000),
                "loaded_images": sum(
                    isinstance(image, ImageAttachment) for image in result
                ),
                "omitted_images": sum(
                    isinstance(image, ImageOmission) for image in result
                ),
                "image_bytes": sum(
                    len(image.data)
                    for image in result
                    if isinstance(image, ImageAttachment)
                ),
            },
        )
        return result

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
        return await self._load_many(
            [(source.message_id, image_file(source)) for source in sources],
            chat_id=message.chat.id,
            request_message_id=message.message_id,
            phase="request",
        )

    async def load_context(
        self,
        messages: tuple[MemoryMessage | OriginalMessage, ...],
        *,
        images: tuple[ImageAttachment | ImageOmission, ...] = (),
        request_message_id: int | None = None,
        max_bytes: int | None = None,
    ) -> tuple[ImageAttachment | ImageOmission, ...]:
        """Load selected media concurrently, preserving input and newest-first priority."""
        allowed = {message.message_id for message in messages}
        supplied = tuple(image for image in images if image.message_id in allowed)
        supplied_ids = {image.message_id for image in supplied}
        request = next(
            (message for message in messages if message.message_id == request_message_id),
            None,
        )
        parent_id = request.reply_to_id if request is not None else None
        pending = [
            (message.message_id, message.image_file)
            for message in reversed(messages)
            if message.image_file is not None and message.message_id not in supplied_ids
        ]
        pending.sort(
            key=lambda item: (
                0 if item[0] == request_message_id else 1 if item[0] == parent_id else 2
            )
        )
        return await self._load_many(
            pending,
            chat_id=(messages[0].chat_id or 0) if messages else 0,
            request_message_id=request_message_id,
            phase="context",
            images=supplied,
            max_bytes=max_bytes,
        )
