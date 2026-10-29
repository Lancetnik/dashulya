import asyncio
from collections.abc import AsyncGenerator
from contextlib import AsyncExitStack
from dataclasses import replace
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, Mock, call, patch

import pytest
from aiogram import Bot
from aiogram.types import File, Message

from dashulya.bot.images import ImageInputError, TelegramImages
from dashulya.bot.media_frames import first_frame
from dashulya.bot.messages import image_file
from dashulya.domain.memory import MemoryMessage
from dashulya.domain.messages import ImageAttachment, ImageFile, ImageOmission
from tests.media_fixtures import animation_bytes

PNG = b"\x89PNG\r\n\x1a\n" + b"image-data"
STICKER: dict[str, Any] = {
    "file_id": "sticker",
    "file_unique_id": "unique-sticker",
    "type": "regular",
    "width": 512,
    "height": 512,
    "is_animated": False,
    "is_video": False,
}
THUMBNAIL = {
    "file_id": "thumbnail",
    "file_unique_id": "unique-thumb",
    "width": 100,
    "height": 100,
}
ANIMATION: dict[str, Any] = {
    "file_id": "animation",
    "file_unique_id": "unique-animation",
    "width": 320,
    "height": 240,
    "duration": 3,
    "thumbnail": THUMBNAIL,
}


def image_message(**fields: Any) -> Message:
    return Message.model_validate({
        "message_id": 1,
        "date": 1700000000,
        "chat": {"id": 42, "type": "private"},
        "photo": [
            {"file_id": "photo", "file_unique_id": "unique", "width": 2, "height": 2}
        ],
        **fields,
    })


@pytest.mark.usefixtures("_setup")
class TestImage:
    @classmethod
    def saved_media(cls, message: Message) -> MemoryMessage:
        return replace(
            cls.saved_image(message.message_id),
            image_file=image_file(message),
            media_type=message.content_type,
        )

    @staticmethod
    def saved_image(number: int, *, size: int | None = None) -> MemoryMessage:
        return MemoryMessage(
            chat_id=-123,
            message_id=number,
            text="",
            sent_at=datetime(2026, 10, 25, tzinfo=UTC),
            author_id=99,
            author="Boris",
            chat_title="Group",
            media_type="photo",
            image_file=ImageFile(f"photo-{number}", size),
        )

    @pytest.fixture
    async def _setup(self, cleanup: AsyncExitStack) -> None:
        self.cleanup = cleanup
        self.bot = Bot("123:TEST")
        self.cleanup.push_async_callback(self.bot.session.close)
        self.get_file = AsyncMock(
            return_value=File(
                file_id="photo", file_unique_id="unique", file_path="photos/image.png"
            )
        )
        self.chunks = [PNG]
        self.read_chunks = 0
        self.closed = False

        async def stream(*args: Any, **kwargs: Any) -> AsyncGenerator[bytes]:
            try:
                for chunk in self.chunks:
                    self.read_chunks += 1
                    yield chunk
            finally:
                self.closed = True

        self.stream = Mock(side_effect=stream)
        for target, name, value in (
            (self.bot, "get_file", self.get_file),
            (self.bot.session, "stream_content", self.stream),
        ):
            patcher = patch.object(target, name, value)
            patcher.start()
            self.cleanup.callback(patcher.stop)
        self.images = TelegramImages(self.bot, max_bytes=len(PNG))

    async def test_image_at_limit_stays_in_memory_and_uses_content_mime(self) -> None:
        images = await self.images.load(image_message())
        assert len(images) == 1
        assert isinstance(images[0], ImageAttachment)
        assert images[0].data == PNG
        assert images[0].media_type == "image/png"
        assert images[0].message_id == 1
        self.stream.assert_called_once()
        assert self.stream.call_args.args[0] == (
            "https://api.telegram.org/file/bot123:TEST/photos/image.png"
        )
        assert self.closed
        assert "image-data" not in repr(images)

    async def test_context_images_share_budget_and_prefer_recent_sources(self) -> None:
        images = await self.images.load_context((
            self.saved_image(1),
            self.saved_image(2),
        ))
        assert images == (ImageAttachment(PNG, "image/png", 2), ImageOmission(1))
        self.get_file.assert_awaited_once_with("photo-2")

    async def test_current_image_is_not_downloaded_twice_or_given_a_separate_budget(
        self,
    ) -> None:
        current = ImageAttachment(PNG, "image/png", 2)
        images = await self.images.load_context(
            (self.saved_image(1), self.saved_image(2)), images=(current,)
        )
        assert images == (current, ImageOmission(1))
        self.get_file.assert_not_awaited()

    async def test_context_filters_supplied_ids_before_loading_and_skips_large_files(
        self,
    ) -> None:
        images = await self.images.load_context(
            (self.saved_image(1, size=len(PNG) + 1),),
            images=(ImageAttachment(PNG, "image/png", 999),),
        )
        assert images == (ImageOmission(1),)
        self.get_file.assert_not_awaited()

    async def test_unavailable_context_image_does_not_hide_other_images_or_credentials(
        self,
    ) -> None:
        self.get_file.side_effect = [
            RuntimeError(self.bot.token),
            self.get_file.return_value,
        ]
        images = await self.images.load_context((
            self.saved_image(1),
            self.saved_image(2),
        ))
        assert images == (
            ImageOmission(2, "unavailable"),
            ImageAttachment(PNG, "image/png", 1),
        )
        assert self.bot.token not in repr(images)

    async def test_context_cancellation_propagates(self) -> None:
        self.get_file.side_effect = asyncio.CancelledError
        with pytest.raises(asyncio.CancelledError):
            await self.images.load_context((self.saved_image(1),))

    async def test_oversized_update_and_getfile_size_skip_download(self) -> None:
        oversized = image_message(
            photo=[
                {
                    "file_id": "photo",
                    "file_unique_id": "unique",
                    "width": 2,
                    "height": 2,
                    "file_size": len(PNG) + 1,
                }
            ]
        )
        assert await self.images.load(oversized) == (ImageOmission(1),)
        self.get_file.assert_not_awaited()
        self.get_file.return_value = self.get_file.return_value.model_copy(
            update={"file_size": len(PNG) + 1}
        )
        assert await self.images.load(image_message()) == (ImageOmission(1),)
        self.stream.assert_not_called()

    async def test_unknown_or_false_size_cannot_bypass_stream_limit(
        self, subtests: pytest.Subtests
    ) -> None:
        self.chunks = [PNG, b"overflow", b"must-not-be-read"]
        for size in (None, 1):
            for partial in (False, True):
                with subtests.test(size=size, partial=partial):
                    self.read_chunks = 0
                    self.closed = False
                    self.get_file.return_value = self.get_file.return_value.model_copy(
                        update={"file_size": size}
                    )
                    if partial:
                        images = TelegramImages(self.bot, max_bytes=2 * len(PNG) - 1)
                        current = ImageAttachment(PNG, "image/png", 2)
                        result = await images.load_context(
                            (self.saved_image(1), self.saved_image(2)), images=(current,)
                        )
                        assert result == (current, ImageOmission(1))
                    else:
                        assert await self.images.load(image_message()) == (
                            ImageOmission(1),
                        )
                    assert self.read_chunks == (1 if partial else 2)
                    assert self.closed

    async def test_image_document_mime_is_derived_from_bytes(self) -> None:
        message = image_message(
            photo=None,
            document={
                "file_id": "doc",
                "file_unique_id": "doc",
                "file_name": "picture.jpg",
                "mime_type": "image/jpeg",
            },
        )
        images = await self.images.load(message)
        assert isinstance(images[0], ImageAttachment)
        assert images[0].media_type == "image/png"
        self.get_file.assert_awaited_once_with("doc")

    async def test_rejects_unsupported_documents_and_nonimage_bytes(self) -> None:
        document = image_message(
            photo=None,
            document={
                "file_id": "doc",
                "file_unique_id": "doc",
                "file_name": "report.pdf",
                "mime_type": "application/pdf",
            },
        )
        with pytest.raises(ImageInputError):
            await self.images.load(document)
        self.get_file.assert_not_awaited()
        self.chunks = [b"not an image"]
        with pytest.raises(ImageInputError):
            await self.images.load(image_message())

    async def test_cancellation_closes_the_download_stream(self) -> None:
        started = asyncio.Event()

        async def waiting_stream(*args: Any, **kwargs: Any) -> AsyncGenerator[bytes]:
            try:
                started.set()
                await asyncio.Event().wait()
                yield PNG
            finally:
                self.closed = True

        self.stream.side_effect = waiting_stream
        task = asyncio.create_task(self.images.load(image_message()))
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert self.closed

    async def test_stalled_current_and_saved_downloads_end_at_the_deadline(
        self, subtests: pytest.Subtests
    ) -> None:
        for saved in (False, True):
            with subtests.test(saved=saved):
                await self.check_stalled_download(saved)

    async def check_stalled_download(self, saved: bool) -> None:
        started = asyncio.Event()
        self.closed = False
        self.get_file.reset_mock()

        async def stalled(*args: Any, **kwargs: Any) -> AsyncGenerator[bytes]:
            try:
                started.set()
                await asyncio.Event().wait()
                yield PNG
            finally:
                self.closed = True

        self.stream.side_effect = stalled
        loop = asyncio.get_running_loop()
        real_time = loop.time
        elapsed = 0.0

        def time() -> float:
            return real_time() + elapsed

        with patch.object(loop, "time", time):
            task = asyncio.create_task(
                self.images.load_context((self.saved_image(1), self.saved_image(2)))
                if saved
                else self.images.load(image_message())
            )
            try:
                await asyncio.wait_for(started.wait(), timeout=1)
                elapsed = 30.0
                # Let the due timer cancel its task, then let that task
                # close the download and complete its public result.
                for _ in range(4):
                    await asyncio.sleep(0)
                assert task.done(), "Download exceeded its 30-second deadline"
                if saved:
                    assert await task == (
                        ImageOmission(2, "unavailable"),
                        ImageOmission(1, "unavailable"),
                    )
                else:
                    with pytest.raises(TimeoutError):
                        await task
                assert self.closed
                self.get_file.assert_awaited_once_with("photo-2" if saved else "photo")
            finally:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

    async def test_all_sticker_types_and_formats_use_supported_image_bytes(
        self, subtests: pytest.Subtests
    ) -> None:
        self.chunks = [b"RIFF\x08\x00\x00\x00WEBPdata"]
        for sticker_type in ("regular", "mask", "custom_emoji"):
            for sticker_format in ("static", "animated", "video"):
                with subtests.test(
                    sticker_type=sticker_type, sticker_format=sticker_format
                ):
                    sticker = {
                        **STICKER,
                        "type": sticker_type,
                        "is_animated": sticker_format == "animated",
                        "is_video": sticker_format == "video",
                        "thumbnail": THUMBNAIL,
                        "file_size": 20_000_001 if sticker_format != "static" else None,
                    }
                    message = image_message(photo=None, sticker=sticker)
                    for saved in (False, True):
                        with subtests.test(saved=saved):
                            self.get_file.reset_mock()
                            images = (
                                await self.images.load_context((
                                    self.saved_media(message),
                                ))
                                if saved
                                else await self.images.load(message)
                            )
                            assert isinstance(images[0], ImageAttachment)
                            assert images[0].data == self.chunks[0]
                            assert images[0].media_type == "image/webp"
                            assert images[0].preview_kind == (
                                "sticker" if sticker_format != "static" else None
                            )
                            self.get_file.assert_awaited_once_with(
                                "sticker" if sticker_format == "static" else "thumbnail"
                            )

    async def test_saved_oversized_animations_and_stickers_use_thumbnail_size(
        self,
        subtests: pytest.Subtests,
    ) -> None:
        for fields in (
            {"animation": {**ANIMATION, "file_size": 20_000_001}},
            {
                "sticker": {
                    **STICKER,
                    "is_video": True,
                    "file_size": 20_000_001,
                    "thumbnail": THUMBNAIL,
                }
            },
        ):
            with subtests.test(fields=fields):
                self.get_file.reset_mock()
                images = await self.images.load_context((
                    self.saved_media(image_message(photo=None, **fields)),
                ))
                assert isinstance(images[0], ImageAttachment)
                self.get_file.assert_awaited_once_with("thumbnail")

    async def test_saved_animations_and_gif_documents_decode_the_original_first_frame(
        self,
        subtests: pytest.Subtests,
    ) -> None:
        self.chunks = [animation_bytes()]
        self.images = TelegramImages(self.bot, max_bytes=100_000)
        for fields, expected_id in (
            (
                {
                    "animation": ANIMATION,
                    "document": {
                        "file_id": "legacy",
                        "file_unique_id": "legacy",
                        "mime_type": "image/gif",
                    },
                },
                "animation",
            ),
            (
                {
                    "document": {
                        "file_id": "gif",
                        "file_unique_id": "gif",
                        "file_name": "picture.GIF",
                    }
                },
                "gif",
            ),
        ):
            with subtests.test(fields=fields):
                self.get_file.reset_mock()
                images = await self.images.load_context((
                    self.saved_media(image_message(photo=None, **fields)),
                ))
                assert isinstance(images[0], ImageAttachment)
                assert images[0].media_type == "image/jpeg"
                assert images[0].first_frame
                assert images[0].preview_kind == "animation"
                self.get_file.assert_awaited_once_with(expected_id)

    async def test_saved_undecodable_animation_uses_thumbnail_or_omission(self) -> None:
        for thumbnail in (THUMBNAIL, None):
            images = await self.images.load_context((
                self.saved_media(
                    image_message(
                        photo=None, animation={**ANIMATION, "thumbnail": thumbnail}
                    )
                ),
            ))
            if thumbnail:
                assert isinstance(images[0], ImageAttachment)
                assert images[0].data == PNG
                assert not images[0].first_frame
                assert images[0].preview_kind == "animation"
            else:
                assert images == (ImageOmission(1, "decode_failed"),)

    async def test_animated_sticker_without_thumbnail_produces_note_without_download(
        self,
        subtests: pytest.Subtests,
    ) -> None:
        for format_flag in ("is_animated", "is_video"):
            message = image_message(photo=None, sticker={**STICKER, format_flag: True})
            for saved in (False, True):
                with subtests.test(format_flag=format_flag, saved=saved):
                    images = (
                        await self.images.load_context((self.saved_media(message),))
                        if saved
                        else await self.images.load(message)
                    )
                    assert images == (ImageOmission(1, "preview_unavailable"),)
        self.get_file.assert_not_awaited()
        self.stream.assert_not_called()

    async def test_oversized_sticker_original_and_thumbnail_are_not_downloaded(
        self,
    ) -> None:
        for sticker in (
            {**STICKER, "file_size": len(PNG) + 1},
            {
                **STICKER,
                "is_video": True,
                "thumbnail": {**THUMBNAIL, "file_size": len(PNG) + 1},
            },
        ):
            images = await self.images.load(image_message(photo=None, sticker=sticker))
            assert images == (ImageOmission(1),)
        self.get_file.assert_not_awaited()
        self.stream.assert_not_called()

    async def test_reply_sticker_preview_and_current_photo_share_byte_budget(
        self,
    ) -> None:
        parent = image_message(
            photo=None, sticker={**STICKER, "is_animated": True, "thumbnail": THUMBNAIL}
        )
        images = await self.images.load(
            image_message(message_id=2, reply_to_message=parent.model_dump())
        )
        assert isinstance(images[0], ImageAttachment)
        assert images[1] == ImageOmission(1)
        self.get_file.assert_awaited_once_with("photo")

    async def test_oversized_animation_falls_back_to_thumbnail_ignoring_legacy_document(
        self,
        subtests: pytest.Subtests,
    ) -> None:
        for mime in ("image/gif", "video/mp4"):
            with subtests.test(mime=mime):
                self.get_file.reset_mock()
                images = await self.images.load(
                    image_message(
                        photo=None,
                        animation={
                            **ANIMATION,
                            "mime_type": mime,
                            "file_size": 20_000_001,
                        },
                        document={
                            "file_id": "legacy-document",
                            "file_unique_id": "legacy",
                            "mime_type": mime,
                            "file_size": 20_000_001,
                        },
                    )
                )
                assert isinstance(images[0], ImageAttachment)
                assert images[0].preview_kind == "animation"
                self.get_file.assert_awaited_once_with("thumbnail")

    async def test_animation_without_thumbnail_attempts_the_original(self) -> None:
        self.chunks = [animation_bytes()]
        self.images = TelegramImages(self.bot, max_bytes=100_000)
        images = await self.images.load(
            image_message(
                photo=None,
                animation={**ANIMATION, "thumbnail": None},
                document={
                    "file_id": "legacy",
                    "file_unique_id": "legacy",
                    "mime_type": "image/gif",
                },
            )
        )
        self.get_file.assert_awaited_once_with("animation")
        assert len(images) == 1
        assert isinstance(images[0], ImageAttachment)
        assert images[0].message_id == 1
        assert images[0].first_frame

        # The tiny GIF fits in RAM even when its JPEG cannot fit beside another image.
        clip = bytes.fromhex(
            "47494638396101000100800000000000ffffff2c00000000010001000002024401003b"
        )
        frame = first_frame(clip, max_bytes=100_000)
        assert len(clip) < len(frame) - 1
        self.chunks = [clip]
        current = ImageAttachment(PNG, "image/png", 2)
        source = self.saved_media(
            image_message(photo=None, animation={**ANIMATION, "thumbnail": None})
        )
        for remaining in (len(frame), len(frame) - 1):
            loader = TelegramImages(self.bot, max_bytes=len(PNG) + remaining)
            loaded = await loader.load_context(
                (source, self.saved_image(2)), images=(current,)
            )
            expected = (
                ImageAttachment(frame, "image/jpeg", 1, "animation", first_frame=True)
                if remaining == len(frame)
                else ImageOmission(1)
            )
            assert loaded == (current, expected)

    async def test_undecodable_gif_document_uses_preview_or_explicit_failure_note(
        self,
    ) -> None:
        document = {
            "file_id": "gif",
            "file_unique_id": "gif",
            "file_name": "picture.GIF",
            "mime_type": "application/octet-stream",
            "thumbnail": THUMBNAIL,
        }
        images = await self.images.load(image_message(photo=None, document=document))
        assert len(images) == 1
        assert isinstance(images[0], ImageAttachment)
        assert images[0].message_id == 1
        assert images[0].data == PNG
        assert images[0].media_type == "image/png"
        assert images[0].preview_kind == "animation"
        assert self.get_file.await_args_list == [call("gif"), call("thumbnail")]
        assert not images[0].first_frame
        self.get_file.reset_mock()
        assert await self.images.load(
            image_message(photo=None, document={**document, "thumbnail": None})
        ) == (ImageOmission(1, "decode_failed"),)
        self.get_file.assert_awaited_once_with("gif")

    async def test_gif_document_decodes_original_without_a_thumbnail(self) -> None:
        self.chunks = [animation_bytes(format="gif")]
        self.images = TelegramImages(self.bot, max_bytes=100_000)
        images = await self.images.load(
            image_message(
                photo=None,
                document={
                    "file_id": "gif",
                    "file_unique_id": "gif",
                    "mime_type": "image/gif",
                },
            )
        )
        assert len(images) == 1
        assert isinstance(images[0], ImageAttachment)
        assert images[0].first_frame
        self.get_file.assert_awaited_once_with("gif")

    async def test_oversized_animation_preview_skips_download(self) -> None:
        images = await self.images.load(
            image_message(
                photo=None,
                animation={
                    **ANIMATION,
                    "file_size": len(PNG) + 1,
                    "thumbnail": {**THUMBNAIL, "file_size": len(PNG) + 1},
                },
            )
        )
        assert images == (ImageOmission(1),)
        self.get_file.assert_not_awaited()
        self.stream.assert_not_called()

    async def test_reply_animation_uses_first_original_frame_from_same_chat(self) -> None:
        self.chunks = [animation_bytes()]
        self.images = TelegramImages(self.bot, max_bytes=100_000)
        parent = image_message(photo=None, animation=ANIMATION).model_dump()
        images = await self.images.load(
            image_message(
                photo=None, text="Explain", message_id=2, reply_to_message=parent
            )
        )
        assert isinstance(images[0], ImageAttachment)
        assert images[0].message_id == 1
        assert images[0].preview_kind == "animation"
        assert images[0].first_frame
        self.get_file.assert_awaited_once_with("animation")
        self.get_file.reset_mock()
        for changed in (
            {"chat": {"id": 43, "type": "private"}},
            {"message_thread_id": 7},
        ):
            assert (
                await self.images.load(
                    image_message(
                        photo=None, text="Explain", reply_to_message={**parent, **changed}
                    )
                )
                == ()
            )
        self.get_file.assert_not_awaited()

    async def test_ordinary_group_reply_threads_share_image_access(self) -> None:
        chat = {"id": -123, "type": "supergroup", "title": "Group"}
        parent = image_message(chat=chat, message_thread_id=7).model_dump()
        images = await self.images.load(
            image_message(
                chat=chat,
                photo=None,
                message_id=2,
                message_thread_id=8,
                reply_to_message=parent,
            )
        )
        assert isinstance(images[0], ImageAttachment)

    async def test_forum_topics_isolate_images_and_general_topic_is_normalized(
        self,
    ) -> None:
        chat = {"id": -123, "type": "supergroup", "title": "Forum", "is_forum": True}
        parent = image_message(
            chat=chat, message_thread_id=7, is_topic_message=True
        ).model_dump()
        images = await self.images.load(
            image_message(
                chat=chat,
                photo=None,
                message_id=2,
                message_thread_id=8,
                is_topic_message=True,
                reply_to_message=parent,
            )
        )
        assert images == ()
        self.get_file.assert_not_awaited()
        general = image_message(chat=chat, is_topic_message=True).model_dump()
        images = await self.images.load(
            image_message(
                chat=chat,
                photo=None,
                message_id=2,
                message_thread_id=1,
                is_topic_message=True,
                reply_to_message=general,
            )
        )
        assert isinstance(images[0], ImageAttachment)
