import asyncio
import unittest
from collections.abc import AsyncGenerator
from typing import Any
from unittest.mock import AsyncMock, Mock, call, patch

from aiogram import Bot
from aiogram.types import File, Message

from dashulya.bot.images import ImageInputError, TelegramImages
from dashulya.domain.messages import ImageAttachment, ImageOmission
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


class ImageTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.bot = Bot("123:TEST")
        self.addAsyncCleanup(self.bot.session.close)
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
            self.addCleanup(patcher.stop)
        self.images = TelegramImages(self.bot, max_bytes=len(PNG))

    async def test_image_at_limit_stays_in_memory_and_uses_content_mime(self) -> None:
        images = await self.images.load(image_message())
        self.assertEqual(len(images), 1)
        assert isinstance(images[0], ImageAttachment)
        self.assertEqual(images[0].data, PNG)
        self.assertEqual(images[0].media_type, "image/png")
        self.assertEqual(images[0].message_id, 1)
        self.assertTrue(self.closed)
        self.assertNotIn("image-data", repr(images))

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
        self.assertEqual(await self.images.load(oversized), (ImageOmission(1),))
        self.get_file.assert_not_awaited()
        self.get_file.return_value = self.get_file.return_value.model_copy(
            update={"file_size": len(PNG) + 1}
        )
        self.assertEqual(await self.images.load(image_message()), (ImageOmission(1),))
        self.stream.assert_not_called()

    async def test_unknown_or_false_size_cannot_bypass_stream_limit(self) -> None:
        self.chunks = [PNG, b"overflow", b"must-not-be-read"]
        self.get_file.return_value = self.get_file.return_value.model_copy(
            update={"file_size": 1}
        )
        self.assertEqual(await self.images.load(image_message()), (ImageOmission(1),))
        self.assertEqual(self.read_chunks, 2)
        self.assertTrue(self.closed)

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
        self.assertEqual(images[0].media_type, "image/png")
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
        with self.assertRaises(ImageInputError):
            await self.images.load(document)
        self.get_file.assert_not_awaited()
        self.chunks = [b"not an image"]
        with self.assertRaises(ImageInputError):
            await self.images.load(image_message())

    async def test_reply_image_requires_same_chat_and_topic(self) -> None:
        parent = image_message().model_dump()
        current = image_message(
            message_id=2, photo=None, text="What is this?", reply_to_message=parent
        )
        images = await self.images.load(current)
        self.assertEqual(images[0].message_id, 1)
        for changed in (
            {"chat": {"id": 43, "type": "private"}},
            {"message_thread_id": 7},
        ):
            self.get_file.reset_mock()
            current = image_message(
                photo=None, text="Question", reply_to_message={**parent, **changed}
            )
            self.assertEqual(await self.images.load(current), ())
            self.get_file.assert_not_awaited()

    async def test_current_and_reply_images_share_one_byte_budget(self) -> None:
        message = image_message(
            message_id=2, reply_to_message=image_message().model_dump()
        )
        images = await self.images.load(message)
        assert isinstance(images[0], ImageAttachment)
        self.assertEqual(images[0].data, PNG)
        self.assertEqual(images[1], ImageOmission(1))
        self.get_file.assert_awaited_once()
        self.stream.assert_called_once()
        self.assertTrue(self.closed)

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
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertTrue(self.closed)

    async def test_all_sticker_types_and_formats_use_supported_image_bytes(self) -> None:
        self.chunks = [b"RIFF\x08\x00\x00\x00WEBPdata"]
        for sticker_type in ("regular", "mask", "custom_emoji"):
            for sticker_format in ("static", "animated", "video"):
                with self.subTest(
                    sticker_type=sticker_type, sticker_format=sticker_format
                ):
                    self.get_file.reset_mock()
                    sticker = {
                        **STICKER,
                        "type": sticker_type,
                        "is_animated": sticker_format == "animated",
                        "is_video": sticker_format == "video",
                        "thumbnail": THUMBNAIL,
                    }
                    images = await self.images.load(
                        image_message(photo=None, sticker=sticker)
                    )
                    assert isinstance(images[0], ImageAttachment)
                    self.assertEqual(images[0].data, self.chunks[0])
                    self.assertEqual(images[0].media_type, "image/webp")
                    self.assertEqual(
                        images[0].preview_kind,
                        "sticker" if sticker_format != "static" else None,
                    )
                    self.get_file.assert_awaited_once_with(
                        "sticker" if sticker_format == "static" else "thumbnail"
                    )

    async def test_animated_sticker_without_thumbnail_produces_note_without_download(
        self,
    ) -> None:
        for format_flag in ("is_animated", "is_video"):
            with self.subTest(format_flag=format_flag):
                images = await self.images.load(
                    image_message(photo=None, sticker={**STICKER, format_flag: True})
                )
                self.assertEqual(images, (ImageOmission(1, "preview_unavailable"),))
        self.get_file.assert_not_awaited()
        self.stream.assert_not_called()

    async def test_sticker_thumbnail_budget_ignores_size_of_unused_animation(
        self,
    ) -> None:
        images = await self.images.load(
            image_message(
                photo=None,
                sticker={
                    **STICKER,
                    "is_animated": True,
                    "file_size": 20_000_001,
                    "thumbnail": THUMBNAIL,
                },
            )
        )
        assert isinstance(images[0], ImageAttachment)
        self.get_file.assert_awaited_once_with("thumbnail")

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
            self.assertEqual(images, (ImageOmission(1),))
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
        self.assertEqual(images[1], ImageOmission(1))
        self.get_file.assert_awaited_once_with("photo")

    async def test_oversized_animation_falls_back_to_thumbnail_ignoring_legacy_document(
        self,
    ) -> None:
        for mime in ("image/gif", "video/mp4"):
            with self.subTest(mime=mime):
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
                self.assertEqual(images[0].preview_kind, "animation")
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
        self.assertEqual(len(images), 1)
        assert isinstance(images[0], ImageAttachment)
        self.assertEqual(images[0].message_id, 1)
        self.assertTrue(images[0].first_frame)

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
        assert isinstance(images[0], ImageAttachment)
        self.assertEqual(images[0].preview_kind, "animation")
        self.assertEqual(self.get_file.await_args_list, [call("gif"), call("thumbnail")])
        self.assertFalse(images[0].first_frame)
        self.get_file.reset_mock()
        self.assertEqual(
            await self.images.load(
                image_message(photo=None, document={**document, "thumbnail": None})
            ),
            (ImageOmission(1, "decode_failed"),),
        )
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
        self.assertEqual(len(images), 1)
        assert isinstance(images[0], ImageAttachment)
        self.assertTrue(images[0].first_frame)
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
        self.assertEqual(images, (ImageOmission(1),))
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
        self.assertEqual(images[0].message_id, 1)
        self.assertEqual(images[0].preview_kind, "animation")
        self.assertTrue(images[0].first_frame)
        self.get_file.assert_awaited_once_with("animation")
        self.get_file.reset_mock()
        for changed in (
            {"chat": {"id": 43, "type": "private"}},
            {"message_thread_id": 7},
        ):
            self.assertEqual(
                await self.images.load(
                    image_message(
                        photo=None, text="Explain", reply_to_message={**parent, **changed}
                    )
                ),
                (),
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
        self.assertIsInstance(images[0], ImageAttachment)

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
        self.assertEqual(images, ())
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
        self.assertIsInstance(images[0], ImageAttachment)
