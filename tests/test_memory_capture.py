"""Owner private capture and canonical identity regression coverage."""

from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from unittest.mock import AsyncMock, patch

from aiogram import Bot
from aiogram.types import Message, Update

from dashulya.bot.images import TelegramImages
from dashulya.bot.messages import capture_message
from dashulya.bot.middlewares.memory import CaptureMessagesMiddleware
from dashulya.domain.memory import MemoryAccess, MemoryMessage, MessageRecorder
from dashulya.domain.messages import ImageAttachment, ImageFile
from dashulya.memory.history import CanonicalHistory
from dashulya.memory.store import MessageStore
from tests.postgres import PostgresTestCase


class MemoryCaptureTests(PostgresTestCase):
    async def test_image_and_gif_documents_retain_references(
        self,
    ) -> None:
        recorder = AsyncMock(spec=MessageRecorder)
        original = self.incoming().model_copy(update={"text": None}).model_dump()
        for mime, name in (
            ("image/png", "picture"),
            ("application/octet-stream", "photo.JPG"),
        ):
            message = Message.model_validate({
                **original,
                "document": {
                    "file_id": "document",
                    "file_unique_id": "unique",
                    "mime_type": mime,
                    "file_name": name,
                },
            })
            await capture_message(recorder, message, allow_private=True)
            self.assertEqual(
                recorder.record.await_args.args[0].image_file, ImageFile("document")
            )
        gif = Message.model_validate({
            **original,
            "document": {
                "file_id": "gif",
                "file_unique_id": "unique-gif",
                "mime_type": "image/gif",
                "file_name": "photo.gif",
            },
        })
        await capture_message(recorder, gif, allow_private=True)
        self.assertEqual(
            recorder.record.await_args.args[0].image_file,
            ImageFile("gif", preview_kind="animation"),
        )

    async def test_sticker_and_animation_references_reload_after_restart(self) -> None:
        thumbnail = {
            "file_id": "thumbnail",
            "file_unique_id": "thumb",
            "width": 2,
            "height": 2,
            "file_size": 16,
        }
        sticker = {
            "file_id": "sticker",
            "file_unique_id": "sticker",
            "type": "regular",
            "width": 2,
            "height": 2,
            "is_animated": False,
            "is_video": False,
        }
        cases: tuple[tuple[dict[str, Any], str, str | None], ...] = (
            ({"sticker": sticker}, "sticker", None),
            (
                {"sticker": {**sticker, "is_animated": True, "thumbnail": thumbnail}},
                "thumbnail",
                "sticker",
            ),
            (
                {"sticker": {**sticker, "is_video": True, "thumbnail": thumbnail}},
                "thumbnail",
                "sticker",
            ),
            (
                {
                    "animation": {
                        "file_id": "animation",
                        "file_unique_id": "animation",
                        "width": 2,
                        "height": 2,
                        "duration": 1,
                        "thumbnail": thumbnail,
                    }
                },
                "animation",
                "animation",
            ),
            (
                {
                    "document": {
                        "file_id": "gif",
                        "file_unique_id": "gif",
                        "mime_type": "image/gif",
                        "thumbnail": thumbnail,
                    }
                },
                "gif",
                "animation",
            ),
        )
        store = MessageStore(self.database)
        loader = TelegramImages(AsyncMock(spec=Bot), max_bytes=100_000)
        png = b"\x89PNG\r\n\x1a\nimage-data"
        for number, (fields, expected_file, kind) in enumerate(cases, start=1):
            with self.subTest(fields=fields):
                await capture_message(
                    store,
                    Message.model_validate({
                        "message_id": number,
                        "date": 1792929600,
                        "chat": {"id": -123, "type": "supergroup", "title": "Group"},
                        **fields,
                    }),
                )
                restarted = MessageStore(self.database)
                selected = await restarted.context(
                    (-123, None), number, limit=1, parent_hops=0
                )
                reference = selected.messages[0].image_file
                assert reference is not None
                self.assertEqual(reference.preview_kind, kind)
                if kind:
                    self.assertEqual(reference.thumbnail_file_id, "thumbnail")
                    self.assertEqual(reference.thumbnail_file_size, 16)
                with (
                    patch.object(
                        loader, "_read", AsyncMock(return_value=png)
                    ) as download,
                    patch("dashulya.bot.images.first_frame", return_value=b"frame"),
                ):
                    images = await loader.load_context(selected.messages)
                assert isinstance(images[0], ImageAttachment)
                self.assertEqual(images[0].message_id, number)
                self.assertEqual(images[0].preview_kind, kind)
                self.assertEqual(images[0].first_frame, kind == "animation")
                download.assert_awaited_once()
                assert download.await_args is not None
                self.assertEqual(download.await_args.args[0].file_id, expected_file)
                originals = await restarted.originals(MemoryAccess(-123, False))
                self.assertTrue(
                    all("image_file" not in original.render() for original in originals)
                )

    async def test_photo_file_reference_survives_restart_and_respects_context_scope(
        self,
    ) -> None:
        store = MessageStore(self.database)
        photo = Message.model_validate({
            "message_id": 1,
            "date": 1792929600,
            "chat": {
                "id": -123,
                "type": "supergroup",
                "title": "Group",
                "is_forum": True,
            },
            "message_thread_id": 7,
            "is_topic_message": True,
            "photo": [
                {"file_id": "small", "file_unique_id": "small", "width": 1, "height": 1},
                {
                    "file_id": "large",
                    "file_unique_id": "large",
                    "width": 2,
                    "height": 2,
                    "file_size": 42,
                },
            ],
        })
        await capture_message(store, photo)
        restarted = MessageStore(self.database)
        selected = await restarted.context((-123, 7), 2, limit=20, parent_hops=5)
        self.assertEqual(selected.messages[0].image_file, ImageFile("large", 42))
        self.assertEqual(
            (await restarted.originals(MemoryAccess(-123, False)))[0].image_file,
            ImageFile("large", 42),
        )
        self.assertFalse(
            (await restarted.context((-123, 8), 2, limit=20, parent_hops=5)).messages
        )
        self.assertFalse(
            (await restarted.context((-456, 7), 2, limit=20, parent_hops=5)).messages
        )
        self.assertFalse(
            (await restarted.context((-123, 7), 0, limit=20, parent_hops=5)).messages
        )
        async with self.database.connection() as connection:
            row = await (
                await connection.execute("SELECT image_file FROM messages")
            ).fetchone()
        assert row is not None
        self.assertEqual(row["image_file"], {"file_id": "large", "file_size": 42})

    async def test_image_reference_edit_is_not_replaced_by_stale_reply_snapshot(
        self,
    ) -> None:
        store = MessageStore(self.database)
        message = MemoryMessage(
            chat_id=-123,
            message_id=1,
            text="Caption",
            sent_at=datetime(2026, 10, 25, tzinfo=UTC),
            author_id=42,
            author="Alice",
            chat_title="Group",
            media_type="photo",
            image_file=ImageFile("original"),
        )
        await store.record(message)
        await store.record(replace(message, image_file=ImageFile("edited"), update_id=2))
        await store.record(message, if_missing=True)
        selected = await store.context((-123, None), 1, limit=20, parent_hops=5)
        self.assertEqual(selected.messages[0].image_file, ImageFile("edited"))
        await store.record(
            replace(message, image_file=None, media_type=None, update_id=3)
        )
        selected = await store.context((-123, None), 1, limit=20, parent_hops=5)
        self.assertIsNone(selected.messages[0].image_file)

    @staticmethod
    def incoming(sender: int = 42, *, bot: bool = False) -> Message:
        return Message.model_validate({
            "message_id": 7,
            "date": 1792929600,
            "chat": {"id": sender, "type": "private", "first_name": "Owner"},
            "from": {
                "id": sender,
                "is_bot": bot,
                "first_name": "Alice",
                "username": "alice",
            },
            "text": "My project uses SQLite",
        })

    async def test_only_actual_owner_private_messages_enter_memory(self) -> None:
        recorder = AsyncMock(spec=MessageRecorder)
        capture = CaptureMessagesMiddleware(recorder, owner_user_id=42)
        await capture.capture(self.incoming(99))
        await capture.capture(self.incoming(42, bot=True))
        recorder.record.assert_not_awaited()
        await capture.capture(self.incoming(), Update(update_id=5))
        saved = recorder.record.await_args.args[0]
        self.assertEqual(
            (saved.chat_id, saved.author_id, saved.username), (42, 42, "alice")
        )
        self.assertIs(saved.is_bot, False)
        self.assertEqual(saved.update_id, 5)

    async def test_forwarded_owner_identity_does_not_authorize_private_capture(
        self,
    ) -> None:
        recorder = AsyncMock(spec=MessageRecorder)
        message = self.incoming(99).model_copy(
            update={
                "forward_origin": {
                    "type": "user",
                    "sender_user": {"id": 42},
                    "date": 1792929600,
                },
            }
        )
        await CaptureMessagesMiddleware(recorder, owner_user_id=42).capture(message)
        recorder.record.assert_not_awaited()

    async def test_schema_migration_and_metadata_survive_restart(self) -> None:
        with TemporaryDirectory() as directory:
            Path(directory) / "memory.sqlite3"
            store = MessageStore(self.database)
            await store.initialize()
            message = MemoryMessage(
                chat_id=42,
                message_id=1,
                text="Private note",
                sent_at=datetime(2026, 10, 25, tzinfo=UTC),
                author_id=42,
                author="Alice",
                chat_title="Owner",
                chat_username="owner",
                username="alice",
                is_bot=False,
            )
            await store.record(message)
            restarted = MessageStore(self.database)
            await restarted.initialize()
            await restarted.initialize()
            messages = (
                await restarted.context((42, None), 1, limit=20, parent_hops=5)
            ).messages
            self.assertEqual(messages[0].username, "alice")
            self.assertIs(messages[0].is_bot, False)

            expanded = await CanonicalHistory(restarted, MemoryAccess(42, True)).query()
            self.assertIsNone(expanded["messages"][0]["url"])
            self.assertEqual((await restarted.coverage())["message_count"], 1)
