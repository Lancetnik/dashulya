"""Owner private capture and canonical identity regression coverage."""

from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import AsyncMock

from aiogram.types import Message, Update

from dashulya.bot.messages import capture_message
from dashulya.bot.middlewares.memory import CaptureMessagesMiddleware
from dashulya.domain.memory import MemoryAccess, MemoryMessage, MessageRecorder
from dashulya.domain.messages import ImageFile
from dashulya.memory.history import CanonicalHistory
from dashulya.memory.store import MessageStore
from tests.postgres import PostgresTestCase


class MemoryCaptureTests(PostgresTestCase):
    async def test_image_documents_retain_references_without_capturing_animation_files(
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
        self.assertIsNone(recorder.record.await_args.args[0].image_file)

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
