"""Owner private capture and canonical identity regression coverage."""

from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import AsyncMock

from aiogram.types import Message, Update

from dashulya.bot.middlewares.memory import CaptureMessagesMiddleware
from dashulya.domain.memory import MemoryAccess, MemoryMessage, MessageRecorder
from dashulya.memory.history import CanonicalHistory
from dashulya.memory.store import MessageStore
from tests.postgres import PostgresTestCase


class MemoryCaptureTests(PostgresTestCase):
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
