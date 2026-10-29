from contextlib import AsyncExitStack
from datetime import UTC, datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import AsyncMock, patch

import pytest
from aiogram import Bot, Dispatcher
from aiogram.types import Update

from dashulya.database import Database
from dashulya.domain.memory import MessageRecorder
from dashulya.main import create_container
from dashulya.main.config import Settings
from dashulya.memory.live import LiveMemory
from tests.support import TEST_API_KEY, BotOverride, DialogueOverride, RecorderOverride

NOW = datetime(2026, 10, 25, 12, tzinfo=UTC)


@pytest.mark.usefixtures("_setup")
class TestLegacyCapture:
    @pytest.fixture
    async def _setup(self, database: Database, cleanup: AsyncExitStack) -> None:
        self.database = database
        self.cleanup = cleanup
        self.directory = TemporaryDirectory()
        self.cleanup.callback(self.directory.cleanup)
        self.path = Path(self.directory.name) / "live.sqlite3"
        self.store = LiveMemory(self.path)
        await self.store.initialize()

    async def test_non_owner_updates_and_edits_are_recorded_without_replies(self) -> None:
        recorder = AsyncMock(spec=MessageRecorder)
        recorder.record.side_effect = self.store.record
        bot = Bot("123456789:TEST_ONLY_NOT_A_REAL_BOT_TOKEN")
        self.cleanup.push_async_callback(bot.session.close)
        requests = AsyncMock(return_value=True)
        with patch.object(bot.session, "make_request", requests):
            async with create_container(
                Settings(
                    bot_token=bot.token,
                    openai_api_key=TEST_API_KEY,
                    owner_user_id=42,
                ),
                BotOverride(bot),
                RecorderOverride(recorder),
                DialogueOverride(AsyncMock()),
            ) as container:
                dispatcher = await container.get(Dispatcher)
                message = {
                    "message_id": 1,
                    "date": int(NOW.timestamp()),
                    "chat": {"id": -1001001, "type": "supergroup", "title": "Group"},
                    "from": {"id": 99, "is_bot": False, "first_name": "Bob"},
                    "text": "Ordinary group discussion",
                }
                await dispatcher.feed_update(
                    bot, Update.model_validate({"update_id": 100, "message": message})
                )
                await dispatcher.feed_update(
                    bot,
                    Update.model_validate({
                        "update_id": 101,
                        "edited_message": {
                            **message,
                            "text": "Edited",
                            "edit_date": int(NOW.timestamp()) + 1,
                        },
                    }),
                )
                await dispatcher.feed_update(
                    bot,
                    Update.model_validate({
                        "update_id": 102,
                        "message": {
                            **message,
                            "chat": {"id": 99, "type": "private", "first_name": "Bob"},
                        },
                    }),
                )
        assert recorder.record.await_count == 2
        assert recorder.record.await_args_list[0].args[0].update_id == 100
        assert recorder.record.await_args_list[1].args[0].text == "Edited"
        assert recorder.record.await_args_list[1].args[0].edited_at == NOW + timedelta(
            seconds=1
        )
        requests.assert_not_awaited()
        assert (await self.store.coverage())["message_count"] == 1
        saved = await self.store.expand("telegram:-1001001:1", None, None, None)
        assert saved[0]["sources"][0]["text"] == "Edited"
