import asyncio
import os
from contextlib import AsyncExitStack
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import AsyncMock, patch

import pytest
from aiogram import Bot, Dispatcher
from aiogram.types import Update
from dishka import AsyncContainer
from psycopg_pool import PoolClosed

from dashulya.database import Database
from dashulya.main import create_container, run
from dashulya.main.config import Settings
from dashulya.storage.groups import GroupStore
from tests.scopes import ScopeProvider, ScopeTrace, record_update
from tests.support import (
    TEST_API_KEY,
    MemoryOverride,
    RecorderOverride,
)


@pytest.mark.usefixtures("_setup")
class TestDependency:
    @pytest.fixture
    def _setup(self, database: Database, cleanup: AsyncExitStack) -> None:
        self.database = database
        self.cleanup = cleanup
        self.directory = TemporaryDirectory()
        self.cleanup.callback(self.directory.cleanup)
        self.settings = Settings(
            bot_token="123456789:TEST_ONLY_NOT_A_REAL_BOT_TOKEN",
            openai_api_key=TEST_API_KEY,
            owner_user_id=42,
        )

    async def test_feed_update_injects_event_context_and_finalizes_each_request(
        self,
    ) -> None:
        container = create_container(
            self.settings, ScopeProvider(), MemoryOverride(), RecorderOverride()
        )
        self.cleanup.push_async_callback(container.close)
        dispatcher = await container.get(Dispatcher)
        bot = await container.get(Bot)
        dispatcher.message.register(record_update)
        for update_id in (1, 2):
            await dispatcher.feed_update(
                bot,
                Update.model_validate({
                    "update_id": update_id,
                    "message": {
                        "message_id": update_id,
                        "date": 1700000000,
                        "chat": {"id": 42, "type": "private", "first_name": "Anna"},
                        "from": {"id": 42, "is_bot": False, "first_name": "Anna"},
                        "text": "Hello",
                    },
                }),
            )
        trace = await container.get(ScopeTrace)
        assert len(trace.telegram_calls) == 2
        assert trace.telegram_calls[0][2] is not trace.telegram_calls[1][2]
        for message, event, _ in trace.telegram_calls:
            assert message is event
        assert len(trace.closed_calls) == 2

    async def test_runtime_closes_production_resources_on_exit_and_polling_failure(
        self,
        subtests: pytest.Subtests,
    ) -> None:
        for failure in (None, RuntimeError("Polling failed"), asyncio.CancelledError()):
            with subtests.test(failure=failure):
                container = create_container(
                    self.settings, MemoryOverride(), RecorderOverride()
                )
                bot = await container.get(Bot)
                groups = await container.get(GroupStore)
                close = AsyncMock(wraps=bot.session.close)
                with (
                    patch(
                        "dashulya.main.runtime.create_container", return_value=container
                    ),
                    patch.object(bot.session, "close", close),
                    patch.object(
                        Dispatcher, "start_polling", AsyncMock(side_effect=failure)
                    ) as polling,
                ):
                    if failure is None:
                        await run(self.settings)
                    else:
                        with pytest.raises(type(failure)):
                            await run(self.settings)
                polling.assert_awaited_once_with(bot, close_bot_session=False)
                close.assert_awaited_once()
                with pytest.raises(PoolClosed):
                    await groups.list_groups()

    async def test_container_readiness_requires_identity_and_is_cleared_on_shutdown(
        self,
    ) -> None:
        marker = Path(self.directory.name) / "nested" / "health" / "ready"
        settings = replace(self.settings, healthcheck_path=marker)
        container = create_container(settings, MemoryOverride(), RecorderOverride())
        bot = await container.get(Bot)

        async def poll(*args: object, **kwargs: object) -> None:
            assert marker.read_text() == str(os.getpid())

        with (
            patch("dashulya.main.runtime.create_container", return_value=container),
            patch.object(bot, "me", AsyncMock()) as identity,
            patch.object(Dispatcher, "start_polling", AsyncMock(side_effect=poll)),
        ):
            await run(settings)
        identity.assert_awaited_once()
        assert not marker.exists()

    async def test_runtime_passes_settings_into_the_real_graph(self) -> None:
        def build(settings: Settings) -> AsyncContainer:
            return create_container(settings, MemoryOverride(), RecorderOverride())

        async def poll(bot: Bot, **kwargs: object) -> None:
            assert bot.token == self.settings.bot_token

        with (
            patch("dashulya.main.runtime.create_container", side_effect=build),
            patch.object(
                Dispatcher, "start_polling", AsyncMock(side_effect=poll)
            ) as polling,
        ):
            await run(self.settings)
        polling.assert_awaited_once()

    async def test_externally_removed_readiness_marker_does_not_break_shutdown(
        self,
    ) -> None:
        marker = Path(self.directory.name) / "ready"
        settings = replace(self.settings, healthcheck_path=marker)
        container = create_container(settings, MemoryOverride(), RecorderOverride())
        bot = await container.get(Bot)
        close = AsyncMock(wraps=bot.session.close)

        async def poll(*args: object, **kwargs: object) -> None:
            assert marker.is_file()
            marker.unlink()

        with (
            patch("dashulya.main.runtime.create_container", return_value=container),
            patch.object(bot, "me", AsyncMock()),
            patch.object(bot.session, "close", close),
            patch.object(Dispatcher, "start_polling", AsyncMock(side_effect=poll)),
        ):
            await run(settings)
        assert not marker.exists()
        close.assert_awaited_once()
