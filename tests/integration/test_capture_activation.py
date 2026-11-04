"""Group capture requires the owner's first activation, independently of replies."""

from contextlib import AsyncExitStack

import pytest
from aiogram.types import Update

from dashulya.database import Database
from dashulya.database.migrate import migrate
from dashulya.domain.groups import Group, GroupRegistry
from dashulya.domain.memory import MemoryAccess
from dashulya.memory.store import MessageStore
from dashulya.storage.groups import GroupStore
from tests.integration.test_ai_handlers import (
    GROUP,
    USER,
    _AIHandlerFixture,
    administrator,
)


class TestCaptureActivation(_AIHandlerFixture):
    active_group = False

    @pytest.mark.parametrize("chat_type", ["group", "supergroup"])
    async def test_capture_requires_owner_start_and_continues_after_stop(
        self, chat_type: str
    ) -> None:
        store = MessageStore(self.database)
        self.recorder.record.side_effect = store.record
        chat = {**GROUP, "type": chat_type}
        access = MemoryAccess(chat["id"], False)
        member = {**USER, "id": 99}
        parent = {
            "message_id": 100,
            "date": 1700000000,
            "chat": chat,
            "from": member,
            "text": "Parent before activation",
        }
        await self.message(
            "Message before activation", chat=chat, sender=member, reply_to_message=parent
        )
        await self.dispatcher.feed_update(
            self.bot,
            Update.model_validate({
                "update_id": 100,
                "edited_message": {**parent, "edit_date": 1700000001},
            }),
        )
        await self.message("/start@dashulya_bot", chat=chat, sender=member)
        await self.message("/start@another_bot", chat=chat)
        self.administrators[chat["id"]] = [administrator(99)]
        await self.message("/stop", chat=chat, sender=member)
        await self.message("/settings", chat=chat, sender=member)
        assert await store.originals(access) == []
        assert await store.pending(limit=None) == []

        await self.message("/start@dashulya_bot", chat=chat)
        await self.message("Message after activation", chat=chat, sender=member)
        await self.message("/stop", chat=chat, sender=member)
        assert not await self.groups.is_active(chat["id"])
        await self.message("Message while paused", chat=chat, sender=member)
        paused_id = self.update_id
        await self.dispatcher.feed_update(
            self.bot,
            Update.model_validate({
                "update_id": 101,
                "edited_message": {
                    **parent,
                    "message_id": paused_id,
                    "text": "Edited while paused",
                    "edit_date": 1700000001,
                },
            }),
        )
        await self.message("/start", chat=chat)
        await self.message("Message after resuming", chat=chat, sender=member)
        texts = [row.text for row in await store.originals(access)]
        assert "Message after activation" in texts
        assert "Edited while paused" in texts
        assert "Message after resuming" in texts
        assert any("Теперь я отвечаю" in text for text in texts)
        assert any("Мои ответы в этой группе приостановлены" in text for text in texts)
        assert not any("before activation" in text for text in texts)
        assert await store.pending(limit=None)


@pytest.mark.parametrize("incoming_first", [False, True])
async def test_capture_permission_survives_pause_restart_and_migration_until_departure(
    database: Database,
    database_url: str,
    cleanup: AsyncExitStack,
    incoming_first: bool,
) -> None:
    registry = GroupRegistry(GroupStore(database))
    for chat_id in (-123, -100456, -789):
        await registry.observe(Group(chat_id, "Group"))
    await registry.activate(-123)
    await registry.deactivate(-123)
    restarted_database = Database(database_url)
    cleanup.push_async_callback(restarted_database.close)
    await restarted_database.open()
    restarted = GroupRegistry(GroupStore(restarted_database))
    await restarted.observe(Group(-123, "Renamed group", "renamed"))
    await restarted.membership(Group(-123, "Renamed group"), present=True)
    paused = await restarted.get(-123)
    assert paused is not None
    assert paused.capture_enabled
    assert not paused.active
    if incoming_first:
        await restarted.observe(Group(-100456, "New group"), migrated_from=-123)
        await restarted.observe(Group(-123, "Old group"), migrated_to=-100456)
    else:
        await restarted.observe(Group(-123, "Old group"), migrated_to=-100456)
        await restarted.observe(Group(-100456, "New group"), migrated_from=-123)
    migrated = await restarted.get(-100456)
    assert migrated is not None
    assert migrated.capture_enabled
    assert not migrated.active
    assert await restarted.get(-123) is None
    untouched = await restarted.get(-789)
    assert untouched is not None
    assert not untouched.capture_enabled
    await restarted.membership(Group(-100456, "New group"), present=False)
    await restarted.membership(Group(-100456, "New group"), present=True)
    rejoined = await restarted.get(-100456)
    assert rejoined is not None
    assert not rejoined.capture_enabled
    assert not rejoined.active


async def test_upgrade_enables_capture_only_for_currently_active_groups(
    database: Database, database_url: str
) -> None:
    registry = GroupRegistry(GroupStore(database))
    for chat_id in (-123, -456, -789):
        await registry.observe(Group(chat_id, "Group"))
    await registry.activate(-123)
    await registry.activate(-456)
    await registry.deactivate(-456)
    async with database.connection() as connection:
        await connection.execute("ALTER TABLE groups DROP COLUMN capture_enabled")
        await connection.execute(
            "DELETE FROM schema_migrations WHERE name='0010_group_capture.sql'"
        )
    await migrate(database_url)
    assert await registry.get(-123) == Group(
        -123, "Group", active=True, capture_enabled=True
    )
    assert await registry.get(-456) == Group(-456, "Group")
    assert await registry.get(-789) == Group(-789, "Group")
