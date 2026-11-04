from contextlib import AsyncExitStack
from tempfile import TemporaryDirectory
from typing import Any, cast
from unittest.mock import AsyncMock, patch

import pytest
from aiogram import Bot, Dispatcher
from aiogram.types import Update, User

from dashulya.database import Database
from dashulya.database.migrate import migrate
from dashulya.domain.groups import Group, GroupRegistry
from dashulya.main import create_container
from dashulya.main.config import Settings
from dashulya.storage.groups import GroupStore
from tests.support import (
    TEST_API_KEY,
    BotOverride,
    DecisionOverride,
    DialogueOverride,
    RecorderOverride,
)

BOT_TOKEN = "123456789:TEST_ONLY_NOT_A_REAL_BOT_TOKEN"
BOT_USER: dict[str, Any] = {"id": 123456789, "is_bot": True, "first_name": "Dashulya"}
USER: dict[str, Any] = {"id": 42, "is_bot": False, "first_name": "Test"}
GROUP_CHAT: dict[str, Any] = {"id": -123, "type": "group", "title": "Test group"}
PRIVATE_CHAT: dict[str, Any] = {"id": 42, "type": "private", "first_name": "Test"}


@pytest.mark.usefixtures("_setup")
class TestGroupTracking:
    @pytest.fixture
    async def _setup(self, database: Database, cleanup: AsyncExitStack) -> None:
        self.cleanup = cleanup
        self.directory = TemporaryDirectory()
        self.cleanup.callback(self.directory.cleanup)
        self.bot = Bot(BOT_TOKEN)
        self.cleanup.push_async_callback(self.bot.session.close)
        self.requests = AsyncMock(return_value=True)
        request_patch = patch.object(self.bot.session, "make_request", self.requests)
        request_patch.start()
        self.cleanup.callback(request_patch.stop)
        me_patch = patch.object(
            self.bot,
            "me",
            AsyncMock(return_value=User(**BOT_USER, username="dashulya_bot")),
        )
        me_patch.start()
        self.cleanup.callback(me_patch.stop)
        decision = AsyncMock()
        decision.ask.return_value.content = AsyncMock(return_value=False)
        self.container = create_container(
            Settings(
                bot_token=BOT_TOKEN,
                openai_api_key=TEST_API_KEY,
                owner_user_id=USER["id"],
            ),
            BotOverride(self.bot),
            RecorderOverride(),
            DialogueOverride(AsyncMock()),
            DecisionOverride(decision),
        )
        self.cleanup.push_async_callback(self.container.close)
        self.groups = await self.container.get(GroupStore)
        self.dispatcher = await self.container.get(Dispatcher)
        self.update_id = 0

    async def feed(self, **payload: Any) -> Any:
        self.update_id += 1
        update = Update.model_validate({"update_id": self.update_id, **payload})
        return await self.dispatcher.feed_update(self.bot, update)

    async def message(
        self,
        text: str | None = None,
        chat: dict[str, Any] | None = None,
        **fields: Any,
    ) -> Any:
        message = {
            "message_id": self.update_id + 1,
            "date": 1700000000,
            "chat": chat or PRIVATE_CHAT,
            "from": USER,
            **fields,
        }
        if text is not None:
            message["text"] = text
            if text.startswith("/"):
                message["entities"] = [
                    {"type": "bot_command", "offset": 0, "length": len(text.split()[0])}
                ]
        return await self.feed(message=message)

    async def membership(
        self,
        status: str = "member",
        *,
        chat: dict[str, Any] | None = None,
        **fields: Any,
    ) -> Any:
        return await self.feed(
            my_chat_member={
                "chat": chat or GROUP_CHAT,
                "from": USER,
                "date": 1700000000,
                "old_chat_member": {"status": "left", "user": BOT_USER},
                "new_chat_member": {"status": status, "user": BOT_USER, **fields},
            }
        )

    def last_reply(self) -> str:
        return cast("str", self.requests.call_args.args[1].text)

    async def test_empty_list_and_private_chats_are_not_registered(self) -> None:
        await self.message("/groups")
        assert "Пока не знаю ни одной группы" in self.last_reply()
        assert self.last_reply().startswith("📭")
        assert await self.groups.list_groups() == []

    async def test_join_lists_groups_with_names_usernames_and_ids(self) -> None:
        await self.membership()
        await self.membership(
            chat={
                "id": -100456,
                "type": "supergroup",
                "title": "Public group",
                "username": "public_group",
            }
        )
        await self.message("/groups")
        text = self.last_reply()
        assert "🔴 Test group (ID: -123)" in text
        assert "🔴 Public group @public_group (ID: -100456)" in text
        assert text.startswith("📋 Известные мне группы:")
        assert text.count("— ответы выключены (отправь там /start)") == 2
        assert text.index("Public group") < text.index("Test group")

    async def test_existing_group_is_registered_before_answering_command(self) -> None:
        await self.message("/groups@dashulya_bot", chat=GROUP_CHAT)
        self.requests.assert_not_awaited()
        assert await self.groups.list_groups() == [Group(-123, "Test group")]
        await self.message("/start@dashulya_bot", chat=GROUP_CHAT)
        await self.message("/groups@dashulya_bot", chat=GROUP_CHAT)
        assert "🟢 Test group (ID: -123)" in self.last_reply()
        assert "— отвечаю" in self.last_reply()
        assert len(await self.groups.list_groups()) == 1

    async def test_messages_refresh_metadata_without_duplicate_groups(self) -> None:
        await self.membership(chat={**GROUP_CHAT, "username": "old_name"})
        await self.message("Hello", chat={**GROUP_CHAT, "title": "Renamed group"})
        await self.message(
            "Another message", chat={**GROUP_CHAT, "title": "Renamed group"}
        )
        assert await self.groups.list_groups() == [Group(-123, "Renamed group")]
        self.requests.assert_not_awaited()

    async def test_leaving_or_being_kicked_removes_group_and_rejoining_restores_it(
        self,
        subtests: pytest.Subtests,
    ) -> None:
        for status in ("left", "kicked"):
            with subtests.test(status=status):
                await self.membership()
                fields: dict[str, Any] = {"until_date": 0} if status == "kicked" else {}
                await self.membership(status, **fields)
                await self.message("/groups")
                assert "Пока не знаю ни одной группы" in self.last_reply()
                await self.membership()
                assert await self.groups.list_groups() == [Group(-123, "Test group")]

    @pytest.mark.parametrize("chat_type", ["group", "supergroup"])
    @pytest.mark.parametrize("membership_first", [True, False])
    async def test_own_departure_service_message_does_not_restore_a_removed_group(
        self, chat_type: str, membership_first: bool
    ) -> None:
        chat = {**GROUP_CHAT, "type": chat_type}
        await self.membership(chat=chat)
        await self.message("/start", chat=chat)
        if membership_first:
            await self.membership("kicked", chat=chat, until_date=0)
        await self.message(
            chat=chat,
            left_chat_member=BOT_USER,
            **cast("dict[str, Any]", {"from": {**USER, "id": 99}}),
        )
        if not membership_first:
            await self.membership("kicked", chat=chat, until_date=0)
        await self.message("/groups")
        assert "Пока не знаю ни одной группы" in self.last_reply()
        await self.membership(chat=chat)
        await self.message("/groups")
        assert "🔴 Test group (ID: -123)" in self.last_reply()
        assert "— ответы выключены (отправь там /start)" in self.last_reply()

    @pytest.mark.parametrize(
        ("departed", "expected"),
        [
            (BOT_USER, "Пока не знаю ни одной группы"),
            (USER, "🟢 Test group (ID: -123) — отвечаю"),
            ({**BOT_USER, "id": 987654321}, "🟢 Test group (ID: -123) — отвечаю"),
        ],
    )
    async def test_departure_service_without_membership_update_removes_only_this_bot(
        self, departed: dict[str, Any], expected: str
    ) -> None:
        await self.message("/start", chat=GROUP_CHAT)
        await self.message(
            chat=GROUP_CHAT,
            left_chat_member=departed,
            **cast("dict[str, Any]", {"from": {**USER, "id": 99}}),
        )
        await self.message("/groups")
        assert expected in self.last_reply()

    async def test_restricted_membership_uses_is_member(self) -> None:
        permissions: dict[str, Any] = dict.fromkeys(
            (
                "can_send_messages",
                "can_send_audios",
                "can_send_documents",
                "can_send_photos",
                "can_send_videos",
                "can_send_video_notes",
                "can_send_voice_notes",
                "can_send_polls",
                "can_send_other_messages",
                "can_add_web_page_previews",
                "can_react_to_messages",
                "can_edit_tag",
                "can_change_info",
                "can_invite_users",
                "can_pin_messages",
                "can_manage_topics",
            ),
            False,
        )
        await self.membership("restricted", is_member=True, until_date=0, **permissions)
        assert await self.groups.list_groups() == [Group(-123, "Test group")]
        await self.membership("restricted", is_member=False, until_date=0, **permissions)
        assert await self.groups.list_groups() == []

    async def test_private_and_channel_membership_updates_are_ignored(self) -> None:
        await self.membership(chat=PRIVATE_CHAT)
        await self.membership(chat={"id": -100789, "type": "channel", "title": "News"})
        assert await self.groups.list_groups() == []

    async def test_migration_replaces_old_group_in_both_service_messages(self) -> None:
        await self.membership()
        await self.message(chat=GROUP_CHAT, migrate_to_chat_id=-100456)
        assert await self.groups.list_groups() == [Group(-100456, "Test group")]
        await self.message(
            chat={"id": -100456, "type": "supergroup", "title": "Upgraded group"},
            migrate_from_chat_id=-123,
        )
        await self.message("/groups")
        assert await self.groups.list_groups() == [Group(-100456, "Upgraded group")]
        assert "ID: -123" not in self.last_reply()

    async def test_incoming_migration_alone_removes_old_group(self) -> None:
        await self.membership()
        await self.message(
            chat={"id": -100456, "type": "supergroup", "title": "Upgraded group"},
            migrate_from_chat_id=-123,
        )
        assert await self.groups.list_groups() == [Group(-100456, "Upgraded group")]

    async def test_existing_commands_still_work(self) -> None:
        await self.message("/start")
        assert "/help" in self.last_reply()
        await self.message("/help")
        assert "/groups" in self.last_reply()
        assert "/clear" in self.last_reply()
        assert "/ping" not in self.last_reply()
        self.requests.reset_mock()
        for chat in (PRIVATE_CHAT, GROUP_CHAT):
            await self.message("/ping", chat=chat)
        self.requests.assert_not_awaited()

    async def test_polling_subscribes_to_membership_updates(self) -> None:
        assert self.dispatcher.resolve_used_update_types() == [
            "callback_query",
            "edited_message",
            "message",
            "my_chat_member",
        ]


@pytest.mark.usefixtures("_setup")
class TestGroupPersistence:
    @pytest.fixture
    def _setup(
        self, database: Database, database_url: str, cleanup: AsyncExitStack
    ) -> None:
        self.database = database
        self.database_url = database_url
        self.cleanup = cleanup

    async def test_blacklist_survives_restart_and_metadata_refresh_only_in_its_group(
        self,
    ) -> None:
        registry = GroupRegistry(GroupStore(self.database))
        await registry.observe(Group(-123, "Test group"))
        await registry.observe(Group(-456, "Other group"))
        assert not await registry.is_banned(-123, 99)
        await registry.ban(-123, 99)
        await registry.ban(-123, 99)
        reopened_database = Database(self.database_url)
        self.cleanup.push_async_callback(reopened_database.close)
        await reopened_database.open()
        reopened = GroupRegistry(GroupStore(reopened_database))
        await reopened.observe(Group(-123, "Renamed group", "renamed"))
        assert await reopened.is_banned(-123, 99)
        assert not await reopened.is_banned(-456, 99)
        assert not await reopened.is_banned(-123, 100)

    async def test_unban_persists_and_preserves_other_users_and_groups(self) -> None:
        registry = GroupRegistry(GroupStore(self.database))
        for chat_id in (-123, -456):
            await registry.observe(Group(chat_id, "Test group"))
            await registry.ban(chat_id, 99)
        await registry.ban(-123, 100)
        await registry.unban(-123, 99)
        await registry.unban(-123, 99)
        reopened_database = Database(self.database_url)
        self.cleanup.push_async_callback(reopened_database.close)
        await reopened_database.open()
        reopened = GroupRegistry(GroupStore(reopened_database))
        assert not await reopened.is_banned(-123, 99)
        assert await reopened.is_banned(-456, 99)
        assert await reopened.is_banned(-123, 100)

    @pytest.mark.parametrize("incoming_first", [False, True])
    async def test_blacklist_follows_migration_and_resets_after_departure(
        self, incoming_first: bool
    ) -> None:
        registry = GroupRegistry(GroupStore(self.database))
        await registry.observe(Group(-123, "Old group"))
        await registry.ban(-123, 99)
        await registry.ban(-123, 101)
        await registry.observe(Group(-100456, "New group"))
        await registry.ban(-100456, 99)
        await registry.ban(-100456, 100)
        if incoming_first:
            await registry.observe(Group(-100456, "New group"), migrated_from=-123)
            await registry.observe(Group(-123, "Old group"), migrated_to=-100456)
        else:
            await registry.observe(Group(-123, "Old group"), migrated_to=-100456)
            await registry.observe(Group(-100456, "New group"), migrated_from=-123)
        assert await registry.is_banned(-100456, 99)
        assert await registry.is_banned(-100456, 100)
        assert await registry.is_banned(-100456, 101)
        assert not await registry.is_banned(-123, 99)
        await registry.membership(Group(-100456, "New group"), present=False)
        await registry.membership(Group(-100456, "New group"), present=True)
        assert not await registry.is_banned(-100456, 99)
        assert not await registry.is_banned(-100456, 100)

    async def test_activation_survives_restart_metadata_refresh_and_membership_update(
        self,
    ) -> None:
        groups = GroupStore(self.database)
        await groups.remember(Group(-123, "Test group"))
        assert not await groups.is_active(-123)
        assert not await groups.is_active(-999)
        await groups.activate(-123)
        await groups.activate(-123)
        reopened_database = Database(self.database_url)
        self.cleanup.push_async_callback(reopened_database.close)
        await reopened_database.open()
        reopened = GroupStore(reopened_database)
        registry = GroupRegistry(reopened)
        await registry.observe(Group(-123, "Renamed group", "renamed"))
        await registry.membership(Group(-123, "Renamed group", "renamed"), present=True)
        assert await registry.is_active(-123)
        assert await reopened.list_groups() == [
            Group(-123, "Renamed group", "renamed", active=True, capture_enabled=True)
        ]
        await registry.membership(Group(-123, "Renamed group"), present=False)
        await registry.membership(Group(-123, "Renamed group"), present=True)
        assert not await registry.is_active(-123)

    async def test_activation_follows_migration_in_either_service_message_order(
        self,
        subtests: pytest.Subtests,
    ) -> None:
        for incoming_first in (False, True):
            with subtests.test(incoming_first=incoming_first):
                store = GroupStore(self.database)
                registry = GroupRegistry(store)
                await registry.observe(Group(-123, "Old group"))
                await registry.activate(-123)
                # The target may already have been observed before the migration.
                await registry.observe(Group(-100456, "New group"))
                if incoming_first:
                    await registry.observe(
                        Group(-100456, "New group"), migrated_from=-123
                    )
                    await registry.observe(Group(-123, "Old group"), migrated_to=-100456)
                else:
                    await registry.observe(Group(-123, "Old group"), migrated_to=-100456)
                    await registry.observe(
                        Group(-100456, "New group"), migrated_from=-123
                    )
                assert await registry.is_active(-100456)
                assert not await registry.is_active(-123)
                assert len(await store.list_groups()) == 1
                await store.forget(-100456)

    async def test_existing_database_groups_require_activation_after_schema_upgrade(
        self,
    ) -> None:
        async with self.database.connection() as connection:
            await connection.execute("ALTER TABLE groups DROP COLUMN active")
            await connection.execute(
                "INSERT INTO groups VALUES (%s,%s,%s)", (-123, "Existing group", None)
            )
        async with self.database.connection() as connection:
            await connection.execute("DELETE FROM schema_migrations")
        await migrate(self.database_url)
        reopened = Database(self.database_url)
        self.cleanup.push_async_callback(reopened.close)
        await reopened.open()
        groups = GroupStore(reopened)
        assert await groups.list_groups() == [Group(-123, "Existing group")]
        assert not await groups.is_active(-123)
        await groups.activate(-123)
        assert await groups.is_active(-123)

    async def test_large_group_list_is_split_without_losing_entries(self) -> None:
        groups = [Group(-index, f"{index}: " + "💋" * 120) for index in range(1, 101)]
        store = GroupStore(self.database)
        for group in groups:
            await store.remember(group)
        messages = await GroupRegistry(store).messages()
        assert len(messages) > 1
        for message in messages:
            assert len(message.encode("utf-16-le")) // 2 <= 4096
        all_text = "\n".join(messages)
        for group in groups:
            assert all_text.count(f"(ID: {group.chat_id})") == 1
