from tempfile import TemporaryDirectory
from typing import Any, cast
from unittest.mock import AsyncMock, patch

from aiogram import Bot, Dispatcher
from aiogram.types import Update, User

from dashulya.domain.groups import Group, GroupRegistry
from dashulya.main import create_container
from dashulya.main.config import Settings
from dashulya.storage.groups import GroupStore
from tests.postgres import PostgresTestCase
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


class GroupTrackingTests(PostgresTestCase):
    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.bot = Bot(BOT_TOKEN)
        self.addAsyncCleanup(self.bot.session.close)
        self.requests = AsyncMock(return_value=True)
        request_patch = patch.object(self.bot.session, "make_request", self.requests)
        request_patch.start()
        self.addCleanup(request_patch.stop)
        me_patch = patch.object(
            self.bot,
            "me",
            AsyncMock(return_value=User(**BOT_USER, username="dashulya_bot")),
        )
        me_patch.start()
        self.addCleanup(me_patch.stop)
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
        self.addAsyncCleanup(self.container.close)
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
        self.assertIn("I don't know of any groups yet", self.last_reply())
        self.assertEqual(await self.groups.list_groups(), [])

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
        self.assertIn("Test group (ID: -123)", text)
        self.assertIn("Public group @public_group (ID: -100456)", text)
        self.assertLess(text.index("Public group"), text.index("Test group"))

    async def test_existing_group_is_registered_before_answering_command(self) -> None:
        await self.message("/groups@dashulya_bot", chat=GROUP_CHAT)
        self.assertIn("Test group (ID: -123)", self.last_reply())
        self.assertEqual(len(await self.groups.list_groups()), 1)

    async def test_messages_refresh_metadata_without_duplicate_groups(self) -> None:
        await self.membership(chat={**GROUP_CHAT, "username": "old_name"})
        await self.message("Hello", chat={**GROUP_CHAT, "title": "Renamed group"})
        await self.message(
            "Another message", chat={**GROUP_CHAT, "title": "Renamed group"}
        )
        self.assertEqual(await self.groups.list_groups(), [Group(-123, "Renamed group")])
        self.requests.assert_not_awaited()

    async def test_leaving_or_being_kicked_removes_group_and_rejoining_restores_it(
        self,
    ) -> None:
        for status in ("left", "kicked"):
            with self.subTest(status=status):
                await self.membership()
                fields: dict[str, Any] = {"until_date": 0} if status == "kicked" else {}
                await self.membership(status, **fields)
                await self.message("/groups")
                self.assertIn("I don't know of any groups yet", self.last_reply())
                await self.membership()
                self.assertEqual(
                    await self.groups.list_groups(), [Group(-123, "Test group")]
                )

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
        self.assertEqual(await self.groups.list_groups(), [Group(-123, "Test group")])
        await self.membership("restricted", is_member=False, until_date=0, **permissions)
        self.assertEqual(await self.groups.list_groups(), [])

    async def test_private_and_channel_membership_updates_are_ignored(self) -> None:
        await self.membership(chat=PRIVATE_CHAT)
        await self.membership(chat={"id": -100789, "type": "channel", "title": "News"})
        self.assertEqual(await self.groups.list_groups(), [])

    async def test_migration_replaces_old_group_in_both_service_messages(self) -> None:
        await self.membership()
        await self.message(chat=GROUP_CHAT, migrate_to_chat_id=-100456)
        self.assertEqual(await self.groups.list_groups(), [Group(-100456, "Test group")])
        await self.message(
            chat={"id": -100456, "type": "supergroup", "title": "Upgraded group"},
            migrate_from_chat_id=-123,
        )
        await self.message("/groups")
        self.assertEqual(
            await self.groups.list_groups(), [Group(-100456, "Upgraded group")]
        )
        self.assertNotIn("ID: -123", self.last_reply())

    async def test_incoming_migration_alone_removes_old_group(self) -> None:
        await self.membership()
        await self.message(
            chat={"id": -100456, "type": "supergroup", "title": "Upgraded group"},
            migrate_from_chat_id=-123,
        )
        self.assertEqual(
            await self.groups.list_groups(), [Group(-100456, "Upgraded group")]
        )

    async def test_existing_commands_still_work(self) -> None:
        await self.message("/start")
        self.assertIn("/help", self.last_reply())
        await self.message("/help")
        self.assertIn("/groups", self.last_reply())
        self.assertIn("/clear", self.last_reply())
        self.assertNotIn("/ping", self.last_reply())
        self.requests.reset_mock()
        for chat in (PRIVATE_CHAT, GROUP_CHAT):
            await self.message("/ping", chat=chat)
        self.requests.assert_not_awaited()

    async def test_polling_subscribes_to_membership_updates(self) -> None:
        self.assertEqual(
            self.dispatcher.resolve_used_update_types(),
            ["edited_message", "message", "my_chat_member"],
        )


class GroupPersistenceTests(PostgresTestCase):
    async def test_groups_survive_reopening_database(self) -> None:
        groups = GroupStore(self.database)
        await groups.remember(Group(-123, "Test group", "test_group"))
        reopened = GroupStore(self.database)
        self.assertEqual(
            await reopened.list_groups(), [Group(-123, "Test group", "test_group")]
        )
        await reopened.forget(-123)
        self.assertEqual(await groups.list_groups(), [])

    async def test_large_group_list_is_split_without_losing_entries(self) -> None:
        groups = [Group(-index, f"{index}: " + "💋" * 120) for index in range(1, 101)]
        store = GroupStore(self.database)
        for group in groups:
            await store.remember(group)
        messages = await GroupRegistry(store).messages()
        self.assertGreater(len(messages), 1)
        for message in messages:
            self.assertLessEqual(len(message.encode("utf-16-le")) // 2, 4096)
        all_text = "\n".join(messages)
        for group in groups:
            self.assertEqual(all_text.count(f"(ID: {group.chat_id})"), 1)
