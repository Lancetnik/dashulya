"""Per-group audiences through Telegram updates and the persistent registry."""

from collections.abc import AsyncIterator
from contextlib import AsyncExitStack
from datetime import UTC, datetime
from typing import Any
from unittest.mock import patch

import pytest
from ag2.testing import TestConfig, TrackingConfig
from aiogram import Bot, Dispatcher
from aiogram.exceptions import TelegramBadRequest, TelegramNetworkError
from aiogram.methods import (
    AnswerCallbackQuery,
    EditMessageText,
    GetChatAdministrators,
    GetChatMember,
    GetMe,
    SendMessage,
    TelegramMethod,
)
from aiogram.types import InlineKeyboardMarkup, Message, Update, User
from dishka import Provider, provide
from dishka_ag2 import AG2Scope

from dashulya.database import Database
from dashulya.database.migrate import migrate
from dashulya.domain.groups import Group, GroupRegistry
from dashulya.domain.memory import ChatMemoryService, MemoryAccess
from dashulya.main import create_container
from dashulya.main.config import Settings
from dashulya.memory.service import HybridChatMemory
from dashulya.memory.store import MessageStore
from dashulya.memory.vectors import EmbeddingCache
from dashulya.storage.groups import GroupStore
from tests.integration.test_ai_handlers import administrator
from tests.support import TEST_API_KEY, BotOverride, ModelOverrides, reply_json

CHAT: dict[str, Any] = {"id": -100123, "type": "supergroup", "title": "Group"}
OWNER: dict[str, Any] = {"id": 42, "is_bot": False, "first_name": "Owner"}
ADMIN: dict[str, Any] = {"id": 98, "is_bot": False, "first_name": "Administrator"}
MEMBER: dict[str, Any] = {"id": 99, "is_bot": False, "first_name": "Participant"}
BOT: dict[str, Any] = {
    "id": 123,
    "is_bot": True,
    "first_name": "Dashulya",
    "username": "dashulya_bot",
}


async def no_embeddings(texts: list[str]) -> list[list[float]]:
    error_message = "Settings tests must not request embeddings"
    raise AssertionError(error_message)


class SettingsMemory(Provider):
    scope = AG2Scope.APP

    @provide(override=True)
    async def memory(self, store: MessageStore) -> AsyncIterator[ChatMemoryService]:
        memory = HybridChatMemory(
            store, EmbeddingCache(store.database, no_embeddings), owner_user_id=42
        )
        try:
            yield memory
        finally:
            await memory.close()


@pytest.mark.usefixtures("_setup")
class TestGroupSettings:
    @pytest.fixture
    async def _setup(self, database: Database, cleanup: AsyncExitStack) -> None:
        self.bot = Bot("123:TEST")
        cleanup.push_async_callback(self.bot.session.close)
        cleanup.enter_context(
            patch.object(self.bot.session, "make_request", self.telegram)
        )
        self.api: list[TelegramMethod[Any]] = []
        self.admin_ids = [98]
        self.lookup_fails = False
        self.unchanged_panel = False
        self.update_id = 0
        self.model = TrackingConfig(TestConfig(reply_json("Allowed answer")))
        self.decision = TrackingConfig(TestConfig("true"))
        self.container = create_container(
            Settings(
                bot_token=self.bot.token, openai_api_key=TEST_API_KEY, owner_user_id=42
            ),
            BotOverride(self.bot),
            SettingsMemory(),
            ModelOverrides(reply=self.model, decision=self.decision),
        )
        cleanup.push_async_callback(self.container.close)
        self.groups = await self.container.get(GroupRegistry)
        await self.groups.observe(Group(CHAT["id"], CHAT["title"]))
        await self.groups.activate(CHAT["id"])
        self.dispatcher = await self.container.get(Dispatcher)

    async def telegram(self, bot: Bot, method: TelegramMethod[Any], **kwargs: Any) -> Any:
        self.api.append(method)
        if isinstance(method, GetMe):
            return User(**BOT)
        if isinstance(method, GetChatAdministrators):
            if self.lookup_fails:
                raise TelegramNetworkError(method, "Synthetic lookup failure")
            return [administrator(user_id, creator=True) for user_id in self.admin_ids]
        if isinstance(method, GetChatMember):
            return administrator(bot.id)
        if isinstance(method, EditMessageText) and self.unchanged_panel:
            raise TelegramBadRequest(method, "Bad Request: message is not modified")
        if isinstance(method, (SendMessage, EditMessageText)):
            return Message.model_validate({
                "message_id": 1000 + len(self.api),
                "date": datetime.now(UTC),
                "chat": CHAT,
                "from": BOT,
                "text": method.text,
                "reply_markup": method.reply_markup,
            })
        return True

    async def feed(
        self,
        text: str,
        *,
        sender: dict[str, Any] = OWNER,
        hidden: bool = False,
        chat: dict[str, Any] = CHAT,
        **fields: Any,
    ) -> None:
        self.update_id += 1
        entities = []
        if text.startswith("/"):
            entities = [
                {
                    "type": "bot_command",
                    "offset": 0,
                    "length": len(text.split(maxsplit=1)[0]),
                }
            ]
        elif text.startswith("@dashulya_bot"):
            entities = [{"type": "mention", "offset": 0, "length": 13}]
        await self.dispatcher.feed_update(
            self.bot,
            Update.model_validate({
                "update_id": self.update_id,
                "message": {
                    "message_id": 0 if hidden else self.update_id,
                    "ephemeral_message_id": 700 if hidden else None,
                    "date": datetime.now(UTC),
                    "chat": chat,
                    "from": sender,
                    "text": text,
                    "entities": entities,
                    **fields,
                },
            }),
        )

    async def press(
        self, data: str, *, sender: dict[str, Any] = OWNER, chat: dict[str, Any] = CHAT
    ) -> None:
        self.update_id += 1
        await self.dispatcher.feed_update(
            self.bot,
            Update.model_validate({
                "update_id": self.update_id,
                "callback_query": {
                    "id": str(self.update_id),
                    "from": sender,
                    "chat_instance": "group",
                    "data": data,
                    "message": {
                        "message_id": 1000,
                        "date": datetime.now(UTC),
                        "chat": chat,
                        "from": BOT,
                        "text": "Настройки Дашули",
                    },
                },
            }),
        )

    def answers(self) -> list[SendMessage]:
        return [item for item in self.api if isinstance(item, SendMessage)]

    async def choose(self, kind: str, label: str) -> None:
        await self.feed("/settings")
        keyboard = self.answers()[-1].reply_markup
        assert isinstance(keyboard, InlineKeyboardMarkup)
        row = keyboard.inline_keyboard[0 if kind == "reply" else 1]
        button = next(item for item in row if item.text.removeprefix("✓ ") == label)
        assert button.callback_data is not None
        await self.press(button.callback_data)

    @pytest.mark.parametrize(
        "actor", ["owner", "admin", "anonymous", "member", "foreign", "bot"]
    )
    async def test_settings_panel_requires_group_authority_even_while_paused(
        self, actor: str
    ) -> None:
        await self.groups.deactivate(CHAT["id"])
        sender = {"owner": OWNER, "admin": ADMIN}.get(actor, MEMBER)
        fields: dict[str, Any] = {}
        if actor in {"anonymous", "foreign"}:
            sender = {**BOT, "id": 1087968824}
            fields["sender_chat"] = CHAT if actor == "anonymous" else {**CHAT, "id": -456}
        if actor == "bot":
            sender = {**OWNER, "is_bot": True}
        await self.feed("/settings@dashulya_bot", sender=sender, **fields)
        if actor in {"owner", "admin", "anonymous"}:
            assert len(self.answers()) == 1
            panel = self.answers()[0]
            assert "Ответы: приостановлены" in panel.text
            assert "Обычные обращения: всем участникам" in panel.text
            assert "Шёпот (/whisper): всем участникам" in panel.text
            assert isinstance(panel.reply_markup, InlineKeyboardMarkup)
            assert [len(row) for row in panel.reply_markup.inline_keyboard] == [2, 3]
        else:
            assert self.answers() == []

    @pytest.mark.parametrize(
        "role", ["owner", "admin", "member", "demoted", "lookup_failure", "bot"]
    )
    async def test_settings_buttons_check_current_rights_in_the_panel_group(
        self, role: str
    ) -> None:
        await self.feed("/settings", sender=ADMIN)
        panel = self.answers()[-1]
        assert isinstance(panel.reply_markup, InlineKeyboardMarkup)
        data = panel.reply_markup.inline_keyboard[0][1].callback_data
        assert data is not None
        sender = {"owner": OWNER, "member": MEMBER, "bot": {**OWNER, "is_bot": True}}.get(
            role, ADMIN
        )
        if role == "demoted":
            self.admin_ids = []
        self.lookup_fails = role == "lookup_failure"
        await self.press(data, sender=sender)
        group = await self.groups.get(CHAT["id"])
        assert group is not None
        assert group.reply_audience == ("admins" if role in {"owner", "admin"} else "all")
        notices = [item for item in self.api if isinstance(item, AnswerCallbackQuery)]
        assert len(notices) == 1
        assert bool(notices[0].show_alert) == (role not in {"owner", "admin"})

    @pytest.mark.parametrize("reply_admins", [False, True])
    @pytest.mark.parametrize("whisper", ["all", "admins", "disabled"])
    @pytest.mark.parametrize("role", ["owner", "admin", "member"])
    async def test_reply_and_whisper_audiences_are_independent(
        self, reply_admins: bool, whisper: str, role: str
    ) -> None:
        await self.choose(
            "reply", "Только администраторам" if reply_admins else "Всем участникам"
        )
        await self.choose(
            "whisper",
            {
                "all": "Всем участникам",
                "admins": "Только администраторам",
                "disabled": "Выключен",
            }[whisper],
        )
        self.api.clear()
        sender = {"owner": OWNER, "admin": ADMIN, "member": MEMBER}[role]
        await self.feed("@dashulya_bot Public question", sender=sender)
        await self.feed("/whisper Private question", sender=sender, hidden=True)
        answers = self.answers()
        public = [item for item in answers if item.ephemeral_message_parameters is None]
        hidden = [
            item for item in answers if item.ephemeral_message_parameters is not None
        ]
        assert [item.text for item in public] == (
            [] if reply_admins and role == "member" else ["Allowed answer"]
        )
        assert [item.text for item in hidden] == (
            []
            if whisper == "disabled" or (whisper == "admins" and role == "member")
            else ["Allowed answer"]
        )
        assert len(self.model.calls) == len(public) + len(hidden)
        for answer in hidden:
            assert answer.ephemeral_message_parameters is not None
            assert answer.ephemeral_message_parameters.receiver_user_id == sender["id"]

    @pytest.mark.parametrize(
        "blocked", ["paused", "banned_owner", "banned_admin", "banned_member"]
    )
    async def test_pause_and_blacklist_override_both_reply_audiences(
        self, blocked: str
    ) -> None:
        sender = {"banned_owner": OWNER, "banned_admin": ADMIN}.get(blocked, MEMBER)
        if blocked == "paused":
            await self.groups.deactivate(CHAT["id"])
        else:
            await self.groups.ban(CHAT["id"], sender["id"])
        await self.feed("@dashulya_bot Public question", sender=sender)
        await self.feed("/whisper Private question", sender=sender, hidden=True)
        assert self.answers() == []
        assert self.model.calls == []

    async def test_settings_remain_available_while_paused_and_banned_and_resume_preserves_them(
        self,
    ) -> None:
        await self.groups.ban(CHAT["id"], OWNER["id"])
        await self.feed("/stop")
        await self.choose("reply", "Только администраторам")
        await self.choose("whisper", "Выключен")
        group = await self.groups.get(CHAT["id"])
        assert group == Group(
            CHAT["id"],
            "Group",
            active=False,
            reply_audience="admins",
            whisper_audience="disabled",
            capture_enabled=True,
        )
        # Choosing the same value again must acknowledge even when Telegram refuses an identical edit.
        self.unchanged_panel = True
        await self.choose("whisper", "Выключен")
        await self.feed("/start")
        group = await self.groups.get(CHAT["id"])
        assert group == Group(
            CHAT["id"],
            "Group",
            active=True,
            reply_audience="admins",
            whisper_audience="disabled",
            capture_enabled=True,
        )
        await self.feed(
            "/groups", chat={"id": 42, "type": "private", "first_name": "Owner"}
        )
        assert "обычные обращения — только администраторам" in self.answers()[-1].text
        assert "шёпот выключен" in self.answers()[-1].text

    async def test_restricted_public_requests_skip_classification_and_downloads_but_are_captured(
        self,
    ) -> None:
        await self.choose("reply", "Только администраторам")
        self.api.clear()
        await self.feed(
            "Дашуля, что на картинке?",
            sender=MEMBER,
            photo=[
                {
                    "file_id": "photo",
                    "file_unique_id": "unique",
                    "width": 2,
                    "height": 2,
                }
            ],
        )
        assert all(isinstance(item, GetChatAdministrators) for item in self.api)
        assert self.decision.calls == []
        assert self.model.calls == []
        store = await self.container.get(MessageStore)
        originals = await store.originals(MemoryAccess(CHAT["id"], False))
        assert any(
            row.text == "Дашуля, что на картинке?" and row.author_id == 99
            for row in originals
        )

    @pytest.mark.parametrize("hidden", [False, True])
    async def test_public_and_hidden_whisper_use_only_the_whisper_audience(
        self, hidden: bool
    ) -> None:
        await self.choose("reply", "Только администраторам")
        self.api.clear()
        await self.feed("/whisper@dashulya_bot Question", sender=MEMBER, hidden=hidden)
        assert [item.text for item in self.answers()] == ["Allowed answer"]
        assert self.answers()[0].ephemeral_message_parameters is not None
        await self.choose("whisper", "Выключен")
        self.api.clear()
        await self.feed("/whisper@dashulya_bot Question", sender=MEMBER, hidden=hidden)
        assert self.answers() == []


async def test_group_audiences_survive_restart_and_metadata_refresh_in_their_group(
    database: Database, database_url: str, cleanup: AsyncExitStack
) -> None:
    registry = GroupRegistry(GroupStore(database))
    for chat_id in (-123, -456):
        await registry.observe(Group(chat_id, "Group"))
    await registry.activate(-123)
    await registry.set_reply_audience(-123, "admins")
    await registry.set_whisper_audience(-123, "disabled")
    reopened_database = Database(database_url)
    cleanup.push_async_callback(reopened_database.close)
    await reopened_database.open()
    reopened = GroupRegistry(GroupStore(reopened_database))
    await reopened.observe(Group(-123, "Renamed", "renamed"))
    await reopened.membership(Group(-123, "Renamed", "renamed"), present=True)
    assert await reopened.get(-123) == Group(
        -123,
        "Renamed",
        "renamed",
        active=True,
        reply_audience="admins",
        whisper_audience="disabled",
        capture_enabled=True,
    )
    assert await reopened.get(-456) == Group(-456, "Group")
    assert await reopened.get(-999) is None


@pytest.mark.parametrize("incoming_first", [False, True])
async def test_group_audiences_follow_migration_and_reset_after_departure(
    database: Database, incoming_first: bool
) -> None:
    registry = GroupRegistry(GroupStore(database))
    await registry.observe(Group(-123, "Old group"))
    await registry.observe(Group(-100456, "New group"))
    await registry.set_reply_audience(-123, "admins")
    await registry.set_whisper_audience(-123, "disabled")
    if incoming_first:
        await registry.observe(Group(-100456, "New group"), migrated_from=-123)
        await registry.observe(Group(-123, "Old group"), migrated_to=-100456)
    else:
        await registry.observe(Group(-123, "Old group"), migrated_to=-100456)
        await registry.observe(Group(-100456, "New group"), migrated_from=-123)
    migrated = await registry.get(-100456)
    assert migrated is not None
    assert (migrated.reply_audience, migrated.whisper_audience) == ("admins", "disabled")
    assert await registry.get(-123) is None
    await registry.membership(Group(-100456, "New group"), present=False)
    await registry.membership(Group(-100456, "New group"), present=True)
    assert await registry.get(-100456) == Group(-100456, "New group")


async def test_audience_upgrade_preserves_activation_and_defaults_to_everyone(
    database: Database, database_url: str
) -> None:
    registry = GroupRegistry(GroupStore(database))
    await registry.observe(Group(-123, "Existing group"))
    await registry.activate(-123)
    async with database.connection() as connection:
        await connection.execute(
            "ALTER TABLE groups DROP COLUMN reply_audience, DROP COLUMN whisper_audience"
        )
        await connection.execute(
            "DELETE FROM schema_migrations WHERE name='0009_group_audiences.sql'"
        )
    await migrate(database_url)
    assert await registry.get(-123) == Group(
        -123, "Existing group", active=True, capture_enabled=True
    )
