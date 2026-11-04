import logging
from collections.abc import Awaitable, Callable
from typing import Any

from aiogram import BaseMiddleware, Bot
from aiogram.enums import ChatMemberStatus, ChatType
from aiogram.exceptions import TelegramAPIError
from aiogram.filters import Command
from aiogram.types import CallbackQuery, Message, TelegramObject

from dashulya.domain.groups import GroupRegistry
from dashulya.observability import exception_fields

logger = logging.getLogger(__name__)


async def _is_chat_admin(bot: Bot, chat_id: int, user_id: int) -> bool:
    try:
        administrators = await bot.get_chat_administrators(chat_id, request_timeout=5)
    except (TelegramAPIError, TimeoutError) as error:
        logger.warning(
            "Group administrator lookup failed",
            extra={"chat_id": chat_id, "user_id": user_id, **exception_fields(error)},
        )
        return False
    return any(
        member.user.id == user_id
        and member.status in {ChatMemberStatus.CREATOR, ChatMemberStatus.ADMINISTRATOR}
        for member in administrators
    )


class AccessMiddleware(BaseMiddleware):
    def __init__(
        self,
        owner_user_id: int,
        groups: GroupRegistry,
    ) -> None:
        if owner_user_id <= 0:
            error_message = "OWNER_USER_ID must be a positive Telegram user ID."
            raise ValueError(error_message)
        self._owner_user_id = owner_user_id
        self._groups = groups
        self._group_commands = Command("start", "stop", "ban", "unban", "settings")
        self._admin_commands = Command("stop", "ban", "unban", "settings")
        self._whisper_command = Command("whisper")

    async def __call__(
        self,
        handler: Callable[[TelegramObject, dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: dict[str, Any],
    ) -> Any:
        if isinstance(event, CallbackQuery):
            message = event.message
            if (
                event.from_user.is_bot
                or not isinstance(message, Message)
                or message.chat.type not in {ChatType.GROUP, ChatType.SUPERGROUP}
                or (
                    event.from_user.id != self._owner_user_id
                    and not await _is_chat_admin(
                        data["bot"], message.chat.id, event.from_user.id
                    )
                )
            ):
                await event.answer(
                    "Настройки доступны владельцу бота и администраторам этой группы.",
                    show_alert=True,
                )
                return None
            return await handler(event, data)
        # This middleware is registered on the message observer.
        assert isinstance(event, Message)
        if (
            event.chat.type in {ChatType.GROUP, ChatType.SUPERGROUP}
            and event.ephemeral_message_id is None
            and event.sender_chat is not None
            and event.sender_chat.id == event.chat.id
            and await self._admin_commands(event, data["bot"])
        ):
            # Telegram identifies anonymous administrators by their own sender_chat.
            return await handler(event, data)
        sender = event.from_user
        if (
            sender is None
            or sender.is_bot
            or (event.ephemeral_message_id is not None and event.sender_chat is not None)
        ):
            return None
        whisper = event.ephemeral_message_id is not None or bool(
            event.chat.type in {ChatType.GROUP, ChatType.SUPERGROUP}
            and await self._whisper_command(event, data["bot"])
        )
        if sender.id != self._owner_user_id:
            if event.chat.type not in {ChatType.GROUP, ChatType.SUPERGROUP}:
                return None
            # Administrative commands remain available independently of reply access.
            if not whisper and (event.text or event.caption or "").lstrip().startswith(
                "/"
            ):
                bot = data["bot"]
                if not await self._admin_commands(event, bot):
                    return None
                if not await _is_chat_admin(bot, event.chat.id, sender.id):
                    return None
                return await handler(event, data)
        if (
            event.chat.type in {ChatType.GROUP, ChatType.SUPERGROUP}
            and not whisper
            and sender.id == self._owner_user_id
            and await self._group_commands(event, data["bot"])
        ):
            return await handler(event, data)
        if event.chat.type in {ChatType.GROUP, ChatType.SUPERGROUP}:
            group = await self._groups.get(event.chat.id)
            if (
                group is None
                or not group.active
                or await self._groups.is_banned(event.chat.id, sender.id)
            ):
                return None
            audience = group.whisper_audience if whisper else group.reply_audience
            if audience == "disabled" or (
                audience == "admins"
                and sender.id != self._owner_user_id
                and not await _is_chat_admin(data["bot"], event.chat.id, sender.id)
            ):
                return None
        return await handler(event, data)
