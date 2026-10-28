from collections.abc import Awaitable, Callable
from typing import Any

from aiogram import BaseMiddleware
from aiogram.enums import ChatType
from aiogram.filters import CommandStart
from aiogram.types import Message, TelegramObject

from dashulya.domain.groups import GroupRegistry


class AccessMiddleware(BaseMiddleware):
    def __init__(
        self,
        owner_user_id: int,
        groups: GroupRegistry,
        *,
        public_group_ids: tuple[int, ...] = (),
    ) -> None:
        if owner_user_id <= 0:
            error_message = "OWNER_USER_ID must be a positive Telegram user ID."
            raise ValueError(error_message)
        self._owner_user_id = owner_user_id
        self._public_group_ids = frozenset(public_group_ids)
        self._groups = groups
        self._start = CommandStart()

    async def __call__(
        self,
        handler: Callable[[TelegramObject, dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: dict[str, Any],
    ) -> Any:
        # This middleware is registered on the message observer.
        assert isinstance(event, Message)
        sender = event.from_user
        if sender is None or sender.is_bot:
            return None
        if sender.id != self._owner_user_id:
            if (
                event.chat.type not in {ChatType.GROUP, ChatType.SUPERGROUP}
                or event.chat.id not in self._public_group_ids
            ):
                return None
            # Group dialogue access does not grant owner commands or registry access.
            if (event.text or event.caption or "").lstrip().startswith("/"):
                return None
        if (
            event.chat.type in {ChatType.GROUP, ChatType.SUPERGROUP}
            and not await self._groups.is_active(event.chat.id)
            and not (
                sender.id == self._owner_user_id and await self._start(event, data["bot"])
            )
        ):
            return None
        return await handler(event, data)
