from collections.abc import Awaitable, Callable
from typing import Any

from aiogram import BaseMiddleware
from aiogram.enums import ChatType
from aiogram.types import Message, TelegramObject

from dashulya.domain.groups import Group, GroupRegistry

GROUP_TYPES = {ChatType.GROUP, ChatType.SUPERGROUP}


class TrackGroupsMiddleware(BaseMiddleware):
    def __init__(self, groups: GroupRegistry) -> None:
        self._groups = groups

    async def __call__(
        self,
        handler: Callable[[TelegramObject, dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: dict[str, Any],
    ) -> Any:
        # This middleware is registered on the message observer.
        assert isinstance(event, Message)
        if event.chat.type in GROUP_TYPES:
            group = Group(
                event.chat.id,
                event.chat.title or "Группа без названия",
                event.chat.username,
            )
            if (
                event.left_chat_member is not None
                and event.left_chat_member.id == data["bot"].id
            ):
                await self._groups.membership(group, present=False)
            else:
                await self._groups.observe(
                    group,
                    migrated_to=event.migrate_to_chat_id,
                    migrated_from=event.migrate_from_chat_id,
                )
        return await handler(event, data)
