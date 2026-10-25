from collections.abc import Awaitable, Callable
from typing import Any

from aiogram import BaseMiddleware
from aiogram.enums import ChatType
from aiogram.types import Message, TelegramObject

from dashulya.storage.groups import Group, GroupStore

GROUP_TYPES = {ChatType.GROUP, ChatType.SUPERGROUP}


class TrackGroupsMiddleware(BaseMiddleware):
    async def __call__(
        self,
        handler: Callable[[TelegramObject, dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: dict[str, Any],
    ) -> Any:
        # This middleware is registered on the message observer.
        assert isinstance(event, Message)
        if event.chat.type in GROUP_TYPES:
            groups: GroupStore = data["groups"]
            if event.migrate_to_chat_id is not None:
                group = Group(
                    event.migrate_to_chat_id, event.chat.title or "Untitled group"
                )
                groups.remember(group, previous_chat_id=event.chat.id)
            else:
                group = Group(
                    event.chat.id,
                    event.chat.title or "Untitled group",
                    event.chat.username,
                )
                groups.remember(group, previous_chat_id=event.migrate_from_chat_id)
        return await handler(event, data)
