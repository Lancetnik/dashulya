from collections.abc import Awaitable, Callable
from typing import Any

from aiogram import BaseMiddleware
from aiogram.types import Message, TelegramObject


class OwnerOnlyMiddleware(BaseMiddleware):
    def __init__(self, owner_user_id: int) -> None:
        if owner_user_id <= 0:
            error_message = "OWNER_USER_ID must be a positive Telegram user ID."
            raise ValueError(error_message)
        self._owner_user_id = owner_user_id

    async def __call__(
        self,
        handler: Callable[[TelegramObject, dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: dict[str, Any],
    ) -> Any:
        # This middleware is registered on the message observer.
        assert isinstance(event, Message)
        sender = event.from_user
        if sender is None or sender.is_bot or sender.id != self._owner_user_id:
            return None
        return await handler(event, data)
