from collections.abc import Awaitable, Callable
from typing import Any

from aiogram import BaseMiddleware, Bot
from aiogram.client.session.middlewares.base import (
    BaseRequestMiddleware,
    NextRequestMiddlewareType,
)
from aiogram.enums import ChatType
from aiogram.methods import SendMessage, SendRichMessage, TelegramMethod
from aiogram.methods.base import Response, TelegramType
from aiogram.types import Message, TelegramObject, Update

from dashulya.bot.messages import capture_message
from dashulya.domain.memory import MessageRecorder


class CaptureMessagesMiddleware(BaseMiddleware):
    def __init__(
        self, recorder: MessageRecorder, *, owner_user_id: int | None = None
    ) -> None:
        self._recorder = recorder
        self._owner_user_id = owner_user_id

    async def capture(self, message: Message, event_update: Update | None = None) -> None:
        private = message.chat.type == ChatType.PRIVATE
        if private and (
            self._owner_user_id is None
            or message.from_user is None
            or message.from_user.is_bot
            or message.from_user.id != self._owner_user_id
        ):
            return
        parent = message.reply_to_message
        if parent is not None and parent.chat.id == message.chat.id:
            # Telegram supplies only the direct target; later observations may fill its ancestry.
            await capture_message(
                self._recorder, parent, if_missing=True, allow_private=private
            )
        await capture_message(
            self._recorder,
            message,
            update_id=event_update.update_id if event_update else 0,
            allow_private=private,
        )

    async def __call__(
        self,
        handler: Callable[[TelegramObject, dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: dict[str, Any],
    ) -> Any:
        assert isinstance(event, Message)
        await self.capture(event, data.get("event_update"))
        return await handler(event, data)


class CaptureSentMessagesMiddleware(BaseRequestMiddleware):
    def __init__(
        self, recorder: MessageRecorder, *, owner_user_id: int | None = None
    ) -> None:
        self._recorder = recorder
        self._owner_user_id = owner_user_id

    async def __call__(
        self,
        make_request: NextRequestMiddlewareType[TelegramType],
        bot: Bot,
        method: TelegramMethod[TelegramType],
    ) -> Response[TelegramType]:
        response = await make_request(bot, method)
        if isinstance(response, Message) and isinstance(
            method, (SendMessage, SendRichMessage)
        ):
            text = (
                method.text
                if isinstance(method, SendMessage)
                else method.rich_message.markdown
            )
            await capture_message(
                self._recorder,
                response.model_copy(
                    update={
                        "message_thread_id": method.message_thread_id
                        or response.message_thread_id,
                        "is_topic_message": response.is_topic_message
                        or method.message_thread_id is not None,
                    }
                ),
                text=response.text or text,
                reply_to_id=method.reply_parameters.message_id
                if method.reply_parameters
                else None,
                allow_private=response.chat.type == ChatType.PRIVATE
                and response.chat.id == self._owner_user_id,
            )
        return response
