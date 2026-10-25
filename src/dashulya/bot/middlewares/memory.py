from collections.abc import Awaitable, Callable
from typing import Any

from aiogram import BaseMiddleware, Bot
from aiogram.client.session.middlewares.base import (
    BaseRequestMiddleware,
    NextRequestMiddlewareType,
)
from aiogram.methods import SendMessage, SendRichMessage, TelegramMethod
from aiogram.methods.base import Response, TelegramType
from aiogram.types import Message, TelegramObject, Update

from dashulya.bot.messages import capture_message
from dashulya.domain.memory import MessageRecorder


class CaptureMessagesMiddleware(BaseMiddleware):
    def __init__(self, recorder: MessageRecorder) -> None:
        self._recorder = recorder

    async def capture(self, message: Message, event_update: Update | None = None) -> None:
        parent = message.reply_to_message
        if parent is not None and parent.chat.id == message.chat.id:
            # Telegram supplies only the direct target; later observations may fill its ancestry.
            await capture_message(self._recorder, parent, if_missing=True)
        await capture_message(
            self._recorder,
            message,
            update_id=event_update.update_id if event_update else 0,
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
    def __init__(self, recorder: MessageRecorder) -> None:
        self._recorder = recorder

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
                    }
                ),
                text=response.text or text,
                reply_to_id=method.reply_parameters.message_id
                if method.reply_parameters
                else None,
            )
        return response
