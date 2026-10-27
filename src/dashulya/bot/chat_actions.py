"""Best-effort typing whose background task is owned by one reply."""

import asyncio
import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager, suppress

from aiogram import Bot
from aiogram.types import Message

from dashulya.observability import exception_fields

logger = logging.getLogger(__name__)


async def _send_typing(bot: Bot, message: Message) -> None:
    try:
        while True:
            await bot.send_chat_action(
                chat_id=message.chat.id,
                action="typing",
                message_thread_id=message.message_thread_id
                if message.is_topic_message
                else None,
            )
            await asyncio.sleep(5)
    except Exception as error:
        logger.warning(
            "Typing indicator failed",
            extra={
                "chat_id": message.chat.id,
                "message_id": message.message_id,
                **exception_fields(error),
            },
        )


@asynccontextmanager
async def typing(bot: Bot, message: Message) -> AsyncGenerator[None, None]:
    task = asyncio.create_task(_send_typing(bot, message))
    try:
        yield
    finally:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task
