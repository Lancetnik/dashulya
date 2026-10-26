import logging
from time import perf_counter

from aiogram import Bot
from aiogram.exceptions import TelegramBadRequest
from aiogram.types import Message, ReplyParameters
from aiogram.utils.chat_action import ChatActionSender

from dashulya.ai.dialogue import Dialogue
from dashulya.bot.formatting import render_markdown, rich_message
from dashulya.observability import exception_fields

logger = logging.getLogger(__name__)


class ReplyDelivery:
    def __init__(self, bot: Bot, dialogue: Dialogue) -> None:
        self._bot = bot
        self._dialogue = dialogue

    async def reply(
        self,
        message: Message,
        *,
        group_context: dict[str, object] | None = None,
    ) -> None:
        async with self._dialogue.delivery((message.chat.id, message.message_thread_id)):
            await self._reply(message, group_context=group_context)

    async def _reply(
        self, message: Message, *, group_context: dict[str, object] | None
    ) -> None:
        # The router accepts only text messages with a sender.
        assert message.text is not None
        assert message.from_user is not None
        started = perf_counter()
        context: dict[str, object] = {
            "chat_id": message.chat.id,
            "message_id": message.message_id,
        }
        logger.info("AI request started", extra=context)
        reply_to_message_id = message.message_id
        async with ChatActionSender.typing(
            bot=self._bot,
            chat_id=message.chat.id,
            message_thread_id=message.message_thread_id,
        ):
            try:
                reply = await self._dialogue.reply(
                    (message.chat.id, message.message_thread_id),
                    message.text,
                    user_id=message.from_user.id,
                    user_name=message.from_user.full_name,
                    sent_at=message.date,
                    message_id=message.message_id,
                    group_context=group_context,
                )
                text = reply.text
                reply_to_message_id = reply.reply_to_message_id or message.message_id
                logger.info(
                    "AI reply generated",
                    extra={
                        **context,
                        "duration_ms": round((perf_counter() - started) * 1000),
                        "reply_to_message_id": reply_to_message_id,
                    },
                )
            except Exception as error:
                # Provider errors can contain request data; keep those out of logs and replies.
                logger.warning(
                    "AI reply failed",
                    extra={
                        **context,
                        **exception_fields(error),
                        "duration_ms": round((perf_counter() - started) * 1000),
                    },
                )
                text = (
                    "I couldn't get an AI reply right now. Please try again in a moment."
                )
        reply_parameters = ReplyParameters(
            message_id=reply_to_message_id, allow_sending_without_reply=True
        )
        rich = rich_message(text)
        if rich is not None:
            try:
                await message.answer_rich(rich, reply_parameters=reply_parameters)
            except TelegramBadRequest as error:
                # Only a definitive rejection permits fallback; network failures are ambiguous.
                logger.warning(
                    "Rich reply rejected; sending text entities",
                    extra={**context, **exception_fields(error)},
                )
            else:
                return
        for chunk, entities in render_markdown(text):
            await message.answer(
                chunk,
                entities=entities,
                parse_mode=None,
                reply_parameters=reply_parameters,
            )
