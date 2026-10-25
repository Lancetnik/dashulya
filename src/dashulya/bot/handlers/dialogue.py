import logging
from time import perf_counter

from aiogram import Bot, F, Router
from aiogram.enums import ChatType
from aiogram.filters import Command
from aiogram.types import Message, ReplyParameters
from aiogram.utils.chat_action import ChatActionSender

from dashulya.ai.dialogue import Dialogue
from dashulya.bot.formatting import render_markdown
from dashulya.observability import exception_fields

logger = logging.getLogger(__name__)


async def clear(message: Message, dialogue: Dialogue) -> None:
    await dialogue.clear((message.chat.id, message.message_thread_id))
    await message.answer("Conversation cleared. Let's start fresh 💋")


async def chat(message: Message, bot: Bot, dialogue: Dialogue) -> None:
    # The router accepts only text messages with a sender.
    assert message.text is not None
    assert message.from_user is not None
    started = perf_counter()
    context: dict[str, object] = {
        "chat_id": message.chat.id,
        "message_id": message.message_id,
    }
    logger.info("AI request started", extra=context)
    async with ChatActionSender.typing(
        bot=bot, chat_id=message.chat.id, message_thread_id=message.message_thread_id
    ):
        try:
            text = await dialogue.reply(
                (message.chat.id, message.message_thread_id),
                message.text,
                user_id=message.from_user.id,
                user_name=message.from_user.full_name,
                sent_at=message.date,
            )
            logger.info(
                "AI reply generated",
                extra={
                    **context,
                    "duration_ms": round((perf_counter() - started) * 1000),
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
            text = "I couldn't get an AI reply right now. Please try again in a moment."
    for chunk, entities in render_markdown(text):
        await message.answer(
            chunk,
            entities=entities,
            parse_mode=None,
            reply_parameters=ReplyParameters(
                message_id=message.message_id, allow_sending_without_reply=True
            ),
        )


def create_router() -> Router:
    router = Router(name=__name__)
    router.message.filter(F.chat.type == ChatType.PRIVATE)
    router.message.register(clear, Command("clear"))
    router.message.register(
        chat,
        F.text,
        ~F.text.startswith("/"),
        F.from_user,
        ~F.from_user.is_bot,
    )
    return router
