from aiogram import F, Router
from aiogram.filters import Command
from aiogram.types import Message, Update
from dishka.integrations.aiogram import FromDishka, inject

from dashulya.bot.messages import message_text
from dashulya.bot.middlewares.group_tracking import GROUP_TYPES
from dashulya.bot.replies import ReplyDelivery


@inject
async def whisper(
    message: Message, event_update: Update, delivery: FromDishka[ReplyDelivery]
) -> None:
    assert message.bot is not None
    command = await Command("whisper")(message, message.bot)
    text = (
        command["command"].args or ""
        if isinstance(command, dict)
        else message_text(message)
    )
    await delivery.whisper(message, text.strip(), update_id=event_update.update_id)


def create_router() -> Router:
    router = Router(name=__name__)
    router.message.filter(F.chat.type.in_(GROUP_TYPES))
    router.message.register(whisper, Command("whisper"))
    # All hidden updates stay on this path, including replies and other commands.
    router.message.register(whisper, F.ephemeral_message_id.is_not(None))
    return router
