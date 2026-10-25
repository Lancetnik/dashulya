import logging

from aiogram import F, Router
from aiogram.enums import ChatType, MessageEntityType
from aiogram.filters import Command
from aiogram.types import Message
from dishka.integrations.aiogram import FromDishka, inject

from dashulya.ai.addressing import GroupAddressing
from dashulya.ai.dialogue import Dialogue
from dashulya.bot.replies import ReplyDelivery
from dashulya.domain.messages import GroupMessage
from dashulya.observability import exception_fields

logger = logging.getLogger(__name__)


def group_message(message: Message, *, include_reply: bool = True) -> GroupMessage:
    sender = message.from_user
    text = message.text or message.caption or ""
    return GroupMessage(
        message_id=message.message_id,
        text=text,
        sent_at=message.date,
        sender_id=sender.id if sender else None,
        sender_name=sender.full_name if sender else None,
        sender_is_bot=sender.is_bot if sender else False,
        mentioned_usernames=tuple(
            entity.extract_from(text).removeprefix("@")
            for entity in message.entities or []
            if entity.type == MessageEntityType.MENTION
        ),
        mentioned_user_ids=tuple(
            entity.user.id
            for entity in message.entities or []
            if entity.type == MessageEntityType.TEXT_MENTION and entity.user is not None
        ),
        reply_to=(
            group_message(message.reply_to_message, include_reply=False)
            if include_reply and message.reply_to_message is not None
            else None
        ),
    )


@inject
async def group_chat(
    message: Message,
    dialogue: FromDishka[Dialogue],
    addressing: FromDishka[GroupAddressing],
    delivery: FromDishka[ReplyDelivery],
) -> None:
    incoming = group_message(message)
    try:
        should_reply = await addressing.should_reply(
            incoming,
            recent_conversation=await dialogue.recent_context((
                message.chat.id,
                message.message_thread_id,
            )),
        )
    except Exception as error:
        logger.warning(
            "Group addressing decision failed",
            extra={
                "chat_id": message.chat.id,
                "message_id": message.message_id,
                **exception_fields(error),
            },
        )
        return
    logger.info(
        "Group addressing decision completed",
        extra={
            "chat_id": message.chat.id,
            "message_id": message.message_id,
            "should_reply": should_reply,
        },
    )
    if should_reply:
        await delivery.reply(message, group_context=incoming.context())


@inject
async def clear(message: Message, dialogue: FromDishka[Dialogue]) -> None:
    await dialogue.clear((message.chat.id, message.message_thread_id))
    await message.answer("Conversation cleared. Let's start fresh 💋")


@inject
async def chat(message: Message, delivery: FromDishka[ReplyDelivery]) -> None:
    await delivery.reply(message)


def create_router() -> Router:
    router = Router(name=__name__)
    router.message.register(clear, Command("clear"), F.chat.type == ChatType.PRIVATE)
    router.message.register(
        chat,
        F.text,
        ~F.text.startswith("/"),
        F.from_user,
        ~F.from_user.is_bot,
        F.chat.type == ChatType.PRIVATE,
    )
    router.message.register(
        group_chat,
        F.chat.type.in_({ChatType.GROUP, ChatType.SUPERGROUP}),
        F.text,
        ~F.text.startswith("/"),
        F.from_user,
        ~F.from_user.is_bot,
    )
    return router
