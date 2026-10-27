import logging
from time import perf_counter

from aiogram import Bot
from aiogram.types import Message, ReplyParameters

from dashulya.ai.dialogue import Dialogue
from dashulya.bot.chat_actions import typing
from dashulya.bot.formatting import render_markdown
from dashulya.bot.images import ImageInputError, TelegramImages
from dashulya.bot.messages import conversation_key
from dashulya.domain.messages import ImageAttachment, ImageOmission
from dashulya.observability import exception_fields

logger = logging.getLogger(__name__)


class ReplyDelivery:
    def __init__(
        self,
        bot: Bot,
        dialogue: Dialogue,
        images: TelegramImages,
    ) -> None:
        self._bot = bot
        self._dialogue = dialogue
        self._images = images

    async def reply(
        self,
        message: Message,
        *,
        group_context: dict[str, object] | None = None,
    ) -> None:
        async with self._dialogue.delivery(conversation_key(message)):
            await self._reply(message, group_context=group_context)

    async def _load_images(
        self, message: Message
    ) -> tuple[ImageAttachment | ImageOmission, ...]:
        started = perf_counter()
        images = await self._images.load(message)
        logger.info(
            "AI media loaded",
            extra={
                "chat_id": message.chat.id,
                "message_id": message.message_id,
                "duration_ms": round((perf_counter() - started) * 1000),
            },
        )
        return images

    async def _reply(
        self, message: Message, *, group_context: dict[str, object] | None
    ) -> None:
        # Owner access and addressing have already succeeded before any download.
        assert message.from_user is not None
        started = perf_counter()
        context: dict[str, object] = {
            "chat_id": message.chat.id,
            "message_id": message.message_id,
        }
        logger.info("AI request started", extra=context)
        reply_to_message_id = message.message_id
        async with typing(self._bot, message):
            try:
                images = await self._load_images(message)
                reply = await self._dialogue.reply(
                    conversation_key(message),
                    message.text
                    or message.caption
                    or "Please respond to this media in the context of our conversation.",
                    user_id=message.from_user.id,
                    user_name=message.from_user.full_name,
                    sent_at=message.date,
                    message_id=message.message_id,
                    group_context=group_context,
                    **({"images": images} if images else {}),
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
            except ImageInputError as error:
                text = str(error)
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
        if not text.strip():
            # Reaction outcomes were already handled within the agent's turn.
            return
        reply_parameters = ReplyParameters(
            message_id=reply_to_message_id, allow_sending_without_reply=True
        )
        delivery_started = perf_counter()
        for chunk, entities in render_markdown(text):
            await message.answer(
                chunk,
                entities=entities,
                parse_mode=None,
                reply_parameters=reply_parameters,
            )
        logger.info(
            "AI reply delivered",
            extra={
                **context,
                "duration_ms": round((perf_counter() - delivery_started) * 1000),
            },
        )
