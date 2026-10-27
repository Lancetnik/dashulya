import logging
from time import perf_counter

from aiogram import Bot
from aiogram.exceptions import TelegramBadRequest
from aiogram.types import Message, ReplyParameters
from aiogram.utils.chat_action import ChatActionSender

from dashulya.ai.dialogue import Dialogue
from dashulya.bot.formatting import render_markdown, rich_message
from dashulya.bot.images import ImageInputError, TelegramImages
from dashulya.bot.messages import conversation_key
from dashulya.domain.reactions import MessageReactions
from dashulya.observability import exception_fields

logger = logging.getLogger(__name__)


class ReplyDelivery:
    def __init__(
        self,
        bot: Bot,
        dialogue: Dialogue,
        images: TelegramImages,
        reactions: MessageReactions,
    ) -> None:
        self._bot = bot
        self._dialogue = dialogue
        self._images = images
        self._reactions = reactions

    async def reply(
        self,
        message: Message,
        *,
        group_context: dict[str, object] | None = None,
    ) -> None:
        async with self._dialogue.delivery(conversation_key(message)):
            await self._reply(message, group_context=group_context)

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
        reaction: str | None = None
        async with ChatActionSender.typing(
            bot=self._bot,
            chat_id=message.chat.id,
            message_thread_id=message.message_thread_id
            if message.is_topic_message
            else None,
        ):
            try:
                images = await self._images.load(message)
                reply = await self._dialogue.reply(
                    conversation_key(message),
                    message.text
                    or message.caption
                    or (
                        "Please respond to the sticker."
                        if message.sticker
                        else "Please describe the attached image."
                    ),
                    user_id=message.from_user.id,
                    user_name=message.from_user.full_name,
                    sent_at=message.date,
                    message_id=message.message_id,
                    group_context=group_context,
                    **({"images": images} if images else {}),
                )
                text, reaction = reply.text, reply.reaction
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
        if reaction is not None and not text.strip():
            outcome = await self._reactions.set(
                message.chat.id, message.message_id, reaction
            )
            if outcome.status != "rejected":
                return
            text = "I couldn't add that reaction. Please try another emoji."
        elif not text.strip():
            # Reaction tools already performed the turn's visible response.
            return
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
