import logging
from contextlib import nullcontext
from dataclasses import dataclass
from time import perf_counter

from aiogram import Bot
from aiogram.types import Message, ReplyParameters

from dashulya.ai.dialogue import Dialogue
from dashulya.bot.chat_actions import typing
from dashulya.bot.formatting import render_markdown
from dashulya.bot.images import ImageInputError, TelegramImages
from dashulya.bot.messages import conversation_key, message_text
from dashulya.bot.presentation import collapse_entities
from dashulya.bot.stickers import TelegramStickers
from dashulya.domain.messages import ImageAttachment, ImageOmission, Reply, ReplyRequest
from dashulya.observability import exception_fields

logger = logging.getLogger(__name__)


@dataclass
class _DeliveryProgress:
    chunk_count: int = 0
    stage: str = "render"


class ReplyDelivery:
    def __init__(
        self,
        bot: Bot,
        dialogue: Dialogue,
        images: TelegramImages,
        stickers: TelegramStickers,
    ) -> None:
        self._bot = bot
        self._dialogue = dialogue
        self._images = images
        self._stickers = stickers

    async def reply(self, message: Message) -> None:
        # Private turns share history and include delivery in their ordering.
        async with (
            self._dialogue.delivery(conversation_key(message))
            if message.chat.id >= 0
            else nullcontext()
        ):
            await self._reply(message)

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

    async def _reply(self, message: Message) -> None:
        # Owner access and addressing have already succeeded before any download.
        assert message.from_user is not None
        started = perf_counter()
        context: dict[str, object] = {
            "chat_id": message.chat.id,
            "message_id": message.message_id,
        }
        logger.info("AI request started", extra=context)
        text = message_text(message) or (
            "Please respond to this media in the context of our conversation."
        )
        parent = message.reply_to_message
        if message.chat.id > 0 and parent is not None and parent.poll is not None:
            text = f"Reply to poll:\n{message_text(parent)}\n\n{text}"
        async with typing(self._bot, message):
            try:
                images = await self._load_images(message) if message.chat.id > 0 else ()
                reply = await self._dialogue.reply(
                    ReplyRequest(
                        conversation_key(message),
                        text,
                        user_id=message.from_user.id,
                        user_name=message.from_user.full_name,
                        sent_at=message.date,
                        message_id=message.message_id,
                        images=images,
                        reply_to_id=message.reply_to_message.message_id
                        if message.reply_to_message is not None
                        else None,
                    )
                )
                messages = reply.messages
                logger.info(
                    "AI reply generated",
                    extra={
                        **context,
                        "duration_ms": round((perf_counter() - started) * 1000),
                        "message_count": len(messages),
                    },
                )
            except ImageInputError as error:
                messages = (Reply(str(error)),)
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
                messages = (
                    Reply(
                        "Сейчас не удалось подготовить ответ. Попробуй ещё раз чуть позже."
                    ),
                )
        if not messages:
            # Reaction outcomes were already handled within the agent's turn.
            return
        delivery_started = perf_counter()
        progress = _DeliveryProgress()
        try:
            # Group generation is independent; serialize only complete deliveries.
            async with (
                self._dialogue.delivery(conversation_key(message))
                if message.chat.id < 0
                else nullcontext()
            ):
                await self._deliver(message, messages, progress)
        except Exception as error:
            logger.warning(
                "AI reply delivery failed",
                extra={
                    **context,
                    **exception_fields(error),
                    "message_count": len(messages),
                    "chunk_count": progress.chunk_count,
                    "delivery_stage": progress.stage,
                    "duration_ms": round((perf_counter() - delivery_started) * 1000),
                },
            )
            raise
        logger.info(
            "AI reply delivered",
            extra={
                **context,
                "duration_ms": round((perf_counter() - delivery_started) * 1000),
                "message_count": len(messages),
                "chunk_count": progress.chunk_count,
            },
        )

    async def _deliver(
        self, message: Message, messages: tuple[Reply, ...], progress: _DeliveryProgress
    ) -> None:
        for index, item in enumerate(messages):
            target = item.reply_to_message_id
            if target is None and index == 0:
                target = message.message_id
            reply_parameters = (
                ReplyParameters(message_id=target, allow_sending_without_reply=True)
                if target is not None
                else None
            )
            if item.sticker is not None:
                progress.stage = "sticker_lookup"
                sticker = await self._stickers.file_id(item.sticker)
                progress.stage = "send"
                await message.answer_sticker(sticker, reply_parameters=reply_parameters)
                progress.chunk_count += 1
                progress.stage = "render"
                continue
            chunks = list(render_markdown(item.text))
            collapse = sum(len(chunk) for chunk, _ in chunks) > 600 or any(
                entity.type == "pre" for _, entities in chunks for entity in entities
            )
            for chunk_index, (chunk, entities) in enumerate(chunks):
                delivery_entities = (
                    collapse_entities(
                        chunk,
                        entities,
                        markdown=item.text if chunk_index == 0 else None,
                    )
                    if collapse
                    else entities
                )
                progress.stage = "send"
                await message.answer(
                    chunk,
                    entities=delivery_entities,
                    parse_mode=None,
                    direct_messages_topic_id=message.direct_messages_topic.topic_id
                    if message.direct_messages_topic
                    else None,
                    reply_parameters=reply_parameters,
                )
                progress.chunk_count += 1
                progress.stage = "render"
