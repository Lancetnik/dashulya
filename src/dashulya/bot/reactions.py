import logging
from time import perf_counter

from aiogram import Bot
from aiogram.exceptions import TelegramAPIError, TelegramNetworkError, TelegramServerError
from aiogram.types import ReactionTypeEmoji

from dashulya.domain.reactions import ReactionResult
from dashulya.observability import exception_fields

logger = logging.getLogger(__name__)


class TelegramReactions:
    def __init__(self, bot: Bot) -> None:
        self._bot = bot

    async def set(self, chat_id: int, message_id: int, emoji: str) -> ReactionResult:
        # Telegram spells some supported emoji without the presentation selector.
        emoji = emoji.strip().replace("\ufe0f", "")
        if type(message_id) is not int or message_id <= 0:
            return ReactionResult("rejected", message_id, emoji, "invalid_message_id")
        if not emoji or len(emoji) > 16:
            return ReactionResult("rejected", message_id, emoji, "invalid_emoji")
        started = perf_counter()
        context: dict[str, object] = {
            "chat_id": chat_id,
            "message_id": message_id,
            "emoji": emoji,
        }
        try:
            sent = await self._bot.set_message_reaction(
                chat_id=chat_id,
                message_id=message_id,
                reaction=[ReactionTypeEmoji(emoji=emoji)],
            )
        except (TelegramNetworkError, TelegramServerError) as error:
            # Delivery may have succeeded. Neither retry nor send another response.
            logger.warning(
                "Reaction delivery unconfirmed",
                extra={
                    **context,
                    "outcome": "unknown",
                    "duration_ms": round((perf_counter() - started) * 1000),
                    **exception_fields(error),
                },
            )
            return ReactionResult("unknown", message_id, emoji, "delivery_unconfirmed")
        except TelegramAPIError as error:
            # Includes disabled/unsupported reactions, missing messages and permissions.
            logger.warning(
                "Reaction rejected",
                extra={
                    **context,
                    "outcome": "rejected",
                    "duration_ms": round((perf_counter() - started) * 1000),
                    **exception_fields(error),
                },
            )
            return ReactionResult("rejected", message_id, emoji, "telegram_rejected")
        except Exception as error:
            # Free-form errors and credentials must never enter the agent's history.
            logger.warning(
                "Reaction delivery unconfirmed",
                extra={
                    **context,
                    "outcome": "unknown",
                    "duration_ms": round((perf_counter() - started) * 1000),
                    **exception_fields(error),
                },
            )
            return ReactionResult("unknown", message_id, emoji, "delivery_unconfirmed")
        result = ReactionResult("sent" if sent else "rejected", message_id, emoji)
        logger.log(
            logging.INFO if sent else logging.WARNING,
            "Reaction delivered" if sent else "Reaction rejected",
            extra={
                **context,
                "outcome": result.status,
                "duration_ms": round((perf_counter() - started) * 1000),
            },
        )
        return result
