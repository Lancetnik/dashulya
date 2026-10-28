import logging
import re
from time import perf_counter

from aiogram import Bot
from aiogram.exceptions import (
    TelegramAPIError,
    TelegramForbiddenError,
    TelegramNetworkError,
    TelegramRetryAfter,
    TelegramServerError,
    TelegramUnauthorizedError,
)
from aiogram.types import ReactionTypeEmoji

from dashulya.domain.reactions import ReactionResult
from dashulya.observability import exception_fields

logger = logging.getLogger(__name__)


# Match only known error markers; never return Telegram's free-form description.
_REJECTIONS = (
    (
        r"\breaction_invalid\b",
        "reaction_invalid",
        (
            "This emoji is invalid or not allowed for this message or chat. "
            "Choose another allowed emoji; do not repeat this one."
        ),
    ),
    (
        r"\breaction_empty\b",
        "invalid_emoji",
        "An empty reaction is not valid. Provide one supported reaction emoji.",
    ),
    (
        r"\breactions_too_many\b",
        "reactions_too_many",
        (
            "The message has reached its limit of distinct reaction types. "
            "Do not try adding another new reaction type to this message."
        ),
    ),
    (
        r"\b(?:message to react not found|message_id_invalid|msg_id_invalid)\b",
        "message_not_found",
        (
            "The target message was not found or is unavailable to this bot. "
            "Do not retry this message ID."
        ),
    ),
    (
        r"\bchat not found\b",
        "chat_not_found",
        (
            "The current chat was not found or is inaccessible to this bot. "
            "Changing the emoji will not fix this."
        ),
    ),
    (
        r"\bbot was kicked\b",
        "bot_removed",
        "The bot was removed from this chat. It cannot set reactions there.",
    ),
    (
        r"\bbot is not a member\b",
        "bot_not_member",
        "The bot is not a member of this chat. It cannot set reactions there.",
    ),
    (
        r"\bbot was blocked by the user\b",
        "bot_blocked",
        "The user blocked the bot. It cannot set reactions in this private chat.",
    ),
    (
        r"\b(?:not enough rights|chat_admin_required|chat_write_forbidden|user_banned_in_channel)\b",
        "permission_denied",
        (
            "The bot lacks permission to set this reaction. "
            "Changing the emoji will not fix the permission problem."
        ),
    ),
)


def _rejection(error: TelegramAPIError, message_id: int, emoji: str) -> ReactionResult:
    for pattern, reason, message in _REJECTIONS:
        if re.search(pattern, error.message, re.IGNORECASE):
            return ReactionResult("rejected", message_id, emoji, reason, message=message)
    if isinstance(error, TelegramForbiddenError):
        reason = "permission_denied"
        message = "Telegram denied this bot access. Changing the emoji will not fix this."
    elif isinstance(error, TelegramUnauthorizedError):
        reason = "bot_unauthorized"
        message = "Telegram could not authenticate the bot. The owner must fix its configuration."
    else:
        reason = "telegram_rejected"
        message = (
            "Telegram rejected the reaction for an unrecognized reason. "
            "Do not assume another emoji will fix it or claim the reaction succeeded."
        )
    return ReactionResult("rejected", message_id, emoji, reason, message=message)


class TelegramReactions:
    def __init__(self, bot: Bot) -> None:
        self._bot = bot

    async def set(self, chat_id: int, message_id: int, emoji: str) -> ReactionResult:
        # Telegram spells some supported emoji without the presentation selector.
        emoji = emoji.strip().replace("\ufe0f", "")
        if type(message_id) is not int or message_id <= 0:
            return ReactionResult(
                "rejected",
                message_id,
                emoji,
                "invalid_message_id",
                message="Use a positive Telegram message ID observed in this chat.",
            )
        if not emoji or len(emoji) > 16:
            return ReactionResult(
                "rejected",
                message_id,
                emoji,
                "invalid_emoji",
                message="Provide one nonempty supported reaction emoji.",
            )
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
        except TelegramRetryAfter as error:
            logger.warning(
                "Reaction rate limited",
                extra={
                    **context,
                    "outcome": "rate_limited",
                    "retry_after": error.retry_after,
                    "duration_ms": round((perf_counter() - started) * 1000),
                    **exception_fields(error),
                },
            )
            return ReactionResult(
                "rate_limited",
                message_id,
                emoji,
                "telegram_rate_limited",
                retry_after=error.retry_after,
                message=(
                    "Telegram is rate limiting reactions. Wait at least "
                    f"{error.retry_after} seconds before trying again. "
                    "Do not try a different emoji or another message to bypass "
                    "this limit."
                ),
            )
        except TelegramAPIError as error:
            result = _rejection(error, message_id, emoji)
            logger.warning(
                "Reaction rejected",
                extra={
                    **context,
                    "outcome": "rejected",
                    "error_code": result.reason,
                    "duration_ms": round((perf_counter() - started) * 1000),
                    **exception_fields(error),
                },
            )
            return result
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
