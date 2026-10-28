import asyncio
from dataclasses import dataclass, field
from typing import Annotated

from ag2 import Depends, Inject, Variable
from ag2.tools import tool
from dishka_ag2 import FromDishka, inject

from dashulya.domain.reactions import MessageReactions, ReactionResult


@dataclass
class ReactionTurn:
    attempted: bool = False
    uncertain: dict[int, ReactionResult] = field(default_factory=dict)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def set(
        self,
        chat_id: int,
        message_id: int,
        emoji: str,
        reactions: MessageReactions,
    ) -> ReactionResult:
        async with self.lock:
            if message_id in self.uncertain:
                return self.uncertain[message_id]
            result = await reactions.set(chat_id, message_id, emoji)
            self.attempted = True
            if result.status == "unknown":
                self.uncertain[message_id] = result
            return result


@dataclass(frozen=True)
class TurnReactions:
    reactions: MessageReactions
    chat_id: int | None
    turn: ReactionTurn | None

    async def set(self, message_id: int, emoji: str) -> ReactionResult:
        if self.chat_id is None or self.turn is None:
            return ReactionResult("rejected", message_id, emoji, "missing_chat_context")
        return await self.turn.set(self.chat_id, message_id, emoji, self.reactions)


@inject
async def get_turn_reactions(
    chat_id: Annotated[int | None, Variable(default=None)],
    turn: Annotated[ReactionTurn | None, Inject("reaction_turn", default=None)],
    reactions: FromDishka[MessageReactions],
) -> TurnReactions:
    if type(chat_id) is not int or chat_id == 0 or not isinstance(turn, ReactionTurn):
        return TurnReactions(reactions, None, None)
    return TurnReactions(reactions, chat_id, turn)


@tool
async def set_reaction(
    message_id: int,
    emoji: str,
    *,
    reactions: Annotated[TurnReactions, Depends(get_turn_reactions, cast=False)],
) -> ReactionResult:
    """Set this bot's reaction on a message in the current chat, replacing its old one.

    Use an observed message ID or original source from this same chat, including
    IDs outside the selected reply context. Chat identity is application-bound.
    Supported emoji (chat settings may restrict them further):
    ❤ 👍 👎 🔥 🥰 👏 😁 🤔 🤯 😱 🤬 😢 🎉 🤩 🤮 💩 🙏 👌 🕊 🤡 🥱 🥴 😍 🐳
    ❤‍🔥 🌚 🌭 💯 🤣 ⚡ 🍌 🏆 💔 🤨 😐 🍓 🍾 💋 🖕 😈 😴 😭 🤓 👻 👨‍💻 👀 🎃
    🙈 😇 😨 🤝 ✍ 🤗 🫡 🎅 🎄 ☃ 💅 🤪 🗿 🆒 💘 🙉 🦄 😘 💊 🙊 😎 👾 🤷‍♂ 🤷 🤷‍♀ 😡.
    Only sent confirms delivery. For rejected, follow reason/message before
    choosing another allowed emoji. For rate_limited, wait at least retry_after
    seconds; switching emoji or targets cannot bypass it. For unknown, delivery is
    unconfirmed: do not retry that target during this turn.
    """
    return await reactions.set(message_id, emoji)
