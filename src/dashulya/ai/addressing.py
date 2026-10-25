import asyncio
import json
import re

from ag2 import Agent

from dashulya.domain.messages import AddressingDecision, GroupMessage

_AGENT_NAME_PATTERN = re.compile(
    r"(?<!\w)(?:дашул(?:ь|я|и|е|ю|ей|ею)|dashulya)(?!\w)",  # ruff: ignore[ambiguous-unicode-character-string] -- Russian case endings and vocative.
    re.IGNORECASE,
)


class GroupAddressing:
    def __init__(self, agent: Agent[bool], *, bot_id: int, bot_username: str) -> None:
        self._agent = agent
        self._bot_id = bot_id
        self._bot_username = bot_username.casefold()

    async def should_reply(
        self,
        message: GroupMessage,
        *,
        recent_conversation: list[dict[str, str]],
    ) -> bool:
        return (
            await self.decide(message, recent_conversation=recent_conversation)
        ).should_reply

    async def decide(
        self,
        message: GroupMessage,
        *,
        recent_conversation: list[dict[str, str]],
    ) -> AddressingDecision:
        original = message.reply_to
        if (
            original is not None
            and original.sender_is_bot
            and original.sender_id == self._bot_id
        ):
            return AddressingDecision(True, "reply_to_bot")
        if self._bot_id in message.mentioned_user_ids or any(
            username.casefold() == self._bot_username
            for username in message.mentioned_usernames
        ):
            return AddressingDecision(True, "explicit_mention")
        if _AGENT_NAME_PATTERN.search(message.text) is None:
            return AddressingDecision(False, "name_not_found")
        # Each classification has a fresh stream; decisions never enter reply history.
        async with asyncio.timeout(12):
            reply = await self._agent.ask(
                json.dumps(
                    {
                        "current_message": message.context(),
                        "recent_conversation": recent_conversation,
                    },
                    ensure_ascii=False,
                )
            )
            decision = await reply.content()
            if decision is None:
                return AddressingDecision(False, "classifier_no_answer")
            if decision is True:
                return AddressingDecision(True, "classifier_accepted")
            return AddressingDecision(False, "classifier_rejected")
