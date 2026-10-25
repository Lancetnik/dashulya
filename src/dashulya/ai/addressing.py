import asyncio
import json
import re

from ag2 import Agent

from dashulya.domain.messages import GroupMessage

_AGENT_NAME_PATTERN = re.compile(
    r"(?<!\w)(?:дашул(?:я|и|е|ю|ей|ею)|dashulya)(?!\w)",  # ruff: ignore[ambiguous-unicode-character-string] -- Russian case endings.
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
        original = message.reply_to
        if (
            original is not None
            and original.sender_is_bot
            and original.sender_id == self._bot_id
        ):
            return True
        if self._bot_id in message.mentioned_user_ids or any(
            username.casefold() == self._bot_username
            for username in message.mentioned_usernames
        ):
            return True
        if _AGENT_NAME_PATTERN.search(message.text) is None:
            return False
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
            return await reply.content() is True
