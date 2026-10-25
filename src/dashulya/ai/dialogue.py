import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from zoneinfo import ZoneInfo

from ag2 import Agent, MemoryStream

ConversationKey = tuple[int, int | None]


@dataclass
class Conversation:
    stream: MemoryStream = field(default_factory=MemoryStream)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class Dialogue:
    def __init__(self, agent: Agent, *, timezone: str = "Europe/Moscow") -> None:
        self._agent = agent
        self._timezone = ZoneInfo(timezone)
        self._conversations: dict[ConversationKey, Conversation] = {}

    def _conversation(self, key: ConversationKey) -> Conversation:
        if key not in self._conversations:
            self._conversations[key] = Conversation()
        return self._conversations[key]

    async def reply(
        self,
        key: ConversationKey,
        text: str,
        *,
        user_id: int,
        user_name: str,
        sent_at: datetime,
    ) -> str:
        if sent_at.utcoffset() is None:
            error_message = "Message send time must include a timezone."
            raise ValueError(error_message)
        local_time = sent_at.astimezone(self._timezone)
        offset = local_time.strftime("%z")
        conversation = self._conversation(key)
        async with conversation.lock:
            async with asyncio.timeout(90):
                reply = await self._agent.ask(
                    f"Sent at: {local_time:%d %B %Y (%A) %H:%M:%S}\n"
                    f"Timezone: {self._timezone.key} (UTC{offset[:3]}:{offset[3:]})\n"
                    f"{user_name} (Telegram user ID: {user_id}):\n{text}",
                    stream=conversation.stream,
                    variables={"user_id": user_id, "chat_id": key[0]},
                )
            return (
                reply.body or ""
            ).strip() or "I couldn't generate a text reply. Please try again."

    async def clear(self, key: ConversationKey) -> None:
        conversation = self._conversations.get(key)
        if conversation is not None:
            # Reuse the lock so queued messages cannot continue on a discarded stream.
            async with conversation.lock:
                conversation.stream = MemoryStream()
