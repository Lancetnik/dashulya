import asyncio
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from zoneinfo import ZoneInfo

from ag2 import Agent, MemoryStream
from ag2.events import BaseEvent, ModelMessage, ModelRequest, ModelResponse, TextInput

from dashulya.domain.memory import ChatHistory, MemoryMessage
from dashulya.domain.messages import ConversationKey, Reply


@dataclass
class Conversation:
    stream: MemoryStream = field(default_factory=MemoryStream)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    delivery_lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class Dialogue:
    def __init__(
        self,
        agent: Agent[Reply],
        *,
        timezone: str = "Europe/Moscow",
        history: ChatHistory | None = None,
        bot_id: int | None = None,
    ) -> None:
        self._agent = agent
        self._timezone = ZoneInfo(timezone)
        self._history = history
        self._bot_id = bot_id
        self._conversations: dict[ConversationKey, Conversation] = {}

    def _conversation(self, key: ConversationKey) -> Conversation:
        if key not in self._conversations:
            self._conversations[key] = Conversation()
        return self._conversations[key]

    @asynccontextmanager
    async def delivery(self, key: ConversationKey) -> AsyncGenerator[None, None]:
        # Keep the next reply behind persistence of the preceding Telegram delivery.
        async with self._conversation(key).delivery_lock:
            yield

    async def _group_history(
        self, key: ConversationKey, message_id: int
    ) -> list[MemoryMessage]:
        if self._history is None:
            error_message = "Group replies require a chat history store."
            raise ValueError(error_message)
        return await self._history.context(key, message_id, limit=20, parent_hops=5)

    def _text(
        self, text: str, *, sent_at: datetime, user_id: int | None, user_name: str
    ) -> str:
        if sent_at.utcoffset() is None:
            error_message = "Message send time must include a timezone."
            raise ValueError(error_message)
        local_time = sent_at.astimezone(self._timezone)
        offset = local_time.strftime("%z")
        return (
            f"Sent at: {local_time:%d %B %Y (%A) %H:%M:%S}\n"
            f"Timezone: {self._timezone.key} (UTC{offset[:3]}:{offset[3:]})\n"
            f"{user_name} (Telegram user ID: {user_id}):\n{text}"
        )

    def _history_text(self, message: MemoryMessage) -> str:
        text = self._text(
            message.text,
            sent_at=message.sent_at,
            user_id=message.author_id,
            user_name=message.author,
        )
        text += f"\nTelegram message ID: {message.message_id}"
        if message.reply_to_id is not None:
            text += f"\nReply to message ID: {message.reply_to_id}"
        if message.media_type:
            text += f"\n[Attachment: {message.media_type}; contents unavailable]"
        return text

    def _role(self, message: MemoryMessage) -> str:
        return (
            "assistant"
            if self._bot_id is not None and message.author_id == self._bot_id
            else "user"
        )

    async def recent_context(
        self, key: ConversationKey, *, through_message_id: int | None = None
    ) -> list[dict[str, str]]:
        if through_message_id is not None:
            return [
                {"role": self._role(message), "text": self._history_text(message)}
                for message in await self._group_history(key, through_message_id)
                if message.message_id != through_message_id
            ]
        conversation = self._conversation(key)
        async with conversation.lock:
            recent: list[dict[str, str]] = []
            for event in await conversation.stream.history.get_events():
                if isinstance(event, ModelRequest):
                    text = "\n".join(
                        part.content
                        for part in event.parts
                        if isinstance(part, TextInput)
                    )
                    recent.append({"role": "user", "text": text[:1000]})
                elif isinstance(event, ModelResponse) and event.content:
                    recent.append({"role": "assistant", "text": event.content[:1000]})
            return recent[-10:]

    async def reply(
        self,
        key: ConversationKey,
        text: str,
        *,
        user_id: int,
        user_name: str,
        sent_at: datetime,
        group_context: dict[str, object] | None = None,
    ) -> Reply:
        message = self._text(text, sent_at=sent_at, user_id=user_id, user_name=user_name)
        conversation = self._conversation(key)
        async with conversation.lock:
            stream = conversation.stream
            allowed_reply_ids: set[int] = set()
            if group_context is not None:
                message_id = group_context.get("message_id")
                if not isinstance(message_id, int):
                    error_message = "Group replies require a Telegram message ID."
                    raise ValueError(error_message)
                stream = MemoryStream()
                allowed_reply_ids.add(message_id)
                events: list[BaseEvent] = []
                for previous in await self._group_history(key, message_id):
                    allowed_reply_ids.add(previous.message_id)
                    if previous.message_id == message_id:
                        continue
                    content = self._history_text(previous)
                    if self._role(previous) == "assistant":
                        events.append(ModelResponse(ModelMessage(content)))
                    else:
                        events.append(ModelRequest([TextInput(content)]))
                await stream.history.replace(events)
                message += f"\nTelegram message ID: {message_id}"
                parent = group_context.get("reply_to")
                if isinstance(parent, dict):
                    message += f"\nReply to message ID: {parent.get('message_id')}"
            async with asyncio.timeout(90):
                reply = await self._agent.ask(
                    message,
                    stream=stream,
                    variables={"user_id": user_id, "chat_id": key[0]},
                )
                result = await reply.content()
            if result is None or not result.text.strip():
                return Reply("I couldn't generate a text reply. Please try again.")
            return Reply(
                result.text.strip(),
                result.reply_to_message_id
                if result.reply_to_message_id in allowed_reply_ids
                else None,
            )

    async def clear(self, key: ConversationKey) -> None:
        conversation = self._conversations.get(key)
        if conversation is not None:
            # Reuse the lock so queued messages cannot continue on a discarded stream.
            async with conversation.delivery_lock, conversation.lock:
                conversation.stream = MemoryStream()
