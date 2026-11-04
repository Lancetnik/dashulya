from dataclasses import dataclass
from typing import Literal, Protocol

type ReplyAudience = Literal["all", "admins"]
type WhisperAudience = Literal["all", "admins", "disabled"]


@dataclass(frozen=True)
class Group:
    chat_id: int
    title: str
    username: str | None = None
    active: bool = False
    reply_audience: ReplyAudience = "all"
    whisper_audience: WhisperAudience = "all"
    capture_enabled: bool = False


class GroupRepository(Protocol):
    async def remember(
        self, group: Group, *, previous_chat_id: int | None = None
    ) -> None: ...

    async def forget(self, chat_id: int) -> None: ...

    async def list_groups(self) -> list[Group]: ...

    async def get(self, chat_id: int) -> Group | None: ...

    async def set_reply_audience(self, chat_id: int, audience: ReplyAudience) -> None: ...

    async def set_whisper_audience(
        self, chat_id: int, audience: WhisperAudience
    ) -> None: ...

    async def activate(self, chat_id: int) -> None: ...

    async def deactivate(self, chat_id: int) -> None: ...

    async def is_active(self, chat_id: int) -> bool: ...

    async def ban(self, chat_id: int, user_id: int) -> None: ...

    async def unban(self, chat_id: int, user_id: int) -> None: ...

    async def is_banned(self, chat_id: int, user_id: int) -> bool: ...

    async def resolve_username(self, chat_id: int, username: str) -> int | None: ...


class GroupRegistry:
    def __init__(self, store: GroupRepository) -> None:
        self._store = store

    async def observe(
        self,
        group: Group,
        *,
        migrated_to: int | None = None,
        migrated_from: int | None = None,
    ) -> None:
        if migrated_to is not None:
            await self._store.remember(
                Group(migrated_to, group.title), previous_chat_id=group.chat_id
            )
        else:
            await self._store.remember(group, previous_chat_id=migrated_from)

    async def membership(self, group: Group, *, present: bool) -> None:
        if present:
            await self.observe(group)
        else:
            await self._store.forget(group.chat_id)

    async def activate(self, chat_id: int) -> None:
        await self._store.activate(chat_id)

    async def get(self, chat_id: int) -> Group | None:
        return await self._store.get(chat_id)

    async def set_reply_audience(self, chat_id: int, audience: ReplyAudience) -> None:
        await self._store.set_reply_audience(chat_id, audience)

    async def set_whisper_audience(self, chat_id: int, audience: WhisperAudience) -> None:
        await self._store.set_whisper_audience(chat_id, audience)

    async def deactivate(self, chat_id: int) -> None:
        await self._store.deactivate(chat_id)

    async def is_active(self, chat_id: int) -> bool:
        return await self._store.is_active(chat_id)

    async def ban(self, chat_id: int, user_id: int) -> None:
        await self._store.ban(chat_id, user_id)

    async def unban(self, chat_id: int, user_id: int) -> None:
        await self._store.unban(chat_id, user_id)

    async def is_banned(self, chat_id: int, user_id: int) -> bool:
        return await self._store.is_banned(chat_id, user_id)

    async def resolve_username(self, chat_id: int, username: str) -> int | None:
        return await self._store.resolve_username(chat_id, username)

    async def messages(self) -> list[str]:
        groups = await self._store.list_groups()
        messages: list[str] = []
        if not groups:
            return [
                "📭 Пока не знаю ни одной группы. Добавь меня в группу и отправь там /start."
            ]
        text = "📋 Известные мне группы:"
        for group in groups:
            username = f" @{group.username}" if group.username else ""
            status = "🟢" if group.active else "🔴"
            line = f"\n\n{status} {group.title}{username} (ID: {group.chat_id})"
            line += (
                " — отвечаю"
                if group.active
                else " — ответы выключены (отправь там /start)"
            )
            if group.reply_audience == "admins":
                line += "; обычные обращения — только администраторам"
            if group.whisper_audience == "admins":
                line += "; шёпот — только администраторам"
            elif group.whisper_audience == "disabled":
                line += "; шёпот выключен"
            # Leave room below Telegram's 4096-character limit, including emoji.
            if len((text + line).encode("utf-16-le")) > 8000:
                messages.append(text)
                text = "📋 Группы (продолжение):"
            text += line
        messages.append(text)
        return messages
