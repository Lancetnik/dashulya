from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class Group:
    chat_id: int
    title: str
    username: str | None = None
    active: bool = False


class GroupRepository(Protocol):
    async def remember(
        self, group: Group, *, previous_chat_id: int | None = None
    ) -> None: ...

    async def forget(self, chat_id: int) -> None: ...

    async def list_groups(self) -> list[Group]: ...

    async def activate(self, chat_id: int) -> None: ...

    async def is_active(self, chat_id: int) -> bool: ...


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

    async def is_active(self, chat_id: int) -> bool:
        return await self._store.is_active(chat_id)

    async def messages(self) -> list[str]:
        groups = await self._store.list_groups()
        messages: list[str] = []
        if not groups:
            return [
                "📭 I don't know of any groups yet. Add me to a group, then send /start there."
            ]
        text = "📋 Groups I know I'm in:"
        for group in groups:
            username = f" @{group.username}" if group.username else ""
            status = "🟢" if group.active else "🔴"
            line = f"\n\n{status} {group.title}{username} (ID: {group.chat_id})"
            line += " — active" if group.active else " — inactive (send /start there)"
            # Leave room below Telegram's 4096-character limit, including emoji.
            if len((text + line).encode("utf-16-le")) > 8000:
                messages.append(text)
                text = "📋 Groups (continued):"
            text += line
        messages.append(text)
        return messages
