from collections.abc import Iterator
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class Group:
    chat_id: int
    title: str
    username: str | None = None


class GroupRepository(Protocol):
    def remember(self, group: Group, *, previous_chat_id: int | None = None) -> None: ...

    def forget(self, chat_id: int) -> None: ...

    def list_groups(self) -> list[Group]: ...


class GroupRegistry:
    def __init__(self, store: GroupRepository) -> None:
        self._store = store

    def observe(
        self,
        group: Group,
        *,
        migrated_to: int | None = None,
        migrated_from: int | None = None,
    ) -> None:
        if migrated_to is not None:
            self._store.remember(
                Group(migrated_to, group.title), previous_chat_id=group.chat_id
            )
        else:
            self._store.remember(group, previous_chat_id=migrated_from)

    def membership(self, group: Group, *, present: bool) -> None:
        if present:
            self.observe(group)
        else:
            self._store.forget(group.chat_id)

    def messages(self) -> Iterator[str]:
        groups = self._store.list_groups()
        if not groups:
            yield "I don't know of any groups yet. Add me to a group or send /groups in an existing group."
            return
        text = "Groups I know I'm in:"
        for group in groups:
            username = f" @{group.username}" if group.username else ""
            line = f"\n• {group.title}{username} (ID: {group.chat_id})"
            # Leave room below Telegram's 4096-character limit, including emoji.
            if len((text + line).encode("utf-16-le")) > 8000:
                yield text
                text = "Groups (continued):"
            text += line
        yield text
