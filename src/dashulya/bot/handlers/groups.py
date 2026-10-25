from collections.abc import Iterator

from aiogram import F, Router
from aiogram.enums import ChatMemberStatus
from aiogram.filters import Command
from aiogram.types import ChatMemberUpdated, Message

from dashulya.bot.middlewares.group_tracking import GROUP_TYPES
from dashulya.storage.groups import Group, GroupStore


def group_list_messages(groups: list[Group]) -> Iterator[str]:
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


async def list_groups(message: Message, groups: GroupStore) -> None:
    for text in group_list_messages(groups.list_groups()):
        await message.answer(text)


async def track_membership(event: ChatMemberUpdated, groups: GroupStore) -> None:
    member = event.new_chat_member
    is_member = member.status in {
        ChatMemberStatus.CREATOR,
        ChatMemberStatus.ADMINISTRATOR,
        ChatMemberStatus.MEMBER,
    } or (member.status == ChatMemberStatus.RESTRICTED and member.is_member)
    if is_member:
        groups.remember(
            Group(
                event.chat.id, event.chat.title or "Untitled group", event.chat.username
            )
        )
    else:
        groups.forget(event.chat.id)


def create_router() -> Router:
    router = Router(name=__name__)
    router.message.register(list_groups, Command("groups"))
    router.my_chat_member.register(track_membership, F.chat.type.in_(GROUP_TYPES))
    return router
