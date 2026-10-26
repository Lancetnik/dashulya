from aiogram import F, Router
from aiogram.enums import ChatMemberStatus
from aiogram.filters import Command
from aiogram.types import ChatMemberUpdated, Message
from dishka.integrations.aiogram import FromDishka, inject

from dashulya.bot.middlewares.group_tracking import GROUP_TYPES
from dashulya.domain.groups import Group, GroupRegistry


@inject
async def list_groups(message: Message, groups: FromDishka[GroupRegistry]) -> None:
    for text in await groups.messages():
        await message.answer(text)


@inject
async def track_membership(
    event: ChatMemberUpdated, groups: FromDishka[GroupRegistry]
) -> None:
    member = event.new_chat_member
    is_member = member.status in {
        ChatMemberStatus.CREATOR,
        ChatMemberStatus.ADMINISTRATOR,
        ChatMemberStatus.MEMBER,
    } or (member.status == ChatMemberStatus.RESTRICTED and member.is_member)
    await groups.membership(
        Group(event.chat.id, event.chat.title or "Untitled group", event.chat.username),
        present=is_member,
    )


def create_router() -> Router:
    router = Router(name=__name__)
    router.message.register(list_groups, Command("groups"))
    router.my_chat_member.register(track_membership, F.chat.type.in_(GROUP_TYPES))
    return router
