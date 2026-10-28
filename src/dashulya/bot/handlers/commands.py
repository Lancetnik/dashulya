from aiogram import Router
from aiogram.filters import Command, CommandStart
from aiogram.types import Message
from dishka.integrations.aiogram import FromDishka, inject

from dashulya.bot.middlewares.group_tracking import GROUP_TYPES
from dashulya.domain.groups import GroupRegistry


@inject
async def start(message: Message, groups: FromDishka[GroupRegistry]) -> None:
    if message.chat.type in GROUP_TYPES:
        await groups.activate(message.chat.id)
        await message.answer(
            "I'm active in this group now 💋\n"
            "Mention me, reply to me, or call me Dashulya when you need me."
        )
        return
    await message.answer(
        "Hi! I'm Dashulya 💋\n"
        "Send me a message in a private chat to talk to me.\n"
        "Use /help to see what I can do."
    )


async def show_help(message: Message) -> None:
    await message.answer(
        "Here's what I can do 💋\n\n"
        "/start — introduce myself; activate me in this group\n"
        "/help — show this guide\n"
        "/groups — list the groups I know I'm in\n"
        "/clear — reset your private conversation\n\n"
        "Send me a private message, mention my @username, or reply to me in a group. "
        "I also check whether messages naming Dashulya need an answer; "
        "/clear works only in private chats.\n\n"
        "In each group, the owner must send /start before I can respond.\n\n"
        "With chat history connected, I can also find earlier discussions "
        "and link to the original messages."
    )


def create_router() -> Router:
    router = Router(name=__name__)
    router.message.register(start, CommandStart())
    router.message.register(show_help, Command("help"))
    return router
