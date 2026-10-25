from aiogram import Router
from aiogram.filters import Command, CommandStart
from aiogram.types import Message


async def start(message: Message) -> None:
    await message.answer(
        "Hi! I'm Dashulya 💋\n"
        "Send me a message in a private chat to talk to me.\n"
        "Use /help to see what I can do."
    )


async def show_help(message: Message) -> None:
    await message.answer(
        "Here's what I can do 💋\n\n"
        "/start — introduce myself\n"
        "/help — show this guide\n"
        "/groups — list the groups I know I'm in\n"
        "/clear — reset your private conversation\n\n"
        "Send me a private message to chat. Group chats support commands only; "
        "/clear works only in private chats.\n\n"
        "With chat history connected, I can also find earlier discussions "
        "and link to the original messages."
    )


def create_router() -> Router:
    router = Router(name=__name__)
    router.message.register(start, CommandStart())
    router.message.register(show_help, Command("help"))
    return router
