import asyncio
import logging
from contextlib import closing

from aiogram import Bot, Dispatcher

from dashulya.ai.agent import create_agent
from dashulya.ai.dialogue import Dialogue
from dashulya.bot.router import create_router
from dashulya.config import Settings, load_settings
from dashulya.memory.search import MemorySearch
from dashulya.observability import configure_logging, exception_fields
from dashulya.storage.groups import GroupStore

logger = logging.getLogger(__name__)


async def run(settings: Settings) -> None:
    memory = (
        MemorySearch(settings.memory_path) if settings.memory_path.is_file() else None
    )
    logger.info(
        "Bot starting",
        extra={"model": settings.openai_model, "memory_enabled": memory is not None},
    )
    with closing(GroupStore(settings.database_path)) as groups:
        agent = create_agent(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            memory=memory,
        )
        dispatcher = Dispatcher(
            groups=groups, dialogue=Dialogue(agent, timezone=settings.bot_timezone)
        )
        dispatcher.include_router(create_router(owner_user_id=settings.owner_user_id))
        async with Bot(token=settings.bot_token) as bot:
            await dispatcher.start_polling(bot, close_bot_session=False)


def main() -> None:
    try:
        settings = load_settings()
    except ValueError as error:
        raise SystemExit(str(error)) from None

    configure_logging(
        settings.log_path, secrets=(settings.bot_token, settings.openai_api_key)
    )
    try:
        asyncio.run(run(settings))
    except KeyboardInterrupt:
        logger.info("Bot stopped")
    except Exception as error:
        logger.exception("Bot stopped unexpectedly", extra=exception_fields(error))
        error_message = f"Bot stopped unexpectedly. See {settings.log_path}."
        raise SystemExit(error_message) from None
    else:
        logger.info("Bot stopped")
