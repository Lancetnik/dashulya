import asyncio
import logging
import os
from urllib.parse import urlsplit

from aiogram import Bot, Dispatcher

from dashulya.domain.memory import ChatMemoryService
from dashulya.main.config import Settings, load_settings
from dashulya.main.container import create_container
from dashulya.observability import configure_logging, exception_fields

logger = logging.getLogger(__name__)


async def run(settings: Settings) -> None:
    if settings.healthcheck_path:
        settings.healthcheck_path.unlink(missing_ok=True)
    container = create_container(settings)
    try:
        await container.get(ChatMemoryService)
        logger.info(
            "Bot starting",
            extra={
                "model": settings.openai_model,
                "memory_backend": "pgvector",
            },
        )
        bot = await container.get(Bot)
        dispatcher = await container.get(Dispatcher)
        if settings.healthcheck_path:
            await bot.me()
            settings.healthcheck_path.parent.mkdir(parents=True, exist_ok=True)
            settings.healthcheck_path.write_text(str(os.getpid()), encoding="utf-8")
        await dispatcher.start_polling(bot, close_bot_session=False)
    finally:
        try:
            if settings.healthcheck_path:
                settings.healthcheck_path.unlink(missing_ok=True)
        finally:
            await container.close()


def main() -> None:
    try:
        settings = load_settings()
    except ValueError as error:
        raise SystemExit(str(error)) from None

    configure_logging(
        settings.log_path,
        secrets=(
            settings.bot_token,
            settings.openai_api_key,
            settings.exa_api_key,
            settings.database_url,
            urlsplit(settings.database_url).password or "",
        ),
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
