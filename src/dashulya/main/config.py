import os
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from aiogram.utils.token import TokenValidationError, validate_token
from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    bot_token: str = field(repr=False)
    openai_api_key: str = field(repr=False)
    owner_user_id: int
    openai_model: str = "gpt-6-luna"
    openai_decision_model: str = "gpt-6-luna"
    database_url: str = field(
        default_factory=lambda: os.getenv(
            "DATABASE_URL", "postgresql://localhost/dashulya"
        ),
        repr=False,
    )
    log_path: Path = Path("var/logs/dashulya.log")
    bot_timezone: str = "Europe/Moscow"
    healthcheck_path: Path | None = None


def load_settings() -> Settings:
    load_dotenv(Path.cwd() / ".env", override=False)
    token = os.getenv("BOT_TOKEN", "").strip()
    if not token:
        error_message = "Set BOT_TOKEN in .env or the environment."
        raise ValueError(error_message)
    try:
        validate_token(token)
    except TokenValidationError:
        error_message = "BOT_TOKEN is malformed. Check the token from @BotFather."
        raise ValueError(error_message) from None
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        error_message = "Set OPENAI_API_KEY in .env or the environment."
        raise ValueError(error_message)
    owner_id = os.getenv("OWNER_USER_ID", "").strip()
    if not owner_id:
        error_message = "Set OWNER_USER_ID to your numeric Telegram user ID in .env or the environment."
        raise ValueError(error_message)
    try:
        owner_user_id = int(owner_id)
    except ValueError:
        error_message = "OWNER_USER_ID must be a positive Telegram user ID."
        raise ValueError(error_message) from None
    if owner_user_id <= 0:
        error_message = "OWNER_USER_ID must be a positive Telegram user ID."
        raise ValueError(error_message)
    bot_timezone = os.getenv("BOT_TIMEZONE", "").strip() or "Europe/Moscow"
    try:
        ZoneInfo(bot_timezone)
    except (ZoneInfoNotFoundError, ValueError):
        error_message = (
            "BOT_TIMEZONE must be a valid IANA timezone, such as Europe/Moscow."
        )
        raise ValueError(error_message) from None
    database_url = os.getenv("DATABASE_URL", "").strip()
    try:
        parsed = urlsplit(database_url)
        if (
            parsed.scheme not in {"postgresql", "postgres"}
            or not parsed.hostname
            or not parsed.path.strip("/")
        ):
            raise ValueError
    except ValueError:
        error_message = "Set DATABASE_URL to a PostgreSQL connection URL."
        raise ValueError(error_message) from None
    return Settings(
        bot_token=token,
        openai_api_key=api_key,
        owner_user_id=owner_user_id,
        openai_model=os.getenv("OPENAI_MODEL", "").strip() or "gpt-6-luna",
        openai_decision_model=os.getenv("OPENAI_DECISION_MODEL", "").strip()
        or "gpt-6-luna",
        database_url=database_url,
        log_path=Path(os.getenv("LOG_PATH") or "var/logs/dashulya.log"),
        bot_timezone=bot_timezone,
        healthcheck_path=Path(value)
        if (value := os.getenv("HEALTHCHECK_PATH"))
        else None,
    )
