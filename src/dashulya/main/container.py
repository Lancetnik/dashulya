from asyncio import Lock

from dishka import AsyncContainer, Provider, make_async_container
from dishka_ag2 import AG2Provider, AG2Scope

from dashulya.main.config import Settings
from dashulya.main.providers.ai import AgentProvider
from dashulya.main.providers.domain import DomainProvider
from dashulya.main.providers.memory import MemoryProvider
from dashulya.main.providers.telegram import TelegramProvider


def create_container(settings: Settings, *overrides: Provider) -> AsyncContainer:
    """Build one graph shared by Telegram events and AG2 turns/tool calls."""
    return make_async_container(
        DomainProvider(),
        MemoryProvider(),
        AgentProvider(),
        TelegramProvider(),
        AG2Provider(),
        *overrides,
        scopes=AG2Scope,
        context={Settings: settings},
        lock_factory=Lock,
    )
