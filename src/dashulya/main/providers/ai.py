from collections.abc import Callable
from typing import Annotated

from ag2 import Agent, Context, Inject
from ag2.middleware import Middleware
from aiogram import Bot
from dishka import AsyncContainer, Provider, provide
from dishka_ag2 import AG2Scope, DishkaAsyncMiddleware
from fast_depends import inject

from dashulya.ai.addressing import GroupAddressing
from dashulya.ai.agents.factory import create_decision_agent, create_reply_agent
from dashulya.ai.agents.models import AgentModels, create_models
from dashulya.ai.dialogue import Dialogue
from dashulya.ai.message_format import MessageFormatter
from dashulya.ai.tools.history import chat_history
from dashulya.ai.tools.memory import search_memory
from dashulya.ai.tools.reactions import set_reaction
from dashulya.ai.tools.web import create_web_tools
from dashulya.domain.memory import (
    ChatHistory,
    ChatMemoryService,
    HistoryReader,
    MemoryAccess,
    MemoryBackend,
)
from dashulya.domain.messages import Reply
from dashulya.main.config import IMAGE_MAX_BYTES, Settings


def _validate_memory_access(
    access: Annotated[MemoryAccess | None, Inject("memory_access", default=None)],
) -> MemoryAccess:
    error_message = "Memory tools require trusted conversation access."
    if type(access) is not MemoryAccess:
        raise ValueError(error_message)
    fields = vars(access)
    if (
        type(fields["chat_id"]) is not int
        or type(fields["is_private"]) is not bool
        or access.chat_id == 0
        or access.is_private != (access.chat_id > 0)
    ):
        raise ValueError(error_message)
    return access


# Dishka providers are not called by AG2, so resolve their AG2 annotations here.
_memory_access: Callable[..., MemoryAccess] = inject(_validate_memory_access, cast=False)


class AgentProvider(Provider):
    scope = AG2Scope.APP

    @provide
    def message_formatter(self, settings: Settings, bot: Bot) -> MessageFormatter:
        return MessageFormatter(
            settings.bot_timezone,
            bot.id,
            author_users=dict.fromkeys(settings.owner_author_ids, settings.owner_user_id),
        )

    @provide
    def models(self, settings: Settings) -> AgentModels:
        return create_models(
            api_key=settings.openai_api_key,
            reply_model=settings.openai_model,
            decision_model=settings.openai_decision_model,
        )

    @provide
    def decision_agent(
        self, models: AgentModels, container: AsyncContainer, settings: Settings
    ) -> Agent[bool]:
        return create_decision_agent(
            models.decision,
            timezone=settings.bot_timezone,
            middleware=[Middleware(DishkaAsyncMiddleware, container=container)],
        )

    @provide
    def reply_agent(
        self,
        models: AgentModels,
        container: AsyncContainer,
        memory: ChatMemoryService,
        settings: Settings,
    ) -> Agent[Reply]:
        return create_reply_agent(
            models.reply,
            timezone=settings.bot_timezone,
            tools=[
                search_memory,
                chat_history,
                set_reaction,
                *create_web_tools(settings.exa_api_key),
            ],
            middleware=[Middleware(DishkaAsyncMiddleware, container=container)],
        )

    @provide(scope=AG2Scope.SESSION)
    def memory_access(self, context: Context) -> MemoryAccess:
        return _memory_access(__ctx__=context)

    @provide(scope=AG2Scope.SESSION)
    def scoped_memory(
        self, service: ChatMemoryService, access: MemoryAccess
    ) -> MemoryBackend:
        return service.recall(access)

    @provide(scope=AG2Scope.SESSION)
    def scoped_history(
        self, service: ChatMemoryService, access: MemoryAccess
    ) -> HistoryReader:
        return service.history(access)

    @provide
    def dialogue(
        self,
        agent: Agent[Reply],
        history: ChatHistory,
        formatter: MessageFormatter,
    ) -> Dialogue:
        return Dialogue(
            agent,
            history=history,
            max_image_bytes=IMAGE_MAX_BYTES,
            formatter=formatter,
        )

    @provide
    async def addressing(self, agent: Agent[bool], bot: Bot) -> GroupAddressing:
        me = await bot.me()
        return GroupAddressing(agent, bot_id=me.id, bot_username=me.username or "")
