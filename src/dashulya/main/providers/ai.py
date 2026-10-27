from ag2 import Agent, Context
from ag2.middleware import Middleware
from aiogram import Bot
from dishka import AsyncContainer, Provider, provide
from dishka_ag2 import AG2Scope, DishkaAsyncMiddleware

from dashulya.ai.addressing import GroupAddressing
from dashulya.ai.agents.factory import create_decision_agent, create_reply_agent
from dashulya.ai.agents.models import AgentModels, create_models
from dashulya.ai.dialogue import Dialogue
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


class AgentProvider(Provider):
    scope = AG2Scope.APP

    @provide
    def models(self, settings: Settings) -> AgentModels:
        return create_models(
            api_key=settings.openai_api_key,
            reply_model=settings.openai_model,
            decision_model=settings.openai_decision_model,
        )

    @provide
    def decision_agent(
        self, models: AgentModels, container: AsyncContainer
    ) -> Agent[bool]:
        return create_decision_agent(
            models.decision,
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
        access = context.dependencies.get(MemoryAccess)
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
        settings: Settings,
        history: ChatHistory,
        bot: Bot,
        memory: ChatMemoryService,
    ) -> Dialogue:
        return Dialogue(
            agent,
            timezone=settings.bot_timezone,
            history=history,
            bot_id=bot.id,
            memory=memory,
            max_image_bytes=IMAGE_MAX_BYTES,
        )

    @provide
    async def addressing(self, agent: Agent[bool], bot: Bot) -> GroupAddressing:
        me = await bot.me()
        return GroupAddressing(agent, bot_id=me.id, bot_username=me.username or "")
