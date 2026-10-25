from ag2 import Agent
from ag2.middleware import Middleware
from aiogram import Bot
from dishka import AsyncContainer, Provider, provide
from dishka_ag2 import AG2Scope, DishkaAsyncMiddleware

from dashulya.ai.addressing import GroupAddressing
from dashulya.ai.agents.factory import create_decision_agent, create_reply_agent
from dashulya.ai.agents.models import AgentModels, create_models
from dashulya.ai.dialogue import Dialogue
from dashulya.ai.tools.memory import search_memory
from dashulya.domain.memory import MemoryBackend
from dashulya.main.config import Settings


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
        memory: MemoryBackend,
    ) -> Agent[str]:
        return create_reply_agent(
            models.reply,
            tools=[search_memory],
            middleware=[Middleware(DishkaAsyncMiddleware, container=container)],
        )

    @provide
    def dialogue(self, agent: Agent[str], settings: Settings) -> Dialogue:
        return Dialogue(agent, timezone=settings.bot_timezone)

    @provide
    async def addressing(self, agent: Agent[bool], bot: Bot) -> GroupAddressing:
        me = await bot.me()
        return GroupAddressing(agent, bot_id=me.id, bot_username=me.username or "")
