from collections.abc import AsyncIterator

from aiogram import Bot, Dispatcher
from aiogram.types import TelegramObject
from dishka import AsyncContainer, Provider, from_context, provide
from dishka.integrations.aiogram import AiogramMiddlewareData, setup_dishka
from dishka_ag2 import AG2Scope

from dashulya.bot.images import TelegramImages
from dashulya.bot.middlewares.access import OwnerOnlyMiddleware
from dashulya.bot.middlewares.group_tracking import TrackGroupsMiddleware
from dashulya.bot.middlewares.memory import (
    CaptureMessagesMiddleware,
    CaptureSentMessagesMiddleware,
)
from dashulya.bot.replies import ReplyDelivery
from dashulya.bot.router import create_router
from dashulya.domain.memory import MessageRecorder
from dashulya.main.config import IMAGE_MAX_BYTES, Settings


class TelegramProvider(Provider):
    scope = AG2Scope.APP

    tracking = provide(TrackGroupsMiddleware)
    delivery = provide(ReplyDelivery)
    # AiogramProvider uses Scope.REQUEST; both integrations share AG2Scope here.
    event = from_context(TelegramObject, scope=AG2Scope.REQUEST)
    middleware_data = from_context(AiogramMiddlewareData, scope=AG2Scope.REQUEST)

    @provide
    def capture(
        self, recorder: MessageRecorder, settings: Settings
    ) -> CaptureMessagesMiddleware:
        return CaptureMessagesMiddleware(recorder, owner_user_id=settings.owner_user_id)

    @provide
    def capture_sent(
        self, recorder: MessageRecorder, settings: Settings
    ) -> CaptureSentMessagesMiddleware:
        return CaptureSentMessagesMiddleware(
            recorder, owner_user_id=settings.owner_user_id
        )

    @provide
    def images(self, bot: Bot) -> TelegramImages:
        return TelegramImages(bot, max_bytes=IMAGE_MAX_BYTES)

    @provide
    async def bot(self, settings: Settings) -> AsyncIterator[Bot]:
        async with Bot(token=settings.bot_token) as bot:
            yield bot

    @provide
    def access(self, settings: Settings) -> OwnerOnlyMiddleware:
        return OwnerOnlyMiddleware(settings.owner_user_id)

    @provide
    def dispatcher(
        self,
        container: AsyncContainer,
        tracking: TrackGroupsMiddleware,
        access: OwnerOnlyMiddleware,
        capture: CaptureMessagesMiddleware,
        capture_sent: CaptureSentMessagesMiddleware,
        bot: Bot,
    ) -> Dispatcher:
        bot.session.middleware(capture_sent)
        dispatcher = Dispatcher()
        dispatcher.include_router(
            create_router(tracking=tracking, access=access, capture=capture)
        )
        setup_dishka(container=container, router=dispatcher)
        return dispatcher
