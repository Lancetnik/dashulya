from aiogram import Router

from dashulya.bot.handlers import commands, dialogue, groups
from dashulya.bot.middlewares.access import AccessMiddleware
from dashulya.bot.middlewares.group_tracking import TrackGroupsMiddleware
from dashulya.bot.middlewares.memory import CaptureMessagesMiddleware


def create_router(
    *,
    tracking: TrackGroupsMiddleware,
    access: AccessMiddleware,
    capture: CaptureMessagesMiddleware,
) -> Router:
    router = Router(name=__name__)
    router.message.outer_middleware(tracking)
    router.message.outer_middleware(capture)
    router.message.outer_middleware(access)
    router.edited_message.register(capture.capture)
    router.include_routers(
        commands.create_router(),
        groups.create_router(),
        dialogue.create_router(),
    )
    return router
