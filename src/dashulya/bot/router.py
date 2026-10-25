from aiogram import Router

from dashulya.bot.handlers import commands, dialogue, groups
from dashulya.bot.middlewares.access import OwnerOnlyMiddleware
from dashulya.bot.middlewares.group_tracking import TrackGroupsMiddleware


def create_router(
    *, tracking: TrackGroupsMiddleware, access: OwnerOnlyMiddleware
) -> Router:
    router = Router(name=__name__)
    router.message.outer_middleware(tracking)
    router.message.outer_middleware(access)
    router.include_routers(
        commands.create_router(),
        groups.create_router(),
        dialogue.create_router(),
    )
    return router
