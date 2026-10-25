from aiogram import Router

from dashulya.bot.handlers import commands, dialogue, groups
from dashulya.bot.middlewares.access import OwnerOnlyMiddleware
from dashulya.bot.middlewares.group_tracking import TrackGroupsMiddleware


def create_router(*, owner_user_id: int) -> Router:
    router = Router(name=__name__)
    router.message.outer_middleware(TrackGroupsMiddleware())
    router.message.outer_middleware(OwnerOnlyMiddleware(owner_user_id))
    router.include_routers(
        commands.create_router(),
        groups.create_router(),
        dialogue.create_router(),
    )
    return router
