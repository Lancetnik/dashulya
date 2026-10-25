from collections.abc import Iterator
from contextlib import closing

from dishka import Provider, alias, provide
from dishka_ag2 import AG2Scope

from dashulya.domain.groups import GroupRegistry, GroupRepository
from dashulya.main.config import Settings
from dashulya.storage.groups import GroupStore


class DomainProvider(Provider):
    scope = AG2Scope.APP

    registry = provide(GroupRegistry)
    repository = alias(GroupStore, provides=GroupRepository)

    @provide
    def groups(self, settings: Settings) -> Iterator[GroupStore]:
        with closing(GroupStore(settings.database_path)) as groups:
            yield groups
