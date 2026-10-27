from dishka import Provider, alias, provide
from dishka_ag2 import AG2Scope

from dashulya.database import Database
from dashulya.domain.groups import GroupRegistry, GroupRepository
from dashulya.storage.groups import GroupStore


class DomainProvider(Provider):
    scope = AG2Scope.APP

    registry = provide(GroupRegistry)
    repository = alias(GroupStore, provides=GroupRepository)

    @provide
    def groups(self, database: Database) -> GroupStore:
        return GroupStore(database)
