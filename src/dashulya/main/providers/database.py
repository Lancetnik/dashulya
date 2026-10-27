from collections.abc import AsyncIterator

from dishka import Provider, provide
from dishka_ag2 import AG2Scope

from dashulya.database import Database
from dashulya.main.config import Settings


class DatabaseProvider(Provider):
    scope = AG2Scope.APP

    @provide
    async def database(self, settings: Settings) -> AsyncIterator[Database]:
        database = Database(settings.database_url)
        try:
            await database.open()
            yield database
        finally:
            await database.close()
