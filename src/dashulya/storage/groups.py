from dashulya.database import Database
from dashulya.domain.groups import Group


class GroupStore:
    def __init__(self, database: Database) -> None:
        self.database = database

    async def remember(
        self, group: Group, *, previous_chat_id: int | None = None
    ) -> None:
        async with self.database.connection() as connection, connection.transaction():
            if previous_chat_id is not None:
                await connection.execute(
                    "DELETE FROM groups WHERE chat_id=%s", (previous_chat_id,)
                )
            await connection.execute(
                """INSERT INTO groups VALUES (%s,%s,%s)
                ON CONFLICT(chat_id) DO UPDATE SET title=excluded.title,username=excluded.username""",
                (group.chat_id, group.title, group.username),
            )

    async def forget(self, chat_id: int) -> None:
        async with self.database.connection() as connection:
            await connection.execute("DELETE FROM groups WHERE chat_id=%s", (chat_id,))

    async def list_groups(self) -> list[Group]:
        async with self.database.connection() as connection:
            cursor = await connection.execute(
                "SELECT * FROM groups ORDER BY lower(title),chat_id"
            )
            return [
                Group(row["chat_id"], row["title"], row["username"])
                for row in await cursor.fetchall()
            ]
