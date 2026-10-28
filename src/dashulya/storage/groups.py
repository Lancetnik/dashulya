from dashulya.database import Database
from dashulya.domain.groups import Group


class GroupStore:
    def __init__(self, database: Database) -> None:
        self.database = database

    async def remember(
        self, group: Group, *, previous_chat_id: int | None = None
    ) -> None:
        async with self.database.connection() as connection, connection.transaction():
            active = False
            if previous_chat_id is not None:
                cursor = await connection.execute(
                    "DELETE FROM groups WHERE chat_id=%s RETURNING active",
                    (previous_chat_id,),
                )
                previous = await cursor.fetchone()
                active = previous is not None and previous["active"]
            await connection.execute(
                """INSERT INTO groups (chat_id,title,username,active) VALUES (%s,%s,%s,%s)
                ON CONFLICT(chat_id) DO UPDATE SET title=excluded.title,
                    username=excluded.username,active=groups.active OR excluded.active""",
                (group.chat_id, group.title, group.username, active),
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
                Group(row["chat_id"], row["title"], row["username"], row["active"])
                for row in await cursor.fetchall()
            ]

    async def activate(self, chat_id: int) -> None:
        async with self.database.connection() as connection:
            await connection.execute(
                "UPDATE groups SET active=TRUE WHERE chat_id=%s", (chat_id,)
            )

    async def is_active(self, chat_id: int) -> bool:
        async with self.database.connection() as connection:
            cursor = await connection.execute(
                "SELECT active FROM groups WHERE chat_id=%s", (chat_id,)
            )
            row = await cursor.fetchone()
            return row is not None and bool(row["active"])
