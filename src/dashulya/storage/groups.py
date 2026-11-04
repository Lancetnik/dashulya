from dashulya.database import Database
from dashulya.domain.groups import Group, ReplyAudience, WhisperAudience


class GroupStore:
    def __init__(self, database: Database) -> None:
        self.database = database

    async def remember(
        self, group: Group, *, previous_chat_id: int | None = None
    ) -> None:
        async with self.database.connection() as connection, connection.transaction():
            active = False
            capture_enabled = False
            reply_audience = "all"
            whisper_audience = "all"
            previous = None
            if previous_chat_id is not None:
                cursor = await connection.execute(
                    "SELECT active,capture_enabled,reply_audience,whisper_audience "
                    "FROM groups WHERE chat_id=%s FOR UPDATE",
                    (previous_chat_id,),
                )
                previous = await cursor.fetchone()
                active = previous is not None and previous["active"]
                if previous is not None:
                    capture_enabled = previous["capture_enabled"]
                    reply_audience = previous["reply_audience"]
                    whisper_audience = previous["whisper_audience"]
            await connection.execute(
                """INSERT INTO groups
                    (chat_id,title,username,active,capture_enabled,reply_audience,whisper_audience)
                VALUES (%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT(chat_id) DO UPDATE SET title=excluded.title,
                    username=excluded.username,active=groups.active OR excluded.active,
                    capture_enabled=groups.capture_enabled OR excluded.capture_enabled,
                    reply_audience=CASE WHEN %s THEN excluded.reply_audience
                        ELSE groups.reply_audience END,
                    whisper_audience=CASE WHEN %s THEN excluded.whisper_audience
                        ELSE groups.whisper_audience END""",
                (
                    group.chat_id,
                    group.title,
                    group.username,
                    active,
                    capture_enabled,
                    reply_audience,
                    whisper_audience,
                    previous is not None,
                    previous is not None,
                ),
            )
            if previous_chat_id is not None and previous_chat_id != group.chat_id:
                await connection.execute(
                    "INSERT INTO group_blacklist (chat_id,user_id) "
                    "SELECT %s,user_id FROM group_blacklist WHERE chat_id=%s "
                    "ON CONFLICT DO NOTHING",
                    (group.chat_id, previous_chat_id),
                )
                await connection.execute(
                    "DELETE FROM groups WHERE chat_id=%s", (previous_chat_id,)
                )

    async def forget(self, chat_id: int) -> None:
        async with self.database.connection() as connection:
            await connection.execute("DELETE FROM groups WHERE chat_id=%s", (chat_id,))

    async def list_groups(self) -> list[Group]:
        async with self.database.connection() as connection:
            cursor = await connection.execute(
                "SELECT * FROM groups ORDER BY lower(title),chat_id"
            )
            return [Group(**row) for row in await cursor.fetchall()]

    async def get(self, chat_id: int) -> Group | None:
        async with self.database.connection() as connection:
            cursor = await connection.execute(
                "SELECT * FROM groups WHERE chat_id=%s", (chat_id,)
            )
            row = await cursor.fetchone()
            return Group(**row) if row is not None else None

    async def set_reply_audience(self, chat_id: int, audience: ReplyAudience) -> None:
        async with self.database.connection() as connection:
            await connection.execute(
                "UPDATE groups SET reply_audience=%s WHERE chat_id=%s",
                (audience, chat_id),
            )

    async def set_whisper_audience(self, chat_id: int, audience: WhisperAudience) -> None:
        async with self.database.connection() as connection:
            await connection.execute(
                "UPDATE groups SET whisper_audience=%s WHERE chat_id=%s",
                (audience, chat_id),
            )

    async def activate(self, chat_id: int) -> None:
        async with self.database.connection() as connection:
            await connection.execute(
                "UPDATE groups SET active=TRUE,capture_enabled=TRUE WHERE chat_id=%s",
                (chat_id,),
            )

    async def is_active(self, chat_id: int) -> bool:
        async with self.database.connection() as connection:
            cursor = await connection.execute(
                "SELECT active FROM groups WHERE chat_id=%s", (chat_id,)
            )
            row = await cursor.fetchone()
            return row is not None and bool(row["active"])

    async def deactivate(self, chat_id: int) -> None:
        async with self.database.connection() as connection:
            await connection.execute(
                "UPDATE groups SET active=FALSE WHERE chat_id=%s", (chat_id,)
            )

    async def ban(self, chat_id: int, user_id: int) -> None:
        async with self.database.connection() as connection:
            await connection.execute(
                "INSERT INTO group_blacklist (chat_id,user_id) VALUES (%s,%s) "
                "ON CONFLICT DO NOTHING",
                (chat_id, user_id),
            )

    async def unban(self, chat_id: int, user_id: int) -> None:
        async with self.database.connection() as connection:
            await connection.execute(
                "DELETE FROM group_blacklist WHERE chat_id=%s AND user_id=%s",
                (chat_id, user_id),
            )

    async def is_banned(self, chat_id: int, user_id: int) -> bool:
        async with self.database.connection() as connection:
            cursor = await connection.execute(
                "SELECT 1 FROM group_blacklist WHERE chat_id=%s AND user_id=%s",
                (chat_id, user_id),
            )
            return await cursor.fetchone() is not None

    async def resolve_username(self, chat_id: int, username: str) -> int | None:
        async with self.database.connection() as connection:
            cursor = await connection.execute(
                """SELECT author_id FROM (
                    SELECT DISTINCT ON (author_id) author_id,username_folded
                    FROM messages WHERE chat_id=%s AND author_id>0 AND is_bot=FALSE
                    ORDER BY author_id,version_time DESC,update_id DESC,message_id DESC
                ) AS latest WHERE username_folded=%s LIMIT 2""",
                (chat_id, username.lower()),
            )
            matches = await cursor.fetchall()
            return int(matches[0]["author_id"]) if len(matches) == 1 else None
