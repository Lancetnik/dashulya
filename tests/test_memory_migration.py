import hashlib
import json
import sqlite3
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import AsyncMock

from dashulya.domain.memory import MemoryAccess, MemoryMessage
from dashulya.memory.live import LiveMemory
from dashulya.memory.migrate import import_evidence, migrate_sqlite
from dashulya.memory.store import MessageStore
from dashulya.memory.vectors import EmbeddingCache
from dashulya.storage.groups import GroupStore
from tests.postgres import PostgresTestCase

GROUP = -100123
NOW = datetime(2026, 10, 25, tzinfo=UTC)


class MemoryMigrationTests(PostgresTestCase):
    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name)
        self.evidence = self.path / "evidence.sqlite3"
        with closing(sqlite3.connect(self.evidence)) as source:
            source.executescript("""
                CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT);
                INSERT INTO metadata VALUES ('source_sha256','"fixture"'),
                    ('chat_username','"group"');
                CREATE TABLE messages(id INTEGER PRIMARY KEY,body TEXT,sent_at TEXT,
                    user_id INTEGER,author TEXT,reply_to_id INTEGER,url TEXT);
                CREATE TABLE memories(id INTEGER PRIMARY KEY,key TEXT,kind TEXT,body TEXT);
                INSERT INTO memories VALUES (1,'fact:1','fact','Preserved legacy fact');
                CREATE TABLE sources(memory_id INTEGER,message_id INTEGER);
                INSERT INTO sources VALUES(1,1);
            """)
            source.executemany(
                "INSERT INTO messages VALUES (?,?,?,?,?,?,?)",
                [
                    (
                        1,
                        "Archive parent",
                        NOW.isoformat(),
                        10,
                        "Alice",
                        None,
                        "https://t.me/group/1",
                    ),
                    (
                        2,
                        "Old text",
                        NOW.isoformat(),
                        10,
                        "Alice",
                        1,
                        "https://t.me/group/2",
                    ),
                ],
            )
            source.commit()

    async def test_readonly_resumable_migration_preserves_originals_registry_and_vectors(
        self,
    ) -> None:
        live_path = self.path / "live.sqlite3"
        live = LiveMemory(live_path)
        await live.initialize()
        for chat_id, number, text in (
            (GROUP, 2, "Corrected text"),
            (42, 1, "Private note"),
        ):
            await live.record(
                MemoryMessage(
                    chat_id,
                    number,
                    text,
                    NOW,
                    42,
                    "Owner",
                    "Chat",
                    reply_to_id=1 if chat_id < 0 else None,
                )
            )
        groups = self.path / "groups.sqlite3"
        vectors = self.path / "vectors.sqlite3"
        provider = AsyncMock(side_effect=AssertionError("Provider must not be called"))
        cache = EmbeddingCache(self.database, provider)
        key = cache.key("Archive parent")
        with closing(sqlite3.connect(groups)) as source:
            source.execute(
                "CREATE TABLE groups(chat_id INTEGER,title TEXT,username TEXT)"
            )
            source.execute("INSERT INTO groups VALUES(?,?,?)", (GROUP, "Group", "group"))
            source.commit()
        with closing(sqlite3.connect(vectors)) as source:
            source.execute("CREATE TABLE embeddings(cache_key TEXT,vector TEXT)")
            source.execute(
                "INSERT INTO embeddings VALUES(?,?)",
                (key, json.dumps([1.0] + [0.0] * 511)),
            )
            source.commit()
        paths = [self.evidence, live_path, groups, vectors]
        hashes = [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
        for _ in range(2):
            result = await migrate_sqlite(
                self.database,
                evidence=self.evidence,
                live=live_path,
                groups=groups,
                vectors=vectors,
                chat_id=GROUP,
                owner_user_id=42,
            )
            self.assertTrue(result["verified"])
            self.assertEqual(result["source_originals"], 3)
            self.assertEqual(result["available_originals"], 3)
            self.assertEqual(result["provider_calls"], 0)
        self.assertEqual(
            hashes, [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
        )
        store = MessageStore(self.database)
        original = await store.originals(MemoryAccess(GROUP, False))
        self.assertEqual([r.text for r in original], ["Archive parent", "Corrected text"])
        self.assertEqual(len(await store.originals(MemoryAccess(42, True))), 3)
        context = (await store.context((GROUP, None), 2, limit=1, parent_hops=5)).messages
        self.assertEqual([r.message_id for r in context], [1, 2])
        self.assertEqual(
            (await GroupStore(self.database).list_groups())[0].chat_id, GROUP
        )
        self.assertEqual(
            (await cache.vectors(["Archive parent"]))["Archive parent"][0], 1
        )
        provider.assert_not_awaited()
        async with self.database.connection() as connection:
            legacy = await (
                await connection.execute("SELECT * FROM legacy_records")
            ).fetchall()
        self.assertEqual(legacy[0]["body"], "Preserved legacy fact")
        self.assertEqual(legacy[0]["source_message_ids"], [1])

    async def test_binding_adopts_unbound_archive_and_refuses_other_chat(self) -> None:
        store = MessageStore(self.database)
        await import_evidence(store, self.evidence)
        self.assertEqual(await store.originals(MemoryAccess(GROUP, False)), [])
        await import_evidence(store, self.evidence, GROUP)
        self.assertEqual(len(await store.originals(MemoryAccess(42, True))), 2)
        with self.assertRaisesRegex(ValueError, "different chat"):
            await import_evidence(store, self.evidence, -100456)
        with self.assertRaises(ValueError):
            await import_evidence(store, self.evidence, 0)
