from collections.abc import AsyncIterator

from dishka import Provider, provide
from dishka_ag2 import AG2Scope
from openai import AsyncOpenAI

from dashulya.ai.embeddings import MemoryEmbeddings
from dashulya.database import Database
from dashulya.domain.memory import ChatHistory, ChatMemoryService, MessageRecorder
from dashulya.main.config import Settings
from dashulya.memory.service import HybridChatMemory
from dashulya.memory.store import MessageStore
from dashulya.memory.vectors import EmbeddingCache


class MemoryProvider(Provider):
    scope = AG2Scope.APP

    @provide
    async def messages(self, database: Database) -> MessageStore:
        store = MessageStore(database)
        await store.initialize()
        return store

    @provide
    def recorder(self, store: MessageStore) -> MessageRecorder:
        return store

    @provide
    def history(self, store: MessageStore) -> ChatHistory:
        return store

    @provide
    async def memory(
        self, settings: Settings, store: MessageStore
    ) -> AsyncIterator[ChatMemoryService]:
        client = AsyncOpenAI(api_key=settings.openai_api_key, timeout=60, max_retries=1)
        try:
            vectors = EmbeddingCache(store.database, MemoryEmbeddings(client))
            memory = HybridChatMemory(
                store,
                vectors,
                owner_user_id=settings.owner_user_id,
            )
            try:
                await memory.start()
                yield memory
            finally:
                await memory.close()
        finally:
            await client.close()
