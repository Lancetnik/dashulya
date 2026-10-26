from collections.abc import AsyncIterator

from dishka import Provider, provide
from dishka_ag2 import AG2Scope
from openai import AsyncOpenAI

from dashulya.ai.embeddings import MemoryEmbeddings
from dashulya.domain.memory import ChatHistory, ChatMemoryService, MessageRecorder
from dashulya.main.config import Settings
from dashulya.memory.live import LiveMemory
from dashulya.memory.search import MemorySearch
from dashulya.memory.service import HybridChatMemory
from dashulya.memory.vectors import EmbeddingCache


class MemoryProvider(Provider):
    scope = AG2Scope.APP

    @provide
    async def live(self, settings: Settings) -> LiveMemory:
        store = LiveMemory(settings.live_memory_path)
        await store.initialize()
        return store

    @provide
    def recorder(self, store: LiveMemory) -> MessageRecorder:
        return store

    @provide
    def history(self, store: LiveMemory) -> ChatHistory:
        return store

    @provide
    async def memory(
        self, settings: Settings, store: LiveMemory
    ) -> AsyncIterator[ChatMemoryService]:
        imported = (
            await MemorySearch.open(settings.memory_path)
            if settings.memory_path.is_file()
            else None
        )
        if settings.memory_import_chat_id is not None and imported is None:
            error_message = "MEMORY_IMPORT_CHAT_ID requires an existing MEMORY_PATH."
            raise ValueError(error_message)
        client = AsyncOpenAI(api_key=settings.openai_api_key, timeout=60, max_retries=1)
        try:
            vectors = EmbeddingCache(
                settings.memory_vectors_path, MemoryEmbeddings(client)
            )
            memory = HybridChatMemory(
                store,
                vectors,
                owner_user_id=settings.owner_user_id,
                imported=imported,
                import_chat_id=settings.memory_import_chat_id,
            )
            try:
                await memory.start()
                yield memory
            finally:
                await memory.close()
        finally:
            await client.close()
