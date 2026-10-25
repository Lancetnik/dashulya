from collections.abc import AsyncIterator

from dishka import Provider, provide
from dishka_ag2 import AG2Scope

from dashulya.domain.memory import ChatHistory, MemoryBackend, MessageRecorder
from dashulya.main.config import Settings
from dashulya.memory.ingestion import MemoryIngestion
from dashulya.memory.live import LiveMemory
from dashulya.memory.openviking import OpenVikingMemory
from dashulya.memory.search import MemorySearch


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
    ) -> AsyncIterator[MemoryBackend]:
        if not settings.memory_path.is_file():
            error_message = (
                "OpenViking requires the exported SQLite index at MEMORY_PATH."
            )
            raise ValueError(error_message)
        if not settings.openviking_manifest.is_file():
            error_message = (
                "Set OPENVIKING_MANIFEST to a completed OpenViking import manifest."
            )
            raise ValueError(error_message)
        memory = OpenVikingMemory(
            await MemorySearch.open(settings.memory_path),
            settings.openviking_manifest,
            url=settings.openviking_url,
            api_key=settings.openviking_api_key,
            live=store,
        )
        await memory.check_ready()
        ingestion = MemoryIngestion(
            store, url=settings.openviking_url, api_key=settings.openviking_api_key
        )
        await ingestion.start()
        try:
            yield memory
        finally:
            await ingestion.close()
