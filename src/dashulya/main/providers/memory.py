from dishka import Provider, provide
from dishka_ag2 import AG2Scope

from dashulya.domain.memory import MemoryBackend
from dashulya.main.config import Settings
from dashulya.memory.openviking import OpenVikingMemory
from dashulya.memory.search import MemorySearch


class MemoryProvider(Provider):
    scope = AG2Scope.APP

    @provide
    async def memory(self, settings: Settings) -> MemoryBackend:
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
        )
        await memory.check_ready()
        return memory
