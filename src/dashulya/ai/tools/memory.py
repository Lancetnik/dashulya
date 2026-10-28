from typing import Annotated, Any

from ag2 import Variable
from ag2.tools import tool
from dishka_ag2 import FromDishka, inject

from dashulya.ai.message_format import MessageFormatter
from dashulya.domain.memory import MemoryBackend


@tool
@inject
async def search_memory(
    query: str = "",
    memory_id: str | None = None,
    chat: str | None = None,
    author_id: int | None = None,
    after: str | None = None,
    before: str | None = None,
    limit: int = 8,
    *,
    memory: FromDishka[MemoryBackend],
    formatter: FromDishka[MessageFormatter],
    user_id: Annotated[int | None, Variable(default=None)],
) -> dict[str, Any]:
    """Recall saved messages by meaning/keywords, or read a returned memory_id.

    Supply query or memory_id. chat narrows to a collection from chat_history(chats);
    author_id narrows authors. after/before are inclusive timezone-aware ISO bounds.
    limit is 1-8. hits distinguish primary sources from supporting reply/nearby
    context, which may have other authors/dates inside the authorized snapshot.
    Check coverage and pending blocks. Read full sources with chat_history(read).
    """
    result = await memory.search(
        query,
        memory_id=memory_id,
        chat=chat,
        author_id=author_id,
        after=after,
        before=before,
        limit=limit,
    )
    return formatter.result(result, interlocutor_id=user_id)
