from typing import Annotated, Any, Literal

from ag2 import Variable
from ag2.tools import tool
from dishka_ag2 import FromDishka, inject

from dashulya.ai.message_format import MessageFormatter
from dashulya.domain.memory import HistoryReader


@tool
@inject
async def chat_history(
    action: Literal["list", "read", "count", "authors", "chats"] = "list",
    text_contains: str = "",
    name_contains: str | None = None,
    case_sensitive: bool = False,
    author_id: int | None = None,
    after: str | None = None,
    before: str | None = None,
    chat: str | None = None,
    source_id: str | None = None,
    cursor: str | None = None,
    offset: int = 0,
    limit: int = 20,
    topic_id: int | None = None,
    parent_hops: int = 5,
    *,
    history: FromDishka[HistoryReader],
    formatter: FromDishka[MessageFormatter],
    user_id: Annotated[int | None, Variable(default=None)],
) -> dict[str, Any]:
    """Read, page and count canonical saved originals independently of embeddings.

    action:
    - list: chronological pages traversed backwards with next_cursor. offset skips
      rows and cannot accompany cursor. limit is 1-50.
    - read: requires source_id; returns up to 12000 characters at offset, with
      next_offset for continuation. List/context excerpts may be truncated.
    - count: exact matching original count, including blank saved records.
    - authors: numeric IDs, names/usernames and examples. name_contains casefolds
      names/handles (optional leading @); text_contains still filters message bodies.
    - chats: saved collections and examples, not Telegram's complete chat list.

    text_contains is literal; case_sensitive controls matching. after is inclusive,
    before exclusive, both timezone-aware ISO timestamps. author_id/chat/topic_id
    narrow the authorized snapshot; topics are unrestricted by default.
    list/read add reply ancestors as separate context via parent_hops (0 disables;
    any nonnegative depth). Expansion stays in the source chat/topic;
    primary matches, counts and pagination remain independent. Ancestors may have
    other authors/dates. For truncated_thread navigation, read boundary_source_id with
    parent_hops=remaining_hops to retrieve the omitted chain; null means unknown.
    """
    result = await history.query(
        action,
        text_contains=text_contains,
        name_contains=name_contains,
        case_sensitive=case_sensitive,
        author_id=author_id,
        after=after,
        before=before,
        chat=chat,
        source_id=source_id,
        cursor=cursor,
        offset=offset,
        limit=limit,
        topic_id=topic_id,
        parent_hops=parent_hops,
    )
    return formatter.result(result, interlocutor_id=user_id)
