"""Test database validation and helpers for importing canonical history."""

import os
import sys

from dashulya.domain.memory import MemoryAccess
from dashulya.memory.history import CanonicalHistory
from dashulya.memory.migrate import import_evidence
from dashulya.memory.search import MemorySearch
from dashulya.memory.store import MessageStore


def require_database_url() -> str:
    url = os.getenv("TEST_DATABASE_URL", "").strip()
    if not url:
        error = (
            "Set TEST_DATABASE_URL to an isolated pgvector-enabled test database "
            "for the full suite, or run just test-local without PostgreSQL."
        )
        raise RuntimeError(error)
    return url


async def imported_history(
    store: MessageStore,
    access: MemoryAccess,
    *,
    imported: MemorySearch,
    import_chat_id: int | None = None,
) -> CanonicalHistory:
    await import_evidence(store, imported.path, import_chat_id)
    return CanonicalHistory(store, access)


if __name__ == "__main__":
    try:
        require_database_url()
    except RuntimeError as error:
        sys.exit(str(error))
