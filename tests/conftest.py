"""Shared resource lifetimes for synchronous and asynchronous test fixtures."""

from collections.abc import AsyncIterator
from contextlib import AsyncExitStack

import pytest


@pytest.fixture
async def cleanup() -> AsyncIterator[AsyncExitStack]:
    async with AsyncExitStack() as stack:
        yield stack
