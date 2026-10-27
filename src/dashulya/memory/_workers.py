"""Cancellation-safe completion of bounded CPU and SQLite worker operations.

These operations cannot be interrupted once their thread starts. Keep their
caller's resource lease until completion. Provider/network awaits belong on the
owning event loop and must not use this helper.
"""

import asyncio
from collections.abc import Callable


async def _drain[Result](pending: asyncio.Task[Result]) -> bool:
    while not pending.done():
        try:
            await asyncio.shield(pending)
        except asyncio.CancelledError:
            continue
        except Exception:
            break
    return not pending.cancelled() and pending.exception() is None


async def run_in_thread[Result](
    call: Callable[[], Result],
    *,
    on_cancel: Callable[[Result], None] | None = None,
) -> Result:
    """Finish started work and optional resource cleanup before propagating cancellation."""
    pending = asyncio.create_task(asyncio.to_thread(call))
    try:
        return await asyncio.shield(pending)
    except asyncio.CancelledError:
        if await _drain(pending) and on_cancel is not None:
            result = pending.result()
            cleanup = asyncio.create_task(asyncio.to_thread(on_cancel, result))
            await _drain(cleanup)
        raise
