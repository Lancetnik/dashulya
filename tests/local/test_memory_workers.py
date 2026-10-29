import asyncio
import threading

import pytest

from dashulya.memory._workers import run_in_thread


class TestMemoryWorker:
    async def wait_for_thread(self, event: threading.Event) -> None:
        assert await asyncio.wait_for(asyncio.to_thread(event.wait, 2), timeout=3)

    async def test_cancellation_waits_for_started_work_even_after_another_cancel(
        self,
    ) -> None:
        started, release, finished = (threading.Event() for _ in range(3))

        def work() -> int:
            started.set()
            if not release.wait(2):
                raise TimeoutError
            finished.set()
            return 42

        task = asyncio.create_task(run_in_thread(work))
        try:
            await self.wait_for_thread(started)
            for _ in range(2):
                task.cancel()
                for _ in range(4):
                    await asyncio.sleep(0)
                assert not task.done(), "Cancellation released unfinished thread work"
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, timeout=1)
            assert finished.is_set()
        finally:
            release.set()
            await asyncio.gather(task, return_exceptions=True)

    async def test_cancelled_failing_work_preserves_cancellation_without_cleanup(
        self,
    ) -> None:
        started, release = threading.Event(), threading.Event()
        cleaned: list[int] = []

        def work() -> int:
            started.set()
            if not release.wait(2):
                raise TimeoutError
            error_message = "Synthetic worker failure"
            raise ValueError(error_message)

        task = asyncio.create_task(run_in_thread(work, on_cancel=cleaned.append))
        try:
            await self.wait_for_thread(started)
            task.cancel()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, timeout=1)
            assert cleaned == []
        finally:
            release.set()
            await asyncio.gather(task, return_exceptions=True)

    async def test_cancellation_also_drains_cleanup_of_the_completed_result(self) -> None:
        started, release, cleaning, finish_cleanup = (threading.Event() for _ in range(4))
        cleaned: list[int] = []

        def work() -> int:
            started.set()
            if not release.wait(2):
                raise TimeoutError
            return 42

        def cleanup(result: int) -> None:
            cleaning.set()
            if not finish_cleanup.wait(2):
                raise TimeoutError
            cleaned.append(result)

        task = asyncio.create_task(run_in_thread(work, on_cancel=cleanup))
        try:
            await self.wait_for_thread(started)
            task.cancel()
            release.set()
            await self.wait_for_thread(cleaning)
            task.cancel()
            for _ in range(4):
                await asyncio.sleep(0)
            assert not task.done(), "Cancellation abandoned resource cleanup"
            assert cleaned == []
            finish_cleanup.set()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, timeout=1)
            assert cleaned == [42]
        finally:
            release.set()
            finish_cleanup.set()
            await asyncio.gather(task, return_exceptions=True)
