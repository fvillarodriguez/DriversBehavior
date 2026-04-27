from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable


HeartbeatCallback = Callable[[], Awaitable[None] | None]


class HeartbeatLoop:
    def __init__(self, callback: HeartbeatCallback, interval_seconds: float = 5.0):
        self.callback = callback
        self.interval_seconds = interval_seconds
        self._task: asyncio.Task[None] | None = None
        self._stopping = asyncio.Event()

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        self._stopping.set()
        if self._task:
            await self._task

    async def _run(self) -> None:
        while not self._stopping.is_set():
            result = self.callback()
            if asyncio.iscoroutine(result):
                await result
            try:
                await asyncio.wait_for(self._stopping.wait(), timeout=self.interval_seconds)
            except TimeoutError:
                pass

