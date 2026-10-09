"""Virtual-time asyncio event loop.

The loop's clock is a simulated integer-microsecond timestamp. Whenever the loop would
block waiting for its next timer, the selector instead ADVANCES simulated time to that
timer (optionally sleeping a scaled amount of real time). Every ``asyncio.sleep``,
``wait_for`` timeout, order-latency delay, maker TTL and periodic strategy loop therefore
runs on simulated time, unmodified, and fully deterministically.

speed=None  -> maximum-speed deterministic replay (no real sleeping)
speed=1.0   -> 1x real time
speed=k     -> k-times accelerated
"""
from __future__ import annotations

import asyncio
import selectors
import time

from utils.clock import Clock


class _VirtualSelector(selectors.DefaultSelector):
    # Replay has no external I/O; asyncio's self-pipe (thread wake-ups / signals) is only
    # polled every POLL_EVERY iterations to avoid an epoll syscall per loop iteration.
    POLL_EVERY = 256

    def __init__(self) -> None:
        super().__init__()
        self.loop: VirtualTimeEventLoop | None = None
        self._n = 0

    def select(self, timeout=None):
        loop = self.loop
        if loop is not None and timeout is not None and timeout > 0:
            # Round (not ceil): timer deadlines are whole microseconds, so rounding lands
            # exactly on them -- ceil overshot by 1us and made periodic loops drift.
            step_us = max(1, round(timeout * 1e6))
            if loop.speed:
                time.sleep(timeout / loop.speed)
            loop._vt_us += step_us
        self._n += 1
        if self._n % self.POLL_EVERY:
            return []
        return super().select(0)


class VirtualTimeEventLoop(asyncio.SelectorEventLoop):
    def __init__(self, start_ms: int, speed: float | None = None) -> None:
        selector = _VirtualSelector()
        super().__init__(selector)
        selector.loop = self
        self._vt_us = int(start_ms) * 1000
        # loop.time() is measured from the replay start: epoch-scale floats (~1.7e9 s)
        # only resolve ~0.24 us, which made timers due "now" never satisfy asyncio's
        # `when < now + resolution` check (a livelock). Small floats are exact enough.
        self._origin_us = self._vt_us
        self._clock_resolution = 1e-6
        self.speed = speed

    def time(self) -> float:  # noqa: D401 - asyncio API
        return (self._vt_us - self._origin_us) / 1e6

    def now_ms(self) -> int:
        return self._vt_us // 1000

    def try_advance_ms(self, ts_ms: int) -> bool:
        """Fast path: jump the clock to ``ts_ms`` without a loop iteration.

        Only valid when nothing is ready to run and no timer is due at or before the
        target time -- then a sleep would do exactly this and nothing else. Returns False
        (caller must ``await sleep_until_ms``) otherwise.
        """
        target_us = int(ts_ms) * 1000
        if target_us <= self._vt_us:
            return not self._ready
        if self._ready:
            return False
        if self._scheduled:
            first = self._scheduled[0]
            if first._when * 1e6 + self._origin_us <= target_us + 1:
                return False
        self._vt_us = target_us
        return True

    async def sleep_until_ms(self, ts_ms: int) -> None:
        """Advance simulated time to ``ts_ms`` (running every timer due before it)."""
        delta = ts_ms - self.now_ms()
        if delta > 0:
            await asyncio.sleep(delta / 1000.0)
        # Timers due exactly at ts_ms may still be pending in this iteration; yield once.
        await asyncio.sleep(0)


class LoopClock(Clock):
    """Bot clock bound to a virtual-time loop."""

    def __init__(self, loop: VirtualTimeEventLoop) -> None:
        self.loop = loop

    def now_ms(self) -> int:
        return self.loop.now_ms()
