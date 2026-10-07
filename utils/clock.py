"""Injectable clock so simulators and tests can control time."""
from __future__ import annotations

import time


class Clock:
    def now_ms(self) -> int:
        return int(time.time() * 1000)

    def now_s(self) -> float:
        return self.now_ms() / 1000.0


class ManualClock(Clock):
    """Deterministic clock for tests and offline replay."""

    def __init__(self, start_ms: int = 1_700_000_000_000) -> None:
        self._now = start_ms

    def now_ms(self) -> int:
        return self._now

    def set(self, ms: int) -> None:
        self._now = ms

    def advance(self, ms: int) -> None:
        self._now += ms


SYSTEM_CLOCK = Clock()
