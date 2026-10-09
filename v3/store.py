"""Reader for a V3 L2 store (one event store per symbol + ``v3_store.json`` manifest).

Merges the per-symbol EventReaders chronologically, so the existing replay backtester and
the V3 dataset builder can read only the symbols they need.
"""
from __future__ import annotations

import heapq
import json
import os
from typing import Iterator

from data.event_store import EventReader, ReplayEvent, list_event_files

MANIFEST = "v3_store.json"


def is_v3_store(root: str) -> bool:
    return os.path.exists(os.path.join(root, MANIFEST))


def manifest(root: str) -> dict:
    with open(os.path.join(root, MANIFEST), encoding="utf-8") as fh:
        return json.load(fh)


def store_symbols(root: str) -> list[str]:
    return [s for s in manifest(root)["symbols"] if os.path.isdir(os.path.join(root, s))]


def store_days(root: str, symbol: str) -> list[str]:
    d = os.path.join(root, symbol)
    return sorted(x for x in os.listdir(d) if x.isdigit() and len(x) == 8) if os.path.isdir(d) else []


class V3StoreReader:
    def __init__(self, root: str, symbols: list[str] | None = None, **kw) -> None:
        self.root = root
        self.symbols = list(symbols) if symbols else store_symbols(root)
        self.readers = [EventReader(os.path.join(root, s), files=list_event_files(os.path.join(root, s)), **kw)
                        for s in self.symbols]

    def __iter__(self) -> Iterator[ReplayEvent]:
        return heapq.merge(*self.readers, key=lambda e: e.ts)

    def time_range(self) -> tuple[int, int] | None:
        rs = [r.time_range() for r in self.readers]
        rs = [r for r in rs if r]
        return (min(r[0] for r in rs), max(r[1] for r in rs)) if rs else None

    def _sum(self, attr: str) -> int:
        return sum(getattr(r, attr) for r in self.readers)

    n_clamped = property(lambda self: self._sum("n_clamped"))
    n_late = property(lambda self: self._sum("n_late"))
    n_bad_lines = property(lambda self: self._sum("n_bad_lines"))
    n_events = property(lambda self: self._sum("n_events"))
