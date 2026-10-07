"""Stream Binance public-archive files (data.binance.vision) directly into replay.

A directory with an ``archive.json`` manifest (written by ``tools.binance_archive``)
is replayed straight from the downloaded daily zip files -- no intermediate event
files. Payloads are shaped exactly like the WebSocket messages
(``<sym>@aggTrade`` / ``<sym>@bookTicker``), so they go through the same validators
and handlers as live data. Archive rows carry exchange time only.
"""
from __future__ import annotations

import csv
import heapq
import io
import json
import os
import zipfile
from datetime import date, datetime, timezone
from typing import Iterator

from data.event_store import ReplayEvent

MANIFEST = "archive.json"


def _csv_rows(path: str) -> Iterator[list[str]]:
    if path.endswith(".zip"):
        with zipfile.ZipFile(path) as zf:
            for name in zf.namelist():
                with zf.open(name) as fh:
                    yield from csv.reader(io.TextIOWrapper(fh, encoding="utf-8"))
    else:
        with open(path, encoding="utf-8") as fh:
            yield from csv.reader(fh)


def agg_trade_events(path: str, symbol: str) -> Iterator[tuple[int, dict]]:
    """Columns: agg_trade_id,price,quantity,first_trade_id,last_trade_id,transact_time,is_buyer_maker"""
    for row in _csv_rows(path):
        if not row or not row[0].strip().lstrip("-").isdigit():
            continue  # header
        a, p, q, f, l, t, m = row[:7]
        ts = int(t)
        yield ts, {"e": "aggTrade", "E": ts, "s": symbol, "a": int(a), "p": p, "q": q,
                   "f": int(f), "l": int(l), "T": ts, "m": m.strip().lower() == "true"}


def book_ticker_events(path: str, symbol: str) -> Iterator[tuple[int, dict]]:
    """Columns: update_id,best_bid_price,best_bid_qty,best_ask_price,best_ask_qty,transaction_time,event_time"""
    for row in _csv_rows(path):
        if not row or not row[0].strip().isdigit():
            continue
        u, b, bq, a, aq, tt, et = row[:7]
        ev = int(et) if et.strip() else int(tt)
        yield ev, {"e": "bookTicker", "u": int(u), "E": ev, "T": int(tt), "s": symbol,
                   "b": b, "B": bq, "a": a, "A": aq}


def _day_of(path: str) -> str:
    return os.path.basename(path).rsplit(".", 1)[0][-10:]        # ...-YYYY-MM-DD.zip


def _day_ms(day: str) -> int:
    d = date.fromisoformat(day)
    return int(datetime(d.year, d.month, d.day, tzinfo=timezone.utc).timestamp() * 1000)


def is_archive_dir(path: str) -> bool:
    return os.path.exists(os.path.join(path, MANIFEST))


def load_manifest(path: str) -> dict:
    with open(os.path.join(path, MANIFEST), encoding="utf-8") as fh:
        return json.load(fh)


class ArchiveReader:
    """Same interface as EventReader, over manifest-listed archive zips."""

    def __init__(self, root: str, start_ms: int | None = None, end_ms: int | None = None,
                 time_source: str = "exchange", feed_latency_ms: int = 0, reorder_window_ms: int = 0,
                 symbols: list[str] | None = None) -> None:
        self.root = root
        self.manifest = load_manifest(root)
        self.start_ms = start_ms
        self.end_ms = end_ms
        self.feed_latency_ms = feed_latency_ms
        self.symbols = symbols
        self.n_clamped = self.n_late = self.n_bad_lines = 0
        self.n_events = 0

    def _files(self) -> list[tuple[str, str, str]]:
        out = []
        for sym, kinds in self.manifest["files"].items():
            if self.symbols and sym not in self.symbols:
                continue
            for kind, paths in kinds.items():
                for p in paths:
                    d0 = _day_ms(_day_of(p))
                    if self.end_ms is not None and d0 >= self.end_ms:
                        continue
                    if self.start_ms is not None and d0 + 86_400_000 <= self.start_ms:
                        continue
                    out.append((sym, kind, p if os.path.isabs(p) else os.path.join(self.root, p)))
        return sorted(out)

    def __iter__(self) -> Iterator[ReplayEvent]:
        def tagged(sym: str, kind: str, path: str, idx: int):
            gen = agg_trade_events(path, sym) if kind == "aggTrades" else book_ticker_events(path, sym)
            stream = f"{sym.lower()}@{'aggTrade' if kind == 'aggTrades' else 'bookTicker'}"
            for ts, d in gen:
                yield ts, idx, stream, d

        its = [tagged(s, k, p, i) for i, (s, k, p) in enumerate(self._files())]
        seq = 0
        for ts, _, stream, d in heapq.merge(*its, key=lambda x: (x[0], x[1])):
            t = ts + self.feed_latency_ms
            if self.start_ms is not None and t < self.start_ms:
                continue
            if self.end_ms is not None and t >= self.end_ms:
                continue
            seq += 1
            self.n_events += 1
            yield ReplayEvent(t, ts, ts, "detail", stream, d, seq)

    def time_range(self) -> tuple[int, int] | None:
        days = sorted({_day_of(p) for kinds in self.manifest["files"].values()
                       for paths in kinds.values() for p in paths})
        if not days:
            return None
        return _day_ms(days[0]), _day_ms(days[-1]) + 86_400_000 - 1
