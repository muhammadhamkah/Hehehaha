"""Raw market-event store for exact replay.

Every WebSocket message (and every REST depth snapshot used for diff-book sync) is
written verbatim with its local receive time:

    <root>/<YYYYMMDD>/<HH>.jsonl.gz      one JSON object per line:
        {"r": local_receive_ms, "c": "market"|"detail", "s": stream, "d": payload}
    <root>/meta.json                     symbol filters + recording config

The reader merges files back into a single chronological stream keyed on either the
EXCHANGE event time (default) or the LOCAL receive time, using a bounded re-ordering
buffer. Events are never emitted out of order: an event older than what has already
been emitted (beyond the re-order window) is clamped forward and counted, so replay can
never inject information into the past. A message whose delivery delay exceeded the
re-order window is replayed at its actual receive time, since that is when the live bot
could first have acted on it.
"""
from __future__ import annotations

import glob
import gzip
import heapq
import json
import logging
import os
import queue
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterator

log = logging.getLogger(__name__)


def exchange_time(stream: str, data: Any, local_ts: int) -> int:
    """Exchange event time of a payload (falls back to local receive time)."""
    if isinstance(data, dict):
        for key in ("E", "T"):
            v = data.get(key)
            if isinstance(v, int) and not isinstance(v, bool):
                return v
        return local_ts
    if isinstance(data, list):
        es = [x.get("E") for x in data if isinstance(x, dict) and isinstance(x.get("E"), int)]
        return max(es) if es else local_ts
    return local_ts


@dataclass(slots=True)
class ReplayEvent:
    ts: int            # replay time (exchange or local, plus optional feed latency)
    local_ts: int
    exchange_ts: int
    conn: str
    stream: str
    data: Any
    seq: int           # file order, deterministic tie-break


class EventWriter:
    def __init__(self, root: str, meta: dict | None = None, blocking: bool = False) -> None:
        """blocking=False (live): never stall the event loop; drops are counted and logged.
        blocking=True (offline conversion): back-pressure instead of dropping."""
        self.root = root
        self.blocking = blocking
        os.makedirs(root, exist_ok=True)
        if meta is not None:
            self.write_meta(meta)
        self._q: queue.Queue = queue.Queue(maxsize=200_000)
        self._dropped = 0
        self.n_written = 0
        self._thread = threading.Thread(target=self._run, name="event-writer", daemon=True)
        self._thread.start()

    def write_meta(self, meta: dict) -> None:
        path = os.path.join(self.root, "meta.json")
        existing = {}
        if os.path.exists(path):
            with open(path, encoding="utf-8") as fh:
                existing = json.load(fh)
        existing.update(meta)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(existing, fh, indent=1)

    def write(self, conn: str, stream: str, data: Any, local_ts: int) -> None:
        if self.blocking:
            self._q.put((local_ts, conn, stream, data))
            return
        try:
            self._q.put_nowait((local_ts, conn, stream, data))
        except queue.Full:
            self._dropped += 1
            if self._dropped in (1, 100, 10_000) or self._dropped % 100_000 == 0:
                log.error("event writer backlog full; dropped %d events (disk too slow?)", self._dropped)

    def _path(self, local_ts: int) -> str:
        dt = datetime.fromtimestamp(local_ts / 1000, tz=timezone.utc)
        d = os.path.join(self.root, dt.strftime("%Y%m%d"))
        os.makedirs(d, exist_ok=True)
        return os.path.join(d, dt.strftime("%H") + ".jsonl.gz")

    def _run(self) -> None:
        fh = None
        cur = None
        while True:
            item = self._q.get()
            if item is None:
                break
            local_ts, conn, stream, data = item
            path = self._path(local_ts)
            if path != cur:
                if fh is not None:
                    fh.close()
                fh = gzip.open(path, "at", compresslevel=3, encoding="utf-8")
                cur = path
            fh.write(json.dumps({"r": local_ts, "c": conn, "s": stream, "d": data}, separators=(",", ":")) + "\n")
            self.n_written += 1
        if fh is not None:
            fh.close()

    @property
    def dropped(self) -> int:
        return self._dropped

    def close(self) -> None:
        self._q.put(None)
        self._thread.join(timeout=60)
        if self._dropped:
            log.error("event writer dropped %d events in total", self._dropped)


def list_event_files(root: str) -> list[str]:
    return sorted(glob.glob(os.path.join(root, "*", "*.jsonl.gz")) + glob.glob(os.path.join(root, "*.jsonl.gz")))


def load_meta(root: str) -> dict:
    path = os.path.join(root, "meta.json")
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


class EventReader:
    """Chronological event stream over recorded files.

    time_source:      "exchange" (exchange event time E/T) or "local" (receive time)
    feed_latency_ms:  added to exchange time to model exchange->bot delivery delay
    reorder_window_ms: events are buffered this long (in input time) for sorting
    """

    def __init__(self, root: str, start_ms: int | None = None, end_ms: int | None = None,
                 time_source: str = "exchange", feed_latency_ms: int = 0,
                 reorder_window_ms: int = 5000, files: list[str] | None = None) -> None:
        if time_source not in ("exchange", "local"):
            raise ValueError("time_source must be 'exchange' or 'local'")
        self.root = root
        self.files = files if files is not None else list_event_files(root)
        self.start_ms = start_ms
        self.end_ms = end_ms
        self.time_source = time_source
        self.feed_latency_ms = feed_latency_ms
        self.reorder_window_ms = reorder_window_ms
        self.n_clamped = 0
        self.n_late = 0
        self.n_bad_lines = 0
        self.n_events = 0

    def _raw(self) -> Iterator[ReplayEvent]:
        seq = 0
        for path in self.files:
            with gzip.open(path, "rt", encoding="utf-8") as fh:
                for line in fh:
                    try:
                        o = json.loads(line)
                        local = int(o["r"])
                        stream, data, conn = o["s"], o["d"], o.get("c", "detail")
                    except (ValueError, KeyError, TypeError):
                        self.n_bad_lines += 1
                        continue
                    ex = exchange_time(stream, data, local)
                    if self.time_source == "local":
                        ts = local
                    elif local - ex > self.reorder_window_ms:
                        # Abnormally delayed delivery: the bot really only saw it on arrival.
                        ts = local
                        self.n_late += 1
                    else:
                        ts = ex + self.feed_latency_ms
                    seq += 1
                    yield ReplayEvent(ts, local, ex, conn, stream, data, seq)

    def __iter__(self) -> Iterator[ReplayEvent]:
        heap: list[tuple[int, int, ReplayEvent]] = []
        last_emitted = None
        newest_input = None
        for ev in self._raw():
            newest_input = ev.ts if newest_input is None else max(newest_input, ev.ts)
            heapq.heappush(heap, (ev.ts, ev.seq, ev))
            while heap and heap[0][0] <= newest_input - self.reorder_window_ms:
                last_emitted = yield from self._emit(heapq.heappop(heap)[2], last_emitted)
        while heap:
            last_emitted = yield from self._emit(heapq.heappop(heap)[2], last_emitted)

    def _emit(self, ev: ReplayEvent, last_emitted: int | None):
        if last_emitted is not None and ev.ts < last_emitted:
            self.n_clamped += 1          # arrived later than the re-order window allows
            ev.ts = last_emitted
        if self.start_ms is not None and ev.ts < self.start_ms:
            return ev.ts
        if self.end_ms is not None and ev.ts >= self.end_ms:
            return ev.ts
        self.n_events += 1
        yield ev
        return ev.ts

    def time_range(self) -> tuple[int, int] | None:
        """(first, last) replay timestamps of the dataset (one pass, no sorting)."""
        lo = hi = None
        for ev in self._raw():
            lo = ev.ts if lo is None else min(lo, ev.ts)
            hi = ev.ts if hi is None else max(hi, ev.ts)
        return (lo, hi) if lo is not None else None
