"""Strictly-sequenced Binance USDT-M diff-depth book with per-level change attribution.

Sync rules (Binance USDT-M "How to manage a local order book correctly"):
  1. buffer diff events, fetch a REST snapshot (lastUpdateId = L)
  2. drop buffered events with u < L
  3. the first applied event must satisfy U <= L <= u
  4. every later event must satisfy pu == previous u
Any violation -- or a crossed book -- is a GAP: the book is marked invalid, every later
event is buffered, and the owner must fetch a new snapshot. The book never continues
silently on a corrupted state.

Every applied diff returns its level changes ``(side, price, old_qty, new_qty)`` so
queue-depletion / replenishment / cancellation features can be attributed exactly.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass, field

OK = "ok"
IGNORED = "ignored"          # stale event (u < L), already contained in the snapshot
BUFFERING = "buffering"      # no valid state; waiting for a snapshot
GAP = "gap"                  # continuity broken -> snapshot required

BID, ASK = 1, -1
MAX_BUFFER = 20_000


@dataclass(slots=True)
class Change:
    side: int        # BID / ASK
    price: float
    old: float
    new: float


@dataclass
class BookStats:
    applied: int = 0
    ignored: int = 0
    gaps: int = 0
    crossed: int = 0
    snapshots: int = 0
    stale_snapshots: int = 0
    buffer_overflow: int = 0
    gap_reasons: dict = field(default_factory=dict)


class L2Book:
    def __init__(self, symbol: str) -> None:
        self.symbol = symbol
        self.bids: dict[float, float] = {}
        self.asks: dict[float, float] = {}
        self.last_u = 0
        self.valid = False
        self.awaiting_bridge = False
        self.last_event_ms = 0
        self.last_trans_ms = 0
        self.valid_since_ms: int | None = None
        self.stats = BookStats()
        self._buffer: list[dict] = []
        self._top: tuple[list, list] | None = None

    # ------------------------------------------------------------------ sync
    def mark_gap(self, reason: str) -> None:
        self.valid = False
        self.awaiting_bridge = False
        self.valid_since_ms = None
        self.stats.gaps += 1
        self.stats.gap_reasons[reason] = self.stats.gap_reasons.get(reason, 0) + 1

    def on_snapshot(self, snap: dict, local_ms: int) -> tuple[str, list[Change]]:
        """Load a REST snapshot, then replay buffered diffs. Returns (status, changes)."""
        self.stats.snapshots += 1
        self.bids = {float(p): float(q) for p, q in snap["bids"] if float(q) > 0}
        self.asks = {float(p): float(q) for p, q in snap["asks"] if float(q) > 0}
        self.last_u = int(snap["lastUpdateId"])
        self._top = None
        self.valid = False
        self.awaiting_bridge = True
        buffered, self._buffer = self._buffer, []
        changes: list[Change] = []
        status = BUFFERING
        for ev in buffered:
            status, ch = self.on_diff(ev, local_ms)
            changes += ch
            if status == GAP:
                return GAP, changes
        if self.awaiting_bridge:
            # nothing bridged yet: valid only once the next live event satisfies U <= L <= u
            return BUFFERING, changes
        return (OK if self.valid else status), changes

    def on_diff(self, ev: dict, local_ms: int) -> tuple[str, list[Change]]:
        if not self.valid and not self.awaiting_bridge:
            self._buf(ev)
            return BUFFERING, []
        U, u = int(ev["U"]), int(ev["u"])
        if u < self.last_u:
            self.stats.ignored += 1
            return IGNORED, []
        if self.awaiting_bridge:
            if not (U <= self.last_u <= u):
                if U > self.last_u:
                    # snapshot older than the oldest buffered event: need a newer snapshot
                    self.stats.stale_snapshots += 1
                    self.mark_gap("snapshot_not_bridged")
                    self._buf(ev)
                    return GAP, []
                self.stats.ignored += 1
                return IGNORED, []
            self.awaiting_bridge = False
            self.valid = True
            self.valid_since_ms = int(ev.get("E", local_ms))
        else:
            pu = ev.get("pu")
            if pu is not None and int(pu) != self.last_u:
                self.mark_gap("pu_mismatch")
                self._buf(ev)
                return GAP, []
        changes: list[Change] = []
        for side, book, key in ((BID, self.bids, "b"), (ASK, self.asks, "a")):
            for p, q in ev.get(key, ()):
                p, q = float(p), float(q)
                old = book.get(p, 0.0)
                if q <= 0:
                    if p in book:
                        del book[p]
                else:
                    book[p] = q
                if old != q:
                    changes.append(Change(side, p, old, max(q, 0.0)))
        self.last_u = u
        self.last_event_ms = int(ev.get("E", local_ms))
        self.last_trans_ms = int(ev.get("T", self.last_event_ms))
        self._top = None
        self.stats.applied += 1
        if self.bids and self.asks and max(self.bids) >= min(self.asks):
            self.stats.crossed += 1
            self.mark_gap("crossed")
            return GAP, changes
        return OK, changes

    def _buf(self, ev: dict) -> None:
        self._buffer.append(ev)
        if len(self._buffer) > MAX_BUFFER:
            self.stats.buffer_overflow += 1
            self._buffer = self._buffer[-MAX_BUFFER:]

    # ------------------------------------------------------------------ queries
    def top(self, n: int = 20) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
        if self._top is None or len(self._top[0]) < min(n, len(self.bids)) or len(self._top[1]) < min(n, len(self.asks)):
            k = max(n, 50)
            self._top = (heapq.nlargest(k, self.bids.items()), heapq.nsmallest(k, self.asks.items()))
        b, a = self._top
        return b[:n], a[:n]

    def best(self) -> tuple[float, float, float, float]:
        b, a = self.top(1)
        if not b or not a:
            return 0.0, 0.0, 0.0, 0.0
        return b[0][0], b[0][1], a[0][0], a[0][1]

    def walk(self, side: str, notional: float) -> tuple[float, bool]:
        """Average fill price of a taker order of ``notional`` (BUY eats asks); (price, full)."""
        b, a = self.top(200)
        levels = a if side == "BUY" else b
        rem, cost, qty = notional, 0.0, 0.0
        for p, q in levels:
            take = min(p * q, rem)
            cost += take
            qty += take / p
            rem -= take
            if rem <= 1e-9:
                break
        if qty <= 0:
            return 0.0, False
        return cost / qty, rem <= 1e-9

    def walk_bps(self, side: str, notional: float) -> float:
        """Slippage beyond the touch (bps) for a taker order; inf if the visible book is too thin."""
        bid, _, ask, _ = self.best()
        avg, full = self.walk(side, notional)
        if not avg or not full:
            return float("inf")
        return (avg - ask) / ask * 1e4 if side == "BUY" else (bid - avg) / bid * 1e4
