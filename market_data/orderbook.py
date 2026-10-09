"""Local L2 order book for one symbol.

Supports two feeds:
  * partial depth snapshots (``<sym>@depth20@100ms``) via :meth:`apply_snapshot`
  * diff depth (``<sym>@depth@100ms``) with Binance USDT-M sequencing rules via
    :meth:`apply_diff` after an initial REST snapshot.

Binance USDT-M diff-depth sync rules:
  1. Buffer events, fetch REST snapshot (lastUpdateId = L).
  2. Drop events with u < L.
  3. First processed event must satisfy U <= L <= u.
  4. Each subsequent event's ``pu`` must equal previous event's ``u``; else resync.

Also keeps a ring buffer of compact book states used by persistence/depletion features.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable

from utils.mathx import imbalance


@dataclass(slots=True)
class BookState:
    ts_ms: int
    bid: float
    ask: float
    bid_qty: float
    ask_qty: float
    bid_depth5: float    # quote notional, top-5 levels
    ask_depth5: float
    bid_depth10: float
    ask_depth10: float

    @property
    def mid(self) -> float:
        return 0.5 * (self.bid + self.ask)

    @property
    def imbalance5(self) -> float:
        return imbalance(self.bid_depth5, self.ask_depth5)


class SyncStatus:
    OK = "ok"
    IGNORED = "ignored"       # stale event (u < last_update_id)
    RESYNC = "resync"         # gap detected; caller must re-snapshot
    BUFFERING = "buffering"   # waiting for snapshot


class OrderBook:
    def __init__(self, symbol: str, history_len: int = 600, max_levels: int = 1000,
                 history_interval_ms: int = 0) -> None:
        self.symbol = symbol
        self.bids: dict[float, float] = {}
        self.asks: dict[float, float] = {}
        self.last_update_id: int = 0
        self.last_event_ts_ms: int = 0
        self.last_local_ts_ms: int = 0
        self.synced: bool = False
        self.max_levels = max_levels
        self.history: deque[BookState] = deque(maxlen=history_len)
        # >0: conflate history into time buckets (latest state per bucket) so a fixed-length
        # history spans a fixed time window regardless of update rate (tick-level L1 feeds).
        self.history_interval_ms = history_interval_ms
        self._buffer: list[dict] = []
        self._prev_u: int | None = None
        self._sorted_cache: tuple[list[tuple[float, float]], list[tuple[float, float]]] | None = None

    # ------------------------------------------------------------------ updates
    def apply_snapshot(
        self,
        bids: Iterable[tuple[float, float]],
        asks: Iterable[tuple[float, float]],
        last_update_id: int = 0,
        event_ts_ms: int = 0,
        local_ts_ms: int = 0,
    ) -> None:
        self.bids = {p: q for p, q in bids if q > 0}
        self.asks = {p: q for p, q in asks if q > 0}
        self.last_update_id = last_update_id
        self.last_event_ts_ms = event_ts_ms
        self.last_local_ts_ms = local_ts_ms or event_ts_ms
        self.synced = bool(self.bids and self.asks)
        self._prev_u = None
        self._sorted_cache = None
        self._record_state()

    def buffer_diff(self, event: dict) -> None:
        """Buffer diff events while waiting for a REST snapshot."""
        self._buffer.append(event)
        if len(self._buffer) > 5000:
            self._buffer = self._buffer[-5000:]

    def sync_from_snapshot(
        self,
        bids: Iterable[tuple[float, float]],
        asks: Iterable[tuple[float, float]],
        last_update_id: int,
        local_ts_ms: int,
    ) -> str:
        """Apply a REST snapshot then replay buffered diffs per Binance rules."""
        self.apply_snapshot(bids, asks, last_update_id, local_ts_ms=local_ts_ms)
        self.synced = False
        buffered, self._buffer = self._buffer, []
        status = SyncStatus.BUFFERING
        for ev in buffered:
            status = self.apply_diff(ev, local_ts_ms)
            if status == SyncStatus.RESYNC:
                return status
        if not self.synced:
            # No bridging event yet; keep waiting for live events.
            self.synced = False
        return SyncStatus.OK if self.synced else SyncStatus.BUFFERING

    def apply_diff(self, ev: dict, local_ts_ms: int = 0) -> str:
        """Apply a diff-depth event: keys U (first), u (final), pu (prev final), b, a, E."""
        if self.last_update_id == 0:
            self.buffer_diff(ev)
            return SyncStatus.BUFFERING
        first_id, final_id = int(ev["U"]), int(ev["u"])
        prev_final = ev.get("pu")
        if final_id < self.last_update_id:
            return SyncStatus.IGNORED
        if not self.synced:
            if not (first_id <= self.last_update_id <= final_id):
                self.synced = False
                return SyncStatus.RESYNC
        else:
            if prev_final is not None and self._prev_u is not None and int(prev_final) != self._prev_u:
                self.synced = False
                return SyncStatus.RESYNC
        for p, q in ev.get("b", []):
            self._set(self.bids, float(p), float(q))
        for p, q in ev.get("a", []):
            self._set(self.asks, float(p), float(q))
        self._prev_u = final_id
        self.last_update_id = final_id
        self.last_event_ts_ms = int(ev.get("E", local_ts_ms))
        self.last_local_ts_ms = local_ts_ms or self.last_event_ts_ms
        self.synced = bool(self.bids and self.asks)
        self._sorted_cache = None
        self._trim()
        if self.crossed():
            self.synced = False
            return SyncStatus.RESYNC
        self._record_state()
        return SyncStatus.OK

    def update_bbo(self, bid: float, bid_qty: float, ask: float, ask_qty: float, ts_ms: int) -> None:
        """Overlay a faster bookTicker update onto the top of book."""
        if bid <= 0 or ask <= 0 or bid >= ask:
            return
        for p in [p for p in self.bids if p > bid]:
            del self.bids[p]
        for p in [p for p in self.asks if p < ask]:
            del self.asks[p]
        self._set(self.bids, bid, bid_qty)
        self._set(self.asks, ask, ask_qty)
        self.last_local_ts_ms = max(self.last_local_ts_ms, ts_ms)
        self._sorted_cache = None

    @staticmethod
    def _set(side: dict[float, float], price: float, qty: float) -> None:
        if qty <= 0:
            side.pop(price, None)
        else:
            side[price] = qty

    def _trim(self) -> None:
        if len(self.bids) > self.max_levels * 2:
            keep = sorted(self.bids, reverse=True)[: self.max_levels]
            self.bids = {p: self.bids[p] for p in keep}
        if len(self.asks) > self.max_levels * 2:
            keep = sorted(self.asks)[: self.max_levels]
            self.asks = {p: self.asks[p] for p in keep}

    def _record_state(self) -> None:
        if not (self.bids and self.asks):
            return
        b, a = self.sorted_levels()
        bid, bq = b[0]
        ask, aq = a[0]
        state = (
            BookState(
                ts_ms=self.last_local_ts_ms,
                bid=bid,
                ask=ask,
                bid_qty=bq,
                ask_qty=aq,
                bid_depth5=sum(p * q for p, q in b[:5]),
                ask_depth5=sum(p * q for p, q in a[:5]),
                bid_depth10=sum(p * q for p, q in b[:10]),
                ask_depth10=sum(p * q for p, q in a[:10]),
            )
        )
        iv = self.history_interval_ms
        if iv and self.history and self.history[-1].ts_ms // iv == state.ts_ms // iv:
            self.history[-1] = state
        else:
            self.history.append(state)

    # ------------------------------------------------------------------ queries
    def sorted_levels(self, n: int | None = None) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
        if self._sorted_cache is None:
            self._sorted_cache = (
                sorted(self.bids.items(), key=lambda kv: -kv[0]),
                sorted(self.asks.items(), key=lambda kv: kv[0]),
            )
        b, a = self._sorted_cache
        return (b[:n], a[:n]) if n else (b, a)

    def crossed(self) -> bool:
        return bool(self.bids and self.asks) and max(self.bids) >= min(self.asks)

    @property
    def best_bid(self) -> float:
        return max(self.bids) if self.bids else 0.0

    @property
    def best_ask(self) -> float:
        return min(self.asks) if self.asks else 0.0

    def best(self) -> tuple[float, float, float, float]:
        b, a = self.sorted_levels(1)
        if not b or not a:
            return 0.0, 0.0, 0.0, 0.0
        return b[0][0], b[0][1], a[0][0], a[0][1]

    @property
    def mid(self) -> float:
        bid, _, ask, _ = self.best()
        return 0.5 * (bid + ask) if bid and ask else 0.0

    @property
    def spread_bps(self) -> float:
        bid, _, ask, _ = self.best()
        if not bid or not ask:
            return float("inf")
        return (ask - bid) / (0.5 * (ask + bid)) * 1e4

    def microprice(self) -> float:
        """Size-weighted mid: leans toward the side with less resting quantity."""
        bid, bq, ask, aq = self.best()
        if not bid or not ask:
            return 0.0
        tot = bq + aq
        if tot <= 0:
            return 0.5 * (bid + ask)
        return (bid * aq + ask * bq) / tot

    def depth_within_bps(self, band_bps: float) -> tuple[float, float]:
        """Quote notional resting within ``band_bps`` of mid on each side."""
        mid = self.mid
        if not mid:
            return 0.0, 0.0
        lo = mid * (1 - band_bps / 1e4)
        hi = mid * (1 + band_bps / 1e4)
        b, a = self.sorted_levels()
        bid_n = 0.0
        for p, q in b:
            if p < lo:
                break
            bid_n += p * q
        ask_n = 0.0
        for p, q in a:
            if p > hi:
                break
            ask_n += p * q
        return bid_n, ask_n

    def walk(self, side: str, notional: float) -> tuple[float, float, bool]:
        """Simulate a taker order consuming the book.

        side: "BUY" consumes asks, "SELL" consumes bids.
        Returns (avg_price, filled_base_qty, fully_filled).
        """
        b, a = self.sorted_levels()
        levels = a if side == "BUY" else b
        remaining = notional
        cost = 0.0
        qty = 0.0
        for p, q in levels:
            lvl_notional = p * q
            take = min(lvl_notional, remaining)
            cost += take
            qty += take / p
            remaining -= take
            if remaining <= 1e-12:
                break
        if qty <= 0:
            return 0.0, 0.0, False
        return cost / qty, qty, remaining <= 1e-9

    def slippage_bps(self, side: str, notional: float) -> float:
        """Expected taker slippage vs mid, in bps (includes half-spread)."""
        mid = self.mid
        avg, _, full = self.walk(side, notional)
        if not mid or not avg:
            return float("inf")
        if not full:
            return float("inf")
        return (avg - mid) / mid * 1e4 if side == "BUY" else (mid - avg) / mid * 1e4

    def is_stale(self, now_ms: int, max_age_ms: int) -> bool:
        return (not self.synced) or now_ms - self.last_local_ts_ms > max_age_ms
