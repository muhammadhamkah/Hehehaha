"""Rolling executed-trade state for one symbol (fed by aggTrade).

Window aggregates use prefix-sum arrays, so each window query is O(log n) instead of a
scan over every trade in the window. This matters for replay speed.
"""
from __future__ import annotations

from array import array
from bisect import bisect_left, bisect_right
from collections import deque
from dataclasses import dataclass


@dataclass(slots=True)
class Trade:
    ts_ms: int
    price: float
    qty: float
    is_buyer_maker: bool   # True -> seller was the aggressor

    @property
    def notional(self) -> float:
        return self.price * self.qty

    @property
    def aggressor(self) -> int:
        """+1 aggressive buy, -1 aggressive sell."""
        return -1 if self.is_buyer_maker else 1


@dataclass(slots=True)
class FlowWindow:
    buy_notional: float = 0.0
    sell_notional: float = 0.0
    buy_count: int = 0
    sell_count: int = 0
    first_price: float = 0.0
    last_price: float = 0.0
    vwap: float = 0.0

    @property
    def total_notional(self) -> float:
        return self.buy_notional + self.sell_notional

    @property
    def count(self) -> int:
        return self.buy_count + self.sell_count


class TradeFlow:
    def __init__(self, symbol: str, history_s: float = 120.0) -> None:
        self.symbol = symbol
        self.history_ms = int(history_s * 1000)
        self.trades: deque[Trade] = deque()
        self.last_ts_ms: int = 0
        self.last_local_ts_ms: int = 0
        self.last_trade_id: int = 0
        # Parallel arrays from absolute index ``_base``; cumulative sums include the
        # element itself. ``_c*[i]`` - ``_c*[j]`` aggregates trades (j, i].
        self._base = 0
        self._ts: list[int] = []
        self._px: list[float] = []
        self._notional = array("d")   # typed buffer: numpy views for medians
        self._qty: list[float] = []
        self._sell: list[bool] = []
        self._cb: list[float] = []   # cumulative aggressive-buy notional
        self._cs: list[float] = []   # cumulative aggressive-sell notional
        self._nb: list[int] = []     # cumulative buy count
        self._ns: list[int] = []     # cumulative sell count
        self._cq: list[float] = []   # cumulative base qty

    def add(self, trade: Trade, local_ts_ms: int = 0, trade_id: int = 0) -> None:
        if trade_id and trade_id <= self.last_trade_id:
            return  # duplicate after reconnect
        if trade_id:
            self.last_trade_id = trade_id
        if self._ts and trade.ts_ms < self._ts[-1]:
            # Out-of-order print: clamp so arrays stay sorted (ordering within ms is not
            # guaranteed across reconnects).
            trade = Trade(self._ts[-1], trade.price, trade.qty, trade.is_buyer_maker)
        self.trades.append(trade)
        self.last_ts_ms = trade.ts_ms
        self.last_local_ts_ms = local_ts_ms or trade.ts_ms
        n = trade.notional
        sell = trade.is_buyer_maker
        if self._ts:
            cb, cs, nb, ns, cq = self._cb[-1], self._cs[-1], self._nb[-1], self._ns[-1], self._cq[-1]
        else:
            cb = cs = cq = 0.0
            nb = ns = 0
        self._ts.append(trade.ts_ms)
        self._px.append(trade.price)
        self._notional.append(n)
        self._qty.append(trade.qty)
        self._sell.append(sell)
        self._cb.append(cb + (0.0 if sell else n))
        self._cs.append(cs + (n if sell else 0.0))
        self._nb.append(nb + (0 if sell else 1))
        self._ns.append(ns + (1 if sell else 0))
        self._cq.append(cq + trade.qty)
        self._prune(trade.ts_ms - self.history_ms)

    def _prune(self, cutoff: int) -> None:
        while self.trades and self.trades[0].ts_ms < cutoff:
            self.trades.popleft()
        drop = bisect_left(self._ts, cutoff)
        if drop > 2048 and drop > len(self._ts) // 2:
            # Keep one extra element so prefix differences at the boundary stay valid.
            k = drop - 1
            for name in ("_ts", "_px", "_notional", "_qty", "_sell", "_cb", "_cs", "_nb", "_ns", "_cq"):
                setattr(self, name, getattr(self, name)[k:])
            self._base += k

    def _range(self, start_ms: int, end_ms: int) -> tuple[int, int]:
        """Array indexes [i, j) of trades with start_ms < ts <= end_ms."""
        return bisect_right(self._ts, start_ms), bisect_right(self._ts, end_ms)

    def window(self, now_ms: int, window_s: float, end_offset_s: float = 0.0) -> FlowWindow:
        """Aggregate trades in (now - offset - window, now - offset]."""
        end = now_ms - int(end_offset_s * 1000)
        start = end - int(window_s * 1000)
        i, j = self._range(start, end)
        w = FlowWindow()
        if j <= i:
            return w
        hi = j - 1
        lo = i - 1
        w.buy_notional = self._cb[hi] - self._prior(self._cb, lo, 0)
        w.sell_notional = self._cs[hi] - self._prior(self._cs, lo, 1)
        w.buy_count = self._nb[hi] - int(self._prior(self._nb, lo, 2))
        w.sell_count = self._ns[hi] - int(self._prior(self._ns, lo, 3))
        qv = self._cq[hi] - self._prior(self._cq, lo, 4)
        w.first_price = self._px[i]
        w.last_price = self._px[hi]
        w.vwap = w.total_notional / qv if qv > 0 else 0.0
        return w

    def _prior(self, arr: list, idx: int, kind: int) -> float:
        """Cumulative value at ``idx``; for idx == -1, the value just before element 0."""
        if idx >= 0:
            return arr[idx]
        n, sell = self._notional[0], self._sell[0]
        own = (
            (0.0 if sell else n),
            (n if sell else 0.0),
            (0 if sell else 1),
            (1 if sell else 0),
            self._qty[0],
        )[kind]
        return arr[0] - own

    def last_price(self) -> float:
        return self._px[-1] if self._px else 0.0

    def price_at(self, ts_ms: int) -> float:
        """Last traded price at or before ``ts_ms`` (0 if none)."""
        i = bisect_right(self._ts, ts_ms) - 1
        return self._px[i] if i >= 0 else 0.0

    def recent_sizes(self, now_ms: int, window_s: float) -> list[float]:
        """Notionals of trades with now - window <= ts <= now (never future prints)."""
        i = bisect_left(self._ts, now_ms - int(window_s * 1000))
        j = bisect_right(self._ts, now_ms)
        return self._notional[i:j]

    def sides_and_sizes(self, now_ms: int, window_s: float) -> tuple[list[float], list[bool]]:
        """(notionals, is_sell_aggressor) for trades with now - window < ts <= now."""
        i, j = self._range(now_ms - int(window_s * 1000), now_ms)
        return self._notional[i:j], self._sell[i:j]

    def is_stale(self, now_ms: int, max_age_ms: int) -> bool:
        return now_ms - self.last_local_ts_ms > max_age_ms
