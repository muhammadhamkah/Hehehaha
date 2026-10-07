"""Rolling executed-trade state for one symbol (fed by aggTrade)."""
from __future__ import annotations

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
    max_trade_notional: float = 0.0

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

    def add(self, trade: Trade, local_ts_ms: int = 0, trade_id: int = 0) -> None:
        if trade_id and trade_id <= self.last_trade_id:
            return  # duplicate after reconnect
        if trade_id:
            self.last_trade_id = trade_id
        self.trades.append(trade)
        self.last_ts_ms = trade.ts_ms
        self.last_local_ts_ms = local_ts_ms or trade.ts_ms
        cutoff = trade.ts_ms - self.history_ms
        while self.trades and self.trades[0].ts_ms < cutoff:
            self.trades.popleft()

    def _start_index(self, start_ms: int) -> int:
        # deque supports indexing; binary search on timestamps
        lo, hi = 0, len(self.trades)
        while lo < hi:
            mid = (lo + hi) // 2
            if self.trades[mid].ts_ms < start_ms:
                lo = mid + 1
            else:
                hi = mid
        return lo

    def window(self, now_ms: int, window_s: float, end_offset_s: float = 0.0) -> FlowWindow:
        """Aggregate trades in (now - offset - window, now - offset]."""
        end = now_ms - int(end_offset_s * 1000)
        start = end - int(window_s * 1000)
        w = FlowWindow()
        i = self._start_index(start + 1)
        pv = 0.0
        qv = 0.0
        n = len(self.trades)
        while i < n:
            t = self.trades[i]
            if t.ts_ms > end:
                break
            nt = t.notional
            if t.is_buyer_maker:
                w.sell_notional += nt
                w.sell_count += 1
            else:
                w.buy_notional += nt
                w.buy_count += 1
            if not w.first_price:
                w.first_price = t.price
            w.last_price = t.price
            pv += nt
            qv += t.qty
            if nt > w.max_trade_notional:
                w.max_trade_notional = nt
            i += 1
        w.vwap = pv / qv if qv else 0.0
        return w

    def last_price(self) -> float:
        return self.trades[-1].price if self.trades else 0.0

    def price_at(self, ts_ms: int) -> float:
        """Last traded price at or before ``ts_ms`` (0 if none)."""
        i = self._start_index(ts_ms + 1) - 1
        return self.trades[i].price if i >= 0 else 0.0

    def recent_sizes(self, now_ms: int, window_s: float) -> list[float]:
        start = now_ms - int(window_s * 1000)
        i = self._start_index(start)
        return [self.trades[k].notional for k in range(i, len(self.trades))]

    def is_stale(self, now_ms: int, max_age_ms: int) -> bool:
        return now_ms - self.last_local_ts_ms > max_age_ms

