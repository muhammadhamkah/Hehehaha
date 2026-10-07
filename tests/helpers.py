"""Shared builders for tests."""
from __future__ import annotations

from market_data.orderbook import OrderBook
from market_data.tradeflow import Trade, TradeFlow


def make_book(symbol: str = "TESTUSDT", bid: float = 100.0, tick: float = 0.01, levels: int = 20,
              bid_qty: float = 50.0, ask_qty: float = 50.0, ts: int = 1_000_000, spread_ticks: int = 1,
              history: OrderBook | None = None) -> OrderBook:
    book = history or OrderBook(symbol)
    ask = bid + spread_ticks * tick
    bids = [(round(bid - i * tick, 8), bid_qty) for i in range(levels)]
    asks = [(round(ask + i * tick, 8), ask_qty) for i in range(levels)]
    book.apply_snapshot(bids, asks, last_update_id=1, event_ts_ms=ts, local_ts_ms=ts)
    return book


def feed_trades(flow: TradeFlow, start_ms: int, n: int, price: float, qty: float, buy_ratio: float,
                step_ms: int = 100, price_step: float = 0.0) -> int:
    ts = start_ms
    buys = int(round(n * buy_ratio))
    for i in range(n):
        is_buy = i < buys if buy_ratio >= 0.5 else i >= n - buys
        flow.add(Trade(ts, price + i * price_step, qty, is_buyer_maker=not is_buy), ts)
        ts += step_ms
    return ts
