"""Single implementation of "apply one market message to book/flow state".

Used by the live/replay bot AND the V2 dataset builder, so model features are computed
from exactly the same state in training and in serving.
"""
from __future__ import annotations

from dataclasses import dataclass

from exchange.schemas import FeedValidator
from market_data.orderbook import OrderBook, SyncStatus
from market_data.tradeflow import Trade, TradeFlow


@dataclass(slots=True)
class Applied:
    kind: str                  # "trade" | "book" | "resync" | "rejected" | "unexpected"
    trade: Trade | None = None


def make_book(symbol: str, history_len: int, depth_mode: str, bbo_history_interval_ms: int) -> OrderBook:
    return OrderBook(symbol, history_len,
                     history_interval_ms=bbo_history_interval_ms if depth_mode == "bbo" else 0)


def apply_detail(book: OrderBook, flow: TradeFlow, feed: FeedValidator, depth_mode: str,
                 symbol: str, kind: str, stream: str, data, ts: int) -> Applied:
    if kind == "aggTrade":
        pt = feed.agg_trade(stream, data, ts, symbol)
        if pt is None:
            return Applied("rejected")
        t = Trade(pt.trade_ts, pt.price, pt.qty, pt.is_buyer_maker)
        flow.add(t, ts, pt.agg_id)
        return Applied("trade", t)
    if kind.startswith("depth"):
        pd_ = feed.depth(stream, data, ts, symbol, partial=(depth_mode == "partial"))
        if pd_ is None:
            return Applied("rejected")
        if depth_mode == "partial":
            book.apply_snapshot(pd_.bids, pd_.asks, pd_.final_id, pd_.event_ts, ts)
        elif depth_mode == "diff":
            status = book.apply_diff(data, ts)
            if status in (SyncStatus.RESYNC, SyncStatus.BUFFERING):
                if status == SyncStatus.RESYNC:
                    book.last_update_id = 0
                    book.buffer_diff(data)
                return Applied("resync")
        else:
            feed.report_unexpected(stream, "depth message in bbo mode")
            return Applied("unexpected")
        return Applied("book")
    if kind == "bookTicker":
        bt = feed.book_ticker(stream, data, ts, symbol)
        if bt is None:
            return Applied("rejected")
        if depth_mode == "bbo":
            # L1-only data (e.g. Binance public archive): top of book IS the book.
            book.apply_snapshot([(bt.bid, bt.bid_qty)], [(bt.ask, bt.ask_qty)], bt.update_id, bt.event_ts, ts)
        else:
            book.update_bbo(bt.bid, bt.bid_qty, bt.ask, bt.ask_qty, ts)
        return Applied("book")
    feed.report_unexpected(stream, data)
    return Applied("unexpected")
