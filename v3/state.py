"""Per-symbol V3 state shared by the dataset builder and the bot (train/serve parity).

Holds the full-depth ``V3Engine`` and an L1-only view of the same messages (bookTicker
book + trade flow), from which V2's features are computed exactly as in the V2 dataset.
Evaluations happen on a fixed grid (default 250 ms) of the replay clock; an evaluation at
time t sees only messages with timestamp < t.
"""
from __future__ import annotations

from collections import Counter
from typing import Callable

from config import BotConfig
from exchange.schemas import FeedValidator
from features.microstructure_features import compute_features
from features.v2_features import FeatureHistory, v2_features
from market_data.state import apply_detail, make_book
from market_data.tradeflow import TradeFlow
from v3.features import V3Engine

DIFF_KIND = "depth@100ms"


class SymbolState:
    def __init__(self, symbol: str, tick: float | None = None, eval_ms: int = 250, cfg: BotConfig | None = None,
                 want_row: Callable[[int], bool] | None = None, want_timeline: Callable[[int], bool] | None = None
                 ) -> None:
        self.symbol = symbol
        self.cfg = cfg or BotConfig()
        md = self.cfg.market_data
        self.engine = V3Engine(symbol, tick=tick)
        self.book1 = make_book(symbol, md.book_history_len, "bbo", md.bbo_history_interval_ms)
        self.flow = TradeFlow(symbol, md.trade_history_s)
        self.feed = FeedValidator()
        self.hist = FeatureHistory()
        self.eval_ms = eval_ms
        self.next_eval: int | None = None
        self.want_row = want_row or (lambda t: True)
        self.want_timeline = want_timeline or (lambda t: False)
        self.last_row: dict | None = None
        self.last_eval_ts = 0
        self.rows: list[dict] = []          # collected rows (dataset builder)
        self.timeline: list[dict] = []
        self.collect = False
        self.why = Counter()
        self.quotes: tuple[list, list, list] = ([], [], [])
        self.record_quotes = False

    # ------------------------------------------------------------------ input
    def on_message(self, stream: str, data, ts: int) -> None:
        self.advance(ts)
        head, _, kind = stream.partition("@")
        if head.startswith("__"):
            self.engine.on_message(head, data, ts)
            return
        if kind == DIFF_KIND:
            self.engine.on_message(kind, data, ts)
        elif kind == "aggTrade":
            self.engine.on_message(kind, data, ts)
            apply_detail(self.book1, self.flow, self.feed, "bbo", self.symbol, kind, stream, data, ts)
        elif kind == "bookTicker":
            res = apply_detail(self.book1, self.flow, self.feed, "bbo", self.symbol, kind, stream, data, ts)
            if res.kind == "book" and self.record_quotes:
                b, _, a, _ = self.book1.best()
                self.quotes[0].append(ts)
                self.quotes[1].append(b)
                self.quotes[2].append(a)

    def advance(self, ts: int) -> None:
        if self.next_eval is None:
            self.next_eval = (ts // self.eval_ms + 1) * self.eval_ms
        while self.next_eval <= ts:
            self._evaluate(self.next_eval)
            self.next_eval += self.eval_ms

    # ------------------------------------------------------------------ evaluation
    def _evaluate(self, now: int) -> None:
        self.last_eval_ts = now
        self.last_row = None
        md = self.cfg.market_data
        if self.book1.is_stale(now, md.stale_after_ms) or len(self.book1.history) < 20:
            self.why["l1_stale"] += 1
            return
        base = compute_features(self.book1, self.flow, self.cfg.features, now)
        if not base:
            self.why["l1_nofeat"] += 1
            return
        base["last_price"] = self.flow.last_price()
        if self.want_timeline(now):
            t = self.engine.timeline(now)
            if t is not None:
                self.timeline.append({"ts_ms": now, **t})
        if self.want_row(now):
            f3 = self.engine.features(now)
            if f3 is None:
                self.why["l2_invalid"] += 1
            else:
                bb, _, ba, _ = self.engine.book.best()
                f2 = v2_features(self.book1, self.flow, base, self.hist, now)
                self.last_row = {"ts_ms": now, "mid0": 0.5 * (bb + ba), **f2, **f3}
                if self.collect:
                    self.rows.append(self.last_row)
        self.hist.add(now, base)
