"""V2 feature vector: current microstructure state + short feature history.

Shared by the dataset builder (training) and the bot (serving) -> identical features.
Every value is computed from state at or before ``now``; nothing looks ahead.

Groups
  base      V1's features, made scale-free across symbols (absolute prices dropped,
            notional/size features log-transformed)
  book_lag  top-of-book changes vs the state L ms ago (from conflated book history)
  flow      aggressive-flow imbalance / intensity over short windows, plus
            acceleration (window vs the preceding window)
  roll      rolling mean / min / max / slope / persistence / change of key features
            over 1 s, 3 s, 10 s, from the per-symbol FeatureHistory (eval cadence)
  l2        depth-profile features (NaN on L1-only data) -- see features/l2_features.py
"""
from __future__ import annotations

import math
from bisect import bisect_right
from collections import deque

from features.l2_features import l2_features
from market_data.orderbook import BookState, OrderBook
from market_data.tradeflow import TradeFlow

NAN = float("nan")
LOOKBACKS_MS = (100, 250, 500, 1000, 2000, 3000, 5000, 10000)
ROLL_WINDOWS_MS = (1000, 3000, 10000)
# V1 features whose recent trajectory is tracked.
ROLL_KEYS = ("imb_l1", "imb_weighted", "micro_tilt", "ofi_1s", "ofi_3s", "flow_imb_1s", "flow_imb_3s",
             "depletion_asym", "replenish_asym", "persistence", "liquidity_change", "rv_1s_bps",
             "vol_accel", "large_trade_imb", "spread_bps", "mom_1s_bps")
# Absolute price levels / raw sizes: non-stationary and symbol-scale dependent.
DROP_BASE = {"bid", "ask", "mid", "microprice", "last_price", "bid_qty", "ask_qty"}
LOG_PREFIXES = ("bid_depth_", "ask_depth_", "buy_notional_", "sell_notional_", "notional_per_s_", "avg_trade_")


class FeatureHistory:
    """Per-symbol history of selected base features at the evaluation cadence."""

    def __init__(self, maxlen: int = 80) -> None:
        self.ts: deque[int] = deque(maxlen=maxlen)
        self.rows: deque[dict[str, float]] = deque(maxlen=maxlen)

    def add(self, ts: int, base: dict[str, float]) -> None:
        self.ts.append(ts)
        self.rows.append({k: base.get(k, NAN) for k in ROLL_KEYS})

    def window(self, now: int, window_ms: int) -> list[dict[str, float]]:
        ts = list(self.ts)
        i = bisect_right(ts, now - window_ms)
        j = bisect_right(ts, now)
        return list(self.rows)[i:j]

    def at(self, ts_ms: int) -> dict[str, float] | None:
        ts = list(self.ts)
        i = bisect_right(ts, ts_ms) - 1
        return self.rows[i] if i >= 0 else None


def _log1p(x: float) -> float:
    return math.log1p(x) if x > 0 else 0.0


def _state_at(history: list[BookState], ts_ms: int) -> BookState | None:
    i = bisect_right(history, ts_ms, key=lambda s: s.ts_ms) - 1
    return history[i] if i >= 0 else None


def _imb(a: float, b: float) -> float:
    s = a + b
    return (a - b) / s if s > 0 else 0.0


def _slope(xs: list[float], dt_s: float) -> float:
    """OLS slope per second of equally spaced samples."""
    n = len(xs)
    if n < 3:
        return NAN
    mx = (n - 1) / 2.0
    my = sum(xs) / n
    num = sum((i - mx) * (x - my) for i, x in enumerate(xs))
    den = sum((i - mx) ** 2 for i in range(n))
    step = dt_s / max(n - 1, 1)
    return num / den / step if den > 0 and step > 0 else NAN


def v2_features(book: OrderBook, flow: TradeFlow, base: dict[str, float], hist: FeatureHistory,
                now: int) -> dict[str, float]:
    if not base:
        return {}
    f: dict[str, float] = {}
    # ---- base (scale-free)
    for k, v in base.items():
        if k in DROP_BASE:
            continue
        f[k] = _log1p(v) if k.startswith(LOG_PREFIXES) else v
    bid, ask, mid = base["bid"], base["ask"], base["mid"]
    bq, aq = base["bid_qty"], base["ask_qty"]
    f["l1_bid_notional_log"] = _log1p(bid * bq)
    f["l1_ask_notional_log"] = _log1p(ask * aq)

    # ---- book lags (state L ms ago from the conflated book history)
    hist_states = list(book.history)
    for lag in LOOKBACKS_MS:
        st = _state_at(hist_states, now - lag)
        tag = f"{lag}ms"
        if st is None or st.mid <= 0:
            for name in ("mid_chg", "imb1_chg", "spread_chg", "bidq_chg", "askq_chg"):
                f[f"{name}_{tag}"] = NAN
            continue
        f[f"mid_chg_{tag}"] = (mid - st.mid) / st.mid * 1e4
        f[f"imb1_chg_{tag}"] = base["imb_l1"] - _imb(st.bid * st.bid_qty, st.ask * st.ask_qty)
        f[f"spread_chg_{tag}"] = base["spread_bps"] - (st.ask - st.bid) / st.mid * 1e4
        f[f"bidq_chg_{tag}"] = math.log((bid * bq + 1.0) / (st.bid * st.bid_qty + 1.0))
        f[f"askq_chg_{tag}"] = math.log((ask * aq + 1.0) / (st.ask * st.ask_qty + 1.0))

    # ---- short-window flow + acceleration
    for lag in LOOKBACKS_MS:
        w_s = lag / 1000.0
        cur = flow.window(now, w_s)
        prev = flow.window(now, w_s, end_offset_s=w_s)
        tag = f"{lag}ms"
        f[f"fimb_{tag}"] = _imb(cur.buy_notional, cur.sell_notional)
        f[f"ntr_rate_{tag}"] = cur.count / w_s
        f[f"vol_rate_log_{tag}"] = _log1p(cur.total_notional / w_s)
        f[f"cnt_accel_{tag}"] = math.log((cur.count + 1.0) / (prev.count + 1.0))
        f[f"vol_accel_{tag}"] = math.log((cur.total_notional + 1.0) / (prev.total_notional + 1.0))
        f[f"tret_{tag}"] = ((cur.last_price - cur.first_price) / cur.first_price * 1e4
                            if cur.first_price else 0.0)

    # ---- rolling transforms of key features (history at eval cadence + current)
    for w in ROLL_WINDOWS_MS:
        rows = hist.window(now, w)
        past = hist.at(now - w)
        for k in ROLL_KEYS:
            cur_v = base.get(k, NAN)
            vals = [r[k] for r in rows if r[k] == r[k]] + ([cur_v] if cur_v == cur_v else [])
            tag = f"{k}_{w}ms"
            if not vals:
                for s in ("mean", "max", "min", "slope", "pers", "chg"):
                    f[f"r{s}_{tag}"] = NAN
                continue
            f[f"rmean_{tag}"] = sum(vals) / len(vals)
            f[f"rmax_{tag}"] = max(vals)
            f[f"rmin_{tag}"] = min(vals)
            f[f"rslope_{tag}"] = _slope(vals, w / 1000.0)
            sgn = 1 if cur_v > 0 else -1 if cur_v < 0 else 0
            f[f"rpers_{tag}"] = (sum(1 for v in vals if (v > 0) - (v < 0) == sgn) / len(vals)) if sgn else 0.0
            f[f"rchg_{tag}"] = cur_v - past[k] if past is not None and past[k] == past[k] else NAN

    f.update(l2_features(book))
    return f
