"""Order-book derived features.

All functions are pure: they read an :class:`OrderBook` (and its ``history``) and
return floats. Signs follow the convention: positive == bullish pressure.

Single resting walls are deliberately NOT used as signals on their own (they can be
cancelled / spoofed). Book features are combined with executed-flow features and
persistence measures downstream.
"""
from __future__ import annotations

import math
from typing import Sequence

from market_data.orderbook import BookState, OrderBook
from utils.mathx import clip, imbalance, mean


def level_imbalance(book: OrderBook, levels: int) -> float:
    """Quote-notional imbalance over the top ``levels`` levels."""
    b, a = book.sorted_levels(levels)
    return imbalance(sum(p * q for p, q in b), sum(p * q for p, q in a))


def weighted_imbalance(book: OrderBook, levels: int, decay: float) -> float:
    """Exponentially decayed multi-level imbalance (near-touch levels dominate)."""
    b, a = book.sorted_levels(levels)
    wb = sum((decay ** i) * p * q for i, (p, q) in enumerate(b))
    wa = sum((decay ** i) * p * q for i, (p, q) in enumerate(a))
    return imbalance(wb, wa)


def microprice_features(book: OrderBook) -> tuple[float, float, float]:
    """Returns (microprice, offset_bps vs mid, tilt in [-1, 1] relative to half-spread)."""
    bid, _, ask, _ = book.best()
    if not bid or not ask:
        return 0.0, 0.0, 0.0
    mid = 0.5 * (bid + ask)
    mp = book.microprice()
    half = 0.5 * (ask - bid)
    tilt = clip((mp - mid) / half, -1.0, 1.0) if half > 0 else 0.0
    return mp, (mp - mid) / mid * 1e4, tilt


def _window(history: Sequence[BookState], now_ms: int, window_s: float) -> list[BookState]:
    start = now_ms - int(window_s * 1000)
    out: list[BookState] = []
    for st in reversed(history):
        if st.ts_ms < start:
            break
        out.append(st)
    out.reverse()
    return out


def order_flow_imbalance(history: Sequence[BookState], now_ms: int, window_s: float) -> float:
    """Cont–Kukanov–Stoikov OFI from consecutive best-quote states, normalized.

    e_n = 1[Pb_n >= Pb_{n-1}] qb_n - 1[Pb_n <= Pb_{n-1}] qb_{n-1}
        - 1[Pa_n <= Pa_{n-1}] qa_n + 1[Pa_n >= Pa_{n-1}] qa_{n-1}
    Normalized by mean top-of-book depth and squashed with tanh to [-1, 1].
    """
    states = _window(history, now_ms, window_s)
    if len(states) < 2:
        return 0.0
    ofi = 0.0
    for prev, cur in zip(states, states[1:]):
        e = 0.0
        if cur.bid >= prev.bid:
            e += cur.bid_qty
        if cur.bid <= prev.bid:
            e -= prev.bid_qty
        if cur.ask <= prev.ask:
            e -= cur.ask_qty
        if cur.ask >= prev.ask:
            e += prev.ask_qty
        ofi += e
    depth = mean([0.5 * (s.bid_qty + s.ask_qty) for s in states])
    if depth <= 0:
        return 0.0
    return math.tanh(ofi / (depth * 3.0))


def depletion(history: Sequence[BookState], now_ms: int, lookback_s: float) -> tuple[float, float, float]:
    """Relative depth depletion of each side over the lookback.

    Returns (bid_depletion, ask_depletion, asymmetry) where depletion > 0 means the side
    got thinner. asymmetry = ask_depl - bid_depl (positive: asks being eaten -> bullish).
    """
    if not history:
        return 0.0, 0.0, 0.0
    cur = history[-1]
    target = now_ms - int(lookback_s * 1000)
    then = None
    for st in reversed(history):
        if st.ts_ms <= target:
            then = st
            break
    if then is None:
        then = history[0]
    bid_d = (then.bid_depth5 - cur.bid_depth5) / then.bid_depth5 if then.bid_depth5 > 0 else 0.0
    ask_d = (then.ask_depth5 - cur.ask_depth5) / then.ask_depth5 if then.ask_depth5 > 0 else 0.0
    bid_d, ask_d = clip(bid_d, -1.0, 1.0), clip(ask_d, -1.0, 1.0)
    return bid_d, ask_d, clip(ask_d - bid_d, -1.0, 1.0)


def replenishment(history: Sequence[BookState], now_ms: int, window_s: float) -> tuple[float, float]:
    """How quickly each side's best level refills after being hit (price unchanged).

    Ratio of quantity added to quantity removed at an unchanged best price. ~1 means
    liquidity is resilient; << 1 means it is being consumed without refill.
    Returns (bid_replenish, ask_replenish) clipped to [0, 3].
    """
    states = _window(history, now_ms, window_s)
    add_b = rem_b = add_a = rem_a = 0.0
    for prev, cur in zip(states, states[1:]):
        if cur.bid == prev.bid:
            d = cur.bid_qty - prev.bid_qty
            if d > 0:
                add_b += d
            else:
                rem_b -= d
        if cur.ask == prev.ask:
            d = cur.ask_qty - prev.ask_qty
            if d > 0:
                add_a += d
            else:
                rem_a -= d
    rb = clip(add_b / rem_b, 0.0, 3.0) if rem_b > 0 else 1.0
    ra = clip(add_a / rem_a, 0.0, 3.0) if rem_a > 0 else 1.0
    return rb, ra


def pressure_persistence(history: Sequence[BookState], n: int) -> float:
    """2 * (fraction of last n states with bid-heavy top-5 book) - 1, in [-1, 1]."""
    if not history:
        return 0.0
    states = list(history)[-n:]
    pos = sum(1 for s in states if s.imbalance5 > 0)
    neg = sum(1 for s in states if s.imbalance5 < 0)
    tot = pos + neg
    return (pos - neg) / tot if tot else 0.0


def liquidity_change(history: Sequence[BookState], now_ms: int, window_s: float) -> float:
    """Current top-10 depth vs its window average, minus 1 (negative: liquidity drying up)."""
    states = _window(history, now_ms, window_s)
    if len(states) < 2:
        return 0.0
    avg = mean([s.bid_depth10 + s.ask_depth10 for s in states])
    cur = states[-1].bid_depth10 + states[-1].ask_depth10
    return clip(cur / avg - 1.0, -1.0, 5.0) if avg > 0 else 0.0


def mid_momentum_bps(history: Sequence[BookState], now_ms: int, window_s: float) -> float:
    if not history:
        return 0.0
    cur = history[-1].mid
    target = now_ms - int(window_s * 1000)
    then = None
    for st in reversed(history):
        if st.ts_ms <= target:
            then = st
            break
    if then is None:
        then = history[0]
    return (cur - then.mid) / then.mid * 1e4 if then.mid else 0.0


def realized_vol_bps_1s(history: Sequence[BookState], now_ms: int, window_s: float) -> float:
    """Per-second realized volatility of mid (bps), from ~1s-resampled mid path."""
    states = _window(history, now_ms, window_s)
    if len(states) < 3:
        return 0.0
    # Resample: last state in each 1-second bucket.
    buckets: dict[int, float] = {}
    for s in states:
        buckets[s.ts_ms // 1000] = s.mid
    keys = sorted(buckets)
    if len(keys) < 3:
        # Fall back to raw-state returns scaled by elapsed time.
        mids = [s.mid for s in states]
        span_s = max((states[-1].ts_ms - states[0].ts_ms) / 1000.0, 1e-3)
        ss = sum(math.log(b / a) ** 2 for a, b in zip(mids, mids[1:]) if a > 0 and b > 0)
        return math.sqrt(ss / span_s) * 1e4
    rets = []
    for k0, k1 in zip(keys, keys[1:]):
        p0, p1 = buckets[k0], buckets[k1]
        if p0 > 0 and p1 > 0:
            rets.append(math.log(p1 / p0) / math.sqrt(k1 - k0))
    if not rets:
        return 0.0
    return math.sqrt(sum(r * r for r in rets) / len(rets)) * 1e4
