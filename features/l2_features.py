"""Depth-profile (L2) features. NaN on L1-only data, so models trained on L1 datasets
and later L2 recordings share one schema; adding L2 data adds information without a
pipeline rewrite.

Implemented from a single book snapshot (works with depth20@100ms recordings):
  l2_bid_cum_k / l2_ask_cum_k   log cumulative notional at levels 1, 5, 10, 20
  l2_wimb_k                     distance-weighted depth imbalance over k levels
  l2_slope_bid / l2_slope_ask   slope of log cumulative depth vs distance from mid (bps)
  l2_convex_bid / _ask          curvature: depth beyond level 5 relative to levels 1-5

Planned (need consecutive L2 snapshots + trades; see docs): cancellation rate, queue
depletion, replenishment velocity, sweep detection.
"""
from __future__ import annotations

import math

from market_data.orderbook import OrderBook

NAN = float("nan")
LEVELS = (1, 5, 10, 20)
NAMES = ([f"l2_bid_cum_{k}" for k in LEVELS] + [f"l2_ask_cum_{k}" for k in LEVELS]
         + [f"l2_wimb_{k}" for k in (5, 10, 20)]
         + ["l2_slope_bid", "l2_slope_ask", "l2_convex_bid", "l2_convex_ask"])


def l2_features(book: OrderBook) -> dict[str, float]:
    b, a = book.sorted_levels(20)
    if len(b) < 5 or len(a) < 5:
        return {k: NAN for k in NAMES}
    mid = 0.5 * (b[0][0] + a[0][0])
    f: dict[str, float] = {}
    for side, lv in (("bid", b), ("ask", a)):
        cum = 0.0
        cums = []
        for p, q in lv:
            cum += p * q
            cums.append(cum)
        for k in LEVELS:
            f[f"l2_{side}_cum_{k}"] = math.log1p(cums[min(k, len(cums)) - 1])
        dist = [abs(p - mid) / mid * 1e4 for p, _ in lv]
        ys = [math.log1p(c) for c in cums]
        n = len(ys)
        mx = sum(dist) / n
        my = sum(ys) / n
        den = sum((x - mx) ** 2 for x in dist)
        f[f"l2_slope_{side}"] = sum((x - mx) * (y - my) for x, y in zip(dist, ys)) / den if den > 0 else NAN
        f[f"l2_convex_{side}"] = math.log((cums[-1] - cums[4] + 1.0) / (cums[4] + 1.0))
    for k in (5, 10, 20):
        wb = sum(p * q / (1.0 + i) for i, (p, q) in enumerate(b[:k]))
        wa = sum(p * q / (1.0 + i) for i, (p, q) in enumerate(a[:k]))
        f[f"l2_wimb_{k}"] = (wb - wa) / (wb + wa) if wb + wa > 0 else 0.0
    return f
