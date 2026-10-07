"""Executed-trade (aggressor flow) features. Positive == aggressive buying."""
from __future__ import annotations

import numpy as np

from market_data.tradeflow import TradeFlow
from utils.mathx import clip, imbalance, safe_div


def flow_features(flow: TradeFlow, now_ms: int, window_s: float) -> dict[str, float]:
    w = flow.window(now_ms, window_s)
    tag = _tag(window_s)
    return {
        f"buy_notional_{tag}": w.buy_notional,
        f"sell_notional_{tag}": w.sell_notional,
        f"flow_imb_{tag}": imbalance(w.buy_notional, w.sell_notional),
        f"aggr_buy_ratio_{tag}": safe_div(w.buy_notional, w.total_notional, 0.5),
        f"trades_per_s_{tag}": w.count / window_s,
        f"notional_per_s_{tag}": w.total_notional / window_s,
        f"avg_trade_{tag}": safe_div(w.total_notional, w.count),
        f"trade_ret_bps_{tag}": safe_div(w.last_price - w.first_price, w.first_price) * 1e4,
    }


def volume_acceleration(flow: TradeFlow, now_ms: int, short_s: float, long_s: float) -> float:
    """Recent notional rate divided by the longer-window rate (1.0 == steady)."""
    s = flow.window(now_ms, short_s).total_notional / short_s
    lg = flow.window(now_ms, long_s).total_notional / long_s
    if lg <= 0:
        return 1.0 if s <= 0 else 3.0
    return clip(s / lg, 0.0, 10.0)


def trade_velocity_change(flow: TradeFlow, now_ms: int, window_s: float) -> float:
    """Trades/s in the latest window relative to the window before it."""
    cur = flow.window(now_ms, window_s).count
    prev = flow.window(now_ms, window_s, end_offset_s=window_s).count
    if prev == 0:
        return 1.0 if cur == 0 else 3.0
    return clip(cur / prev, 0.0, 10.0)


def large_trade_imbalance(flow: TradeFlow, now_ms: int, window_s: float, mult: float, ref_window_s: float = 60.0) -> tuple[float, float]:
    """(share of notional from large trades, imbalance of large-trade aggressors)."""
    sizes = flow.recent_sizes(now_ms, ref_window_s)
    if len(sizes) < 10:
        return 0.0, 0.0
    thresh = float(np.median(np.frombuffer(sizes, dtype=np.float64))) * mult
    notionals, sells = flow.sides_and_sizes(now_ms, window_s)
    big_buy = big_sell = 0.0
    tot = sum(notionals)
    for n, sell in zip(notionals, sells):
        if n >= thresh:
            if sell:
                big_sell += n
            else:
                big_buy += n
    return safe_div(big_buy + big_sell, tot), imbalance(big_buy, big_sell)


def _tag(window_s: float) -> str:
    return f"{int(window_s)}s" if float(window_s).is_integer() else f"{int(window_s * 1000)}ms"
