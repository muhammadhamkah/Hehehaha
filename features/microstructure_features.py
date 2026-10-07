"""Assemble a flat feature vector (dict[str, float]) for one symbol at one instant.

The flat dict is what the predictor consumes, what the recorder stores (as JSON), and
what offline research / ML training reads back. Keep names stable.
"""
from __future__ import annotations

from config import FeatureConfig
from features import orderbook_features as obf
from features import trade_features as tf
from market_data.orderbook import OrderBook
from market_data.tradeflow import TradeFlow
from utils.mathx import imbalance


def compute_features(book: OrderBook, flow: TradeFlow, cfg: FeatureConfig, now_ms: int) -> dict[str, float]:
    bid, bid_qty, ask, ask_qty = book.best()
    if not bid or not ask:
        return {}
    mid = 0.5 * (bid + ask)
    f: dict[str, float] = {
        "bid": bid,
        "ask": ask,
        "bid_qty": bid_qty,
        "ask_qty": ask_qty,
        "mid": mid,
        "spread_bps": (ask - bid) / mid * 1e4,
        "book_age_ms": float(now_ms - book.last_local_ts_ms),
        "flow_age_ms": float(now_ms - flow.last_local_ts_ms) if flow.last_local_ts_ms else 1e9,
    }

    # --- book shape
    for lv in cfg.imbalance_levels:
        f[f"imb_l{lv}"] = obf.level_imbalance(book, lv)
    f["imb_weighted"] = obf.weighted_imbalance(book, max(cfg.imbalance_levels), cfg.imbalance_decay)
    for band in cfg.depth_bands_bps:
        bd, ad = book.depth_within_bps(band)
        tag = f"{int(band)}bps"
        f[f"bid_depth_{tag}"] = bd
        f[f"ask_depth_{tag}"] = ad
        f[f"depth_imb_{tag}"] = imbalance(bd, ad)
    mp, mp_off, tilt = obf.microprice_features(book)
    f["microprice"] = mp
    f["microprice_offset_bps"] = mp_off
    f["micro_tilt"] = tilt

    # --- book dynamics
    h = list(book.history)   # one snapshot; the helpers binary-search it
    f["ofi_1s"] = obf.order_flow_imbalance(h, now_ms, 1.0)
    f["ofi_3s"] = obf.order_flow_imbalance(h, now_ms, 3.0)
    f["ofi_10s"] = obf.order_flow_imbalance(h, now_ms, 10.0)
    bd_, ad_, asym = obf.depletion(h, now_ms, cfg.depletion_lookback_s)
    f["bid_depletion"] = bd_
    f["ask_depletion"] = ad_
    f["depletion_asym"] = asym
    rb, ra = obf.replenishment(h, now_ms, 5.0)
    f["bid_replenish"] = rb
    f["ask_replenish"] = ra
    f["replenish_asym"] = max(-1.0, min(1.0, (rb - ra) / 2.0))
    f["persistence"] = obf.pressure_persistence(h, cfg.persistence_window)
    f["liquidity_change"] = obf.liquidity_change(h, now_ms, 10.0)
    for w in cfg.momentum_windows_s:
        f[f"mom_{int(w)}s_bps"] = obf.mid_momentum_bps(h, now_ms, w)
    f["rv_1s_bps"] = obf.realized_vol_bps_1s(h, now_ms, cfg.vol_window_s)

    # --- executed flow
    for w in cfg.flow_windows_s:
        f.update(tf.flow_features(flow, now_ms, w))
    f["vol_accel"] = tf.volume_acceleration(flow, now_ms, 3.0, 30.0)
    f["velocity_change"] = tf.trade_velocity_change(flow, now_ms, 5.0)
    share, big_imb = tf.large_trade_imbalance(flow, now_ms, 10.0, cfg.large_trade_quantile_mult)
    f["large_trade_share"] = share
    f["large_trade_imb"] = big_imb

    # --- price response to flow: did aggressive buying actually lift price?
    f["flow_price_agree_3s"] = _agree(f.get("flow_imb_3s", 0.0), f.get("mom_3s_bps", 0.0))
    return f


def _agree(flow_imb: float, mom_bps: float) -> float:
    if flow_imb == 0 or mom_bps == 0:
        return 0.0
    return 1.0 if (flow_imb > 0) == (mom_bps > 0) else -1.0
