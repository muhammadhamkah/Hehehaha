"""Statistical analyses on replay output (``replay.sqlite``): symbols, calibration,
feature predictive power, and minimum tradable edge.

All functions are pure (DataFrame in, dict/DataFrame out) so they can be applied to
any chronological segment. Statistical caution built in:
  * forward-return samples overlap in time, so t-statistics use a THINNED sample
    (one observation per symbol per horizon window)
  * many features x horizons are tested, so "significant" requires |t| >= 3 AND the
    same sign in both chronological halves
"""
from __future__ import annotations

import json
import math
import sqlite3
from typing import Any

import numpy as np
import pandas as pd


CONF_BUCKETS = [0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 1.0001]
CONF_LABELS = ["50-60%", "60-65%", "65-70%", "70-75%", "75-80%", ">80%"]

FEATURE_FAMILIES: dict[str, list[str]] = {
    "order-book imbalance": ["imb_l1", "imb_l5", "imb_weighted", "depth_imb_10bps"],
    "order-flow imbalance (OFI)": ["ofi_1s", "ofi_3s", "ofi_10s"],
    "aggressive volume ratio": ["aggr_buy_ratio_3s", "flow_imb_3s", "flow_imb_10s"],
    "microprice displacement": ["micro_tilt", "microprice_offset_bps"],
    "depth depletion": ["depletion_asym", "bid_depletion", "ask_depletion"],
    "replenishment": ["replenish_asym"],
    "trade velocity": ["trades_per_s_3s", "velocity_change"],
    "volume acceleration": ["vol_accel"],
    "short-term volatility": ["rv_1s_bps"],
    "spread": ["spread_bps"],
    "momentum": ["mom_1s_bps", "mom_3s_bps", "mom_10s_bps"],
}
# Signed (directional) features: positive == bullish. The rest are conditioning variables.
NON_DIRECTIONAL = {"trades_per_s_3s", "velocity_change", "vol_accel", "rv_1s_bps", "spread_bps",
                   "bid_depletion", "ask_depletion"}


# ---------------------------------------------------------------------- loading
def load(db_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    conn = sqlite3.connect(db_path)
    try:
        trades = pd.read_sql_query("SELECT * FROM trades ORDER BY entry_ts_ms", conn)
        signals = pd.read_sql_query("SELECT * FROM signals WHERE labeled >= 1 ORDER BY ts_ms", conn)
    finally:
        conn.close()
    return trades, signals


def expand_features(signals: pd.DataFrame) -> pd.DataFrame:
    if signals.empty:
        return pd.DataFrame()
    feats = pd.json_normalize(signals["features_json"].map(json.loads))
    feats.index = signals.index
    return feats


# ---------------------------------------------------------------------- helpers
def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - r) / d, (c + r) / d)


def thin(df: pd.DataFrame, horizon_s: int) -> pd.DataFrame:
    """Keep at most one sample per symbol per horizon window (non-overlapping outcomes)."""
    if df.empty:
        return df
    bucket = df["ts_ms"] // (horizon_s * 1000)
    return df.loc[~pd.DataFrame({"s": df["symbol"], "b": bucket}).duplicated()]


def rank_ic(x: pd.Series, y: pd.Series) -> tuple[float, int]:
    m = x.notna() & y.notna()
    n = int(m.sum())
    if n < 30 or x[m].nunique() < 3 or y[m].nunique() < 3:
        return float("nan"), n
    return float(x[m].rank().corr(y[m].rank())), n


def t_of_ic(ic: float, n: int) -> float:
    if not np.isfinite(ic) or n < 4 or abs(ic) >= 1:
        return float("nan")
    return ic * math.sqrt((n - 2) / (1 - ic * ic))


def cost_bps_series(signals: pd.DataFrame, maker_fee: float, taker_fee: float,
                    extra_bps: float = 0.8) -> pd.Series:
    """Round trip, maker entry + taker exit: fees + half spread + latency/buffer."""
    return (maker_fee + taker_fee) * 1e4 + signals["spread_bps"].fillna(0) / 2 + extra_bps


# ---------------------------------------------------------------------- symbols
def per_symbol(trades: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame(columns=["symbol", "trades"])
    t = trades.copy()
    t["entry_spread_bps"] = t["features_json"].map(lambda j: json.loads(j).get("spread_bps") if j else None)
    t["slip_bps"] = t["actual_slippage_usdt"] / t["notional"] * 1e4
    rows = []
    for sym, g in t.groupby("symbol"):
        wins = g.loc[g.net_pnl > 0, "net_pnl"].sum()
        losses = -g.loc[g.net_pnl <= 0, "net_pnl"].sum()
        rows.append({
            "symbol": sym, "trades": len(g),
            "net_pnl": round(g.net_pnl.sum(), 4),
            "expectancy": round(g.net_pnl.mean(), 5),
            "median_net": round(g.net_pnl.median(), 5),
            "win_rate": round((g.net_pnl > 0).mean(), 3),
            "profit_factor": round(wins / losses, 3) if losses > 0 else float("inf"),
            "avg_entry_spread_bps": round(g.entry_spread_bps.mean(), 3),
            "avg_slippage_bps": round(g.slip_bps.mean(), 3),
            "avg_fees": round((g.entry_fee + g.exit_fee).mean(), 4),
            "avg_hold_s": round(g.holding_s.mean(), 2),
            "maker_entry_pct": round(g.entry_maker.mean(), 3),
        })
    out = pd.DataFrame(rows)
    if not signals.empty:
        sp = signals.groupby("symbol")["spread_bps"].mean().round(3).rename("avg_market_spread_bps")
        out = out.merge(sp, left_on="symbol", right_index=True, how="left")
    return out.sort_values("expectancy", ascending=False)


# ---------------------------------------------------------------------- calibration
def calibration(signals: pd.DataFrame, horizon_s: int = 10) -> dict[str, Any]:
    """Predicted probabilities vs observed frequencies."""
    out: dict[str, Any] = {}
    s = signals[signals["direction"].fillna(0) != 0].copy()
    if s.empty:
        return {"note": "no directional signals"}
    s["confidence"] = s[["p_long", "p_short"]].max(axis=1)
    s["bucket"] = pd.cut(s["confidence"], CONF_BUCKETS, labels=CONF_LABELS, right=False)

    # (a) directional confidence vs realised direction at the horizon (zero moves excluded)
    col = f"ret_{horizon_s}s"
    d = thin(s.dropna(subset=[col]), horizon_s)
    d = d[d[col] != 0]
    rows = []
    for b, g in d.groupby("bucket", observed=True):
        k = int(((g[col] > 0) == (g["direction"] > 0)).sum())
        lo, hi = wilson(k, len(g))
        rows.append({"bucket": str(b), "n": len(g), "predicted": round(g.confidence.mean(), 3),
                     "realised": round(k / len(g), 3), "ci95": (round(lo, 3), round(hi, 3)),
                     "consistent": bool(lo <= g.confidence.mean() <= hi)})
    out["direction"] = rows

    # (b) P(target before stop) vs realised, using each opportunity's OWN target/stop
    p = s.dropna(subset=["p_target", "tp_plan"])
    p = thin(p, 60)
    p = p.assign(pb=pd.cut(p["p_target"], CONF_BUCKETS, labels=CONF_LABELS, right=False))
    rows = []
    for b, g in p.groupby("pb", observed=True):
        k = int((g["tp_plan"] == 1).sum())
        lo, hi = wilson(k, len(g))
        rows.append({"bucket": str(b), "n": len(g), "predicted": round(g.p_target.mean(), 3),
                     "realised": round(k / len(g), 3), "ci95": (round(lo, 3), round(hi, 3)),
                     "consistent": bool(lo <= g.p_target.mean() <= hi)})
    out["target_before_stop"] = rows
    if len(p):
        out["brier_target_before_stop"] = round(float(((p.p_target - (p.tp_plan == 1)) ** 2).mean()), 4)
    trusted = [r for r in rows if r["n"] >= 30]
    out["verdict"] = (
        "INSUFFICIENT DATA" if not trusted else
        "CALIBRATED" if all(r["consistent"] for r in trusted) else
        "OVERCONFIDENT" if np.mean([r["predicted"] - r["realised"] for r in trusted]) > 0 else
        "MISCALIBRATED"
    )
    return out


# ---------------------------------------------------------------------- features
def feature_analysis(signals: pd.DataFrame, maker_fee: float, taker_fee: float,
                     horizons: tuple[int, ...] = (5, 10, 30, 60)) -> dict[str, Any]:
    if signals.empty:
        return {"rows": [], "verdict": "NO DATA"}
    feats = expand_features(signals)
    s = signals.copy()
    s["cost_bps"] = cost_bps_series(s, maker_fee, taker_fee)
    half = s["ts_ms"].median()
    rows = []
    for family, cols in FEATURE_FAMILIES.items():
        for c in cols:
            if c not in feats:
                continue
            x = pd.to_numeric(feats[c], errors="coerce")
            directional = c not in NON_DIRECTIONAL
            for h in horizons:
                col = f"ret_{h}s"
                base = s.assign(x=x).dropna(subset=["x", col])
                th = thin(base, h)
                if directional:
                    ic, n = rank_ic(th["x"], th[col])
                    ic1, _ = rank_ic(th.loc[th.ts_ms <= half, "x"], th.loc[th.ts_ms <= half, col])
                    ic2, _ = rank_ic(th.loc[th.ts_ms > half, "x"], th.loc[th.ts_ms > half, col])
                    sign = np.sign(th["x"])
                    aligned_net = (sign * th[col] - th["cost_bps"])[sign != 0]
                    q = _quintiles(th, "x", col, signed=True)
                    tp = np.where(th["x"] > 0, th["tp_long"], np.where(th["x"] < 0, th["tp_short"], np.nan))
                    tp_rate = float(np.nanmean(tp == 1)) if len(th) else float("nan")
                else:
                    ic, n = rank_ic(th["x"], th[col].abs())
                    ic1, _ = rank_ic(th.loc[th.ts_ms <= half, "x"], th.loc[th.ts_ms <= half, col].abs())
                    ic2, _ = rank_ic(th.loc[th.ts_ms > half, "x"], th.loc[th.ts_ms > half, col].abs())
                    d = th["direction"].replace(0, np.nan)
                    aligned_net = (d * th[col] - th["cost_bps"]).dropna()
                    q = _quintiles(th, "x", col, signed=False)
                    tp_rate = float((th["tp_pred"] == 1).mean()) if len(th) else float("nan")
                t = t_of_ic(ic, n)
                stable = bool(np.isfinite(ic1) and np.isfinite(ic2) and np.sign(ic1) == np.sign(ic2) != 0)
                rows.append({
                    "family": family, "feature": c, "horizon_s": h, "directional": directional,
                    "n_thinned": n, "ic": _r(ic, 4), "t": _r(t, 2), "ic_first_half": _r(ic1, 4),
                    "ic_second_half": _r(ic2, 4), "stable_sign": stable,
                    "significant": bool(np.isfinite(t) and abs(t) >= 3 and stable),
                    "aligned_net_bps_mean": _r(float(aligned_net.mean()) if len(aligned_net) else float("nan"), 3),
                    "top_quintile_net_bps": q.get("top_net"),
                    "aligned_tp_rate": _r(tp_rate, 3),
                    "quintile_mean_ret_bps": q.get("means"),
                })
    sig = [r for r in rows if r["significant"]]
    tradable = [r for r in sig if r["directional"] and (r["top_quintile_net_bps"] or -1) > 0]
    verdict = ("USABLE PREDICTIVE STRUCTURE (directional feature significant, stable, and net-positive "
               "in its top quintile)" if tradable else
               "STATISTICAL STRUCTURE BUT NOT NET-PROFITABLE AFTER COSTS" if sig else
               "NO STATISTICALLY MEANINGFUL PREDICTIVE STRUCTURE")
    return {"rows": rows, "significant": sig, "tradable": tradable, "verdict": verdict,
            "note": "t uses thinned (non-overlapping) samples; significant = |t|>=3 and same sign in both halves"}


def _quintiles(df: pd.DataFrame, xcol: str, ycol: str, signed: bool) -> dict[str, Any]:
    if len(df) < 50 or df[xcol].nunique() < 5:
        return {}
    try:
        qs = pd.qcut(df[xcol], 5, labels=False, duplicates="drop")
    except ValueError:
        return {}
    y = df[ycol] if signed else df[ycol].abs()
    means = [round(float(v), 3) for v in y.groupby(qs).mean().values]
    out: dict[str, Any] = {"means": means}
    if signed:
        top = df[qs == qs.max()]
        bot = df[qs == qs.min()]
        # trade long the top quintile and short the bottom quintile
        net = pd.concat([top[ycol] - top["cost_bps"], -bot[ycol] - bot["cost_bps"]])
        out["top_net"] = round(float(net.mean()), 3) if len(net) else None
    return out


def _r(x: float, nd: int) -> float | None:
    return round(float(x), nd) if x is not None and np.isfinite(x) else None


# ---------------------------------------------------------------------- min edge
class MinEdgeObserver:
    """Replay observer: samples, from the LIVE book state, the mid move required for
    min_net_profit at several notionals using the bot's own cost model."""

    def __init__(self, notionals=(50, 100, 150, 200, 250), min_net: float | None = None) -> None:
        self.notionals = tuple(notionals)
        self.min_net = min_net
        self.rows: list[tuple] = []

    def __call__(self, bot, now_ms: int) -> None:
        min_net = self.min_net if self.min_net is not None else bot.cfg.entry.min_net_profit_usdt
        for sym in bot.scanner.selected:
            book = bot.books.get(sym)
            if book is None or book.is_stale(now_ms, bot.cfg.market_data.stale_after_ms):
                continue
            spread = book.spread_bps
            for n in self.notionals:
                for maker in (True, False):
                    c = bot.costs.estimate(book, 1, n, entry_maker=maker)
                    req = bot.costs.required_move_bps(c, min_net) if np.isfinite(c.total_usdt) else float("inf")
                    self.rows.append((now_ms, sym, n, maker, spread, c.exit_slippage_bps, c.fees, req))

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows, columns=["ts_ms", "symbol", "notional", "maker_entry", "spread_bps",
                                                "exit_slip_bps", "fees", "required_bps"])


def min_edge_summary(df: pd.DataFrame, signals: pd.DataFrame | None = None) -> dict[str, Any]:
    if df.empty:
        return {"note": "no samples"}
    out: dict[str, Any] = {"by_notional": [], "by_symbol": []}
    for (n, maker), g in df.groupby(["notional", "maker_entry"]):
        r = g["required_bps"].replace(np.inf, np.nan)
        row = {"notional": n, "entry": "maker" if maker else "taker", "samples": len(g),
               "required_bps_p25": _r(r.quantile(0.25), 2), "required_bps_median": _r(r.median(), 2),
               "required_bps_p75": _r(r.quantile(0.75), 2), "required_bps_p90": _r(r.quantile(0.9), 2),
               "unfillable_share": round(float(g["required_bps"].isin([np.inf]).mean()), 4),
               "avg_spread_bps": _r(g.spread_bps.mean(), 3), "avg_exit_slip_bps": _r(g.exit_slip_bps.mean(), 3)}
        if signals is not None and not signals.empty and "ret_60s" in signals:
            moves = signals["ret_60s"].abs().dropna()
            med = r.median()
            if len(moves) and np.isfinite(med):
                row["share_of_60s_moves_exceeding_median_required"] = round(float((moves >= med).mean()), 4)
        out["by_notional"].append(row)
    g = df[df.maker_entry].groupby(["symbol", "notional"])["required_bps"].median().unstack()
    out["by_symbol_maker_median_required_bps"] = {s: {int(k): _r(v, 2) for k, v in r.items()} for s, r in g.iterrows()}
    return out


def main() -> None:
    import argparse

    from config import load_config

    ap = argparse.ArgumentParser(description="Symbol / calibration / feature analysis of a replay database")
    ap.add_argument("--db", required=True, help="replay.sqlite produced by backtest.replay")
    ap.add_argument("--config")
    ap.add_argument("--out", help="write JSON here")
    args = ap.parse_args()
    cfg = load_config(args.config)
    trades, signals = load(args.db)
    res = {
        "per_symbol": per_symbol(trades, signals).to_dict("records"),
        "calibration": calibration(signals),
        "features": feature_analysis(signals, cfg.costs.maker_fee, cfg.costs.taker_fee),
    }
    pd.set_option("display.width", 220)
    print("PER SYMBOL\n", per_symbol(trades, signals).to_string(index=False))
    print("\nCALIBRATION:", res["calibration"].get("verdict"))
    for k in ("direction", "target_before_stop"):
        print(pd.DataFrame(res["calibration"].get(k, [])).to_string(index=False))
    fa = res["features"]
    print("\nFEATURES:", fa["verdict"])
    rows = pd.DataFrame(fa["rows"])
    if not rows.empty:
        rows = rows.dropna(subset=["t"]).reindex(rows["t"].abs().sort_values(ascending=False).index)
        print(rows[["feature", "horizon_s", "n_thinned", "ic", "t", "stable_sign", "significant",
                    "top_quintile_net_bps", "aligned_tp_rate"]].head(25).to_string(index=False))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=1, default=str)


if __name__ == "__main__":
    main()
