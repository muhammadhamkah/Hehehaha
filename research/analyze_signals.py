"""Phase 5: does a statistically meaningful edge exist AFTER costs?

Reads labelled signals from the recorder database and reports, for each bucket of
signal strength, the direction-adjusted forward returns at every horizon, the realised
target-before-stop rate, and the expected NET result after a conservative cost model.
Results are shown separately for an in-sample (earlier) and out-of-sample (later)
period — only trust buckets that hold up out of sample.

    python -m research.analyze_signals --db data/microstructure.sqlite
"""
from __future__ import annotations

import argparse
import json
import math
import sqlite3

import numpy as np
import pandas as pd

from config import HORIZONS_S, load_config


def load_signals(db_path: str, include_partial: bool = False) -> pd.DataFrame:
    """labeled=1: full 60s window observed; labeled=2: partial (feed stopped / shutdown)."""
    where = "labeled >= 1" if include_partial else "labeled = 1"
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query(f"SELECT * FROM signals WHERE {where} ORDER BY ts_ms", conn)
    finally:
        conn.close()
    return df


def add_cost_columns(df: pd.DataFrame, cfg) -> pd.DataFrame:
    c = cfg.costs
    fee_bps = (c.maker_fee + c.taker_fee) * 1e4 if cfg.execution.entry_mode == "maker_first" else 2 * c.taker_fee * 1e4
    entry_bps = c.maker_adverse_selection_bps if cfg.execution.entry_mode == "maker_first" else 0.0
    # Exit as taker: half the spread plus latency + buffer (entry taker would add the same again).
    df["cost_bps"] = fee_bps + entry_bps + df["spread_bps"].fillna(0) / 2 + c.latency_slippage_bps + c.extra_slippage_bps
    if cfg.execution.entry_mode != "maker_first":
        df["cost_bps"] += df["spread_bps"].fillna(0) / 2 + c.latency_slippage_bps + c.extra_slippage_bps
    return df


def add_directional(df: pd.DataFrame) -> pd.DataFrame:
    d = df["direction"].replace(0, np.nan)
    for h in HORIZONS_S:
        df[f"dret_{h}s"] = df[f"ret_{h}s"] * d
    df["abs_score"] = df["score"].abs()
    df["confidence"] = df[["p_long", "p_short"]].max(axis=1)
    return df


def bucket_table(df: pd.DataFrame, by: str, notional: float) -> pd.DataFrame:
    rows = []
    for key, g in df.groupby(by, observed=True):
        row = {by: key, "n": len(g)}
        for h in HORIZONS_S:
            x = g[f"dret_{h}s"].dropna()
            row[f"ret{h}s"] = round(x.mean(), 3) if len(x) else np.nan
        x = g["dret_10s"].dropna()
        row["hit10s"] = round((x > 0).mean(), 3) if len(x) else np.nan
        tp = g["tp_pred"].dropna()
        row["tp_rate"] = round((tp == 1).mean(), 3) if len(tp) else np.nan
        row["sl_rate"] = round((tp == -1).mean(), 3) if len(tp) else np.nan
        # Barrier-label PnL: +target on TP, -stop on SL, 60s return on timeout, minus costs.
        tgt, stp = g["label_target_bps"], g["label_stop_bps"]
        barrier = np.where(g["tp_pred"] == 1, tgt, np.where(g["tp_pred"] == -1, -stp, g["dret_60s"].fillna(0)))
        net = barrier - g["cost_bps"].values
        row["net_bps"] = round(float(np.nanmean(net)), 3) if len(net) else np.nan
        row["net_usdt"] = round(float(np.nanmean(net)) * notional / 1e4, 4) if len(net) else np.nan
        sd = float(np.nanstd(net, ddof=1)) if len(net) > 1 else 0.0
        row["t"] = round(float(np.nanmean(net)) / (sd / math.sqrt(len(net))), 2) if sd > 1e-6 and len(net) >= 10 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def feature_ic(df: pd.DataFrame, horizon: int = 10, top: int = 25) -> pd.DataFrame:
    feats = pd.json_normalize(df["features_json"].map(json.loads))
    target = df[f"ret_{horizon}s"].reset_index(drop=True)
    out = []
    for col in feats.columns:
        x = pd.to_numeric(feats[col], errors="coerce")
        mask = x.notna() & target.notna()
        if mask.sum() < 50 or x[mask].nunique() < 5:
            continue
        ic = x[mask].rank().corr(target[mask].rank())
        out.append({"feature": col, f"spearman_ic_{horizon}s": round(ic, 4), "n": int(mask.sum())})
    res = pd.DataFrame(out)
    if res.empty:
        return res
    return res.reindex(res[f"spearman_ic_{horizon}s"].abs().sort_values(ascending=False).index).head(top)


def calibration(df: pd.DataFrame) -> pd.DataFrame:
    g = df.dropna(subset=["p_target", "tp_pred"])
    if g.empty:
        return pd.DataFrame()
    g = g.assign(bucket=pd.cut(g["p_target"], [0, 0.4, 0.5, 0.55, 0.6, 0.7, 0.8, 1.0]))
    return g.groupby("bucket", observed=True).agg(
        n=("tp_pred", "size"), predicted=("p_target", "mean"), realised=("tp_pred", lambda s: (s == 1).mean())
    ).round(3).reset_index()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/microstructure.sqlite")
    ap.add_argument("--config", default=None)
    ap.add_argument("--split", type=float, default=0.7, help="in-sample fraction (time ordered)")
    ap.add_argument("--include-partial", action="store_true",
                    help="also use signals whose 60s label window was cut short (biased)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    notional = cfg.sizing.position_notional_usdt

    df = load_signals(args.db, args.include_partial)
    if df.empty:
        print("No labelled signals yet. Run the bot in record/paper mode first.")
        return
    df = add_directional(add_cost_columns(df, cfg))
    df["score_bucket"] = pd.cut(df["abs_score"], [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1.0])
    df["conf_bucket"] = pd.cut(df["confidence"], [0.5, 0.55, 0.6, 0.65, 0.7, 0.8, 1.0])
    cut = int(len(df) * args.split)
    ins, oos = df.iloc[:cut], df.iloc[cut:]
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)

    print(f"signals: {len(df)}  (in-sample {len(ins)}, out-of-sample {len(oos)})")
    print(f"symbols: {df['symbol'].nunique()}  span: {(df.ts_ms.max() - df.ts_ms.min()) / 3.6e6:.2f} h")
    print(f"mean round-trip cost estimate: {df['cost_bps'].mean():.2f} bps "
          f"({df['cost_bps'].mean() * notional / 1e4:.4f} USDT at {notional:.0f} notional)\n")
    for name, part in (("IN-SAMPLE", ins), ("OUT-OF-SAMPLE", oos)):
        if part.empty:
            continue
        print(f"=== {name}: by |score| ===")
        print(bucket_table(part, "score_bucket", notional).to_string(index=False))
        print(f"\n=== {name}: by confidence ===")
        print(bucket_table(part, "conf_bucket", notional).to_string(index=False))
        print()
    print("=== by decision (all) ===")
    print(bucket_table(df, "decision", notional).to_string(index=False))
    print("\n=== by rejection reason (all) ===")
    print(bucket_table(df[df["rejection_reason"] != ""], "rejection_reason", notional).to_string(index=False))
    print("\n=== by sample type ===")
    print(bucket_table(df, "sample_type", notional).to_string(index=False))
    print("\n=== p_target calibration ===")
    print(calibration(df).to_string(index=False))
    print("\n=== feature information coefficients (vs 10s mid return) ===")
    print(feature_ic(df, 10).to_string(index=False))
    print("\nInterpretation: an edge is credible only if net_usdt >= the entry threshold "
          f"({cfg.entry.min_net_profit_usdt} USDT), t > 2, and it persists OUT-OF-SAMPLE.")


if __name__ == "__main__":
    main()
