"""Train a logistic direction model from recorded signals (time-ordered split).

Output JSON is consumed by ``strategy.predictor.LinearModelPredictor`` (set
``strategy.predictor = "linear"``). Only switch if out-of-sample metrics beat the
rule-based engine AFTER costs (re-run research.analyze_signals with the new model in
paper mode).

    python -m research.train_model --db data/microstructure.sqlite --horizon 10
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3

import numpy as np
import pandas as pd

DEFAULT_FEATURES = [
    "imb_l1", "imb_l5", "imb_l10", "imb_weighted", "depth_imb_5bps", "depth_imb_10bps",
    "micro_tilt", "ofi_1s", "ofi_3s", "ofi_10s", "depletion_asym", "replenish_asym",
    "persistence", "liquidity_change", "mom_1s_bps", "mom_3s_bps", "mom_10s_bps",
    "flow_imb_1s", "flow_imb_3s", "flow_imb_10s", "flow_imb_30s", "vol_accel",
    "velocity_change", "large_trade_imb", "spread_bps", "rv_1s_bps",
]


def load(db: str, horizon: int) -> pd.DataFrame:
    conn = sqlite3.connect(db)
    try:
        df = pd.read_sql_query(
            f"SELECT ts_ms, features_json, ret_{horizon}s AS ret FROM signals "
            f"WHERE labeled = 1 AND ret_{horizon}s IS NOT NULL ORDER BY ts_ms", conn)
    finally:
        conn.close()
    feats = pd.json_normalize(df["features_json"].map(json.loads))
    return pd.concat([df[["ts_ms", "ret"]].reset_index(drop=True), feats], axis=1)


def fit_logistic(X: np.ndarray, y: np.ndarray, l2: float = 1.0, iters: int = 500, lr: float = 0.1):
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0
    for _ in range(iters):
        z = np.clip(X @ w + b, -30, 30)
        p = 1 / (1 + np.exp(-z))
        g = p - y
        w -= lr * (X.T @ g / n + l2 * w / n)
        b -= lr * g.mean()
    return w, b


def auc(y: np.ndarray, s: np.ndarray) -> float:
    order = np.argsort(s)
    ranks = np.empty(len(s))
    ranks[order] = np.arange(1, len(s) + 1)
    pos = y == 1
    n1, n0 = pos.sum(), (~pos).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/microstructure.sqlite")
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--split", type=float, default=0.7)
    ap.add_argument("--l2", type=float, default=5.0)
    ap.add_argument("--min-move-bps", type=float, default=0.5, help="ignore near-zero moves when fitting")
    ap.add_argument("--out", default="models/linear_model.json")
    args = ap.parse_args()

    df = load(args.db, args.horizon)
    feats = [f for f in DEFAULT_FEATURES if f in df.columns]
    df = df.dropna(subset=feats + ["ret"])
    df = df[df["ret"].abs() >= args.min_move_bps]
    if len(df) < 500:
        print(f"only {len(df)} usable samples; record more data first (need >= 500).")
        return
    cut = int(len(df) * args.split)
    tr, te = df.iloc[:cut], df.iloc[cut:]
    mu = tr[feats].mean().values
    sd = tr[feats].std().replace(0, 1).values
    Xtr = (tr[feats].values - mu) / sd
    Xte = (te[feats].values - mu) / sd
    ytr = (tr["ret"].values > 0).astype(float)
    yte = (te["ret"].values > 0).astype(float)
    w, b = fit_logistic(Xtr, ytr, l2=args.l2)

    def evaluate(X, y, ret):
        p = 1 / (1 + np.exp(-(X @ w + b)))
        edge = 2 * p - 1
        return {
            "n": int(len(y)),
            "auc": round(auc(y, p), 4),
            "accuracy": round(float(((p > 0.5) == (y == 1)).mean()), 4),
            "mean_dir_ret_bps_conf>0.6": round(float((np.sign(edge) * ret)[np.abs(edge) > 0.2].mean()), 3)
            if (np.abs(edge) > 0.2).any() else None,
        }

    print("train:", evaluate(Xtr, ytr, tr["ret"].values))
    print("test :", evaluate(Xte, yte, te["ret"].values))

    p_tr = 1 / (1 + np.exp(-(Xtr @ w + b)))
    edge_tr = 2 * p_tr - 1
    sigma = np.maximum(tr["rv_1s_bps"].values if "rv_1s_bps" in tr else np.ones(len(tr)), 0.3)
    target = tr["ret"].values / (sigma * args.horizon)
    drift_per_edge = float((edge_tr * target).sum() / max((edge_tr ** 2).sum(), 1e-9))
    print(f"drift_per_edge (sigma/s per unit edge): {drift_per_edge:.4f}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({
            "features": feats, "mean": mu.tolist(), "std": sd.tolist(), "coef": w.tolist(),
            "intercept": float(b), "drift_per_edge": max(drift_per_edge, 0.0),
            "horizon_s": args.horizon, "trained_on": int(len(tr)),
        }, fh, indent=2)
    print("saved", args.out)
    for name, coef in sorted(zip(feats, w), key=lambda kv: -abs(kv[1]))[:12]:
        print(f"  {name:<20} {coef:+.4f}")


if __name__ == "__main__":
    main()
