"""V2 diagnostics: model variants, feature importance/SHAP, V1-feature verdicts, entry
timing, regimes, inference latency. All computed on TRAIN/validation days only."""
from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd

from config import StrategyConfig
from research.v2_dataset import tp_first
from research.v2_models import LGBMModel, auc, brier, feature_columns, matrix
from strategy.predictor import RuleBasedPredictor

V1_FEATURES = {   # V1 rule-based components and the sign V1 assumes (+1: positive == bullish)
    "imb_weighted": 1, "ofi_3s": 1, "flow_imb_3s": 1, "flow_imb_10s": 1, "micro_tilt": 1,
    "mom_3s_bps": 1, "persistence": 1, "depletion_asym": 1,
}


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 50:
        return float("nan")
    return float(pd.Series(x[m]).rank().corr(pd.Series(y[m]).rank()))


# ---------------------------------------------------------------------- variants
def variants(tr, ca, se, cols, stop: int, target: int, threads: int) -> dict[str, Any]:
    out: dict[str, Any] = {}
    xtr, xca, xse = matrix(tr, cols), matrix(ca, cols), matrix(se, cols)
    # (a) separate LONG / SHORT
    sep = {}
    for side in ("long", "short"):
        m = LGBMModel(threads)
        m.fit(xtr, tp_first(tr, side, target, stop), xca, tp_first(ca, side, target, stop))
        y = tp_first(se, side, target, stop)
        p = m.predict(xse)
        sep[side] = {"auc": round(auc(y, p), 4), "brier": round(brier(y, p), 5), "base_rate": round(float(y.mean()), 4)}
    out["separate_sides"] = sep

    # (b) one model, direction as a feature (features sign-flipped is NOT assumed)
    def stack(df, x):
        yl, ys = tp_first(df, "long", target, stop), tp_first(df, "short", target, stop)
        d = np.concatenate([np.ones(len(df)), -np.ones(len(df))]).reshape(-1, 1)
        return np.hstack([np.vstack([x, x]), d]).astype(np.float32), np.concatenate([yl, ys])
    xs_tr, ys_tr = stack(tr, xtr)
    xs_ca, ys_ca = stack(ca, xca)
    xs_se, ys_se = stack(se, xse)
    m = LGBMModel(threads)
    m.fit(xs_tr, ys_tr, xs_ca, ys_ca)
    p = m.predict(xs_se)
    n = len(se)
    out["direction_feature"] = {
        "long": {"auc": round(auc(ys_se[:n], p[:n]), 4), "brier": round(brier(ys_se[:n], p[:n]), 5)},
        "short": {"auc": round(auc(ys_se[n:], p[n:]), 4), "brier": round(brier(ys_se[n:], p[n:]), 5)}}

    # (c) per-symbol models vs global (long side)
    per = {}
    g = LGBMModel(threads)
    g.fit(xtr, tp_first(tr, "long", target, stop), xca, tp_first(ca, "long", target, stop))
    for sym in sorted(se["symbol"].unique()):
        mtr, mca, mse = (tr["symbol"] == sym).to_numpy(), (ca["symbol"] == sym).to_numpy(), (se["symbol"] == sym).to_numpy()
        y = tp_first(se[mse], "long", target, stop)
        row = {"global_auc": round(auc(y, g.predict(xse[mse])), 4), "n_sel": int(mse.sum()),
               "base_rate": round(float(y.mean()), 4)}
        try:
            ms = LGBMModel(threads)
            ms.fit(xtr[mtr], tp_first(tr[mtr], "long", target, stop), xca[mca], tp_first(ca[mca], "long", target, stop))
            row["symbol_model_auc"] = round(auc(y, ms.predict(xse[mse])), 4)
        except Exception as exc:  # noqa: BLE001
            row["symbol_model_auc"] = f"failed: {exc}"
        per[sym] = row
    out["per_symbol_long"] = per
    return out


# ---------------------------------------------------------------------- importance
def importance(model: LGBMModel, se: pd.DataFrame, cols: list[str], n_shap: int = 20_000) -> dict[str, Any]:
    b = model.b
    gain = b.feature_importance("gain", iteration=b.best_iteration)
    sample = se.sample(min(n_shap, len(se)), random_state=7)
    contrib = b.predict(matrix(sample, cols), pred_contrib=True, num_iteration=b.best_iteration)
    shap = np.abs(contrib[:, :-1]).mean(0)
    df = pd.DataFrame({"feature": cols, "gain": gain, "mean_abs_shap": shap})
    df["gain_share"] = df["gain"] / max(df["gain"].sum(), 1e-12)
    df["shap_share"] = df["mean_abs_shap"] / max(df["mean_abs_shap"].sum(), 1e-12)
    df["group"] = df["feature"].map(feature_group)
    top = df.sort_values("mean_abs_shap", ascending=False)
    groups = df.groupby("group")[["gain_share", "shap_share"]].sum().sort_values("shap_share", ascending=False)
    return {"top": top.head(30).round(5).to_dict("records"), "by_group": groups.round(4).reset_index().to_dict("records"),
            "table": top}


def feature_group(name: str) -> str:
    if name.startswith(("rmean_", "rmax_", "rmin_", "rslope_", "rpers_", "rchg_")):
        return "rolling history"
    if name.startswith(("mid_chg_", "imb1_chg_", "spread_chg_", "bidq_chg_", "askq_chg_")):
        return "book lags (100ms-10s)"
    if name.startswith(("fimb_", "ntr_rate_", "vol_rate_log_", "cnt_accel_", "vol_accel_", "tret_")):
        return "short-window flow"
    if name.startswith("l2_"):
        return "L2 depth profile"
    return "V1 base features"


def v1_feature_verdicts(tr: pd.DataFrame, se: pd.DataFrame, imp: pd.DataFrame, stop: int, target: int) -> list[dict]:
    """useful / redundant / reversed / lagging / noisy for each V1 component."""
    out = []
    shap = dict(zip(imp["feature"], imp["shap_share"]))
    rank = {f: i for i, f in enumerate(imp["feature"])}
    for f, sign in V1_FEATURES.items():
        if f not in se:
            continue
        x = se[f].to_numpy(float)
        yl = tp_first(se, "long", target, stop).astype(float)
        ys = tp_first(se, "short", target, stop).astype(float)
        ic_long, ic_short = _spearman(x, yl), _spearman(x, ys)
        ic_fwd = _spearman(x, se["fret_10s"].to_numpy(float))
        ic_past = _spearman(x, se["mid_chg_5000ms"].to_numpy(float)) if "mid_chg_5000ms" in se else float("nan")
        # redundancy: strongest correlation with a more important feature
        corr_best, corr_with = 0.0, None
        for g in imp["feature"].head(40):
            if g == f or g not in tr:
                continue
            c = _spearman(tr[f].to_numpy(float)[:50_000], tr[g].to_numpy(float)[:50_000])
            if abs(c) > abs(corr_best) and rank.get(g, 1e9) < rank.get(f, 1e9):
                corr_best, corr_with = c, g
        verdicts = []
        if np.isfinite(ic_fwd) and np.sign(ic_fwd) == -sign and abs(ic_fwd) > 0.02:
            verdicts.append("REVERSED (sign opposite to V1 assumption)")
        if np.isfinite(ic_past) and np.isfinite(ic_fwd) and abs(ic_past) > 3 * max(abs(ic_fwd), 0.01):
            verdicts.append("LAGGING (reflects the past move far more than the next one)")
        if abs(corr_best) > 0.9:
            verdicts.append(f"REDUNDANT (rho={corr_best:.2f} with {corr_with})")
        if shap.get(f, 0) < 0.002 and (not np.isfinite(ic_fwd) or abs(ic_fwd) < 0.02):
            verdicts.append("NOISY / uninformative")
        if not verdicts:
            verdicts.append("USEFUL" if shap.get(f, 0) >= 0.005 else "WEAK")
        out.append({"feature": f, "v1_sign": sign, "ic_fwd_10s": round(ic_fwd, 4), "ic_past_5s": round(ic_past, 4),
                    "ic_long_tp": round(ic_long, 4), "ic_short_tp": round(ic_short, 4),
                    "shap_share": round(shap.get(f, 0.0), 5), "verdict": "; ".join(verdicts)})
    return out


# ---------------------------------------------------------------------- entry timing
def entry_timing(df: pd.DataFrame, threshold: float = 0.35) -> dict[str, Any]:
    """Event study around V1-style entries reconstructed with the V1 predictor on every row.

    For rows where |V1 score| first crosses ``threshold`` (direction d), average
    d * (mid move) over [-5 s, +60 s] relative to the signal, and d * key features at
    -5..+5 s. Rows are 1 s apart within (symbol, day), so offsets index neighbours.
    """
    pred = RuleBasedPredictor(StrategyConfig())
    scores = np.array([pred.predict(r).score for r in df[list(_v1_inputs(df))].to_dict("records")])
    df = df.assign(v1_score=scores)
    out_rows = []
    feats = ["imb_l1", "micro_tilt", "fimb_1000ms", "ofi_3s", "flow_imb_3s"]
    for (sym, day), g in df.groupby(["symbol", "day"], sort=False):
        g = g.sort_values("ts_ms")
        s = g["v1_score"].to_numpy()
        mid = g["mid0"].to_numpy(float)
        ts = g["ts_ms"].to_numpy()
        prev = np.concatenate([[0.0], s[:-1]])
        idx = np.where((np.abs(s) >= threshold) & (np.abs(prev) < threshold))[0]
        for i in idx:
            d = np.sign(s[i])
            row = {"symbol": sym, "dir": d}
            for k in (-5, -3, -1, 0, 1, 3, 5, 10, 30, 60):
                j = i + k
                if 0 <= j < len(g) and abs(ts[j] - ts[i] - k * 1000) <= 1:
                    row[f"move_{k}s"] = d * (mid[j] - mid[i]) / mid[i] * 1e4
                    if -5 <= k <= 5:
                        for f in feats:
                            if f in g:
                                row[f"{f}_{k}s"] = d * g[f].iat[j]
            out_rows.append(row)
    ev = pd.DataFrame(out_rows)
    if ev.empty:
        return {"events": 0}
    path = {k: round(float(ev[f"move_{k}s"].mean()), 3) for k in (-5, -3, -1, 0, 1, 3, 5, 10, 30, 60)
            if f"move_{k}s" in ev}
    feat_path = {f: {k: round(float(ev[f"{f}_{k}s"].mean()), 4) for k in (-5, -3, -1, 0, 1, 3, 5)
                     if f"{f}_{k}s" in ev} for f in feats}
    lead_in = -path.get(-5, 0.0)
    follow = path.get(5, 0.0)
    verdict = ("LATE: most of the move happened before the signal and it does not continue"
               if lead_in > 0.5 and follow < 0.5 * lead_in else
               "REVERSAL: price moves against the signal after entry" if follow < -0.2 else
               "NOT LATE: price continues in the signal direction after entry")
    return {"events": len(ev), "threshold": threshold, "avg_move_bps_vs_signal_time": path,
            "direction_adjusted_feature_path": feat_path, "lead_in_bps_5s": round(lead_in, 3),
            "follow_through_bps_5s": round(follow, 3), "verdict": verdict}


def _v1_inputs(df: pd.DataFrame) -> list[str]:
    need = ["imb_weighted", "ofi_3s", "flow_imb_3s", "flow_imb_10s", "micro_tilt", "mom_3s_bps", "persistence",
            "depletion_asym", "rv_1s_bps", "trades_per_s_10s", "flow_price_agree_3s"]
    return [c for c in need if c in df]


# ---------------------------------------------------------------------- regimes
def regimes(trades: pd.DataFrame) -> dict[str, list[dict]]:
    if trades.empty:
        return {}
    out = {}
    t = trades.copy()
    t["utc_hour"] = (t["ts_ms"] // 3_600_000) % 24
    dims = {"symbol": "symbol", "utc_hour": "utc_hour", "side": "side", "target": "target"}
    for name, col in (("volatility", "rv_1s_bps"), ("spread", "spread_bps"), ("liquidity", "l1_bid_notional_log"),
                      ("trade_activity", "trades_per_s_10s"), ("flow_imbalance", "flow_imb_3s")):
        if col in t and t[col].nunique() > 3:
            t[name] = pd.qcut(t[col], 3, labels=["low", "mid", "high"], duplicates="drop")
            dims[name] = name
    for name, col in dims.items():
        rows = []
        for k, g in t.groupby(col, observed=True):
            rows.append({"bucket": str(k), "trades": len(g), "expectancy": round(float(g["net_usdt"].mean()), 5),
                         "net": round(float(g["net_usdt"].sum()), 4), "tp_rate": round(float(g["tp"].mean()), 4)})
        out[name] = rows
    return out


# ---------------------------------------------------------------------- latency
def latency(models: dict[str, Any], cols: list[str], sample: np.ndarray, n: int = 2000) -> dict[str, dict]:
    """Single-row inference latency (one (side, target) model) in microseconds."""
    out = {}
    row = sample[:1].astype(np.float32)
    for kind, obj in models.items():
        fn = _single_row_fn(kind, obj)
        if fn is None:
            continue
        for _ in range(50):
            fn(row)
        ts = []
        for i in range(n):
            r = sample[i % len(sample): i % len(sample) + 1]
            t0 = time.perf_counter_ns()
            fn(r)
            ts.append((time.perf_counter_ns() - t0) / 1000)
        ts = np.array(ts)
        out[kind] = {"median_us": round(float(np.median(ts)), 1), "p95_us": round(float(np.percentile(ts, 95)), 1),
                     "p99_us": round(float(np.percentile(ts, 99)), 1)}
    return out


def _single_row_fn(kind: str, m: Any):
    if kind == "logistic":
        med, mu, sd = m.med, m.mu, m.sd
        w, b = m.m.coef_[0], m.m.intercept_[0]

        def f(r):
            z = np.where(np.isnan(r[0]), med, r[0])
            return 1 / (1 + np.exp(-(np.clip((z - mu) / sd, -8, 8) @ w + b)))
        return f
    if kind == "lightgbm":
        b = m.b   # (Booster.reset_parameter segfaults on predict-only boosters; pass threads per call)
        return lambda r: b.predict(r, num_iteration=b.best_iteration, num_threads=1)
    if kind == "xgboost":
        booster = m.m.get_booster()
        booster.set_param({"nthread": 1})
        return lambda r: booster.inplace_predict(r)
    if kind == "mlp":
        med, mu, sd = m.med, m.mu, m.sd
        W, B = m.m.coefs_, m.m.intercepts_

        def f(r):
            h = np.clip((np.where(np.isnan(r[0]), med, r[0]) - mu) / sd, -8, 8)
            for k, (w, b) in enumerate(zip(W, B)):
                h = h @ w + b
                if k < len(W) - 1:
                    h = np.maximum(h, 0)
            return 1 / (1 + np.exp(-h))
        return f
    return None


__all__ = ["variants", "importance", "v1_feature_verdicts", "entry_timing", "regimes", "latency",
           "feature_columns"]
