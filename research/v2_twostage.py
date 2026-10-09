"""Two-stage V2 research: what has the model learned -- move MAGNITUDE or DIRECTION?

    python -m research.v2_twostage --data data/v2/ds_t15 --out runs/v2_twostage \\
        --train-days 2024-03-21 ... 2024-03-26 --cal-day 2024-03-27 --sel-day 2024-03-28 \\
        --forbid-days 2024-03-29 2024-03-30 --direct-models models/v2 --export models/v2

Chronological roles (TEST days are refused here; they are only ever replayed once per
frozen configuration by research.compare_v1_v2 --final-test):
  TRAIN       model fitting
  CALIBRATION early stopping, probability calibration, Stage-A quantile thresholds
  SELECTION   every reported metric and every choice (thresholds, dynamic targets)

Labels (all on EXECUTABLE prices after the entry latency, as in research.v2_dataset):
  large(X, H) = a +X bps (long: bid vs entry ask) or -X bps (short: ask vs entry bid)
                excursion is touched within H seconds      -> Stage A, P(|move| >= X)
  up(X)       = the long barrier is touched first           -> Stage B, P(UP | large move)
Stage B is trained ONLY on samples where the large move actually happened; at inference
it is applied after Stage A, with no conditioning on the future.
"""
from __future__ import annotations

import os as _os

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    _os.environ.setdefault(_var, "1")

import argparse
import json
import logging
import os
import re
import time
from typing import Any

import numpy as np
import pandas as pd

from research.v2_dataset import check_days, load_dataset, tp_first
from research.v2_models import (LGBMModel, XGBModel, auc, brier, ece, feature_columns, fit_calibrators, matrix,
                                realised_bps, simulate)

log = logging.getLogger("v2twostage")

XS = (10, 15, 20, 25, 30)
HS = (5, 10, 30, 60)
TOP = (0.20, 0.10, 0.05, 0.02, 0.01, 0.005)
TRADE_TARGETS = (15, 20, 25, 30)
NOTIONALS_ECON = (100, 150, 200, 250, 500)
INF = np.iinfo(np.int64).max


# ====================================================================== labels
def large(df: pd.DataFrame, X: int, H: int) -> np.ndarray:
    t = np.minimum(df[f"ttp_long_{X}"].to_numpy(), df[f"ttp_short_{X}"].to_numpy())
    return (t <= H * 1000).astype(np.int8)


def up(df: pd.DataFrame, X: int) -> np.ndarray:
    return (df[f"ttp_long_{X}"].to_numpy() < df[f"ttp_short_{X}"].to_numpy()).astype(np.int8)


def tie(df: pd.DataFrame, X: int) -> np.ndarray:
    a, b = df[f"ttp_long_{X}"].to_numpy(), df[f"ttp_short_{X}"].to_numpy()
    return (a == b) & (a != INF)


# ====================================================================== feature taxonomy
SIGNED_BASES = ("imb_", "depth_imb_", "microprice_offset", "micro_tilt", "ofi_", "depletion_asym", "replenish_asym",
                "persistence", "mom_", "flow_imb_", "aggr_buy_ratio", "trade_ret_bps", "large_trade_imb",
                "mid_chg_", "imb1_chg_", "fimb_", "tret_", "l2_wimb_", "flow_price_agree")
SIDE_PAIR = ("bid_", "ask_", "buy_notional", "sell_notional", "bidq_", "askq_", "l1_bid", "l1_ask", "l2_bid",
             "l2_ask", "l2_slope_", "l2_convex_")
INTENSITY = ("spread", "rv_", "trades_per_s", "notional_per_s", "avg_trade", "vol_accel", "velocity_change",
             "large_trade_share", "liquidity_change", "ntr_rate_", "vol_rate_log_", "cnt_accel_")


def feature_kind(name: str) -> str:
    """intensity (unsigned) / signed (directional) / side_pair (bid-vs-ask raw levels)."""
    base = re.sub(r"^r(mean|max|min|slope|pers|chg)_", "", name)
    base = re.sub(r"_\d+ms$", "", base) if name.startswith(("rmean_", "rmax_", "rmin_", "rslope_", "rpers_", "rchg_")) \
        else base
    if base.startswith(INTENSITY):
        return "intensity"
    if base.startswith(SIGNED_BASES):
        return "signed"
    if base.startswith(SIDE_PAIR):
        return "side_pair"
    return "other"


# ====================================================================== helpers
def _fit(kind: str, x, y, xc, yc, threads: int):
    m = LGBMModel(threads) if kind == "lightgbm" else XGBModel(threads)
    m.fit(x, y, xc, yc)
    return m


def _r(v, k=4):
    return None if v is None or not np.isfinite(v) else round(float(v), k)


def topk(y: np.ndarray, s: np.ndarray, fracs=TOP) -> list[dict]:
    out = []
    base = float(y.mean())
    for f in fracs:
        thr = np.quantile(s, 1 - f)
        m = s >= thr
        prec = float(y[m].mean()) if m.any() else float("nan")
        out.append({"top": f"{f * 100:g}%", "n": int(m.sum()), "hit_rate": _r(prec),
                    "recall": _r(y[m].sum() / max(y.sum(), 1)), "lift": _r(prec / base if base else np.nan, 2)})
    return out


def reliability(y: np.ndarray, p: np.ndarray, q: int = 10) -> list[dict]:
    """Quantile-binned reliability (base rates are small, so fixed 50..90% buckets are empty)."""
    edges = np.unique(np.quantile(p, np.linspace(0, 1, q + 1)))
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
    rows = []
    for b in range(len(edges) - 1):
        m = idx == b
        if m.any():
            rows.append({"bin": b + 1, "n": int(m.sum()), "predicted": _r(p[m].mean()), "actual": _r(y[m].mean())})
    return rows


def cluster_t(values: np.ndarray, sym: np.ndarray, ts: np.ndarray) -> tuple[float, float, int]:
    """Mean and t-stat with (symbol, minute) clusters: overlapping 1 s samples are not independent."""
    m = np.isfinite(values)
    if m.sum() < 2:
        return float("nan"), float("nan"), 0
    g = pd.DataFrame({"v": values[m], "k": pd.Series(sym[m]).astype(str) + "_" + (ts[m] // 60_000).astype(str)})
    cm = g.groupby("k")["v"].mean()
    if len(cm) < 2 or cm.std(ddof=1) == 0:
        return float(values[m].mean()), float("nan"), len(cm)
    return float(values[m].mean()), float(cm.mean() / (cm.std(ddof=1) / np.sqrt(len(cm)))), len(cm)


def walk_bps(df: pd.DataFrame, cfg_costs) -> dict[int, np.ndarray]:
    """Book-walk slippage (both sides) implied by the dataset's taker cost columns."""
    fixed = 2 * cfg_costs.taker_fee * 1e4 + 2 * (cfg_costs.latency_slippage_bps + cfg_costs.extra_slippage_bps)
    return {n: 0.5 * (df[f"cost_bps_long_{n}"].to_numpy(float) + df[f"cost_bps_short_{n}"].to_numpy(float)) - fixed
            for n in (100, 150, 250)}


# ====================================================================== Stage A / Stage B
def stage_a(kind, tr, ca, se, xtr, xca, xse, X, H, threads, cols_mask=None) -> dict[str, Any]:
    ytr, yca, yse = large(tr, X, H), large(ca, X, H), large(se, X, H)
    sl = slice(None) if cols_mask is None else cols_mask
    m = _fit(kind, xtr[:, sl], ytr, xca[:, sl], yca, threads)
    p_ca, p_se = m.predict(xca[:, sl]), m.predict(xse[:, sl])
    cal = fit_calibrators(p_ca, yca).get("isotonic")
    pc = cal(p_se) if cal is not None else p_se
    q = {f: float(np.quantile(p_ca, 1 - f)) for f in TOP}       # thresholds from the CAL distribution
    return {"model": m, "cal": cal, "p_ca": p_ca, "p_se": p_se, "pc_se": pc, "q_cal": q,
            "metrics": {"X": X, "H": H, "base_rate": _r(yse.mean()), "auc": _r(auc(yse, p_se)),
                        "brier_iso": _r(brier(yse, pc), 5), "ece_iso": _r(ece(yse, pc)),
                        "topk": topk(yse, p_se), "reliability": reliability(yse, pc)}}


def stage_b(kind, tr, ca, se, xtr, xca, xse, X, H, threads, cols_mask=None,
            min_pop: tuple[int, int, int] = (2000, 200, 200)) -> dict[str, Any] | None:
    sl = slice(None) if cols_mask is None else cols_mask
    pops = []
    for d in (tr, ca, se):
        pops.append((large(d, X, H) == 1) & ~tie(d, X))
    if any(p.sum() < k for p, k in zip(pops, min_pop)):
        return None
    ytr, yca, yse = up(tr[pops[0]], X), up(ca[pops[1]], X), up(se[pops[2]], X)
    m = _fit(kind, xtr[pops[0]][:, sl], ytr, xca[pops[1]][:, sl], yca, threads)
    p_ca_pop = m.predict(xca[pops[1]][:, sl])
    cal = fit_calibrators(p_ca_pop, yca).get("platt")
    p = m.predict(xse[pops[2]][:, sl])
    pc = cal(p) if cal is not None else p
    conf = np.abs(pc - 0.5)
    acc = ((pc >= 0.5) == (yse == 1))
    rows = []
    for f in (1.0, 0.5, 0.2, 0.1, 0.05):
        thr = np.quantile(conf, 1 - f)
        mm = conf >= thr
        rows.append({"most_confident": f"{f * 100:g}%", "n": int(mm.sum()), "accuracy": _r(acc[mm].mean()),
                     "mean_conf": _r(np.maximum(pc[mm], 1 - pc[mm]).mean())})
    base = {}
    sp = se[pops[2]]
    for f in ("mom_3s_bps", "ofi_3s", "flow_imb_3s", "imb_l1", "micro_tilt"):
        if f in sp:
            v = sp[f].to_numpy(float)
            nz = np.isfinite(v) & (v != 0)
            base[f"sign({f})"] = {"accuracy": _r(((v[nz] > 0) == (yse[nz] == 1)).mean()), "coverage": _r(nz.mean())}
    return {"model": m, "cal": cal, "metrics": {
        "X": X, "H": H, "n_train": int(pops[0].sum()), "n_sel": int(pops[2].sum()), "up_rate": _r(yse.mean()),
        "majority_acc": _r(max(yse.mean(), 1 - yse.mean())), "auc": _r(auc(yse, p)), "accuracy": _r(acc.mean()),
        "brier_platt": _r(brier(yse, pc), 5), "by_confidence": rows, "simple_rules": base}}


# ====================================================================== top-bucket analysis
def bucket_table(se: pd.DataFrame, score: np.ndarray, p_up: np.ndarray, X: int, H: int, stops: tuple[int, ...],
                 notional: int = 150) -> list[dict]:
    """Inside the top-k predicted large-move rows: move rate, direction accuracy, MFE/MAE, net if traded."""
    lg = large(se, X, H).astype(bool)
    upl = up(se, X).astype(bool)
    side_long = p_up >= 0.5
    mfe = np.where(side_long, se["mfe_long"].to_numpy(float), se["mfe_short"].to_numpy(float))
    mae = np.where(side_long, se["mae_long"].to_numpy(float), se["mae_short"].to_numpy(float))
    sym, ts = se["symbol"].to_numpy(), se["ts_ms"].to_numpy()
    nets = {}
    for S in stops:
        nl = realised_bps(se, "long", X, S, se[f"cost_bps_long_{notional}"].to_numpy(float))
        ns = realised_bps(se, "short", X, S, se[f"cost_bps_short_{notional}"].to_numpy(float))
        tpl, tps = tp_first(se, "long", X, S), tp_first(se, "short", X, S)
        fin = lambda v: np.where(np.isfinite(v), v, np.nan)        # untradeable rows (no cost) are excluded
        nets[S] = (fin(np.where(side_long, nl, ns)), np.where(side_long, tpl, tps), fin(np.where(upl, nl, ns)))
    rows = []
    for f in TOP[:5]:
        m = score >= np.quantile(score, 1 - f)
        r = {"top": f"{f * 100:g}%", "n": int(m.sum()), "large_move_rate": _r(lg[m].mean()),
             "base_large_rate": _r(lg.mean()),
             "dir_acc_given_move": _r((side_long[m & lg] == upl[m & lg]).mean()) if (m & lg).any() else None,
             "long_share": _r(side_long[m].mean()), "avg_mfe_bps": _r(np.nanmean(mfe[m]), 2),
             "avg_mae_bps": _r(np.nanmean(mae[m]), 2)}
        for S, (net, tp, oracle) in nets.items():
            mean, t, k = cluster_t(net[m] * notional / 1e4, sym[m], ts[m])
            r[f"p_tp_S{S}"] = _r(tp[m].mean())
            r[f"net_usdt_S{S}"] = _r(mean)
            r[f"t_S{S}"] = _r(t, 2)
            r[f"oracle_dir_net_usdt_S{S}"] = _r(np.nanmean(oracle[m]) * notional / 1e4)
        rows.append(r)
    return rows


# ====================================================================== economics
def economics(se: pd.DataFrame, cfg_costs, stop: int = 8, min_net: float = 0.10) -> list[dict]:
    w = walk_bps(se, cfg_costs)
    med = {n: float(np.nanmedian(v[np.isfinite(v)])) for n, v in w.items()}
    med[200] = med[150] + 0.5 * (med[250] - med[150])
    med[500] = med[250] + 2.5 * (med[250] - med[150])        # L1-only data: linear extrapolation (flagged)
    rows = []
    lat = cfg_costs.latency_slippage_bps + cfg_costs.extra_slippage_bps
    for n in NOTIONALS_ECON:
        for mode in ("taker", "maker"):
            if mode == "taker":
                c = 2 * cfg_costs.taker_fee * 1e4 + 2 * lat + med[n]
            else:   # maker entry (if filled) + taker exit
                c = (cfg_costs.maker_fee + cfg_costs.taker_fee) * 1e4 + cfg_costs.maker_adverse_selection_bps \
                    + lat + 0.5 * med[n]
            for T in TRADE_TARGETS:
                e = min_net * 1e4 / n
                rows.append({"notional": n, "entry": mode, "target_bps": T, "stop_bps": stop,
                             "cost_bps": round(c, 2), "cost_usdt": round(c * n / 1e4, 4),
                             "net_if_tp_usdt": round((T - c) * n / 1e4, 4),
                             "loss_if_sl_usdt": round(-(stop + c) * n / 1e4, 4),
                             "p_breakeven": round((stop + c) / (T + stop), 4),
                             "p_for_+0.10": round((e + stop + c) / (T + stop), 4),
                             "walk_extrapolated": n == 500})
    return rows


# ====================================================================== direct-model diagnosis
def load_direct(model_dir: str):
    import lightgbm as lgb

    with open(os.path.join(model_dir, "spec.json"), encoding="utf-8") as fh:
        spec = json.load(fh)
    return spec, {k: lgb.Booster(model_file=os.path.join(model_dir, e["file"])) for k, e in spec["models"].items()}


def shap_groups(booster, x: np.ndarray, cols: list[str], lgb_kind=True) -> dict[str, float]:
    if lgb_kind:
        contrib = booster.predict(x, pred_contrib=True)
    else:
        import xgboost as xgb
        contrib = booster.predict(xgb.DMatrix(x), pred_contribs=True)
    s = np.abs(contrib[:, :-1]).mean(0)
    tot = max(s.sum(), 1e-12)
    out: dict[str, float] = {}
    for c, v in zip(cols, s):
        k = feature_kind(c)
        out[k] = out.get(k, 0.0) + v / tot
    top = sorted(zip(cols, s / tot), key=lambda z: -z[1])[:12]
    return {"group_share": {k: round(v, 4) for k, v in sorted(out.items(), key=lambda z: -z[1])},
            "top": [{"feature": c, "kind": feature_kind(c), "share": round(float(v), 4)} for c, v in top]}


def direct_diagnosis(se: pd.DataFrame, direct_dir: str, stop: int, xs_sample_idx: np.ndarray) -> dict[str, Any]:
    spec, boosters = load_direct(direct_dir)
    cols = spec["features"]
    x = matrix(se, cols)
    out: dict[str, Any] = {"targets": {}}
    for T in spec["targets"]:
        if f"long_{T}" not in boosters:
            continue
        pl, ps = boosters[f"long_{T}"].predict(x), boosters[f"short_{T}"].predict(x)
        yl, ys = tp_first(se, "long", T, stop), tp_first(se, "short", T, stop)
        Xm = T if T in XS else None
        r = {"corr_p_long_p_short": _r(np.corrcoef(pl, ps)[0, 1]),
             "spearman_p_long_p_short": _r(pd.Series(pl).rank().corr(pd.Series(ps).rank())),
             "auc_long_model_on_long_tp": _r(auc(yl, pl)), "auc_long_model_on_short_tp": _r(auc(ys, pl)),
             "auc_short_model_on_short_tp": _r(auc(ys, ps)), "auc_short_model_on_long_tp": _r(auc(yl, ps))}
        any_tp = np.maximum(yl, ys)
        r["auc_(pl+ps)_on_any_tp"] = _r(auc(any_tp, pl + ps))
        both = (yl + ys) == 1                                  # exactly one side wins -> direction question
        r["direction_auc_(pl-ps)_given_one_side_tp"] = _r(auc(yl[both], (pl - ps)[both]))
        r["direction_acc_given_one_side_tp"] = _r(((pl > ps)[both] == (yl[both] == 1)).mean())
        if Xm is not None:
            lg = large(se, Xm, 60).astype(bool) & ~tie(se, Xm)
            r["direction_auc_(pl-ps)_given_large_move"] = _r(auc(up(se, Xm)[lg], (pl - ps)[lg]))
        out["targets"][T] = r
    for T in (20, 30):
        if f"long_{T}" in boosters:
            out[f"shap_long_{T}"] = shap_groups(boosters[f"long_{T}"], x[xs_sample_idx], cols)
            out[f"shap_short_{T}"] = shap_groups(boosters[f"short_{T}"], x[xs_sample_idx], cols)
    return out


def univariate(se: pd.DataFrame, cols: list[str], X: int = 20, H: int = 60) -> dict[str, Any]:
    lg = large(se, X, H)
    pop = (lg == 1) & ~tie(se, X)
    u = up(se, X)[pop]
    rows = []
    for c in cols:
        v = se[c].to_numpy(float)
        ok = np.isfinite(v)
        if ok.sum() < 1000 or np.nanstd(v) == 0:
            continue
        rk = pd.Series(np.where(ok, v, np.nan)).rank().to_numpy()
        mag = pd.Series(rk[ok]).corr(pd.Series(lg[ok].astype(float)))
        okp = ok[pop]
        dirc = pd.Series(rk[pop][okp]).corr(pd.Series(u[okp].astype(float))) if okp.sum() > 200 else np.nan
        rows.append({"feature": c, "kind": feature_kind(c), "rho_large_move": mag, "rho_up_given_move": dirc})
    df = pd.DataFrame(rows)
    df["abs_mag"] = df["rho_large_move"].abs()
    df["abs_dir"] = df["rho_up_given_move"].abs()
    g = df.groupby("kind")[["abs_mag", "abs_dir"]].agg(["mean", "max", "count"]).round(4)
    g.columns = ["_".join(c) for c in g.columns]
    return {"X": X, "H": H,
            "by_kind": g.reset_index().to_dict("records"),
            "top_magnitude": df.sort_values("abs_mag", ascending=False).head(15)[
                ["feature", "kind", "rho_large_move", "rho_up_given_move"]].round(4).to_dict("records"),
            "top_direction": df.sort_values("abs_dir", ascending=False).head(15)[
                ["feature", "kind", "rho_large_move", "rho_up_given_move"]].round(4).to_dict("records")}


# ====================================================================== combined rule
def combined_probs(A: dict[int, dict], B: dict[int, dict], df_x: np.ndarray, a_thr: dict[int, float], b_thr: float,
                   calib: dict[int, Any], targets) -> dict[tuple[str, int], np.ndarray]:
    """Stage A first; Stage B only where A passes; calibrated P(TP first) for the predicted side."""
    probs = {}
    for T in targets:
        pa = A[T]["model"].predict(df_x)
        pb = B[T]["cal"](B[T]["model"].predict(df_x))
        conf = np.maximum(pb, 1 - pb)
        gate = (pa >= a_thr[T]) & (conf >= b_thr)
        p_tp = calib[T](pa * conf)
        probs[("long", T)] = np.where(gate & (pb >= 0.5), p_tp, 0.0)
        probs[("short", T)] = np.where(gate & (pb < 0.5), p_tp, 0.0)
    return probs


def fit_combo_calib(A, B, ca, xca, targets, stop):
    from sklearn.isotonic import IsotonicRegression

    out = {}
    for T in targets:
        pa = A[T]["model"].predict(xca)
        pb = B[T]["cal"](B[T]["model"].predict(xca))
        side_long = pb >= 0.5
        y = np.where(side_long, tp_first(ca, "long", T, stop), tp_first(ca, "short", T, stop))
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(pa * np.maximum(pb, 1 - pb), y)
        out[T] = iso
    return out


def select_combined(kind, A, B, ca, se, xca, xse, stop, notional, min_net, targets) -> dict[str, Any]:
    calib_iso = fit_combo_calib(A, B, ca, xca, targets, stop)
    calib = {T: (lambda iso: (lambda s: iso.predict(s)))(iso) for T, iso in calib_iso.items()}
    grid = []
    best = None
    for f in TOP:
        a_thr = {T: A[T]["q_cal"][f] for T in targets}
        for b_thr in (0.50, 0.55, 0.60, 0.65, 0.70):
            for tset in [list(targets)] + [[T] for T in targets]:
                probs = combined_probs(A, B, xse, a_thr, b_thr, calib, tset)
                st, _ = simulate(se, probs, stop, notional, 0.0, min_net)
                row = {"stage_a_top": f"{f * 100:g}%", "b_thr": b_thr, "targets": tset, **st}
                grid.append(row)
                key = (st.get("trades", 0) >= 20, st.get("expectancy") or -1e9)
                if best is None or key > best[0]:
                    best = (key, row, a_thr, b_thr, tset)
    _, row, a_thr, b_thr, tset = best
    return {"grid": grid, "selected": row, "a_thr": a_thr, "b_thr": b_thr, "targets": tset, "calib": calib_iso}


def export_twostage(out_dir: str, kind: str, cols: list[str], A, B, sel: dict, stop: int, notional: float,
                    min_net: float) -> str:
    d = os.path.join(out_dir, f"twostage_{kind}")
    os.makedirs(d, exist_ok=True)
    spec = {"kind": "twostage", "base": kind, "features": cols, "stop_bps": stop, "notional": notional,
            "min_net_usdt": min_net, "threshold": 0.0, "targets": sel["targets"], "b_thr": sel["b_thr"],
            "stage_a_horizon_s": 60, "a_thr": {}, "stage_a": {}, "stage_b": {}, "combo_calib": {},
            "selected_on": "selection day (validation)", "sel_metrics": {k: v for k, v in sel["selected"].items()
                                                                          if not isinstance(v, (dict, list))}}
    for T in sel["targets"]:
        for stage, D, store in (("a", A, spec["stage_a"]), ("b", B, spec["stage_b"])):
            m = D[T]["model"]
            ext = "txt" if kind == "lightgbm" else "json"
            path = os.path.join(d, f"stage{stage}_{T}.{ext}")
            if kind == "lightgbm":
                m.b.save_model(path, num_iteration=m.b.best_iteration)
            else:
                m.m.get_booster().save_model(path)
            entry = {"file": os.path.basename(path)}
            if kind == "xgboost":
                entry["best_iteration"] = int(getattr(m.m, "best_iteration", 0) or 0)
            if stage == "b":
                c = D[T]["cal"]
                entry["calibrator"] = {"method": c.method, "a": c.a, "b": c.b, "xs": c.xs, "ys": c.ys}
            store[str(T)] = entry
        spec["a_thr"][str(T)] = float(sel["a_thr"][T])
        iso = sel["calib"][T]
        spec["combo_calib"][str(T)] = {"method": "isotonic", "xs": iso.X_thresholds_.tolist(),
                                       "ys": iso.y_thresholds_.tolist()}
    with open(os.path.join(d, "spec.json"), "w", encoding="utf-8") as fh:
        json.dump(spec, fh)
    return d


# ====================================================================== dynamic target study
def dynamic_targets(se: pd.DataFrame, ca: pd.DataFrame, A: dict, B: dict, xse, xca, stop: int, notional: int,
                    strength_X: int = 20, targets=TRADE_TARGETS) -> dict[str, Any]:
    """Strength = Stage-A P(|move| >= 20 bps, 60 s). Buckets from CAL quantiles; best T chosen on SEL."""
    pa_ca = A[strength_X]["model"].predict(xca)
    q = np.quantile(pa_ca, [0.90, 0.97, 0.99])
    pa = A[strength_X]["model"].predict(xse)
    bucket = np.digitize(pa, q)                      # 0 weak, 1 moderate, 2 strong, 3 very strong
    names = ["weak (<p90)", "moderate (p90-p97)", "strong (p97-p99)", "very strong (>=p99)"]
    sym, ts = se["symbol"].to_numpy(), se["ts_ms"].to_numpy()
    rows, mapping = [], {}
    for b, name in enumerate(names):
        m = bucket == b
        r = {"bucket": name, "n": int(m.sum())}
        best = ("skip", 0.0)
        for T in targets:
            pb = B[T]["cal"](B[T]["model"].predict(xse[m]))
            side_long = pb >= 0.5
            net = np.where(side_long,
                           realised_bps(se[m], "long", T, stop, se[m][f"cost_bps_long_{notional}"].to_numpy(float)),
                           realised_bps(se[m], "short", T, stop, se[m][f"cost_bps_short_{notional}"].to_numpy(float)))
            usd = net * notional / 1e4
            mean, t, _ = cluster_t(usd, sym[m], ts[m])
            r[f"T{T}_net_usdt"] = _r(mean)
            r[f"T{T}_t"] = _r(t, 2)
            if np.isfinite(mean) and mean > best[1] and (np.isfinite(t) and t > 2):
                best = (f"T{T}", mean)
        r["chosen_on_validation"] = best[0]
        mapping[name] = best[0]
        rows.append(r)
    return {"strength": f"Stage A P(|move|>={strength_X}bps in 60s)", "cal_quantiles": [float(v) for v in q],
            "table": rows, "mapping": mapping}


# ====================================================================== latency
def latency_us(model_dirs: dict[str, str], se: pd.DataFrame, n: int = 1500) -> dict[str, dict]:
    from strategy.v2 import load_v2_model

    out = {}
    rows = se.sample(min(n, len(se)), random_state=3)
    for name, d in model_dirs.items():
        if not os.path.exists(os.path.join(d, "spec.json")):
            continue
        mdl = load_v2_model(d)
        fs = [{c: float(v) for c, v in zip(mdl.features, r)} for r in rows[mdl.features].to_numpy(float)]
        for f in fs[:30]:
            mdl.predict(f)
        ts = []
        for f in fs:
            t0 = time.perf_counter_ns()
            mdl.predict(f)
            ts.append((time.perf_counter_ns() - t0) / 1000)
        ts = np.array(ts)
        out[name] = {"median_us": _r(np.median(ts), 1), "p95_us": _r(np.percentile(ts, 95), 1),
                     "p99_us": _r(np.percentile(ts, 99), 1), "n_models": len(getattr(mdl, "keys", [])) or None}
    return out


# ====================================================================== main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--train-days", nargs="+", required=True)
    ap.add_argument("--cal-day", required=True)
    ap.add_argument("--sel-day", required=True)
    ap.add_argument("--forbid-days", nargs="+", default=["2024-03-29", "2024-03-30"])
    ap.add_argument("--direct-models", default="models/v2")
    ap.add_argument("--export", default="models/v2")
    ap.add_argument("--stop", type=int, default=8)
    ap.add_argument("--notional", type=float, default=150.0)
    ap.add_argument("--min-net", type=float, default=0.10)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--xgb", action="store_true", help="also fit the two-stage XGBoost pipeline")
    ap.add_argument("--grid-x", nargs="+", type=int, default=list(XS))
    ap.add_argument("--grid-h", nargs="+", type=int, default=list(HS))
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    from config import CostConfig

    check_days(args.train_days + [args.cal_day, args.sel_day], args.forbid_days)
    os.makedirs(args.out, exist_ok=True)
    t0 = time.perf_counter()
    tr = load_dataset(args.data, args.train_days)
    ca = load_dataset(args.data, [args.cal_day])
    se = load_dataset(args.data, [args.sel_day])
    cols = feature_columns(tr)
    xtr, xca, xse = matrix(tr, cols), matrix(ca, cols), matrix(se, cols)
    log.info("rows train=%d cal=%d sel=%d features=%d (%.0fs)", len(tr), len(ca), len(se), len(cols),
             time.perf_counter() - t0)
    kinds = pd.Series({c: feature_kind(c) for c in cols})
    rep: dict[str, Any] = {"split": {"train": args.train_days, "calibration": args.cal_day, "selection": args.sel_day,
                                     "test": "UNTOUCHED here (" + ", ".join(args.forbid_days) + ")"},
                           "rows": {"train": len(tr), "cal": len(ca), "sel": len(se)},
                           "feature_kinds": kinds.value_counts().to_dict(), "stop_bps": args.stop,
                           "notional": args.notional}
    shap_idx = np.random.default_rng(7).choice(len(se), min(15_000, len(se)), replace=False)

    def save():
        with open(os.path.join(args.out, "twostage.json"), "w", encoding="utf-8") as fh:
            json.dump(rep, fh, indent=1, default=str)

    # ---- economics + volatility diagnosis of the existing direct models (no fitting)
    rep["economics"] = economics(se, CostConfig(), args.stop, args.min_net)
    rep["univariate"] = univariate(se, cols)
    if os.path.exists(os.path.join(args.direct_models, "lightgbm", "spec.json")):
        log.info("direct LightGBM diagnosis ...")
        rep["direct_lightgbm"] = direct_diagnosis(se, os.path.join(args.direct_models, "lightgbm"), args.stop, shap_idx)
    save()

    # ---- Stage A grid (LightGBM) + Stage B grid
    kinds_run = ["lightgbm"] + (["xgboost"] if args.xgb else [])
    A: dict[str, dict] = {k: {} for k in kinds_run}
    B: dict[str, dict] = {k: {} for k in kinds_run}
    rep["stage_a"], rep["stage_b"] = {}, {}
    for kind in kinds_run:
        grid = [(X, H) for X in args.grid_x for H in args.grid_h] if kind == "lightgbm" else \
            [(X, 60) for X in TRADE_TARGETS if X in args.grid_x]
        for X, H in grid:
            tk = time.perf_counter()
            a = stage_a(kind, tr, ca, se, xtr, xca, xse, X, H, args.threads)
            A[kind][(X, H)] = a
            rep["stage_a"].setdefault(kind, []).append(a["metrics"])
            log.info("A %s X=%d H=%d auc=%s base=%s (%.0fs)", kind, X, H, a["metrics"]["auc"],
                     a["metrics"]["base_rate"], time.perf_counter() - tk)
            if kind == "xgboost" or H in (10, 60):
                b = stage_b(kind, tr, ca, se, xtr, xca, xse, X, H, args.threads)
                if b is not None:
                    B[kind][(X, H)] = b
                    rep["stage_b"].setdefault(kind, []).append(b["metrics"])
                    log.info("B %s X=%d H=%d auc=%s acc=%s", kind, X, H, b["metrics"]["auc"], b["metrics"]["accuracy"])
            save()

    # ---- SHAP of Stage A / Stage B (LightGBM, X=20, H=60)
    for name, D in (("stage_a", A["lightgbm"]), ("stage_b", B["lightgbm"])):
        if (20, 60) in D:
            rep.setdefault("shap", {})[f"{name}_X20_H60"] = shap_groups(D[(20, 60)]["model"].b, xse[shap_idx], cols)

    # ---- ablation: intensity-only vs signed-only vs all (LightGBM, X=20, H=60)
    log.info("ablation ...")
    masks = {"intensity_only": (kinds == "intensity").to_numpy(),
             "signed_and_side_only": kinds.isin(["signed", "side_pair"]).to_numpy(),
             "all": np.ones(len(cols), bool)}
    rep["ablation"] = {}
    for name, mk in masks.items():
        a = stage_a("lightgbm", tr, ca, se, xtr, xca, xse, 20, 60, args.threads, np.where(mk)[0]) \
            if name != "all" else A["lightgbm"][(20, 60)]
        b = stage_b("lightgbm", tr, ca, se, xtr, xca, xse, 20, 60, args.threads, np.where(mk)[0]) \
            if name != "all" else B["lightgbm"].get((20, 60))
        rep["ablation"][name] = {"n_features": int(mk.sum()), "stage_a_auc": a["metrics"]["auc"],
                                 "stage_a_top1pct_hit": a["metrics"]["topk"][4]["hit_rate"],
                                 "stage_b_auc": b["metrics"]["auc"] if b else None,
                                 "stage_b_acc": b["metrics"]["accuracy"] if b else None}
    save()

    # ---- top-bucket analysis (Stage A rank, Stage B direction), and the direct model's own ranking
    rep["top_buckets"] = {}
    for kind in kinds_run:
        for X in TRADE_TARGETS:
            if (X, 60) in A[kind] and (X, 60) in B[kind]:
                pb = B[kind][(X, 60)]["cal"](B[kind][(X, 60)]["model"].predict(xse))
                rep["top_buckets"][f"{kind}_twostage_X{X}_H60"] = bucket_table(
                    se, A[kind][(X, 60)]["p_se"], pb, X, 60, (args.stop, 12), int(args.notional))
    if os.path.exists(os.path.join(args.direct_models, "lightgbm", "spec.json")):
        spec, boosters = load_direct(os.path.join(args.direct_models, "lightgbm"))
        xd = matrix(se, spec["features"])
        for X in (20, 25, 30):
            if f"long_{X}" in boosters:
                pl, ps = boosters[f"long_{X}"].predict(xd), boosters[f"short_{X}"].predict(xd)
                rep["top_buckets"][f"direct_lightgbm_T{X}"] = bucket_table(
                    se, np.maximum(pl, ps), (pl >= ps).astype(float), X, 60, (args.stop, 12), int(args.notional))
    save()

    # ---- dynamic target study (validation only)
    both = [X for X in TRADE_TARGETS if (X, 60) in A["lightgbm"] and (X, 60) in B["lightgbm"]]
    if 20 in both:
        rep["dynamic_targets"] = dynamic_targets(
            se, ca, {X: A["lightgbm"][(X, 60)] for X in both}, {X: B["lightgbm"][(X, 60)] for X in both},
            xse, xca, args.stop, int(args.notional), targets=both)
    save()

    # ---- combined rule: thresholds chosen on SEL; export for the replay
    rep["combined"] = {}
    exported = {}
    for kind in kinds_run:
        Ak = {X: A[kind][(X, 60)] for X in TRADE_TARGETS if (X, 60) in A[kind]}
        Bk = {X: B[kind][(X, 60)] for X in TRADE_TARGETS if (X, 60) in B[kind]}
        tg = sorted(set(Ak) & set(Bk))
        sel = select_combined(kind, Ak, Bk, ca, se, xca, xse, args.stop, args.notional, args.min_net, tg)
        rep["combined"][kind] = {"selected": sel["selected"], "grid": sel["grid"]}
        exported[f"twostage_{kind}"] = export_twostage(args.export, kind, cols, Ak, Bk, sel, args.stop,
                                                       args.notional, args.min_net)
        log.info("combined %s selected: %s", kind, sel["selected"])
    save()

    # ---- inference latency of the full serving path (one decision = all targets, both sides)
    dirs = {k: os.path.join(args.direct_models, k) for k in ("lightgbm", "xgboost", "mlp")}
    dirs.update(exported)
    rep["latency_us"] = latency_us(dirs, se)
    rep["seconds"] = round(time.perf_counter() - t0)
    save()
    print(f"report: {os.path.join(args.out, 'twostage.json')}")


if __name__ == "__main__":
    main()
