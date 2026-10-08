"""V3 research: does full L2 / queue-dynamics data add DIRECTIONAL information beyond L1?

    python -m v3.research --data data/v3/ds --out runs/v3 \\
        --train-days D1 ... --cal-days D --sel-days D ... --forbid-days T1 T2 ... \\
        [--notional 150] [--min-net 0.10] [--export models/v3] [--max-train-rows 1500000]

Chronological roles (test days are refused here; they are only replayed once per frozen
configuration by ``v3.evaluate --final-test``):
  TRAIN        fitting
  CALIBRATION  early stopping, probability calibration, Stage-A quantile thresholds
  SELECTION    every reported number and every choice (feature set, barrier pair, thresholds)

Models: logistic regression and LightGBM only (no other family until LightGBM shows real
directional discrimination).

Sections of the output (``v3_research.json`` + ``V3_RESEARCH.md``):
  1  Stage A  P(|move| >= X within H): V2's L1 definition, and with all V3 features
  2  Direction P(UP first | large move), per feature set (ablation) and model
  3  Direction INSIDE Stage A's strongest predicted large-move states (top 20/10/5/2/1%):
     direct model vs model trained only inside the Stage-A gate; cluster-bootstrap CI of
     the AUC gain of every set over L1-only
  4  Explicit directional events (depletion, pulling, absorption): univariate direction AUC
  5  Entry timing: signed feature/price paths from -5 s to +5 s around confident signals
  6  Economics by notional with depth-based costs; break-even P(TP) per barrier pair
  7  Execution-aware selection on the selection days: pair models, EV gate, optional
     Stage-A gate -> frozen configs exported for the replay simulator
  8  Suggested classification (criteria stated; the final call is made in the report)
"""
from __future__ import annotations

import os as _os

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    _os.environ.setdefault(_var, "1")

import argparse
import json
import logging
import os
import time
from typing import Any

import numpy as np
import pandas as pd

from research.v2_models import LGBMModel, LogisticModel, auc, brier, ece, fit_calibrators
from v3.dataset import load
from v3.features import feature_group
from v3.labels import HORIZONS_S, PAIR_HORIZON_MS, PAIRS, XS, hold_ms, large, pair_outcome, tie, up_first

log = logging.getLogger("v3.research")

SETS = {"L1": {"l1"}, "L2": {"l2"}, "FLOW": {"flow"}, "L1+L2": {"l1", "l2"}, "ALL": {"l1", "l2", "flow", "l2flow"}}
TOP = (0.20, 0.10, 0.05, 0.02, 0.01)
BUCKETS = (1.0,) + TOP
META = {"symbol", "day", "ts_ms", "mid0", "entry_ask", "entry_bid", "quote_gap_ms", "book_age_ms", "flow_age_ms"}
LABEL_PREFIXES = ("t_up_", "t_dn_", "tp_", "sl_", "hret_", "mfe", "mae", "fret_", "cost_", "costm_")
PRIMARY = ((10, 10), (15, 30), (20, 60), (25, 60), (30, 60))
NOTIONALS = (100, 150, 250, 500, 1000)


# ====================================================================== columns / helpers
def feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in META and not c.startswith(LABEL_PREFIXES)
            and pd.api.types.is_numeric_dtype(df[c])]


def set_cols(cols: list[str], name: str) -> list[str]:
    g = SETS[name]
    return [c for c in cols if feature_group(c) in g]


def v2_cols(cols: list[str]) -> list[str]:
    return [c for c in cols if not c.startswith(("v3d_", "v3f_", "v3t_", "v3x_"))]


def mat(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    x = df[cols].to_numpy(np.float32, copy=True)
    x[~np.isfinite(x)] = np.nan
    return x


def _r(v, k=4):
    try:
        return None if v is None or not np.isfinite(v) else round(float(v), k)
    except TypeError:
        return None


def top_mask(score: np.ndarray, frac: float) -> np.ndarray:
    """Exactly the top ``frac`` of rows by score (deterministic random tie-break)."""
    k = max(int(round(len(score) * frac)), 1)
    jitter = np.random.default_rng(11).random(len(score)) * 1e-12
    order = np.lexsort((jitter, -np.asarray(score, float)))
    m = np.zeros(len(score), bool)
    m[order[:k]] = True
    return m


def fit(kind: str, x, y, xc, yc, threads: int):
    m = LGBMModel(threads) if kind == "lightgbm" else LogisticModel()
    m.fit(x, y, xc, yc)
    return m


def clusters(df: pd.DataFrame) -> np.ndarray:
    return pd.factorize(df["symbol"].astype(str) + "_" + (df["ts_ms"] // 60_000).astype(str))[0]


def boot_auc(y: np.ndarray, scores: dict[str, np.ndarray], cl: np.ndarray, base: str, B: int = 200,
             seed: int = 7) -> dict[str, dict]:
    """Cluster (symbol x minute) bootstrap of AUC and of the AUC gain over ``base``."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(cl)
    if len(uniq) < 10 or y.min() == y.max():
        return {}
    members = pd.Series(np.arange(len(cl))).groupby(cl).apply(np.asarray).to_dict()
    draws = {k: [] for k in scores}
    gains = {k: [] for k in scores if k != base}
    for _ in range(B):
        pick = rng.choice(uniq, len(uniq), replace=True)
        idx = np.concatenate([members[c] for c in pick])
        yy = y[idx]
        if yy.min() == yy.max():
            continue
        a = {k: auc(yy, s[idx]) for k, s in scores.items()}
        for k in scores:
            draws[k].append(a[k])
        for k in gains:
            gains[k].append(a[k] - a[base])
    out = {}
    for k in scores:
        d = np.array(draws[k])
        out[k] = {"auc": _r(auc(y, scores[k])), "ci": [_r(np.percentile(d, 2.5)), _r(np.percentile(d, 97.5))]}
        if k in gains:
            g = np.array(gains[k])
            out[k]["gain_vs_" + base] = _r(auc(y, scores[k]) - auc(y, scores[base]))
            out[k]["gain_ci"] = [_r(np.percentile(g, 2.5)), _r(np.percentile(g, 97.5))]
    return out


# ====================================================================== 1. Stage A
def stage_a(tr, ca, se, cols, X, H, threads, kind="lightgbm"):
    xtr, xca, xse = mat(tr, cols), mat(ca, cols), mat(se, cols)
    ytr, yca, yse = large(tr, X, H), large(ca, X, H), large(se, X, H)
    m = fit(kind, xtr, ytr, xca, yca, threads)
    p_tr, p_ca, p_se = m.predict(xtr), m.predict(xca), m.predict(xse)
    hit = {}
    for f in TOP:
        mm = top_mask(p_se, f)
        hit[f"{f * 100:g}%"] = _r(yse[mm].mean())
    return {"model": m, "cols": cols, "p_tr": p_tr, "p_ca": p_ca, "p_se": p_se,
            "metrics": {"X": X, "H": H, "base_rate": _r(yse.mean()), "auc": _r(auc(yse, p_se)), "top_hit": hit}}


# ====================================================================== 2/3. direction
def direction_models(tr, ca, se, cols_by_set, X, H, threads, kinds=("logistic", "lightgbm"), gate=None):
    """P(UP first | realised large move). gate=(p_tr, p_ca, frac): train only inside Stage-A's top frac."""
    pops = [(large(d, X, H) == 1) & ~tie(d, X) for d in (tr, ca, se)]
    if gate is not None:
        p_tr, p_ca, frac = gate
        thr = np.quantile(p_tr, 1 - frac)
        pops[0] = pops[0] & (p_tr >= thr)
        pops[1] = pops[1] & (p_ca >= np.quantile(p_ca, 1 - frac))
    if pops[0].sum() < 500 or pops[1].sum() < 100:
        return {}
    ytr, yca = up_first(tr[pops[0]], X), up_first(ca[pops[1]], X)
    out = {}
    for name, cols in cols_by_set.items():
        if not cols:
            continue
        xtr, xca = mat(tr[pops[0]], cols), mat(ca[pops[1]], cols)
        for kind in kinds:
            m = fit(kind, xtr, ytr, xca, yca, threads)
            out[(name, kind)] = {"model": m, "cols": cols, "score_se": m.predict(mat(se, cols))}
    return out


def direction_metrics(se, X, H, models, mask=None) -> dict[str, dict]:
    pop = (large(se, X, H) == 1) & ~tie(se, X)
    if mask is not None:
        pop = pop & mask
    y = up_first(se, X)[pop]
    res = {}
    if pop.sum() < 30 or y.min() == y.max():
        return res
    for (name, kind), m in models.items():
        s = m["score_se"][pop]
        res[f"{name}|{kind}"] = {"n": int(pop.sum()), "up_rate": _r(y.mean()), "auc": _r(auc(y, s)),
                                 "acc": _r(((s >= 0.5) == (y == 1)).mean()),
                                 "acc_by_rank": _r(((s >= np.median(s)) == (y == 1)).mean())}
    return res


def bucket_direction(se, X, H, pA, models, base_key) -> list[dict]:
    """Direction AUC/accuracy inside Stage A's top-k predicted large-move states (SEL)."""
    rows = []
    cl = clusters(se)
    for f in BUCKETS:
        mask = top_mask(pA, f)
        pop = mask & (large(se, X, H) == 1) & ~tie(se, X)
        y = up_first(se, X)[pop]
        r = {"top": f"{f * 100:g}%", "rows": int(mask.sum()), "move_rate": _r(large(se, X, H)[mask].mean()),
             "n_moves": int(pop.sum())}
        if pop.sum() >= 30 and y.min() != y.max():
            scores = {f"{k[0]}|{k[1]}": m["score_se"][pop] for k, m in models.items()}
            r["boot"] = boot_auc(y, scores, cl[pop], base_key) if base_key in scores else {}
            r["acc"] = {k: _r(((s >= 0.5) == (y == 1)).mean()) for k, s in scores.items()}
            r["majority"] = _r(max(y.mean(), 1 - y.mean()))
        rows.append(r)
    return rows


# ====================================================================== 4. events
EVENT_FEATURES = ("v3x_depletion_net", "v3x_pull_net", "v3x_absorb_net", "v3x_pressure_net", "v3d_can_asym_1s",
                  "v3d_pull_asym_1s", "v3d_dep_asym_1s", "v3f_sweepimb_3s", "v3f_largeimb_3s", "v3d_imb_5",
                  "v3d_wimb_20", "v3d_micro5_bps", "imb_l1", "ofi_3s", "flow_imb_3s", "mom_3s_bps")


def event_study(se, X=20, H=60) -> list[dict]:
    pop = (large(se, X, H) == 1) & ~tie(se, X)
    y = up_first(se, X)[pop]
    rows = []
    for c in EVENT_FEATURES:
        if c not in se or y.min() == y.max():
            continue
        v = se[c].to_numpy(float)[pop]
        ok = np.isfinite(v)
        if ok.sum() < 30:
            continue
        a = auc(y[ok], v[ok])
        hi = v[ok] >= np.quantile(v[ok], 0.9)
        lo = v[ok] <= np.quantile(v[ok], 0.1)
        rows.append({"feature": c, "group": feature_group(c), "dir_auc": _r(a), "up_rate_top_decile": _r(y[ok][hi].mean()),
                     "up_rate_bottom_decile": _r(y[ok][lo].mean())})
    return sorted(rows, key=lambda r: -abs((r["dir_auc"] or 0.5) - 0.5))


# ====================================================================== 5. entry timing
OFFSETS_MS = (-5000, -3000, -2000, -1000, -500, -250, 0, 500, 1000, 2000, 5000)


def entry_timing(data: str, se: pd.DataFrame, score: np.ndarray, frac: float = 0.05) -> dict[str, Any]:
    """Signed paths (in the predicted direction) around the most confident direction signals."""
    conf = np.abs(score - 0.5)
    m = top_mask(conf, frac)
    sig = se[m].copy()
    sig["dir"] = np.where(score[m] >= 0.5, 1.0, -1.0)
    tl_dir = os.path.join(data, "timeline")
    acc: dict[int, dict[str, list]] = {o: {} for o in OFFSETS_MS}
    for (sym, day), g in sig.groupby(["symbol", "day"]):
        path = os.path.join(tl_dir, f"{sym}_{day}.parquet")
        if not os.path.exists(path):
            continue
        tl = pd.read_parquet(path).sort_values("ts_ms")
        ts = tl["ts_ms"].to_numpy()
        for _, r in g.iterrows():
            t0, d = int(r["ts_ms"]), r["dir"]
            i0 = np.searchsorted(ts, t0 - 5000)
            if i0 >= len(ts) or abs(ts[i0] - (t0 - 5000)) > 300:
                continue
            ref = tl.iloc[i0]
            for o in OFFSETS_MS:
                i = np.searchsorted(ts, t0 + o)
                if i >= len(ts) or abs(ts[i] - (t0 + o)) > 300:
                    continue
                row = tl.iloc[i]
                a = acc[o]
                a.setdefault("mid_from_-5s_bps", []).append(d * (row["mid"] - ref["mid"]) / ref["mid"] * 1e4)
                for k in ("imb5", "wimb20", "micro5", "can_asym1s", "dep_asym1s", "flow_imb1s"):
                    # all oriented so that positive = bullish (ask-side cancels/depletion are bullish)
                    a.setdefault(k, []).append(d * row[k])
    table = []
    for o in OFFSETS_MS:
        r = {"offset_ms": o}
        for k, v in acc[o].items():
            r[k] = _r(np.mean(v), 3)
        r["n"] = len(acc[o].get("mid_from_-5s_bps", []))
        table.append(r)
    pre = next((r.get("mid_from_-5s_bps") for r in table if r["offset_ms"] == 0), None)
    post = next((r.get("mid_from_-5s_bps") for r in table if r["offset_ms"] == 5000), None)
    verdict, share = None, None
    if pre is not None and post is not None:
        if post <= 1.0:
            verdict = "no meaningful signed move around the signals (direction not predicted)"
        else:
            share = round(pre / post, 3)
            verdict = ("late: most of the signed move (-5 s..+5 s) happened BEFORE the signal" if share > 0.6 else
                       "early: most of the signed move happens AFTER the signal (pressure precedes price)"
                       if share < 0.4 else "mixed: the signal fires mid-move")
    return {"signals": int(len(sig)), "table": table, "move_before_entry_bps": pre,
            "move_by_+5s_bps": post, "share_of_move_before_signal": share, "verdict": verdict}


# ====================================================================== 6. economics
def economics(se: pd.DataFrame, min_net: float) -> list[dict]:
    rows = []
    for n in NOTIONALS:
        for mode, pre in (("taker", "cost"), ("maker", "costm")):
            c = 0.5 * (se[f"{pre}_long_{n}"].to_numpy(float) + se[f"{pre}_short_{n}"].to_numpy(float))
            c = c[np.isfinite(c)]
            if not len(c):
                continue
            med, p90 = float(np.median(c)), float(np.percentile(c, 90))
            for T, S in PAIRS:
                e = min_net * 1e4 / n
                rows.append({"notional": n, "entry": mode, "pair": f"{T}/{S}", "cost_bps_median": round(med, 2),
                             "cost_bps_p90": round(p90, 2), "p_breakeven": round((S + med) / (T + S), 4),
                             "p_for_min_net": round((e + S + med) / (T + S), 4)})
    return rows


# ====================================================================== 7. execution-aware selection
def pair_models(tr, ca, se, cols, threads, kind="lightgbm") -> dict:
    xtr, xca, xse = mat(tr, cols), mat(ca, cols), mat(se, cols)
    out = {}
    for T, S in PAIRS:
        for side in ("long", "short"):
            ytr, _ = pair_outcome(tr, side, T, S)
            yca, _ = pair_outcome(ca, side, T, S)
            yse, _ = pair_outcome(se, side, T, S)
            if ytr.sum() < 50 or yca.sum() < 10:
                continue
            m = fit(kind, xtr, ytr, xca, yca, threads)
            p_ca = m.predict(xca)
            cal = fit_calibrators(p_ca, yca).get("isotonic") or fit_calibrators(p_ca, yca)["raw"]
            p = cal(m.predict(xse))
            out[(side, T, S)] = {"model": m, "cal": cal, "p_se": p, "auc": _r(auc(yse, p)),
                                 "brier": _r(brier(yse, p), 5), "ece": _r(ece(yse, p)), "base": _r(yse.mean())}
    return out


def simulate(df, probs: dict, notional: int, thr: float, min_net: float, gate: np.ndarray | None = None):
    """EV-gated, best-EV pair/side per row, non-overlapping per symbol; realised from labels."""
    n = len(df)
    best = np.full(n, -np.inf)
    pick = np.full(n, -1)
    keys = list(probs)
    for j, (side, T, S) in enumerate(keys):
        p = probs[(side, T, S)]
        c = df[f"cost_{side}_{notional}"].to_numpy(float)
        ev = notional / 1e4 * (p * (T - c) - (1 - p) * (S + c))
        ok = (p >= thr) & (ev >= min_net) & np.isfinite(ev) & (ev > best) & (notional / 1e4 * (T - c) >= min_net)
        if gate is not None:
            ok &= gate
        best = np.where(ok, ev, best)
        pick = np.where(ok, j, pick)
    cand = np.where(pick >= 0)[0]
    sym, ts = df["symbol"].to_numpy(), df["ts_ms"].to_numpy()
    busy: dict[str, int] = {}
    taken = []
    for i in cand:
        if ts[i] < busy.get(sym[i], -1):
            continue
        side, T, S = keys[pick[i]]
        busy[sym[i]] = ts[i] + int(hold_ms(df.iloc[[i]], side, T, S)[0]) + 100
        taken.append(i)
    if not taken:
        return {"trades": 0}, pd.DataFrame()
    t = df.iloc[taken][["symbol", "day", "ts_ms"]].copy()
    nets, tps, holds, costs, ps = [], [], [], [], []
    for i in taken:
        side, T, S = keys[pick[i]]
        row = df.iloc[[i]]
        win, gross = pair_outcome(row, side, T, S)
        c = float(row[f"cost_{side}_{notional}"].iat[0])
        nets.append((gross[0] - c) * notional / 1e4)
        tps.append(win[0])
        holds.append(hold_ms(row, side, T, S)[0])
        costs.append(c * notional / 1e4)
        ps.append(probs[(side, T, S)][i])
    t["net"] = nets
    t["tp"] = tps
    t["p"] = ps
    t["side"] = [keys[pick[i]][0] for i in taken]
    t["pair"] = [f"{keys[pick[i]][1]}/{keys[pick[i]][2]}" for i in taken]
    return stats(np.array(nets), np.array(tps), np.array(holds), np.array(costs), t), t


def stats(net, tp, hold, cost, t) -> dict:
    n = len(net)
    cum = np.cumsum(net)
    dd = float(np.max(np.maximum.accumulate(np.concatenate([[0], cum])) - np.concatenate([[0], cum])))
    sd = float(np.std(net, ddof=1)) if n > 1 else 0.0
    wins, losses = net[net > 0], net[net <= 0]
    by_day = t.groupby("day")["net"].agg(["count", "mean"]).round(4).reset_index().to_dict("records")
    return {"trades": n, "net_pnl": _r(net.sum()), "expectancy": _r(net.mean(), 5),
            "t_stat": _r(net.mean() / (sd / np.sqrt(n)), 2) if sd > 0 else None, "win_rate": _r((net > 0).mean()),
            "profit_factor": _r(wins.sum() / -losses.sum(), 3) if losses.sum() < 0 else None,
            "max_drawdown": _r(dd), "tp_rate": _r(tp.mean()), "avg_hold_s": _r(hold.mean() / 1000, 1),
            "cost_per_trade": _r(cost.mean()), "long_share": _r((t["side"] == "long").mean()) if "side" in t else None,
            "by_day": by_day}


def select_trading(se, pm: dict, pA_se, pA_ca, notional, min_net, min_trades=30) -> dict:
    grid, best = [], None
    gates = {"none": None}
    for f in (0.20, 0.10, 0.05):
        gates[f"stageA_top{f * 100:g}%"] = pA_se >= np.quantile(pA_ca, 1 - f)
    pair_sets = {"dynamic": list(PAIRS), **{f"{T}/{S}": [(T, S)] for T, S in PAIRS}}
    for gname, gate in gates.items():
        for pname, pairs in pair_sets.items():
            probs = {k: v["p_se"] for k, v in pm.items() if (k[1], k[2]) in pairs}
            if not probs:
                continue
            for thr in (0.0, 0.4, 0.5, 0.6, 0.7):
                st, _ = simulate(se, probs, notional, thr, min_net, gate)
                row = {"gate": gname, "pairs": pname, "threshold": thr, **{k: v for k, v in st.items() if k != "by_day"}}
                grid.append(row)
                key = (st.get("trades", 0) >= min_trades, st.get("expectancy") or -1e9)
                if best is None or key > best[0]:
                    best = (key, row, gname, pairs, thr)
    return {"grid": grid, "selected": best[1], "gate": best[2], "pairs": best[3], "threshold": best[4]}


def export(out_dir: str, name: str, cols: list[str], pm: dict, sel: dict, stage: dict | None, pA_ca, notional, min_net):
    d = os.path.join(out_dir, name)
    os.makedirs(d, exist_ok=True)
    spec = {"kind": "v3", "name": name, "features": cols, "notional": notional, "min_net_usdt": min_net,
            "threshold": sel["threshold"], "pairs": [list(p) for p in sel["pairs"]], "horizon_s": PAIR_HORIZON_MS / 1000,
            "entry": "taker", "models": {}, "stage_a": None, "selected_on": "selection days",
            "sel_metrics": sel["selected"]}
    for (side, T, S), m in pm.items():
        if (T, S) not in sel["pairs"]:
            continue
        key = f"{side}_{T}_{S}"
        m["model"].b.save_model(os.path.join(d, f"{key}.txt"), num_iteration=m["model"].b.best_iteration)
        c = m["cal"]
        spec["models"][key] = {"file": f"{key}.txt", "calibrator": {"method": c.method, "a": c.a, "b": c.b,
                                                                     "xs": c.xs, "ys": c.ys}}
    if sel["gate"] != "none" and stage is not None:
        frac = float(sel["gate"].split("top")[1].rstrip("%")) / 100
        stage["model"].b.save_model(os.path.join(d, "stage_a.txt"), num_iteration=stage["model"].b.best_iteration)
        spec["stage_a"] = {"file": "stage_a.txt", "features": stage["cols"], "X": stage["metrics"]["X"],
                           "H": stage["metrics"]["H"], "threshold": float(np.quantile(pA_ca, 1 - frac))}
    with open(os.path.join(d, "spec.json"), "w", encoding="utf-8") as fh:
        json.dump(spec, fh)
    return d


# ====================================================================== 8. classification
def suggest_classification(rep: dict) -> dict:
    """Mechanical suggestion from SELECTION-day evidence (the report makes the final call
    after the locked test replay)."""
    best_gain, best_acc = None, None
    for key, rows in rep.get("top_buckets", {}).items():
        for r in rows:
            for name, b in (r.get("boot") or {}).items():
                if name.startswith("ALL|") and b.get("gain_ci"):
                    if best_gain is None or b["gain_vs_L1|lightgbm"] > best_gain[0]:
                        best_gain = (b["gain_vs_L1|lightgbm"], b["gain_ci"], key, r["top"], b["auc"])
            for name, a in (r.get("acc") or {}).items():
                if name.startswith("ALL|") and a is not None and (best_acc is None or a > best_acc[0]):
                    best_acc = (a, key, r["top"])
    material = bool(best_gain and best_gain[0] >= 0.03 and best_gain[1][0] is not None and best_gain[1][0] > 0)
    sel = {k: v["selected"] for k, v in rep.get("trading", {}).items()}
    positive = any((s.get("expectancy") or -1) > 0 and (s.get("t_stat") or 0) >= 2 and s.get("trades", 0) >= 100
                   for s in sel.values())
    acc_ok = bool(best_acc and best_acc[0] >= 0.56)
    if material and acc_ok and positive:
        c = "1 STRONG L2 DIRECTIONAL EDGE (pending locked-test replay)"
    elif material and acc_ok:
        c = "2/3 L2 IMPROVES DIRECTION -- decide after costs on the locked test"
    elif material:
        c = "3 L2 IMPROVES DIRECTION BUT NOT ENOUGH AFTER COSTS (likely)"
    else:
        c = "4 NO USEFUL DIRECTIONAL EDGE (L2 does not materially beat L1)"
    return {"suggested": c, "best_auc_gain_vs_L1": best_gain, "best_direction_accuracy": best_acc,
            "criteria": "material = AUC gain over L1 >= 0.03 with 95% cluster-bootstrap CI > 0 inside a Stage-A "
                        "top bucket; direction accuracy >= 0.56; selection-day expectancy > 0 with t >= 2 and "
                        ">= 100 trades. Direction 50-55% => the strategy class is unlikely to work."}


# ====================================================================== main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--train-days", nargs="+", required=True)
    ap.add_argument("--cal-days", nargs="+", required=True)
    ap.add_argument("--sel-days", nargs="+", required=True)
    ap.add_argument("--forbid-days", nargs="+", default=[])
    ap.add_argument("--symbols", nargs="+")
    ap.add_argument("--notional", type=int, default=150)
    ap.add_argument("--min-net", type=float, default=0.10)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--max-train-rows", type=int, default=1_500_000)
    ap.add_argument("--export", default="models/v3")
    ap.add_argument("--min-trades", type=int, default=30)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    days = a.train_days + a.cal_days + a.sel_days
    bad = sorted(set(days) & set(a.forbid_days))
    if bad:
        raise SystemExit(f"refusing to use TEST days {bad} in research")
    if not (max(a.train_days) < min(a.cal_days) and max(a.cal_days) < min(a.sel_days)):
        raise SystemExit("days must be chronological: train < calibration < selection")
    os.makedirs(a.out, exist_ok=True)
    t0 = time.perf_counter()
    tr = load(a.data, a.train_days, a.symbols)
    if len(tr) > a.max_train_rows:
        tr = tr.sample(a.max_train_rows, random_state=7).sort_values(["symbol", "ts_ms"]).reset_index(drop=True)
    ca = load(a.data, a.cal_days, a.symbols)
    se = load(a.data, a.sel_days, a.symbols)
    cols = feature_cols(tr)
    by_set = {k: set_cols(cols, k) for k in SETS}
    v2c = v2_cols(cols)
    rep: dict[str, Any] = {"split": {"train": a.train_days, "calibration": a.cal_days, "selection": a.sel_days,
                                     "test": f"UNTOUCHED ({', '.join(a.forbid_days)})"},
                           "rows": {"train": len(tr), "cal": len(ca), "sel": len(se)},
                           "feature_sets": {k: len(v) for k, v in by_set.items()}, "notional": a.notional}
    log.info("rows %s, sets %s (%.0fs)", rep["rows"], rep["feature_sets"], time.perf_counter() - t0)

    def save():
        with open(os.path.join(a.out, "v3_research.json"), "w", encoding="utf-8") as fh:
            json.dump(rep, fh, indent=1, default=str)

    # 1. Stage A
    A, A3 = {}, {}
    rep["stage_a_v2def"], rep["stage_a_all"] = [], []
    for X in XS:
        for H in HORIZONS_S:
            A[(X, H)] = stage_a(tr, ca, se, v2c, X, H, a.threads)
            rep["stage_a_v2def"].append(A[(X, H)]["metrics"])
        A3[(X, 60)] = stage_a(tr, ca, se, by_set["ALL"], X, 60, a.threads)
        rep["stage_a_all"].append(A3[(X, 60)]["metrics"])
        log.info("Stage A X=%d: v2def auc(60s)=%s all=%s", X, A[(X, 60)]["metrics"]["auc"], A3[(X, 60)]["metrics"]["auc"])
    save()

    # 2/3. direction: direct and Stage-A-gated, per feature set
    rep["direction"], rep["top_buckets"], rep["top_buckets_gated"] = {}, {}, {}
    D_keep = {}
    for X, H in PRIMARY:
        key = f"X{X}_H{H}"
        D = direction_models(tr, ca, se, by_set, X, H, a.threads)
        if not D:
            continue
        D_keep[(X, H)] = D
        rep["direction"][key] = direction_metrics(se, X, H, D)
        pA = A[(X, H)]["p_se"]
        rep["top_buckets"][key] = bucket_direction(se, X, H, pA, D, "L1|lightgbm")
        G = direction_models(tr, ca, se, {k: by_set[k] for k in ("L1", "ALL")}, X, H, a.threads, ("lightgbm",),
                             gate=(A[(X, H)]["p_tr"], A[(X, H)]["p_ca"], 0.20))
        if G:
            rep["top_buckets_gated"][key] = bucket_direction(se, X, H, pA, G, "L1|lightgbm")
        log.info("direction %s: %s", key, {k: v["auc"] for k, v in rep["direction"][key].items()})
        save()

    # 4. explicit events, 5. timing
    rep["events"] = event_study(se)
    if (20, 60) in D_keep:
        rep["entry_timing"] = entry_timing(a.data, se, D_keep[(20, 60)][("ALL", "lightgbm")]["score_se"])
    # 6. economics
    rep["economics"] = economics(se, a.min_net)
    save()

    # 7. execution-aware selection + export
    rep["trading"], rep["exported"] = {}, {}
    gate_stage = A[(20, 60)]
    for name in ("L1", "ALL"):
        pm = pair_models(tr, ca, se, by_set[name], a.threads)
        sel = select_trading(se, pm, gate_stage["p_se"], gate_stage["p_ca"], a.notional, a.min_net, a.min_trades)
        _, trades = simulate(se, {k: v["p_se"] for k, v in pm.items() if (k[1], k[2]) in sel["pairs"]},
                             a.notional, sel["threshold"], a.min_net,
                             None if sel["gate"] == "none" else
                             gate_stage["p_se"] >= np.quantile(gate_stage["p_ca"],
                                                               1 - float(sel["gate"].split("top")[1].rstrip("%")) / 100))
        rep["trading"][name] = {"pair_models": {f"{k[0]}_{k[1]}_{k[2]}": {kk: vv for kk, vv in v.items()
                                                                        if kk in ("auc", "brier", "ece", "base")}
                                                for k, v in pm.items()},
                                "selected": sel["selected"], "grid": sel["grid"],
                                "per_symbol": trades.groupby("symbol")["net"].agg(["count", "mean"]).round(4)
                                .reset_index().to_dict("records") if len(trades) else []}
        rep["exported"][name] = export(a.export, f"v3_{name.lower().replace('+', '_')}", by_set[name], pm, sel,
                                       gate_stage, gate_stage["p_ca"], a.notional, a.min_net)
        log.info("trading %s selected: %s", name, sel["selected"])
        save()

    rep["classification"] = suggest_classification(rep)
    rep["seconds"] = round(time.perf_counter() - t0)
    save()
    write_markdown(rep, os.path.join(a.out, "V3_RESEARCH.md"))
    print(f"report: {os.path.join(a.out, 'V3_RESEARCH.md')}")


def write_markdown(rep: dict, path: str) -> None:
    L = ["# V3 research (TRAIN / CALIBRATION / SELECTION only; TEST untouched)\n",
         f"Split: {json.dumps(rep['split'])}  \nRows: {rep['rows']}  \nFeature sets: {rep['feature_sets']}\n",
         f"## Suggested classification\n\n**{rep['classification']['suggested']}**\n\n{rep['classification']['criteria']}\n",
         "## Stage A (V2 L1 definition) — AUC / top-1% hit\n",
         pd.DataFrame([{"X": m["X"], "H": m["H"], "base": m["base_rate"], "auc": m["auc"], "top1%": m["top_hit"]["1%"]}
                       for m in rep["stage_a_v2def"]]).to_string(index=False), "",
         "## Direction AUC on all realised large moves (selection days)\n"]
    for key, d in rep["direction"].items():
        L.append(f"**{key}**  " + ", ".join(f"{k}: {v['auc']} (acc {v['acc']})" for k, v in d.items()))
    L.append("\n## Direction inside Stage A's top buckets (direct models)\n")
    for key, rows in rep["top_buckets"].items():
        L.append(f"**{key}**")
        for r in rows:
            if r.get("boot"):
                L.append(f"- top {r['top']}: moves {r['n_moves']} (move rate {r['move_rate']}), majority {r['majority']}; "
                         + "; ".join(f"{k} AUC {b['auc']} {b['ci']}" + (f" gain {b.get('gain_vs_L1|lightgbm')} {b.get('gain_ci')}"
                                                                         if b.get("gain_ci") else "")
                                     for k, b in r["boot"].items() if k.endswith("lightgbm")))
    L.append("\n## Explicit directional events (direction AUC, X=20 H=60)\n")
    L.append(pd.DataFrame(rep["events"]).to_string(index=False) if rep["events"] else "_none_")
    if rep.get("entry_timing"):
        L.append("\n## Entry timing (signed in the predicted direction)\n")
        L.append(pd.DataFrame(rep["entry_timing"]["table"]).to_string(index=False))
        L.append(f"\n{rep['entry_timing']['verdict']}\n")
    L.append("\n## Execution-aware selection (selection days, offline)\n")
    for k, v in rep["trading"].items():
        L.append(f"**{k}**: {json.dumps(v['selected'])}")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
