"""V2 model research: direct P(TP before SL) classifiers, calibration, EV-gated evaluation.

    python -m research.v2_models --data data/v2/ds --out models/v2 \\
        --train-days 2024-03-21 ... 2024-03-26 --cal-day 2024-03-27 --sel-day 2024-03-28

Chronological roles (no shuffling, no test data):
  TRAIN days      model fitting
  CAL day (val A) early stopping + probability calibration (Platt and isotonic)
  SEL day (val B) model / calibration-method / threshold selection and all reported
                  validation metrics -- never used for fitting anything
Models (deliberately small): logistic regression, LightGBM, XGBoost, MLP(64, 32).
Variants: separate LONG/SHORT models (default) vs one model with a direction feature;
global vs per-symbol (LightGBM, primary target).

Offline trade evaluation on SEL mirrors the V2 entry gate:
  calibrated p >= threshold  AND  EV = N/1e4 * (p*(T - c) - (1-p)*(S + c)) >= min_net
  dynamic target = argmax EV over targets meeting the gate; one position per symbol at a
  time; realised outcome from the barrier labels (TP -> T - c, SL -> -(S + c),
  timeout -> executable return at the horizon - c). The full replay is the final arbiter.
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
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from research.v2_dataset import STOPS_BPS, TARGETS_BPS, load_dataset, tp_first

log = logging.getLogger("v2models")

META_PREFIXES = ("ttp_", "tsl_", "mfe_", "mae_", "hret_", "fret_", "cost_bps_", "req_bps_")
META_COLS = {"symbol", "day", "ts_ms", "bid0", "ask0", "mid0", "entry_lag_ms", "book_age_ms", "flow_age_ms"}
PROB_BUCKETS = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 1.0001]
BUCKET_LABELS = ["50-55%", "55-60%", "60-65%", "65-70%", "70-75%", "75-80%", "80-85%", "85-90%", ">90%"]
SIDES = ("long", "short")


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in META_COLS and not c.startswith(META_PREFIXES)
            and pd.api.types.is_numeric_dtype(df[c])]


def matrix(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    x = df[cols].to_numpy(np.float32, copy=True)
    x[~np.isfinite(x)] = np.nan
    return x


# ====================================================================== models
class Model:
    kind = "base"

    def fit(self, x, y, x_cal, y_cal) -> None: ...

    def predict(self, x) -> np.ndarray: ...


class LogisticModel(Model):
    kind = "logistic"

    def fit(self, x, y, x_cal, y_cal) -> None:
        from sklearn.linear_model import LogisticRegression

        self.med = np.nanmedian(x, axis=0)
        self.med[~np.isfinite(self.med)] = 0.0
        z = self._prep(x, fit=True)
        self.m = LogisticRegression(C=0.05, max_iter=400)
        self.m.fit(z, y)

    def _prep(self, x, fit=False):
        z = np.where(np.isnan(x), self.med, x)
        if fit:
            self.mu = z.mean(0)
            self.sd = z.std(0)
            self.sd[self.sd < 1e-9] = 1.0
        return np.clip((z - self.mu) / self.sd, -8, 8)

    def predict(self, x) -> np.ndarray:
        return self.m.predict_proba(self._prep(x))[:, 1]

    def export(self) -> dict:
        return {"med": self.med.tolist(), "mu": self.mu.tolist(), "sd": self.sd.tolist(),
                "coef": self.m.coef_[0].tolist(), "intercept": float(self.m.intercept_[0])}


class LGBMModel(Model):
    kind = "lightgbm"

    def __init__(self, threads: int = 4) -> None:
        self.threads = threads

    def fit(self, x, y, x_cal, y_cal) -> None:
        import lightgbm as lgb

        params = dict(objective="binary", learning_rate=0.05, num_leaves=15, max_depth=4,
                      min_data_in_leaf=300, feature_fraction=0.5, bagging_fraction=0.8, bagging_freq=1,
                      lambda_l2=5.0, verbose=-1, num_threads=self.threads, seed=7)
        dtr = lgb.Dataset(x, y)
        dva = lgb.Dataset(x_cal, y_cal, reference=dtr)
        self.b = lgb.train(params, dtr, num_boost_round=400, valid_sets=[dva],
                           callbacks=[lgb.early_stopping(40, verbose=False)])

    def predict(self, x) -> np.ndarray:
        return self.b.predict(x, num_iteration=self.b.best_iteration)


class XGBModel(Model):
    kind = "xgboost"

    def __init__(self, threads: int = 4) -> None:
        self.threads = threads

    def fit(self, x, y, x_cal, y_cal) -> None:
        import xgboost as xgb

        self.m = xgb.XGBClassifier(n_estimators=400, learning_rate=0.05, max_depth=4, min_child_weight=50,
                                   subsample=0.8, colsample_bytree=0.5, reg_lambda=5.0, tree_method="hist",
                                   n_jobs=self.threads, early_stopping_rounds=40, eval_metric="logloss",
                                   random_state=7)
        self.m.fit(x, y, eval_set=[(x_cal, y_cal)], verbose=False)

    def predict(self, x) -> np.ndarray:
        return self.m.predict_proba(x)[:, 1]


class MLPModel(Model):
    kind = "mlp"

    def fit(self, x, y, x_cal, y_cal) -> None:
        from sklearn.neural_network import MLPClassifier

        self.med = np.nanmedian(x, axis=0)
        self.med[~np.isfinite(self.med)] = 0.0
        z = np.where(np.isnan(x), self.med, x)
        self.mu = z.mean(0)
        self.sd = z.std(0)
        self.sd[self.sd < 1e-9] = 1.0
        self.m = MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-3, batch_size=1024, learning_rate_init=1e-3,
                               max_iter=30, early_stopping=True, validation_fraction=0.1, n_iter_no_change=4,
                               random_state=7)
        self.m.fit(self._prep(x), y)

    def _prep(self, x):
        z = np.where(np.isnan(x), self.med, x)
        return np.clip((z - self.mu) / self.sd, -8, 8)

    def predict(self, x) -> np.ndarray:
        return self.m.predict_proba(self._prep(x))[:, 1]

    def export(self) -> dict:
        return {"med": self.med.tolist(), "mu": self.mu.tolist(), "sd": self.sd.tolist(),
                "W": [w.tolist() for w in self.m.coefs_], "b": [b.tolist() for b in self.m.intercepts_]}


MODEL_FACTORIES = {"logistic": LogisticModel, "lightgbm": LGBMModel, "xgboost": XGBModel, "mlp": MLPModel}


# ====================================================================== calibration
@dataclass
class Calibrator:
    method: str
    a: float = 1.0
    b: float = 0.0
    xs: list | None = None
    ys: list | None = None

    def __call__(self, p: np.ndarray) -> np.ndarray:
        p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
        if self.method == "platt":
            z = self.a * np.log(p / (1 - p)) + self.b
            return 1 / (1 + np.exp(-z))
        if self.method == "isotonic":
            return np.interp(p, self.xs, self.ys)
        return p


def fit_calibrators(p_cal: np.ndarray, y_cal: np.ndarray) -> dict[str, Calibrator]:
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression

    out = {"raw": Calibrator("raw")}
    lp = np.log(np.clip(p_cal, 1e-6, 1 - 1e-6) / (1 - np.clip(p_cal, 1e-6, 1 - 1e-6))).reshape(-1, 1)
    if len(np.unique(y_cal)) == 2:
        lr = LogisticRegression(C=1e6, max_iter=200).fit(lp, y_cal)
        out["platt"] = Calibrator("platt", float(lr.coef_[0, 0]), float(lr.intercept_[0]))
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(p_cal, y_cal)
        out["isotonic"] = Calibrator("isotonic", xs=iso.X_thresholds_.tolist(), ys=iso.y_thresholds_.tolist())
    return out


# ====================================================================== metrics
def auc(y: np.ndarray, s: np.ndarray) -> float:
    y = np.asarray(y)
    if y.min() == y.max():
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s))
    ranks[order] = np.arange(1, len(s) + 1)
    n1 = y.sum()
    n0 = len(y) - n1
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def brier(y, p) -> float:
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def ece(y, p, bins=10) -> float:
    y, p = np.asarray(y), np.asarray(p)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    e = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(e)


def realised_bps(df: pd.DataFrame, side: str, target: int, stop: int, cost_bps: np.ndarray,
                 horizon_ms: int = 60_000) -> np.ndarray:
    """Net bps of a barrier trade: TP -> T - c, SL -> -(S + c), timeout -> exec return - c."""
    ttp = df[f"ttp_{side}_{target}"].to_numpy()
    tsl = df[f"tsl_{side}_{stop}"].to_numpy()
    tp = (ttp < tsl) & (ttp <= horizon_ms)
    sl = (~tp) & (tsl <= horizon_ms)
    hret = df[f"hret_{side}"].to_numpy(float)
    gross = np.where(tp, target, np.where(sl, -stop, np.nan_to_num(hret)))
    return gross - cost_bps


def hold_ms(df, side, target, stop, horizon_ms=60_000) -> np.ndarray:
    ttp = df[f"ttp_{side}_{target}"].to_numpy().astype(float)
    tsl = df[f"tsl_{side}_{stop}"].to_numpy().astype(float)
    return np.minimum(np.minimum(ttp, tsl), horizon_ms)


def trade_stats(net_usdt: np.ndarray, costs_usdt: np.ndarray, hold: np.ndarray, tp: np.ndarray) -> dict[str, Any]:
    n = len(net_usdt)
    if n == 0:
        return {"trades": 0}
    wins = net_usdt[net_usdt > 0]
    losses = net_usdt[net_usdt <= 0]
    cum = np.cumsum(net_usdt)
    dd = float(np.max(np.maximum.accumulate(np.concatenate([[0], cum])) - np.concatenate([[0], cum])))
    sd = float(np.std(net_usdt, ddof=1)) if n > 1 else 0.0
    return {
        "trades": n, "net_pnl": round(float(net_usdt.sum()), 4), "expectancy": round(float(net_usdt.mean()), 5),
        "t_stat": round(float(net_usdt.mean() / (sd / np.sqrt(n))), 2) if sd > 0 else None,
        "win_rate": round(float((net_usdt > 0).mean()), 4),
        "profit_factor": round(float(wins.sum() / -losses.sum()), 3) if losses.sum() < 0 else None,
        "avg_winner": round(float(wins.mean()), 4) if len(wins) else None,
        "avg_loser": round(float(losses.mean()), 4) if len(losses) else None,
        "median_trade": round(float(np.median(net_usdt)), 5), "max_drawdown": round(dd, 4),
        "tp_before_sl_rate": round(float(tp.mean()), 4), "avg_hold_s": round(float(hold.mean() / 1000), 2),
        "costs_per_trade": round(float(costs_usdt.mean()), 4),
    }


# ====================================================================== EV-gated simulation
def simulate(df: pd.DataFrame, probs: dict[tuple[str, int], np.ndarray], stop: int, notional: float,
             threshold: float, min_net: float, horizon_ms: int = 60_000) -> tuple[dict, pd.DataFrame]:
    """One decision per row; dynamic target/side by best EV; non-overlapping per symbol."""
    n = len(df)
    best_ev = np.full(n, -np.inf)
    best_side = np.full(n, "", dtype=object)
    best_t = np.zeros(n, int)
    best_p = np.zeros(n)
    for (side, T), p in probs.items():
        c = df[f"cost_bps_{side}_{int(notional)}"].to_numpy(float)
        ev = notional / 1e4 * (p * (T - c) - (1 - p) * (stop + c))
        ok = (p >= threshold) & (ev >= min_net) & np.isfinite(ev) & (ev > best_ev)
        best_ev = np.where(ok, ev, best_ev)
        best_side = np.where(ok, side, best_side)
        best_t = np.where(ok, T, best_t)
        best_p = np.where(ok, p, best_p)
    cand = np.where(best_side != "")[0]
    taken = []
    busy_until: dict[str, int] = {}
    sym = df["symbol"].to_numpy()
    ts = df["ts_ms"].to_numpy()
    for i in cand:                                      # rows are time-ordered within symbol
        if ts[i] < busy_until.get(sym[i], -1):
            continue
        side, T = best_side[i], best_t[i]
        h = hold_ms(df.iloc[[i]], side, T, stop, horizon_ms)[0]
        busy_until[sym[i]] = ts[i] + int(h) + 100
        taken.append(i)
    if not taken:
        return {"trades": 0}, pd.DataFrame()
    t = df.iloc[taken].copy()
    t["side"] = best_side[taken]
    t["target"] = best_t[taken]
    t["p"] = best_p[taken]
    t["ev"] = best_ev[taken]
    net_bps = np.array([realised_bps(t.iloc[[k]], t["side"].iat[k], int(t["target"].iat[k]), stop,
                                     t[f"cost_bps_{t['side'].iat[k]}_{int(notional)}"].to_numpy(float))[0]
                        for k in range(len(t))])
    cost_bps = np.array([t[f"cost_bps_{s}_{int(notional)}"].iat[k] for k, s in enumerate(t["side"])])
    hold = np.array([hold_ms(t.iloc[[k]], s, int(T), stop, horizon_ms)[0]
                     for k, (s, T) in enumerate(zip(t["side"], t["target"]))])
    tp = np.array([tp_first(t.iloc[[k]], s, int(T), stop, horizon_ms)[0] for k, (s, T) in
                   enumerate(zip(t["side"], t["target"]))])
    t["net_usdt"] = net_bps * notional / 1e4
    t["tp"] = tp
    t["hold_ms"] = hold
    st = trade_stats(t["net_usdt"].to_numpy(), cost_bps * notional / 1e4, hold, tp)
    st["long_share"] = round(float((t["side"] == "long").mean()), 3)
    st["targets_used"] = t["target"].value_counts().sort_index().to_dict()
    return st, t


def calibration_table(y: np.ndarray, p: np.ndarray, net_usdt: np.ndarray) -> list[dict]:
    rows = []
    b = pd.cut(p, PROB_BUCKETS, labels=BUCKET_LABELS, right=False)
    for lab in BUCKET_LABELS:
        m = np.asarray(b == lab)
        if not m.any():
            rows.append({"bucket": lab, "n": 0})
            continue
        rows.append({"bucket": lab, "n": int(m.sum()), "predicted": round(float(p[m].mean()), 4),
                     "actual_tp_rate": round(float(y[m].mean()), 4),
                     "calibration_error": round(float(p[m].mean() - y[m].mean()), 4),
                     "avg_net_usdt_if_traded": round(float(np.nanmean(net_usdt[m])), 4)})
    return rows


# ====================================================================== orchestration
def run(data: str, out: str, train_days: list[str], cal_day: str, sel_day: str, stop: int, notional: float,
        min_net: float, kinds: list[str], targets: list[int], mlp_targets: list[int], threads: int,
        thresholds: list[float]) -> dict[str, Any]:
    os.makedirs(out, exist_ok=True)
    t0 = time.perf_counter()
    tr = load_dataset(data, train_days)
    ca = load_dataset(data, [cal_day])
    se = load_dataset(data, [sel_day])
    cols = feature_columns(tr)
    xtr, xca, xse = matrix(tr, cols), matrix(ca, cols), matrix(se, cols)
    log.info("rows train=%d cal=%d sel=%d features=%d (%.0fs load)", len(tr), len(ca), len(se), len(cols),
             time.perf_counter() - t0)
    report: dict[str, Any] = {"train_days": train_days, "cal_day": cal_day, "sel_day": sel_day, "stop_bps": stop,
                              "notional": notional, "min_net_usdt": min_net, "n_features": len(cols),
                              "rows": {"train": len(tr), "cal": len(ca), "sel": len(se)}, "models": {}}
    base_rates = {f"{s}_{T}": round(float(tp_first(se, s, T, stop).mean()), 4) for s in SIDES for T in targets}
    report["sel_base_rates"] = base_rates
    artifacts: dict[str, dict] = {}
    for kind in kinds:
        tlist = mlp_targets if kind == "mlp" else targets
        probs_cal: dict[tuple[str, int], dict[str, np.ndarray]] = {}
        per_target = []
        arts = {}
        tk = time.perf_counter()
        for side in SIDES:
            for T in tlist:
                ytr, yca, yse = (tp_first(d, side, T, stop) for d in (tr, ca, se))
                m = MODEL_FACTORIES[kind](threads) if kind in ("lightgbm", "xgboost") else MODEL_FACTORIES[kind]()
                m.fit(xtr, ytr, xca, yca)
                p_ca, p_se = m.predict(xca), m.predict(xse)
                cals = fit_calibrators(p_ca, yca)
                probs_cal[(side, T)] = {k: c(p_se) for k, c in cals.items()}
                net = realised_bps(se, side, T, stop, se[f"cost_bps_{side}_{int(notional)}"].to_numpy(float)) \
                    * notional / 1e4
                row = {"side": side, "target": T, "base_rate": round(float(yse.mean()), 4),
                       "auc": round(auc(yse, p_se), 4)}
                for k, pc in probs_cal[(side, T)].items():
                    row[f"brier_{k}"] = round(brier(yse, pc), 5)
                    row[f"ece_{k}"] = round(ece(yse, pc), 4)
                per_target.append(row)
                arts[(side, T)] = (m, cals, net, yse)
        best = None
        sims = {}
        for method in ("raw", "platt", "isotonic"):
            probs = {k: v[method] for k, v in probs_cal.items() if method in v}
            for thr in thresholds:
                st, _ = simulate(se, probs, stop, notional, thr, min_net)
                sims[f"{method}@{thr}"] = st
                key = (st.get("trades", 0) >= 20, st.get("expectancy") or -1e9)
                if best is None or key > best[0]:
                    best = (key, method, thr, st)
        _, method, thr, st = best
        primary = 16 if 16 in tlist else tlist[len(tlist) // 2]
        cal_tables = {}
        for side in SIDES:
            for T in sorted({primary, *tlist[:1]}):
                m, cals, net, yse = arts[(side, T)]
                cal_tables[f"{side}_{T}"] = calibration_table(yse, probs_cal[(side, T)][method], net)
        report["models"][kind] = {"per_target": per_target, "simulations": sims,
                                  "selected": {"calibration": method, "threshold": thr, "sel_metrics": st},
                                  "calibration_tables": cal_tables, "train_seconds": round(time.perf_counter() - tk, 1)}
        artifacts[kind] = {"arts": arts, "method": method, "threshold": thr}
        save_model(out, kind, cols, arts, method, thr, stop, notional, min_net, tlist)
        log.info("%s: selected %s@%.2f -> %s", kind, method, thr, st)
    report["_artifacts"] = artifacts
    return report


def save_model(out: str, kind: str, cols: list[str], arts: dict, method: str, thr: float, stop: int,
               notional: float, min_net: float, targets: list[int]) -> None:
    d = os.path.join(out, kind)
    os.makedirs(d, exist_ok=True)
    spec = {"kind": kind, "features": cols, "stop_bps": stop, "notional": notional, "min_net_usdt": min_net,
            "threshold": thr, "calibration": method, "targets": targets, "sides": list(SIDES), "models": {}}
    for (side, T), (m, cals, _, _) in arts.items():
        key = f"{side}_{T}"
        c = cals.get(method, cals["raw"])
        entry = {"calibrator": {"method": c.method, "a": c.a, "b": c.b, "xs": c.xs, "ys": c.ys}}
        if kind == "lightgbm":
            path = os.path.join(d, f"{key}.txt")
            m.b.save_model(path, num_iteration=m.b.best_iteration)
            entry["file"] = os.path.basename(path)
        elif kind == "xgboost":
            path = os.path.join(d, f"{key}.json")
            m.m.get_booster().save_model(path)
            entry["file"] = os.path.basename(path)
            entry["best_iteration"] = int(getattr(m.m, "best_iteration", 0) or 0)
        else:
            entry["params"] = m.export()
        spec["models"][key] = entry
    with open(os.path.join(d, "spec.json"), "w", encoding="utf-8") as fh:
        json.dump(spec, fh)


# ====================================================================== CLI / report
def write_report(rep: dict[str, Any], path: str) -> None:
    L: list[str] = []
    w = L.append
    w("# V2 model study (TRAIN + VALIDATION only; TEST untouched)\n")
    w(f"Train days: {', '.join(rep['train_days'])} · calibration day: {rep['cal_day']} · selection day: "
      f"{rep['sel_day']} · stop {rep['stop_bps']} bps · notional {rep['notional']} USDT · min net "
      f"{rep['min_net_usdt']} USDT · {rep['n_features']} features · rows {rep['rows']}\n")
    w("## Model comparison on the selection day (EV-gated offline simulation)\n")
    cols = ["model", "calibration", "threshold", "trades", "expectancy", "net_pnl", "t_stat", "win_rate",
            "profit_factor", "median_trade", "max_drawdown", "tp_before_sl_rate", "avg_hold_s", "costs_per_trade",
            "mean_auc", "mean_brier_cal", "latency_median_us", "latency_p99_us"]
    w("| " + " | ".join(cols) + " |")
    w("|" + "---|" * len(cols))
    for kind, m in rep["models"].items():
        s = m["selected"]
        st = s["sel_metrics"]
        aucs = [r["auc"] for r in m["per_target"] if r["auc"] == r["auc"]]
        meth = s["calibration"]
        briers = [r.get(f"brier_{meth}") for r in m["per_target"] if r.get(f"brier_{meth}") is not None]
        lat = rep.get("latency_us", {}).get(kind, {})
        vals = {"model": kind, "calibration": meth, "threshold": s["threshold"], **st,
                "mean_auc": round(float(np.mean(aucs)), 4) if aucs else None,
                "mean_brier_cal": round(float(np.mean(briers)), 5) if briers else None,
                "latency_median_us": lat.get("median_us"), "latency_p99_us": lat.get("p99_us")}
        w("| " + " | ".join("" if vals.get(c) is None else str(vals.get(c)) for c in cols) + " |")
    w("\nBase rates of TP-before-SL on the selection day: `" + json.dumps(rep["sel_base_rates"]) + "`\n")
    for kind, m in rep["models"].items():
        w(f"### {kind}: per-target discrimination (selection day)\n")
        pt = pd.DataFrame(m["per_target"])
        w(pt.to_string(index=False))
        w("")
        for key, tab in m["calibration_tables"].items():
            w(f"Calibration table {kind} {key} ({m['selected']['calibration']}):\n")
            w(pd.DataFrame(tab).to_string(index=False))
            w("")
    for sec in ("variants", "feature_groups", "top_features", "v1_feature_verdicts", "entry_timing", "regimes",
                "latency_us", "feature_latency_us"):
        if sec in rep:
            w(f"## {sec.replace('_', ' ').title()}\n")
            v = rep[sec]
            if isinstance(v, list) and v and isinstance(v[0], dict):
                w(pd.DataFrame(v).to_string(index=False))
            else:
                w("```\n" + json.dumps(v, indent=1, default=str) + "\n```")
            w("")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")


def main() -> None:
    from research import v2_analysis as VA

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--train-days", nargs="+", required=True)
    ap.add_argument("--cal-day", required=True)
    ap.add_argument("--sel-day", required=True)
    ap.add_argument("--stop", type=int, default=8, choices=STOPS_BPS)
    ap.add_argument("--notional", type=float, default=150.0)
    ap.add_argument("--min-net", type=float, default=0.10)
    ap.add_argument("--models", nargs="+", default=["logistic", "lightgbm", "xgboost", "mlp"])
    ap.add_argument("--targets", nargs="+", type=int, default=list(TARGETS_BPS))
    ap.add_argument("--mlp-targets", nargs="+", type=int, default=[12, 16, 20, 25])
    ap.add_argument("--thresholds", nargs="+", type=float, default=[0.5, 0.55, 0.6, 0.65, 0.7])
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    rep = run(args.data, args.out, args.train_days, args.cal_day, args.sel_day, args.stop, args.notional,
              args.min_net, args.models, args.targets, args.mlp_targets, args.threads, args.thresholds)
    arts = rep.pop("_artifacts")
    tr = load_dataset(args.data, args.train_days)
    ca = load_dataset(args.data, [args.cal_day])
    se = load_dataset(args.data, [args.sel_day])
    cols = feature_columns(tr)
    primary = 16 if 16 in args.targets else args.targets[len(args.targets) // 2]
    log.info("variants (direction feature, per-symbol) ...")
    rep["variants"] = VA.variants(tr, ca, se, cols, args.stop, primary, args.threads)
    if "lightgbm" in arts:
        log.info("importance / SHAP ...")
        imp = VA.importance(arts["lightgbm"]["arts"][("long", primary)][0], se, cols)
        rep["top_features"] = imp["top"]
        rep["feature_groups"] = imp["by_group"]
        rep["v1_feature_verdicts"] = VA.v1_feature_verdicts(tr, se, imp["table"], args.stop, primary)
    log.info("entry timing ...")
    rep["entry_timing"] = VA.entry_timing(tr)
    best_kind = max(rep["models"], key=lambda k: rep["models"][k]["selected"]["sel_metrics"].get("expectancy") or -1e9)
    a = arts[best_kind]
    probs = {k: (v[1].get(a["method"], v[1]["raw"]))(v[0].predict(matrix(se, cols))) for k, v in a["arts"].items()}
    _, trades = simulate(se, probs, args.stop, args.notional, a["threshold"], args.min_net)
    rep["regimes"] = {"model": best_kind, **VA.regimes(trades)}
    log.info("latency ...")
    sample = matrix(se.sample(min(2000, len(se)), random_state=1), cols)
    rep["latency_us"] = VA.latency({k: v["arts"][next(iter(v["arts"]))][0] for k, v in arts.items()}, cols, sample)
    with open(os.path.join(args.out, "v2_model_report.json"), "w", encoding="utf-8") as fh:
        json.dump(rep, fh, indent=1, default=str)
    write_report(rep, os.path.join(args.out, "V2_MODEL_REPORT.md"))
    print(f"report: {os.path.join(args.out, 'V2_MODEL_REPORT.md')}")


if __name__ == "__main__":
    main()
