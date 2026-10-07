"""V1 vs V2 comparison through the SAME replay framework, same days/symbols/window.

    python -m research.compare_v1_v2 --events data/arch_5x10 --out runs/v1_vs_v2 \\
        --v1-config runs/real_l1_protocol/chosen_config.json --v2-models models/v2 \\
        --symbols ... --val-days 2024-03-27 2024-03-28 --test-days 2024-03-29 2024-03-30 \\
        --window 13 17 [--final-test]

Every configuration is frozen BEFORE the test run (V1: its own protocol's choice;
V2: the threshold/calibration selected on the selection day). The test days are
replayed once per frozen configuration; test_lock.json records each config hash and a
changed configuration is refused on the same test period.
"""
from __future__ import annotations

import os as _os

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    _os.environ.setdefault(_var, "1")

import argparse
import copy
import json
import logging
import os
import time
from datetime import date, datetime, timezone
from typing import Any


import pandas as pd

from config import BotConfig, _merge, load_config
from research.validate import Runner, config_hash, load_many, metrics

log = logging.getLogger("compare")
DAY_MS = 86_400_000
BUCKETS = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 1.0001]
LABELS = ["50-55%", "55-60%", "60-65%", "65-70%", "70-75%", "75-80%", "80-85%", "85-90%", ">90%"]


def _ms(day: str) -> int:
    d = date.fromisoformat(day)
    return int(datetime(d.year, d.month, d.day, tzinfo=timezone.utc).timestamp() * 1000)


def v2_config(base: BotConfig, model_dir: str) -> BotConfig:
    c = copy.deepcopy(base)
    c.strategy.predictor = "v2"
    c.strategy.v2_model_dir = model_dir
    c.execution.entry_mode = "taker"        # V2 labels assume taker entry after latency
    c.exit.mode = "barrier"                 # and barrier exits
    with open(os.path.join(model_dir, "spec.json"), encoding="utf-8") as fh:
        spec = json.load(fh)
    c.sizing.position_notional_usdt = spec["notional"]
    c.entry.min_net_profit_usdt = spec["min_net_usdt"]
    return c


def calibration_from_signals(signals: pd.DataFrame) -> list[dict]:
    """Predicted P(TP first) of the chosen (side, target) vs realised tp_plan, from replay."""
    s = signals.dropna(subset=["p_target", "tp_plan"])
    if s.empty:
        return []
    b = pd.cut(s["p_target"], BUCKETS, labels=LABELS, right=False)
    out = []
    for lab in LABELS:
        g = s[b == lab]
        if len(g):
            out.append({"bucket": lab, "n": len(g), "predicted": round(float(g.p_target.mean()), 4),
                        "actual": round(float((g.tp_plan == 1).mean()), 4),
                        "error": round(float(g.p_target.mean() - (g.tp_plan == 1).mean()), 4)})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--events", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base-config", default="configs/l1_adapted.json")
    ap.add_argument("--v1-config", help="frozen V1 config JSON (e.g. the protocol's chosen_config.json)")
    ap.add_argument("--v2-models", default="models/v2")
    ap.add_argument("--kinds", nargs="+", default=["logistic", "lightgbm", "xgboost", "mlp"])
    ap.add_argument("--symbols", nargs="+", required=True)
    ap.add_argument("--val-days", nargs="+", required=True)
    ap.add_argument("--test-days", nargs="+", required=True)
    ap.add_argument("--window", nargs=2, type=int, default=[13, 17])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--final-test", action="store_true")
    ap.add_argument("--no-v1", action="store_true", help="omit the V1 baseline (its test result is already locked)")
    ap.add_argument("--latency-json", help="model study report (for the inference-latency column)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for noisy in ("bot", "feed", "analytics.trade_logger", "risk.risk_manager"):
        logging.getLogger(noisy).setLevel(logging.ERROR)

    base = load_config(args.base_config)
    base.execution.sim_latency_ms = 100
    v1 = copy.deepcopy(base)
    if args.v1_config:
        with open(args.v1_config, encoding="utf-8") as fh:
            d = json.load(fh)
        for k in ("exchange", "recorder", "mode", "dry_run", "log_level"):
            d.pop(k, None)
        _merge(v1, d)
    configs: dict[str, BotConfig] = {} if args.no_v1 else {"V1 rule-based": v1}
    for kind in args.kinds:
        md = os.path.join(args.v2_models, kind)
        if os.path.exists(os.path.join(md, "spec.json")):
            configs[f"V2 {kind}"] = v2_config(base, md)
    bounds = {"validation": (_ms(min(args.val_days)), _ms(max(args.val_days)) + DAY_MS),
              "test": (_ms(min(args.test_days)), _ms(max(args.test_days)) + DAY_MS)}
    R = Runner(args.events, args.out, bounds, args.symbols, 120_000, args.workers, tuple(args.window))
    report: dict[str, Any] = {"bounds": bounds, "symbols": args.symbols, "window_utc": args.window,
                              "configs": {k: config_hash(c) for k, c in configs.items()}, "created": time.time()}

    def research(c: BotConfig) -> BotConfig:
        c = copy.deepcopy(c)
        c.risk.research_mode = True
        c.strategy.record_score_threshold = 0.0
        return c

    names = list(configs)
    log.info("VALIDATION: %d configs (production risk) + research-mode", len(names))
    reqs = [(configs[n], "validation", f"{i}_prod", n, {}) for i, n in enumerate(names)]
    reqs += [(research(configs[n]), "validation", f"{i}_research", n + " [RESEARCH-ONLY]", {}) for i, n in enumerate(names)]
    res = R.run(reqs)
    report["validation"] = {n: metrics(r) for n, r in zip(names, res[:len(names)])}
    report["validation_research_only"] = {n: metrics(r) for n, r in zip(names, res[len(names):])}
    report["validation_calibration"] = {}
    for n, r in zip(names, res[len(names):]):
        _, sig = load_many(r["dirs"])
        report["validation_calibration"][n] = calibration_from_signals(sig)

    if args.final_test:
        lock_path = os.path.join(args.out, "test_lock.json")
        lock = {}
        if os.path.exists(lock_path):
            with open(lock_path, encoding="utf-8") as fh:
                lock = json.load(fh)
        for n, c in configs.items():
            h = config_hash(c)
            if n in lock and lock[n] != h:
                raise SystemExit(f"TEST already used for {n} with config {lock[n]}; refusing changed config {h}")
            lock[n] = h
        with open(lock_path, "w", encoding="utf-8") as fh:
            json.dump(lock, fh, indent=1)
        log.info("TEST (once per frozen config): %s", names)
        tres = R.run([(configs[n], "test", f"{i}_prod", n, {}) for i, n in enumerate(names)])
        report["test"] = {n: metrics(r) for n, r in zip(names, tres)}
        report["test_per_symbol"] = {}
        from research import analysis as A
        for n, r in zip(names, tres):
            tr, sig = load_many(r["dirs"])
            report["test_per_symbol"][n] = A.per_symbol(tr, sig).to_dict("records") if len(tr) else []
            report.setdefault("test_calibration", {})[n] = calibration_from_signals(sig)
    else:
        report["test"] = "UNTOUCHED"
    if args.latency_json and os.path.exists(args.latency_json):
        with open(args.latency_json, encoding="utf-8") as fh:
            report["inference_latency_us"] = json.load(fh).get("latency_us", {})
    with open(os.path.join(args.out, "comparison.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1, default=str)
    write_markdown(report, os.path.join(args.out, "V1_VS_V2.md"))
    print(f"report: {os.path.join(args.out, 'V1_VS_V2.md')}")


def write_markdown(rep: dict[str, Any], path: str) -> None:
    rows = ["trades", "net_pnl", "expectancy", "t_stat", "win_rate", "profit_factor", "median_net", "max_drawdown",
            "target_before_stop_rate", "fees", "slippage", "maker_fill_rate"]
    lines = ["# V1 vs V2 — same replay framework, same data\n",
             f"Symbols: {', '.join(rep['symbols'])} · window {rep['window_utc'][0]:02d}:00–{rep['window_utc'][1]:02d}:00 UTC"
             " · production risk controls unless marked RESEARCH-ONLY\n"]
    for sec, title in (("test", "LOCKED TEST (each frozen config evaluated once)"),
                       ("validation", "Validation (production risk)"),
                       ("validation_research_only", "Validation, RESEARCH-ONLY (no consecutive-loss halt)")):
        data = rep.get(sec)
        if not isinstance(data, dict):
            lines.append(f"## {title}\n\n{data}\n")
            continue
        names = list(data)
        lines.append(f"## {title}\n")
        lines.append("| Metric | " + " | ".join(names) + " |")
        lines.append("|---|" + "---:|" * len(names))
        for m in rows:
            lines.append(f"| {m} | " + " | ".join("" if data[n].get(m) is None else str(data[n].get(m))
                                                for n in names) + " |")
        if sec == "test" and rep.get("inference_latency_us"):
            lat = rep["inference_latency_us"]
            lines.append("| inference p50 / p99 (us) | " + " | ".join(
                "rule-based (no model)" if n.startswith("V1") else
                f"{lat.get(n.split()[-1], {}).get('median_us')} / {lat.get(n.split()[-1], {}).get('p99_us')}"
                for n in names) + " |")
        lines.append("")
    for sec in ("validation_calibration", "test_calibration"):
        if rep.get(sec):
            lines.append(f"## {sec.replace('_', ' ').title()} (predicted P(TP first) vs realised)\n")
            for n, tab in rep[sec].items():
                lines.append(f"**{n}**\n")
                lines.append(pd.DataFrame(tab).to_string(index=False) if tab else "_no labelled entries_")
                lines.append("")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")




if __name__ == "__main__":
    main()
