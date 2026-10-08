"""V3 execution-aware evaluation through the SAME replay simulator as V1/V2.

    python -m v3.evaluate --store data/l2 --out runs/v3_replay --models models/v3 \\
        --kinds v3_all v3_l1 --symbols BTCUSDT ETHUSDT SOLUSDT \\
        --val-days D ... --test-days T ... [--latencies 50 100 250] [--final-test]

Each configuration is frozen before the test run (its spec was selected on the selection
days). Validation days are replayed with production risk controls and in research-only
mode (no consecutive-loss halt). ``--final-test`` replays the TEST days exactly once per
frozen configuration; ``test_lock.json`` records each configuration hash and refuses a
changed configuration on the same test period. Live trading stays disabled.
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
from typing import Any

from config import BotConfig
from research.compare_v1_v2 import _ms, calibration_from_signals
from research.validate import Runner, config_hash, load_many, metrics

log = logging.getLogger("v3.evaluate")
DAY_MS = 86_400_000
WARMUP_MS = 15 * 60_000


def v3_config(model_dir: str, latency_ms: int = 100) -> BotConfig:
    with open(os.path.join(model_dir, "spec.json"), encoding="utf-8") as fh:
        spec = json.load(fh)
    c = BotConfig()
    c.mode = "paper"
    c.dry_run = True
    c.market_data.depth_mode = "diff"
    c.strategy.predictor = "v3"
    c.v3.model_dir = model_dir
    c.execution.entry_mode = "taker"
    c.execution.sim_latency_ms = latency_ms
    c.exit.mode = "barrier"
    c.sizing.position_notional_usdt = spec["notional"]
    c.entry.min_net_profit_usdt = spec["min_net_usdt"]
    c.strategy.record_score_threshold = 0.0
    return c


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", default="models/v3")
    ap.add_argument("--kinds", nargs="+", default=["v3_all", "v3_l1"])
    ap.add_argument("--symbols", nargs="+", required=True)
    ap.add_argument("--val-days", nargs="+", required=True)
    ap.add_argument("--test-days", nargs="+", required=True)
    ap.add_argument("--latencies", nargs="+", type=int, default=[100])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--final-test", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for noisy in ("bot", "feed", "analytics.trade_logger", "risk.risk_manager"):
        logging.getLogger(noisy).setLevel(logging.ERROR)
    if set(a.val_days) & set(a.test_days) or max(a.val_days) >= min(a.test_days):
        raise SystemExit("validation days must precede and not overlap the test days")
    configs: dict[str, BotConfig] = {}
    for kind in a.kinds:
        md = os.path.join(a.models, kind)
        if not os.path.exists(os.path.join(md, "spec.json")):
            raise SystemExit(f"missing model {md}")
        for lat in a.latencies:
            configs[f"{kind}@{lat}ms"] = v3_config(md, lat)
    bounds = {"validation": (_ms(min(a.val_days)), _ms(max(a.val_days)) + DAY_MS),
              "test": (_ms(min(a.test_days)), _ms(max(a.test_days)) + DAY_MS)}
    os.makedirs(a.out, exist_ok=True)
    R = Runner(a.store, a.out, bounds, a.symbols, WARMUP_MS, a.workers, None)
    names = list(configs)
    rep: dict[str, Any] = {"bounds": bounds, "symbols": a.symbols, "configs": {n: config_hash(c) for n, c in configs.items()},
                           "created": time.time()}

    def research(c: BotConfig) -> BotConfig:
        c = copy.deepcopy(c)
        c.risk.research_mode = True
        return c

    reqs = [(configs[n], "validation", f"{i}_prod", n, {}) for i, n in enumerate(names)]
    reqs += [(research(configs[n]), "validation", f"{i}_research", n + " [RESEARCH-ONLY]", {}) for i, n in enumerate(names)]
    res = R.run(reqs)
    rep["validation"] = {n: metrics(r) for n, r in zip(names, res[:len(names)])}
    rep["validation_research_only"] = {n: metrics(r) for n, r in zip(names, res[len(names):])}
    rep["validation_calibration"] = {n: calibration_from_signals(load_many(r["dirs"])[1])
                                     for n, r in zip(names, res[len(names):])}
    if a.final_test:
        lock_path = os.path.join(a.out, "test_lock.json")
        lock = json.load(open(lock_path, encoding="utf-8")) if os.path.exists(lock_path) else {}
        for n, c in configs.items():
            h = config_hash(c)
            if n in lock and lock[n] != h:
                raise SystemExit(f"TEST already used for {n} with config {lock[n]}; refusing changed config {h}")
            lock[n] = h
        with open(lock_path, "w", encoding="utf-8") as fh:
            json.dump(lock, fh, indent=1)
        tres = R.run([(configs[n], "test", f"{i}_prod", n, {}) for i, n in enumerate(names)])
        rep["test"] = {n: metrics(r) for n, r in zip(names, tres)}
        from research import analysis as A
        rep["test_per_symbol"], rep["test_calibration"] = {}, {}
        for n, r in zip(names, tres):
            tr, sig = load_many(r["dirs"])
            rep["test_per_symbol"][n] = A.per_symbol(tr, sig).to_dict("records") if len(tr) else []
            rep["test_calibration"][n] = calibration_from_signals(sig)
    else:
        rep["test"] = "UNTOUCHED"
    with open(os.path.join(a.out, "v3_replay.json"), "w", encoding="utf-8") as fh:
        json.dump(rep, fh, indent=1, default=str)
    print(json.dumps({k: rep[k] for k in ("validation", "test")}, indent=1, default=str)[:4000])


if __name__ == "__main__":
    main()
