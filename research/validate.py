"""End-to-end strategy validation on recorded data with strict out-of-sample discipline.

    python -m research.validate --events data/events --out runs/v1 [--workers 4]
        [--grid grid.json] [--max-combos 40] [--symbols BTCUSDT ...] [--final-test]

Protocol
  1. Split the dataset CHRONOLOGICALLY into TRAIN / VALIDATION / TEST (default 60/20/20).
  2. TRAIN: baseline replay (research recording on) -> feature analysis, calibration,
     minimum tradable edge, per-symbol results, decision diagnostics.
  3. TRAIN: parameter sweep (full replays through the real bot). Ranked by EXPECTED NET
     PnL PER TRADE after all costs, subject to a minimum trade count.
  4. VALIDATION: the top-K train configurations are replayed; the best VALIDATION
     expectancy (min trades) is selected. Latency robustness is checked here.
  5. TEST (only with --final-test): the single selected configuration is evaluated ONCE
     on the untouched test period, at every latency scenario. A lock file records this;
     evaluating a different configuration on the same test period is refused, because the
     test set would no longer be untouched.

Nothing is tuned on TEST, and thresholds are never loosened to manufacture trades: if
no configuration shows positive validation expectancy, that is the reported result.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import logging
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from typing import Any

from backtest.replay import ReplaySpec, run_replay
from config import BotConfig, _merge, load_config
from data.event_store import EventReader

log = logging.getLogger("validate")

# User-facing parameter names -> config paths
PARAMS = {
    "target_bps": "entry.min_target_bps",
    "stop_bps": "entry.stop_bps",
    "min_probability": "entry.min_p_target_before_stop",
    "min_confidence": "entry.min_confidence",
    "max_spread_bps": "entry.max_spread_bps",
    "imbalance_threshold": "entry.min_book_imbalance",
    "flow_threshold": "entry.min_flow_imbalance",
    "max_hold_s": "strategy.max_hold_s",
    "maker_ttl_ms": "execution.maker_ttl_ms",
    "taker_fallback": "execution.ttl_fallback",
    "latency_ms": "execution.sim_latency_ms",
    "notional": "sizing.position_notional_usdt",
}

DEFAULT_GRID: dict[str, list] = {
    "target_bps": [6.0, 12.0],
    "stop_bps": [6.0, 10.0],
    "min_probability": [0.55, 0.65],
    "imbalance_threshold": [-1.0, 0.2],
    "flow_threshold": [-1.0, 0.2],
    "maker_ttl_ms": [800, 2000],
    "taker_fallback": ["skip", "taker_if_edge"],
    "notional": [150.0],
}
LATENCIES = [20, 50, 100, 250, 500]
PRIMARY_LATENCY = 100


# ---------------------------------------------------------------------- helpers
def nested(overrides: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, value in overrides.items():
        path = PARAMS.get(name, name).split(".")
        d = out
        for p in path[:-1]:
            d = d.setdefault(p, {})
        d[path[-1]] = value
    return out


def apply(cfg: BotConfig, overrides: dict[str, Any]) -> BotConfig:
    c = copy.deepcopy(cfg)
    _merge(c, nested(overrides))
    return c


def config_hash(cfg: BotConfig) -> str:
    d = cfg.to_dict()
    for k in ("recorder", "exchange", "log_level"):
        d.pop(k, None)
    return hashlib.sha256(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:16]


def split_bounds(t0: int, t1: int, fractions: tuple[float, float, float]) -> dict[str, tuple[int, int]]:
    if abs(sum(fractions) - 1.0) > 1e-6:
        raise ValueError("split fractions must sum to 1")
    span = t1 - t0
    a = t0 + int(span * fractions[0])
    b = a + int(span * fractions[1])
    return {"train": (t0, a), "validation": (a, b), "test": (b, t1 + 1)}


def combos(grid: dict[str, list], max_combos: int | None, seed: int = 7) -> list[dict[str, Any]]:
    keys = sorted(grid)
    allc = [dict(zip(keys, vals)) for vals in itertools.product(*(grid[k] for k in keys))]
    if max_combos and len(allc) > max_combos:
        rng = random.Random(seed)
        allc = rng.sample(allc, max_combos)
    return allc


def research_cfg(cfg: BotConfig) -> BotConfig:
    """Recording settings for unbiased research data. Recording never affects trading."""
    c = copy.deepcopy(cfg)
    c.strategy.record_score_threshold = 0.0      # record every evaluated symbol...
    c.strategy.record_min_interval_ms = 1000     # ...at most once per second
    return c


def metrics(res: dict[str, Any]) -> dict[str, Any]:
    p = res.get("performance", {})
    e = res.get("execution", {})
    return {
        "trades": p.get("trades", 0), "win_rate": p.get("win_rate"), "net_pnl": p.get("net_pnl"),
        "profit_factor": p.get("profit_factor"), "expectancy": p.get("expectancy"),
        "max_drawdown": p.get("max_drawdown"), "sharpe_per_trade": p.get("sharpe_per_trade"),
        "t_stat": p.get("t_stat"), "median_net": p.get("median_net"), "fees": p.get("fees_paid"),
        "slippage": p.get("slippage_cost_actual"), "maker_fill_rate": e.get("maker_fill_rate_any"),
        "target_before_stop_rate": p.get("target_before_stop_rate"),
    }


def _run(args: tuple) -> dict[str, Any]:
    cfg, spec_kwargs, label = args
    logging.getLogger().setLevel(logging.ERROR)
    spec = ReplaySpec(**spec_kwargs, label=label)
    t = time.perf_counter()
    res = run_replay(cfg, spec)
    res["_wall"] = time.perf_counter() - t
    return res


def run_many(jobs: list[tuple], workers: int) -> list[dict[str, Any]]:
    if workers <= 1 or len(jobs) == 1:
        return [_run(j) for j in jobs]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(_run, jobs))


# ---------------------------------------------------------------------- protocol
def validate(cfg: BotConfig, events: str, out: str, grid: dict[str, list], max_combos: int | None,
             workers: int, symbols: list[str], fractions=(0.6, 0.2, 0.2), top_k: int = 5,
             min_trades: int = 30, final_test: bool = False, force_retest: bool = False,
             latencies: list[int] = LATENCIES, warmup_ms: int = 120_000) -> dict[str, Any]:
    os.makedirs(out, exist_ok=True)
    rng = EventReader(events).time_range()
    if rng is None:
        raise SystemExit("no events found")
    bounds = split_bounds(rng[0] + warmup_ms, rng[1], fractions)
    report: dict[str, Any] = {"events": events, "bounds": bounds, "symbols": symbols,
                              "grid": grid, "min_trades": min_trades, "created": time.time()}
    base = copy.deepcopy(cfg)
    base.execution.sim_latency_ms = PRIMARY_LATENCY

    def spec(seg: str, sub: str) -> dict[str, Any]:
        s, e = bounds[seg]
        return {"events_dir": events, "out_dir": os.path.join(out, seg, sub), "start_ms": s, "end_ms": e,
                "warmup_ms": warmup_ms, "symbols": symbols}

    # -- 2. baseline on TRAIN with research recording + min-edge sampling
    from research import analysis as A

    log.info("baseline replay on TRAIN %s", bounds["train"])
    obs = A.MinEdgeObserver()
    bspec = ReplaySpec(**spec("train", "baseline"), label="baseline", observers=[(1000, obs)])
    bres = run_replay(research_cfg(base), bspec)
    trades, signals = A.load(os.path.join(out, "train", "baseline", "replay.sqlite"))
    report["baseline_train"] = {
        "metrics": metrics(bres), "decisions": bres["decisions"][:25], "evaluations": bres["evaluations"],
        "feed_health": bres["feed_health"]["malformed_by_kind"],
        "per_symbol": A.per_symbol(trades, signals).to_dict("records"),
        "calibration": A.calibration(signals),
        "features": A.feature_analysis(signals, base.costs.maker_fee, base.costs.taker_fee),
        "min_edge": A.min_edge_summary(obs.frame(), signals),
    }

    # -- 3. sweep on TRAIN
    cands = combos(grid, max_combos)
    log.info("sweep: %d configurations on TRAIN (workers=%d)", len(cands), workers)
    jobs = [(apply(base, c), spec("train", f"sweep_{i:03d}"), json.dumps(c)) for i, c in enumerate(cands)]
    train_res = run_many(jobs, workers)
    sweep = []
    for c, r in zip(cands, train_res):
        m = metrics(r)
        sweep.append({"params": c, "train": m, "eligible": (m["trades"] or 0) >= min_trades})
    ranked = sorted(sweep, key=lambda s: (s["eligible"], s["train"]["expectancy"] or -1e9), reverse=True)
    report["sweep_train"] = ranked

    # -- 4. VALIDATION on top-K eligible
    top = [s for s in ranked if s["eligible"]][:top_k]
    if not top:
        log.warning("no configuration reached %d trades on TRAIN", min_trades)
    vjobs = [(apply(base, s["params"]), spec("validation", f"cand_{i}"), json.dumps(s["params"]))
             for i, s in enumerate(top)]
    for s, r in zip(top, run_many(vjobs, workers)):
        s["validation"] = metrics(r)
    valid = [s for s in top if (s["validation"]["trades"] or 0) >= max(10, min_trades // 3)]
    chosen = max(valid, key=lambda s: s["validation"]["expectancy"] or -1e9) if valid else None
    report["validation"] = top
    report["chosen"] = chosen
    if chosen is not None:
        ccfg = apply(base, chosen["params"])
        ljobs = [(apply(ccfg, {"latency_ms": lat}), spec("validation", f"latency_{lat}"), f"lat{lat}")
                 for lat in latencies]
        report["validation_latency"] = {lat: metrics(r) for lat, r in zip(latencies, run_many(ljobs, workers))}
        report["chosen_config_hash"] = config_hash(ccfg)
        with open(os.path.join(out, "chosen_config.json"), "w", encoding="utf-8") as fh:
            json.dump(asdict(ccfg) | {"exchange": {}}, fh, indent=1, default=str)

    # -- 5. TEST (once)
    lock_path = os.path.join(out, "test_lock.json")
    if final_test:
        if chosen is None:
            report["test"] = {"skipped": "no configuration passed validation; nothing to test"}
        else:
            h = report["chosen_config_hash"]
            if os.path.exists(lock_path):
                with open(lock_path, encoding="utf-8") as fh:
                    lock = json.load(fh)
                if lock["config_hash"] != h and not force_retest:
                    raise SystemExit(
                        f"TEST PERIOD ALREADY USED for config {lock['config_hash']}; refusing to evaluate "
                        f"a different config ({h}). The test set is no longer untouched for new choices.")
            with open(lock_path, "w", encoding="utf-8") as fh:
                json.dump({"config_hash": h, "params": chosen["params"], "at": time.time(),
                           "forced": bool(force_retest)}, fh, indent=1)
            ccfg = apply(base, chosen["params"])
            tjobs = [(apply(research_cfg(ccfg), {"latency_ms": lat}), spec("test", f"latency_{lat}"), f"test{lat}")
                     for lat in latencies]
            tres = dict(zip(latencies, run_many(tjobs, workers)))
            report["test"] = {lat: metrics(r) for lat, r in tres.items()}
            ttrades, tsignals = A.load(os.path.join(out, "test", f"latency_{PRIMARY_LATENCY}", "replay.sqlite"))
            report["test_per_symbol"] = A.per_symbol(ttrades, tsignals).to_dict("records")
            report["test_calibration"] = A.calibration(tsignals)
    else:
        report["test"] = {"status": "UNTOUCHED (run with --final-test once the configuration is frozen)"}

    report["verdict"] = verdict(report, min_trades)
    with open(os.path.join(out, "validation_report.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1, default=str)
    from research.report import write_markdown

    write_markdown(report, os.path.join(out, "VALIDATION_REPORT.md"))
    return report


def verdict(report: dict[str, Any], min_trades: int) -> str:
    feats = report.get("baseline_train", {}).get("features", {}).get("verdict", "")
    chosen = report.get("chosen")
    test = report.get("test", {})
    if chosen is None:
        return ("NO EDGE DEMONSTRATED: no configuration achieved enough trades with positive validation "
                f"expectancy. Feature analysis: {feats}.")
    if PRIMARY_LATENCY not in test:
        v = chosen["validation"]
        return (f"VALIDATION ONLY (test untouched): chosen config expectancy {v['expectancy']} USDT/trade over "
                f"{v['trades']} trades. Not evidence of an edge until evaluated once on TEST.")
    t = test[PRIMARY_LATENCY]
    slow = test.get(250, {})
    ok = ((t["trades"] or 0) >= min_trades and (t["expectancy"] or 0) > 0 and (t["t_stat"] or 0) > 2
          and (slow.get("expectancy") or 0) > 0)
    if ok:
        return (f"POSITIVE OUT-OF-SAMPLE EXPECTANCY: {t['expectancy']} USDT/trade over {t['trades']} unseen "
                f"trades (t={t['t_stat']}), still positive at 250 ms latency.")
    return (f"NO EDGE ON UNSEEN DATA: test expectancy {t['expectancy']} USDT/trade over {t['trades']} trades "
            f"(t={t['t_stat']}); 250 ms latency expectancy {slow.get('expectancy')}.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--events", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config")
    ap.add_argument("--grid", help="JSON file: {param: [values]} using names in PARAMS")
    ap.add_argument("--max-combos", type=int, default=40)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--symbols", nargs="*", default=[])
    ap.add_argument("--split", nargs=3, type=float, default=[0.6, 0.2, 0.2])
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--min-trades", type=int, default=30)
    ap.add_argument("--latencies", nargs="*", type=int, default=LATENCIES)
    ap.add_argument("--final-test", action="store_true")
    ap.add_argument("--force-retest", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for noisy in ("bot", "feed", "analytics.trade_logger", "data.recorder", "exchange.execution"):
        logging.getLogger(noisy).setLevel(logging.ERROR)
    cfg = load_config(args.config)
    grid = DEFAULT_GRID
    if args.grid:
        with open(args.grid, encoding="utf-8") as fh:
            grid = json.load(fh)
    rep = validate(cfg, args.events, args.out, grid, args.max_combos, args.workers, args.symbols,
                   tuple(args.split), args.top_k, args.min_trades, args.final_test, args.force_retest,
                   args.latencies)
    print(f"\nVERDICT: {rep['verdict']}\nreport: {os.path.join(args.out, 'VALIDATION_REPORT.md')}")


if __name__ == "__main__":
    main()
