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

import os as _os

# Pin BLAS/OpenMP to one thread BEFORE numpy loads: the barrier model uses tiny matrices,
# and parallel replay workers each spawning a BLAS thread pool oversubscribed the CPU
# (observed: replay slowed to below real time).
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    _os.environ.setdefault(_var, "1")

import argparse
import copy
import itertools
import json
import logging
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any

from backtest.replay import ReplaySpec, run_replay
from config import BotConfig, _merge, load_config
from data.event_store import open_reader

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
    return cfg.fingerprint()


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


DAY_MS = 86_400_000


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


def day_shards(start: int, end: int) -> list[tuple[int, int]]:
    """Split [start, end) at UTC midnights."""
    out, cur = [], start
    while cur < end:
        nxt = min(end, (cur // DAY_MS + 1) * DAY_MS)
        out.append((cur, nxt))
        cur = nxt
    return out


def merge_shards(dirs: list[str], results: list[dict[str, Any]]) -> dict[str, Any]:
    """Combine day-shard replays into one result (trades/orders pooled, decisions summed)."""
    from analytics.decisions import decision_table
    from backtest.replay import execution_stats, performance_with_extras
    from data.database import Database

    trades: list[dict] = []
    orders: list[dict] = []
    for i, d in enumerate(dirs):
        path = os.path.join(d, "replay.sqlite")
        if not os.path.exists(path):
            continue
        db = Database(path)
        try:
            trades += db.query("SELECT * FROM trades")
            orders += [dict(o, client_id=f"{i}:{o['client_id']}") for o in db.query("SELECT * FROM orders ORDER BY id")]
        finally:
            db.close()
    trades.sort(key=lambda t: t["entry_ts_ms"])
    decisions: dict[str, int] = {}
    malformed: dict[str, int] = {}
    evaluations = 0
    for r in results:
        evaluations += r.get("evaluations", 0)
        for row in r.get("decisions", []):
            decisions[row["reason"]] = decisions.get(row["reason"], 0) + row["count"]
        for k, v in r.get("feed_health", {}).get("malformed_by_kind", {}).items():
            malformed[k] = malformed.get(k, 0) + v
    return {"performance": performance_with_extras(trades), "execution": execution_stats(orders),
            "decisions": decision_table(decisions), "evaluations": evaluations,
            "feed_health": {"malformed_by_kind": malformed}, "dirs": dirs,
            "wall_s": round(sum(r.get("_wall", 0) for r in results), 1)}


class Runner:
    """Expands segment runs into day shards, executes all shards in one process pool."""

    def __init__(self, events: str, out: str, bounds: dict[str, tuple[int, int]], symbols: list[str],
                 warmup_ms: int, workers: int, daily_window: tuple[int, int] | None = None,
                 resume: bool = False) -> None:
        self.resume = resume
        self.events, self.out, self.bounds = events, out, bounds
        self.symbols, self.warmup_ms, self.workers = symbols, warmup_ms, workers
        self.daily_window = daily_window       # (start_hour, end_hour) UTC, or None = whole day

    def shards(self, s: int, e: int) -> list[tuple[int, int, int]]:
        """(warmup_from, trade_start, end) per shard."""
        out = []
        for a, b in day_shards(s, e):
            if self.daily_window:
                day = a // DAY_MS * DAY_MS
                a = max(a, day + self.daily_window[0] * 3_600_000)
                b = min(b, day + self.daily_window[1] * 3_600_000)
                start = a
            else:
                start = a + self.warmup_ms if a % DAY_MS == 0 else a   # warm up on same-day data
            if start < b:
                out.append((a, start, b))
        return out

    @staticmethod
    def _reusable(job: tuple) -> dict[str, Any] | None:
        """A completed shard is reused only if its spec, label and (when stored) config match."""
        cfg, spec, label = job
        path = os.path.join(spec["out_dir"], "result.json")
        if not os.path.exists(path):
            return None
        with open(path, encoding="utf-8") as fh:
            r = json.load(fh)
        old = r.get("spec", {})
        same = (r.get("label") == label and all(old.get(k) == spec.get(k) for k in
                ("start_ms", "end_ms", "warmup_ms", "symbols", "events_dir"))
                and r.get("config_hash", cfg.fingerprint()) == cfg.fingerprint())
        if not same:
            return None
        log.info("reusing completed shard %s", spec["out_dir"])
        r["_wall"] = 0.0
        return r

    def run(self, requests: list[tuple[BotConfig, str, str, str, dict]]) -> list[dict[str, Any]]:
        """requests: (cfg, segment, sub_dir, label, extra ReplaySpec kwargs)."""
        jobs, groups = [], []
        for cfg, seg, sub, label, extra in requests:
            s, e = self.bounds[seg]
            idx = []
            for a, start, b in self.shards(s, e):
                d = os.path.join(self.out, seg, sub, datetime.fromtimestamp(a / 1000, tz=timezone.utc)
                                 .strftime("%Y%m%d_%H%M"))
                spec = {"events_dir": self.events, "out_dir": d, "start_ms": start, "end_ms": b,
                        "warmup_ms": self.warmup_ms, "symbols": self.symbols, **extra}
                idx.append(len(jobs))
                jobs.append((cfg, spec, label))
            groups.append(idx)
        results: list[dict[str, Any] | None] = [self._reusable(j) if self.resume else None for j in jobs]
        todo = [i for i, r in enumerate(results) if r is None]
        log.info("running %d shard replays (%d reused, %d requests, workers=%d)", len(todo),
                 len(jobs) - len(todo), len(requests), self.workers)
        for i, r in zip(todo, run_many([jobs[i] for i in todo], self.workers) if todo else []):
            results[i] = r
        return [merge_shards([jobs[i][1]["out_dir"] for i in g], [results[i] for i in g]) for g in groups]


def load_many(dirs: list[str]):
    import pandas as pd

    from research import analysis as A

    ts, ss = [], []
    for d in dirs:
        p = os.path.join(d, "replay.sqlite")
        if os.path.exists(p):
            t, s = A.load(p)
            ts.append(t)
            ss.append(s)
    trades = pd.concat(ts, ignore_index=True) if ts else pd.DataFrame()
    signals = pd.concat(ss, ignore_index=True).sort_values("ts_ms") if ss else pd.DataFrame()
    return trades, signals


# ---------------------------------------------------------------------- protocol
def validate(cfg: BotConfig, events: str, out: str, grid: dict[str, list], max_combos: int | None,
             workers: int, symbols: list[str], fractions=(0.6, 0.2, 0.2), top_k: int = 5,
             min_trades: int = 30, final_test: bool = False, force_retest: bool = False,
             latencies: list[int] = LATENCIES, warmup_ms: int = 120_000,
             daily_window: tuple[int, int] | None = None, resume: bool = False) -> dict[str, Any]:
    import pandas as pd

    from research import analysis as A

    os.makedirs(out, exist_ok=True)
    rng = open_reader(events, symbols=symbols or None).time_range()
    if rng is None:
        raise SystemExit("no events found")
    t0, t1 = rng[0], rng[1] + 1
    bounds = split_bounds(t0, t1, fractions)
    if t1 - t0 >= 3 * DAY_MS:      # multi-day data: split on whole UTC days
        snap = lambda x: int(round(x / DAY_MS)) * DAY_MS   # noqa: E731
        bounds = {"train": (t0, snap(bounds["train"][1])),
                  "validation": (snap(bounds["validation"][0]), snap(bounds["validation"][1])),
                  "test": (snap(bounds["test"][0]), t1)}
    report: dict[str, Any] = {"events": events, "bounds": bounds, "symbols": symbols,
                              "grid": grid, "min_trades": min_trades, "created": time.time(),
                              "daily_window_utc": daily_window}
    base = copy.deepcopy(cfg)
    base.execution.sim_latency_ms = PRIMARY_LATENCY
    R = Runner(events, out, bounds, symbols, warmup_ms, workers, daily_window, resume)

    # -- 2. baseline (research recording) + 3. sweep, all on TRAIN, in one pool
    cands = combos(grid, max_combos) if grid else []     # empty grid: research-only run
    report["research_only"] = not cands
    log.info("TRAIN %s: baseline + %d sweep configurations", bounds["train"], len(cands))
    reqs = [(research_cfg(base), "train", "baseline", "baseline", {"min_edge": True})]
    reqs += [(apply(base, c), "train", f"sweep_{i:03d}", json.dumps(c), {}) for i, c in enumerate(cands)]
    res = R.run(reqs)
    bres, train_res = res[0], res[1:]
    trades, signals = load_many(bres["dirs"])
    me = [pd.read_csv(os.path.join(d, "min_edge.csv")) for d in bres["dirs"]
          if os.path.exists(os.path.join(d, "min_edge.csv"))]
    me_df = pd.concat(me, ignore_index=True) if me else pd.DataFrame()
    report["baseline_train"] = {
        "metrics": metrics(bres), "decisions": bres["decisions"][:25], "evaluations": bres["evaluations"],
        "feed_health": bres["feed_health"]["malformed_by_kind"],
        "per_symbol": A.per_symbol(trades, signals).to_dict("records"),
        "calibration": A.calibration(signals),
        "features": A.feature_analysis(signals, base.costs.maker_fee, base.costs.taker_fee),
        "min_edge": A.min_edge_summary(me_df, signals),
    }
    sweep = []
    for c, r in zip(cands, train_res):
        m = metrics(r)
        sweep.append({"params": c, "train": m, "eligible": (m["trades"] or 0) >= min_trades})
    ranked = sorted(sweep, key=lambda s: (s["eligible"], s["train"]["expectancy"] or -1e9), reverse=True)
    report["sweep_train"] = ranked
    report["ineffective_params"] = ineffective_params(sweep)

    # -- 4. VALIDATION on top-K eligible
    top = [s for s in ranked if s["eligible"]][:top_k]
    if not top:
        log.warning("no configuration reached %d trades on TRAIN", min_trades)
    vres = R.run([(apply(base, s["params"]), "validation", f"cand_{i}", json.dumps(s["params"]), {})
                  for i, s in enumerate(top)]) if top else []
    for s, r in zip(top, vres):
        s["validation"] = metrics(r)
    valid = [s for s in top if (s["validation"]["trades"] or 0) >= max(10, min_trades // 3)]
    chosen = max(valid, key=lambda s: s["validation"]["expectancy"] or -1e9) if valid else None
    report["validation"] = top
    report["chosen"] = chosen
    if chosen is not None:
        ccfg = apply(base, chosen["params"])
        extra = [lat for lat in latencies if lat != PRIMARY_LATENCY]   # primary already run above
        lres = R.run([(apply(ccfg, {"latency_ms": lat}), "validation", f"latency_{lat}", f"lat{lat}", {})
                      for lat in extra]) if extra else []
        report["validation_latency"] = {lat: metrics(r) for lat, r in zip(extra, lres)}
        if PRIMARY_LATENCY in latencies:
            report["validation_latency"][PRIMARY_LATENCY] = chosen["validation"]
        report["validation_latency"] = dict(sorted(report["validation_latency"].items()))
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
            tres = R.run([(apply(research_cfg(ccfg), {"latency_ms": lat}), "test", f"latency_{lat}", f"test{lat}", {})
                          for lat in latencies])
            report["test"] = {lat: metrics(r) for lat, r in zip(latencies, tres)}
            prim = tres[latencies.index(PRIMARY_LATENCY)] if PRIMARY_LATENCY in latencies else tres[0]
            ttrades, tsignals = load_many(prim["dirs"])
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


def ineffective_params(sweep: list[dict[str, Any]]) -> list[str]:
    """Parameters whose tested values never changed TRAIN results (holding the others fixed).

    Typical cause: another constraint binds first (e.g. target_bps below the cost-derived
    required move), so the grid wastes runs on that dimension.
    """
    out = []
    keys = sorted({k for s in sweep for k in s["params"]})
    for k in keys:
        groups: dict[str, set] = {}
        for s in sweep:
            rest = json.dumps({kk: v for kk, v in s["params"].items() if kk != k}, sort_keys=True)
            groups.setdefault(rest, set()).add(json.dumps(s["train"], sort_keys=True, default=str))
        varied = len({json.dumps(s["params"].get(k)) for s in sweep}) > 1
        if varied and all(len(g) == 1 for g in groups.values()):
            out.append(k)
    return out


def verdict(report: dict[str, Any], min_trades: int) -> str:
    feats = report.get("baseline_train", {}).get("features", {}).get("verdict", "")
    if report.get("research_only"):
        m = report.get("baseline_train", {}).get("metrics", {})
        return (f"RESEARCH ONLY (no parameter selection, TEST untouched). Feature analysis: {feats}. "
                f"Strategy as configured on TRAIN: {m.get('trades')} trades, expectancy {m.get('expectancy')} USDT/trade.")
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
    if (t["trades"] or 0) < min_trades:
        return (f"INSUFFICIENT TEST SAMPLE: {t['trades']} trades on the unseen period (minimum {min_trades}); "
                f"expectancy {t['expectancy']} USDT/trade, t={t['t_stat']}. No conclusion either way — "
                "record more data.")
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
    ap.add_argument("--research-only", action="store_true",
                    help="baseline analyses on TRAIN only (features, calibration, min edge); no sweep/test")
    ap.add_argument("--daily-window", nargs=2, type=int, metavar=("START_H", "END_H"),
                    help="replay only this UTC hour window of each day (compute-limited studies)")
    ap.add_argument("--resume", action="store_true",
                    help="reuse completed shard results with identical spec/label/config")
    ap.add_argument("--final-test", action="store_true")
    ap.add_argument("--force-retest", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for noisy in ("bot", "feed", "analytics.trade_logger", "data.recorder", "exchange.execution"):
        logging.getLogger(noisy).setLevel(logging.ERROR)
    cfg = load_config(args.config)
    grid = DEFAULT_GRID
    if args.research_only:
        grid = {}
    elif args.grid:
        with open(args.grid, encoding="utf-8") as fh:
            grid = json.load(fh)
    rep = validate(cfg, args.events, args.out, grid, args.max_combos, args.workers, args.symbols,
                   tuple(args.split), args.top_k, args.min_trades, args.final_test, args.force_retest,
                   args.latencies, daily_window=tuple(args.daily_window) if args.daily_window else None,
                   resume=args.resume)
    print(f"\nVERDICT: {rep['verdict']}\nreport: {os.path.join(args.out, 'VALIDATION_REPORT.md')}")


if __name__ == "__main__":
    main()
