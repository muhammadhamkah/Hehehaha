"""Event-driven tick replay through the SAME bot used live.

    recorded Binance events -> chronological queue (EventReader)
      -> virtual clock (VirtualTimeEventLoop)
      -> TradingBot handlers: order book / trade flow / feed validation
      -> scanner, signal engine (features -> predictor -> entry filter -> risk)
      -> PaperExchange execution (latency, queue position, partial fills, TTL, fees)
      -> exit engine -> trade log -> performance report

Nothing here re-implements strategy logic: ``TradingBot`` runs unmodified with
``offline=True``. The driver only (1) advances simulated time to each event's
timestamp, letting every bot timer that falls before it fire first, and (2) delivers the
event through the bot's own WebSocket handlers -- and only if the bot was subscribed to
that stream at that moment, exactly like live.

No look-ahead: the bot only ever holds state built from events with timestamp <= now.
Forward-return labels in the ``signals`` table are written AFTER the fact by the
recorder and are never read by the strategy.

CLI:
    python -m backtest.replay --events data/events --out runs/r1 [--latency-ms 50]
        [--speed max|1|10] [--start ISO --end ISO] [--symbols BTCUSDT ETHUSDT]
        [--depth-mode partial|bbo] [--time-source exchange|local] [--config cfg.json]
"""
from __future__ import annotations

import os as _os

# Pin BLAS/OpenMP to one thread BEFORE numpy loads: the barrier model uses tiny matrices,
# and parallel replay workers each spawning a BLAS thread pool oversubscribed the CPU
# (observed: replay slowed to below real time).
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    _os.environ.setdefault(_var, "1")

import argparse
import asyncio
import copy
import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

from analytics.decisions import decision_table
from analytics.performance import compute_performance
from backtest.virtual_loop import LoopClock, VirtualTimeEventLoop
from config import BotConfig, load_config
from data.database import Database
from data.event_store import load_meta, open_reader
from exchange.models import SymbolInfo

log = logging.getLogger("replay")

Observer = Callable[[Any, int], None]


@dataclass
class ReplaySpec:
    events_dir: str
    out_dir: str
    start_ms: int | None = None          # trading/recording starts here
    end_ms: int | None = None            # replay stops here (exclusive)
    warmup_ms: int = 120_000             # market state is built from this much earlier data
    time_source: str = "exchange"
    feed_latency_ms: int = 0
    reorder_window_ms: int = 5000
    speed: float | None = None           # None = max-speed deterministic
    symbols: list[str] = field(default_factory=list)
    seed: int = 12345
    observers: list[tuple[int, Observer]] = field(default_factory=list)   # (interval_ms, fn)
    label: str = ""
    fast_clock: bool = True              # skip loop iterations between timers (same semantics)
    min_edge: bool = False               # sample the minimum tradable edge -> out_dir/min_edge.csv


def _prepare_config(cfg: BotConfig, spec: ReplaySpec, meta: dict) -> BotConfig:
    cfg = copy.deepcopy(cfg)
    # Backtests are always simulated: never live, regardless of the input config.
    cfg.mode = "paper"
    cfg.dry_run = True
    cfg.recorder.record_events = False
    cfg.recorder.db_path = os.path.join(spec.out_dir, "replay.sqlite")
    cfg.recorder.trades_jsonl = os.path.join(spec.out_dir, "trades.jsonl")
    cfg.risk.kill_switch_file = ""
    if spec.symbols:
        cfg.scanner.static_symbols = tuple(spec.symbols)
    if meta.get("depth_mode") == "bbo" and cfg.market_data.depth_mode != "bbo":
        log.warning("dataset is L1-only (bbo); switching market_data.depth_mode to 'bbo'")
        cfg.market_data.depth_mode = "bbo"
    return cfg


def _symbol_info(meta: dict) -> dict[str, SymbolInfo]:
    out = {}
    for sym, d in (meta.get("symbol_info") or {}).items():
        out[sym] = SymbolInfo(**d)
    return out


def run_replay(cfg: BotConfig, spec: ReplaySpec) -> dict[str, Any]:
    from main import TradingBot   # local import: main imports a lot

    os.makedirs(spec.out_dir, exist_ok=True)
    db_path = os.path.join(spec.out_dir, "replay.sqlite")
    for suffix in ("", "-wal", "-shm"):
        if os.path.exists(db_path + suffix):
            os.remove(db_path + suffix)
    jl = os.path.join(spec.out_dir, "trades.jsonl")
    if os.path.exists(jl):
        os.remove(jl)
    meta = load_meta(spec.events_dir)
    cfg = _prepare_config(cfg, spec, meta)
    problems = [p for p in cfg.validate() if "live" not in p]
    if problems:
        raise ValueError(f"config problems: {problems}")

    first_ts = spec.start_ms - spec.warmup_ms if spec.start_ms is not None else None
    reader = open_reader(spec.events_dir, start_ms=first_ts, end_ms=spec.end_ms,
                         time_source=spec.time_source, feed_latency_ms=spec.feed_latency_ms,
                         reorder_window_ms=spec.reorder_window_ms, symbols=spec.symbols or None)
    events = iter(reader)
    try:
        first = next(events)
    except StopIteration:
        return {"error": "no events in range", "spec": _spec_dict(spec)}

    observers = list(spec.observers)
    min_edge_obs = None
    if spec.min_edge:
        from research.analysis import MinEdgeObserver

        min_edge_obs = MinEdgeObserver()
        observers.append((1000, min_edge_obs))
    loop = VirtualTimeEventLoop(start_ms=first.ts, speed=spec.speed)
    asyncio.set_event_loop(loop)
    clock = LoopClock(loop)
    bot = TradingBot(cfg, clock=clock, offline=True, rng_seed=spec.seed)
    info = _symbol_info(meta)
    stats = {"events": 0, "delivered": 0, "skipped_unsubscribed": 0, "market": 0}
    wall0 = time.perf_counter()
    active_from = spec.start_ms if spec.start_ms is not None else first.ts + spec.warmup_ms

    async def observe(interval_ms: int, fn: Observer) -> None:
        while True:
            await asyncio.sleep(interval_ms / 1000)
            if clock.now_ms() >= active_from:
                fn(bot, clock.now_ms())

    async def drive() -> None:
        await bot.start_core(info if info else None)
        bot.active_from_ms = active_from
        obs_tasks = [asyncio.ensure_future(observe(i, fn)) for i, fn in observers]
        ev = first
        last_ts = first.ts
        fast = spec.fast_clock
        while ev is not None:
            if not (fast and loop.try_advance_ms(ev.ts)):
                await loop.sleep_until_ms(ev.ts)
            stats["events"] += 1
            last_ts = ev.ts
            if ev.conn == "market":
                stats["market"] += 1
                bot._on_market_msg(ev.stream, ev.data, ev.ts)
            elif ev.stream in bot.detail_ws.streams or ev.stream.startswith("__snapshot__"):
                stats["delivered"] += 1
                bot._on_detail_msg(ev.stream, ev.data, ev.ts)
            else:
                stats["skipped_unsubscribed"] += 1
            ev = next(events, None)
        # Segment over: no further evaluations/entries; let in-flight orders settle, then flatten.
        bot.active_until_ms = spec.end_ms if spec.end_ms is not None else last_ts + 1
        await loop.sleep_until_ms(last_ts + 1000)
        for t in obs_tasks:
            t.cancel()
        stats["last_ts"] = last_ts
        await bot.shutdown()

    try:
        loop.run_until_complete(drive())
    finally:
        asyncio.set_event_loop(None)
        loop.close()

    if min_edge_obs is not None:
        min_edge_obs.frame().to_csv(os.path.join(spec.out_dir, "min_edge.csv"), index=False)
    result = summarize(db_path, bot, stats, reader, spec, active_from, wall0)
    with open(os.path.join(spec.out_dir, "result.json"), "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=1, default=str)
    return result


def _spec_dict(spec: ReplaySpec) -> dict:
    d = {k: v for k, v in spec.__dict__.items() if k != "observers"}
    return d


def execution_stats(orders: list[dict]) -> dict[str, Any]:
    """Maker fill rate etc. from the order event log (last event per order)."""
    final: dict[str, dict] = {}
    for o in orders:
        final[o["client_id"]] = o
    makers = [o for o in final.values() if o["tif"] == "GTX"]
    filled = [o for o in makers if (o["filled_qty"] or 0) > 0]
    full = [o for o in makers if o["status"] == "FILLED"]
    rejected = [o for o in makers if o["detail"] == "post_only_would_take"]
    takers = [o for o in final.values() if o["tif"] == "IOC" or o["type"] == "MARKET"]
    return {
        "maker_orders": len(makers),
        "maker_fill_rate_any": round(len(filled) / len(makers), 4) if makers else None,
        "maker_fill_rate_full": round(len(full) / len(makers), 4) if makers else None,
        "maker_post_only_rejected": len(rejected),
        "taker_orders": len(takers),
        "taker_partial_or_unfilled": sum(1 for o in takers if o["status"] != "FILLED"),
    }


def performance_with_extras(trades: list[dict]) -> dict[str, Any]:
    perf = compute_performance(trades)
    nets = sorted(t["net_pnl"] for t in trades)
    if nets:
        perf["median_net"] = round(nets[len(nets) // 2] if len(nets) % 2 else
                                   0.5 * (nets[len(nets) // 2 - 1] + nets[len(nets) // 2]), 6)
        hits = sum(1 for t in trades if t["exit_reason"] in ("target_profit", "trailing_stop"))
        stops = sum(1 for t in trades if t["exit_reason"] == "stop_loss")
        perf["target_before_stop_rate"] = round(hits / (hits + stops), 4) if hits + stops else None
    return perf


def summarize(db_path: str, bot, stats: dict, reader, spec: ReplaySpec,
              active_from: int, wall0: float) -> dict[str, Any]:
    db = Database(db_path)
    try:
        trades = db.query("SELECT * FROM trades ORDER BY entry_ts_ms")
        orders = db.query("SELECT * FROM orders ORDER BY id")
        n_signals = db.query("SELECT COUNT(*) AS n FROM signals")[0]["n"]
    finally:
        db.close()
    perf = performance_with_extras(trades)
    span_ms = max(stats.get("last_ts", active_from) - active_from, 0)
    return {
        "research_only": bool(bot.cfg.risk.research_mode),
        "label": spec.label,
        "spec": _spec_dict(spec),
        "period": {"from": _iso(active_from), "to": _iso(stats.get("last_ts", active_from)),
                   "hours": round(span_ms / 3.6e6, 3)},
        "replay": {**stats, "reader_clamped": reader.n_clamped, "reader_late": reader.n_late,
                   "reader_bad_lines": reader.n_bad_lines,
                   "wall_s": round(time.perf_counter() - wall0, 2),
                   "speedup": round(span_ms / 1000 / max(time.perf_counter() - wall0, 1e-9), 1)},
        "performance": perf,
        "execution": execution_stats(orders),
        "decisions": decision_table(bot.signals.decisions),
        "evaluations": sum(bot.signals.decisions.values()),
        "signals_recorded": n_signals,
        "feed_health": bot.feed.summary(),
        "risk": bot.risk.status(),
    }


def _iso(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")[:-4] + "Z"


def parse_time(s: str | None) -> int | None:
    if s is None:
        return None
    if s.isdigit():
        return int(s)
    dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def main() -> None:
    ap = argparse.ArgumentParser(description="Event-driven tick replay backtest")
    ap.add_argument("--events", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config")
    ap.add_argument("--start")
    ap.add_argument("--end")
    ap.add_argument("--warmup-s", type=float, default=120)
    ap.add_argument("--latency-ms", type=int, default=None, help="order round-trip latency (sim)")
    ap.add_argument("--feed-latency-ms", type=int, default=0)
    ap.add_argument("--time-source", choices=["exchange", "local"], default="exchange")
    ap.add_argument("--speed", default="max", help="'max' or a multiple of real time (1 = real time)")
    ap.add_argument("--symbols", nargs="*", default=[])
    ap.add_argument("--depth-mode", choices=["partial", "diff", "bbo"])
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("replay").setLevel(logging.INFO)
    cfg = load_config(args.config)
    if args.latency_ms is not None:
        cfg.execution.sim_latency_ms = args.latency_ms
    if args.depth_mode:
        cfg.market_data.depth_mode = args.depth_mode
    spec = ReplaySpec(
        events_dir=args.events, out_dir=args.out, start_ms=parse_time(args.start), end_ms=parse_time(args.end),
        warmup_ms=int(args.warmup_s * 1000), time_source=args.time_source, feed_latency_ms=args.feed_latency_ms,
        speed=None if args.speed == "max" else float(args.speed), symbols=args.symbols, seed=args.seed,
    )
    res = run_replay(cfg, spec)
    from backtest.report import format_replay

    print(format_replay(res))
    print(f"\nfull result: {os.path.join(args.out, 'result.json')}")


if __name__ == "__main__":
    main()
