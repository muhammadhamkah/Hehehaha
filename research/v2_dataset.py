"""V2 training dataset: features at decision time + direct barrier outcomes.

    python -m research.v2_dataset --events data/arch_5x10 --out data/v2/ds \\
        --symbols BTCUSDT ETHUSDT SOLUSDT DOGEUSDT LINKUSDT \\
        --days 2024-03-21 ... 2024-03-28 --window 13 17 --workers 4

Pipeline per (symbol, day) shard -- the same state code and feature functions the bot uses:
  archive events -> market_data.state.apply_detail -> OrderBook / TradeFlow
  every 250 ms (bot eval cadence): base features + FeatureHistory update
  every sample_ms: one row = v2_features(...) at that instant
After the pass, labels are computed from the recorded quote path AFTER the decision:
  entry at the quote prevailing at t + latency (long buys the ask, short sells the bid)
  for each target T and stop S: first-touch times on the executable side within the
  horizon -> TP-before-SL for any (T, S) is  ttp_T < tsl_S  (ties count as stop)
  plus forward mid returns, MFE/MAE, exit return at the horizon, and cost columns:
  round-trip cost bps and required gross move for min net profit, per notional.

TEST days must never be passed here during development (enforced by --forbid-days).
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
from concurrent.futures import ProcessPoolExecutor
from datetime import date, datetime, timezone

import numpy as np
import pandas as pd

from config import BotConfig, load_config
from data.event_store import load_meta, open_reader
from exchange.schemas import FeedValidator
from features.microstructure_features import compute_features
from features.v2_features import FeatureHistory, v2_features
from market_data.state import apply_detail, make_book
from market_data.tradeflow import TradeFlow
from strategy.costs import CostModel

log = logging.getLogger("v2ds")

TARGETS_BPS = (8, 10, 12, 14, 15, 16, 18, 20, 25, 30)
STOPS_BPS = (6, 8, 10, 12, 15)
RET_HORIZONS_S = (1, 3, 5, 10, 30, 60)
NOTIONALS = (100, 150, 250)
INF = np.iinfo(np.int64).max


def _day_ms(day: str) -> int:
    d = date.fromisoformat(day)
    return int(datetime(d.year, d.month, d.day, tzinfo=timezone.utc).timestamp() * 1000)


def build_shard(args: tuple) -> str:
    (events, out_dir, symbol, day, window, cfg, sample_ms, eval_ms, horizon_ms, latency_ms, min_net) = args
    t_wall = time.perf_counter()
    d0 = _day_ms(day)
    w0 = d0 + window[0] * 3_600_000
    w1 = d0 + window[1] * 3_600_000
    warm = 120_000
    md = cfg.market_data
    mode = load_meta(events).get("depth_mode") or md.depth_mode
    book = make_book(symbol, md.book_history_len, mode, md.bbo_history_interval_ms)
    flow = TradeFlow(symbol, md.trade_history_s)
    feed = FeedValidator()
    hist = FeatureHistory()
    costs = CostModel(cfg.costs)
    reader = open_reader(events, start_ms=w0 - warm, end_ms=w1 + horizon_ms + latency_ms + 2000,
                         symbols=[symbol])
    q_ts: list[int] = []
    q_bid: list[float] = []
    q_ask: list[float] = []
    rows: list[dict[str, float]] = []
    next_eval = w0 - warm + eval_ms
    stale = md.stale_after_ms

    def evaluate(now: int) -> None:
        if book.is_stale(now, stale) or len(book.history) < 20:
            return
        base = compute_features(book, flow, cfg.features, now)
        if not base:
            return
        base["last_price"] = flow.last_price()
        if w0 <= now < w1 and (now - w0) % sample_ms == 0:
            f = v2_features(book, flow, base, hist, now)
            meta = {"ts_ms": now, "bid0": base["bid"], "ask0": base["ask"], "mid0": base["mid"]}
            for n in NOTIONALS:
                for d, tag in ((1, "long"), (-1, "short")):
                    c = costs.estimate(book, d, n, entry_maker=False)
                    meta[f"cost_bps_{tag}_{n}"] = c.total_bps
                    meta[f"req_bps_{tag}_{n}"] = costs.required_move_bps(c, min_net)
            rows.append({**meta, **f})
        hist.add(now, base)

    for ev in reader:
        while ev.ts >= next_eval:            # evaluations see only events strictly before them
            evaluate(next_eval)
            next_eval += eval_ms
        sym_lower, _, kind = ev.stream.partition("@")
        if sym_lower.upper() != symbol or ev.conn != "detail":
            continue
        res = apply_detail(book, flow, feed, mode, symbol, kind, ev.stream, ev.data, ev.ts)
        if res.kind == "book":
            b, _, a, _ = book.best()
            q_ts.append(ev.ts)
            q_bid.append(b)
            q_ask.append(a)
    if not rows:
        return ""
    df = pd.DataFrame(rows)
    labels = barrier_labels(df["ts_ms"].to_numpy(np.int64), df["mid0"].to_numpy(),
                            np.asarray(q_ts, np.int64), np.asarray(q_bid), np.asarray(q_ask),
                            horizon_ms, latency_ms)
    df = pd.concat([df, labels], axis=1)
    df.insert(0, "symbol", symbol)
    df.insert(1, "day", day)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{symbol}_{day}.parquet")
    df.astype({c: "float32" for c in df.columns if df[c].dtype == np.float64}).to_parquet(path, index=False)
    log.info("%s %s: %d rows, %d quotes, malformed=%s, %.0fs", symbol, day, len(df), len(q_ts),
             feed.summary()["malformed_by_kind"], time.perf_counter() - t_wall)
    return path


def barrier_labels(t: np.ndarray, mid0: np.ndarray, q_ts: np.ndarray, q_bid: np.ndarray, q_ask: np.ndarray,
                   horizon_ms: int, latency_ms: int) -> pd.DataFrame:
    """First-touch times (ms after entry; INF = not within horizon) and outcome stats."""
    n = len(t)
    out: dict[str, np.ndarray] = {}
    for side in ("long", "short"):
        for T in TARGETS_BPS:
            out[f"ttp_{side}_{T}"] = np.full(n, INF, np.int64)
        for S in STOPS_BPS:
            out[f"tsl_{side}_{S}"] = np.full(n, INF, np.int64)
        out[f"mfe_{side}"] = np.full(n, np.nan)
        out[f"mae_{side}"] = np.full(n, np.nan)
        out[f"hret_{side}"] = np.full(n, np.nan)      # executable return at the horizon (timeout exit)
    for h in RET_HORIZONS_S:
        out[f"fret_{h}s"] = np.full(n, np.nan)
    out["entry_lag_ms"] = np.full(n, np.nan)
    tp_lv = np.asarray(TARGETS_BPS, float) / 1e4
    sl_lv = np.asarray(STOPS_BPS, float) / 1e4
    for i in range(n):
        e = int(np.searchsorted(q_ts, t[i] + latency_ms, side="right")) - 1   # prevailing quote
        if e < 0 or q_ts[e] < t[i] - 5000:
            continue
        t_e = t[i] + latency_ms
        out["entry_lag_ms"][i] = t_e - q_ts[e]
        end = int(np.searchsorted(q_ts, t_e + horizon_ms, side="right"))
        fts = q_ts[e + 1:end]
        if len(fts) == 0:
            continue
        rel = fts - t_e
        bid, ask = q_bid[e + 1:end], q_ask[e + 1:end]
        # LONG: buy at ask[e], exit side = bid
        ent = q_ask[e]
        cmax, cmin = np.maximum.accumulate(bid), np.minimum.accumulate(bid)
        k = np.searchsorted(cmax, ent * (1 + tp_lv), side="left")
        out_ttp = np.where(k < len(rel), rel[np.minimum(k, len(rel) - 1)], INF)
        for j, T in enumerate(TARGETS_BPS):
            out[f"ttp_long_{T}"][i] = out_ttp[j]
        k = np.searchsorted(-cmin, -ent * (1 - sl_lv), side="left")
        out_tsl = np.where(k < len(rel), rel[np.minimum(k, len(rel) - 1)], INF)
        for j, S in enumerate(STOPS_BPS):
            out[f"tsl_long_{S}"][i] = out_tsl[j]
        out["mfe_long"][i] = (cmax[-1] - ent) / ent * 1e4
        out["mae_long"][i] = (cmin[-1] - ent) / ent * 1e4
        out["hret_long"][i] = (bid[-1] - ent) / ent * 1e4
        # SHORT: sell at bid[e], exit side = ask
        ent = q_bid[e]
        cmin_a, cmax_a = np.minimum.accumulate(ask), np.maximum.accumulate(ask)
        k = np.searchsorted(-cmin_a, -ent * (1 - tp_lv), side="left")
        out_ttp = np.where(k < len(rel), rel[np.minimum(k, len(rel) - 1)], INF)
        for j, T in enumerate(TARGETS_BPS):
            out[f"ttp_short_{T}"][i] = out_ttp[j]
        k = np.searchsorted(cmax_a, ent * (1 + sl_lv), side="left")
        out_tsl = np.where(k < len(rel), rel[np.minimum(k, len(rel) - 1)], INF)
        for j, S in enumerate(STOPS_BPS):
            out[f"tsl_short_{S}"][i] = out_tsl[j]
        out["mfe_short"][i] = (ent - cmin_a[-1]) / ent * 1e4
        out["mae_short"][i] = (ent - cmax_a[-1]) / ent * 1e4
        out["hret_short"][i] = (ent - ask[-1]) / ent * 1e4
        # forward mid returns from the DECISION mid (not the entry)
        mids = 0.5 * (bid + ask)
        for h in RET_HORIZONS_S:
            kk = int(np.searchsorted(fts, t[i] + h * 1000, side="left"))
            if kk < len(fts):
                out[f"fret_{h}s"][i] = (mids[kk] - mid0[i]) / mid0[i] * 1e4
    return pd.DataFrame(out)


def tp_first(df: pd.DataFrame, side: str, target: int, stop: int, horizon_ms: int = 60_000) -> np.ndarray:
    """1 if the target is touched before the stop within the horizon (ties -> stop)."""
    ttp = df[f"ttp_{side}_{target}"].to_numpy()
    tsl = df[f"tsl_{side}_{stop}"].to_numpy()
    return ((ttp < tsl) & (ttp <= horizon_ms)).astype(np.int8)


def load_dataset(path: str, days: list[str] | None = None) -> pd.DataFrame:
    files = sorted(f for f in os.listdir(path) if f.endswith(".parquet"))
    if days:
        files = [f for f in files if f.rsplit("_", 1)[1][:10] in days]
    return pd.concat([pd.read_parquet(os.path.join(path, f)) for f in files], ignore_index=True)


def check_days(days: list[str], forbidden: list[str]) -> None:
    bad = sorted(set(days) & set(forbidden))
    if bad:
        raise SystemExit(f"refusing to build dataset rows for forbidden (test) days: {bad}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--events", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config")
    ap.add_argument("--symbols", nargs="+", required=True)
    ap.add_argument("--days", nargs="+", required=True)
    ap.add_argument("--forbid-days", nargs="*", default=[], help="days that must not be used (locked test)")
    ap.add_argument("--window", nargs=2, type=int, default=[0, 24])
    ap.add_argument("--sample-ms", type=int, default=1000)
    ap.add_argument("--latency-ms", type=int, default=100)
    ap.add_argument("--horizon-ms", type=int, default=60_000)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    check_days(args.days, args.forbid_days)
    cfg: BotConfig = load_config(args.config)
    jobs = [(args.events, args.out, s, d, tuple(args.window), cfg, args.sample_ms,
             cfg.strategy.eval_interval_ms, args.horizon_ms, args.latency_ms, cfg.entry.min_net_profit_usdt)
            for d in args.days for s in args.symbols]
    jobs.sort(key=lambda j: {"BTCUSDT": 0, "ETHUSDT": 1, "SOLUSDT": 2}.get(j[2], 3))   # longest first
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        paths = [p for p in ex.map(build_shard, jobs) if p]
    with open(os.path.join(args.out, "dataset_meta.json"), "w", encoding="utf-8") as fh:
        json.dump({"days": args.days, "forbidden_days": args.forbid_days, "symbols": args.symbols,
                   "window_utc": args.window, "sample_ms": args.sample_ms, "latency_ms": args.latency_ms,
                   "horizon_ms": args.horizon_ms, "targets_bps": TARGETS_BPS, "stops_bps": STOPS_BPS,
                   "notionals": NOTIONALS, "min_net_usdt": cfg.entry.min_net_profit_usdt,
                   "files": paths}, fh, indent=1)
    print(f"built {len(paths)} shards -> {args.out}")


if __name__ == "__main__":
    main()
