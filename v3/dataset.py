"""V3 dataset builder: recorded L2 store -> per (symbol, day) parquet of features + labels.

    python -m v3.dataset --store data/l2 --out data/v3/ds --days 2025-01-06 ... \\
        [--symbols BTCUSDT ...] [--sample-ms 1000] [--window 0 24] [--workers 4]

Every row holds, computed only from information available at the decision time:
  * V2 features (unprefixed) from an L1-only view of the SAME recording (bookTicker book +
    trades), i.e. exactly V2's definition -> V2 Stage A and the "L1 only" ablation
  * V3 features (v3d_/v3f_/v3t_/v3x_) from the full diff-depth book + trades
  * depth-based round-trip costs per notional, by walking the actual L2 book
plus forward labels (``v3.labels``). Rows are dropped -- and counted -- when the L2 book is
not continuously valid (gaps, resyncs, warm-up after a resync) or the label path crosses a
disconnect. A timeline parquet (every 250 ms evaluation, a few key series) supports the
entry-timing study.
"""
from __future__ import annotations

import os as _os

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    _os.environ.setdefault(_var, "1")

import argparse
import hashlib
import json
import logging
import os
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import date, datetime, timezone

import numpy as np
import pandas as pd

from config import BotConfig
from data.event_store import load_meta
from v3.features import FEATURE_VERSION
from v3.labels import LABEL_CONFIG, PAIR_HORIZON_MS, labels
from v3.state import SymbolState
from v3.store import V3StoreReader

log = logging.getLogger("v3.dataset")
NOTIONALS = (100, 150, 250, 500, 1000)
WARMUP_MS = 15 * 60_000          # >= the recorder's audit-snapshot interval
MAX_QUOTE_GAP_MS = 3_000


def day_ms(day: str) -> int:
    d = date.fromisoformat(day)
    return int(datetime(d.year, d.month, d.day, tzinfo=timezone.utc).timestamp() * 1000)


def costs_from_book(book, cfg_costs, notional: float) -> dict[str, float]:
    """Round-trip cost in bps beyond the touch: fees + book walk (entry AND exit side, both
    from the actual L2 depth now) + latency/extra buffers. Labels already pay the spread."""
    fee_t = cfg_costs.taker_fee * 1e4
    fee_m = cfg_costs.maker_fee * 1e4
    buf = cfg_costs.latency_slippage_bps + cfg_costs.extra_slippage_bps
    wb, ws = book.walk_bps("BUY", notional), book.walk_bps("SELL", notional)
    n = int(notional)
    return {f"cost_long_{n}": 2 * fee_t + wb + ws + 2 * buf, f"cost_short_{n}": 2 * fee_t + ws + wb + 2 * buf,
            f"costm_long_{n}": fee_m + fee_t + cfg_costs.maker_adverse_selection_bps + ws + buf,
            f"costm_short_{n}": fee_m + fee_t + cfg_costs.maker_adverse_selection_bps + wb + buf}


def build_shard(args: tuple) -> dict:
    store, out, symbol, day, window, sample_ms, eval_ms, latency_ms = args
    t_wall = time.perf_counter()
    cfg = BotConfig()
    d0 = day_ms(day)
    w0, w1 = d0 + window[0] * 3_600_000, d0 + window[1] * 3_600_000
    meta = load_meta(os.path.join(store, symbol))
    tick = (meta.get("symbol_info", {}).get(symbol) or {}).get("tick_size")
    st = SymbolState(symbol, tick=tick, eval_ms=eval_ms, cfg=cfg,
                     want_row=lambda t: w0 <= t < w1 and (t - w0) % sample_ms == 0,
                     want_timeline=lambda t: w0 <= t < w1)
    st.collect = True
    st.record_quotes = True
    reader = V3StoreReader(store, symbols=[symbol], start_ms=w0 - WARMUP_MS,
                           end_ms=w1 + PAIR_HORIZON_MS + latency_ms + 5000)
    rows = []
    for ev in reader:
        n_before = len(st.rows)
        st.on_message(ev.stream, ev.data, ev.ts)
        if len(st.rows) > n_before:                     # attach costs from the book at that instant
            for r in st.rows[n_before:]:
                for n in NOTIONALS:
                    r.update(costs_from_book(st.engine.book, cfg.costs, n))
    rows = st.rows
    q_ts, q_bid, q_ask = st.quotes
    eng = st.engine
    stats = {"symbol": symbol, "day": day, "rows": len(rows), "dropped": dict(st.why), "quotes": len(q_ts),
             "l2_gaps": eng.n_gaps, "gap_reasons": dict(eng.book.stats.gap_reasons),
             "malformed": st.feed.summary()["malformed_by_kind"], "seconds": round(time.perf_counter() - t_wall, 1)}
    if not rows:
        return stats
    df = pd.DataFrame(rows)
    lab = labels(df["ts_ms"].to_numpy(np.int64), df["mid0"].to_numpy(float), np.asarray(q_ts, np.int64),
                 np.asarray(q_bid, float), np.asarray(q_ask, float), latency_ms)
    df = pd.concat([df, lab], axis=1)
    bad = df["quote_gap_ms"].isna() | (df["quote_gap_ms"] > MAX_QUOTE_GAP_MS)
    stats["dropped"]["label_gap"] = int(bad.sum())
    df = df[~bad].reset_index(drop=True)
    stats["rows"] = len(df)
    df.insert(0, "symbol", symbol)
    df.insert(1, "day", day)
    os.makedirs(os.path.join(out, "timeline"), exist_ok=True)
    path = os.path.join(out, f"{symbol}_{day}.parquet")
    df.astype({c: "float32" for c in df.columns if df[c].dtype == np.float64}).to_parquet(path, index=False)
    feats = [c for c in df.columns if c.startswith(("v3d_", "v3f_", "v3t_", "v3x_"))]
    stats.update({"file": os.path.basename(path), "sha256": sha256_file(path), "feature_version": FEATURE_VERSION,
                  "n_columns": len(df.columns), "n_v3_features": len(feats),
                  "columns_sha256": hashlib.sha256("\n".join(df.columns).encode()).hexdigest(),
                  "label_config": LABEL_CONFIG, "sample_ms": sample_ms, "eval_ms": eval_ms, "window_utc_h": list(window),
                  "latency_ms": latency_ms, "expected_rows": int((window[1] - window[0]) * 3_600_000 // sample_ms),
                  "store": os.path.abspath(store), "built_at_ms": int(time.time() * 1000)})
    if st.timeline:
        pd.DataFrame(st.timeline).to_parquet(os.path.join(out, "timeline", f"{symbol}_{day}.parquet"), index=False)
    return stats


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def update_manifest(out: str, entries: list[dict]) -> None:
    """dataset_manifest.json: one entry per (symbol, day) shard, latest build wins."""
    path = os.path.join(out, "dataset_manifest.json")
    man = {"format": "v3_dataset", "shards": {}}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            man = json.load(fh)
    for e in entries:
        man["shards"][f"{e['symbol']}_{e['day']}"] = e
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(man, fh, indent=1)
    os.replace(tmp, path)


def load(path: str, days: list[str] | None = None, symbols: list[str] | None = None) -> pd.DataFrame:
    files = sorted(f for f in os.listdir(path) if f.endswith(".parquet"))
    if days:
        files = [f for f in files if f.rsplit("_", 1)[1][:10] in days]
    if symbols:
        files = [f for f in files if f.rsplit("_", 1)[0] in symbols]
    if not files:
        raise SystemExit(f"no dataset files for days={days} in {path}")
    return pd.concat([pd.read_parquet(os.path.join(path, f)) for f in files], ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--days", nargs="+", required=True)
    ap.add_argument("--symbols", nargs="+")
    ap.add_argument("--window", nargs=2, type=int, default=[0, 24])
    ap.add_argument("--sample-ms", type=int, default=1000,
                    help="training-row spacing (raw data is untouched). 1000: more rows, heavier; 2000: half the rows "
                         "and disk, nearly the same information because adjacent 1 s rows are highly overlapping")
    ap.add_argument("--eval-ms", type=int, default=250)
    ap.add_argument("--latency-ms", type=int, default=100)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    if a.sample_ms % a.eval_ms or a.sample_ms < a.eval_ms:
        raise SystemExit("--sample-ms must be a positive multiple of --eval-ms")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    from v3.store import store_symbols

    syms = a.symbols or store_symbols(a.store)
    jobs = [(a.store, a.out, s, d, tuple(a.window), a.sample_ms, a.eval_ms, a.latency_ms) for s in syms for d in a.days]
    os.makedirs(a.out, exist_ok=True)
    allstats = []
    with ProcessPoolExecutor(a.workers) as ex:
        for st in ex.map(build_shard, jobs):
            log.info("%s %s rows=%d dropped=%s gaps=%d (%ss)", st["symbol"], st["day"], st["rows"], st["dropped"],
                     st["l2_gaps"], st["seconds"])
            allstats.append(st)
            if st.get("sha256"):
                update_manifest(a.out, [st])
    with open(os.path.join(a.out, "build_stats.json"), "a", encoding="utf-8") as fh:
        for st in allstats:
            fh.write(json.dumps(st) + "\n")


if __name__ == "__main__":
    main()
