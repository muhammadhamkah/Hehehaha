"""Download / convert Binance public USDT-M archive data into replayable event files.

Source: https://data.binance.vision  (futures/um/daily/{aggTrades,bookTicker})

    # download + convert 3 days of BTCUSDT and ETHUSDT
    python -m tools.binance_archive --symbols BTCUSDT ETHUSDT --start 2024-03-01 --days 3 \\
        --out data/archive_events

    # convert already-downloaded zip/csv files
    python -m tools.binance_archive --convert-only --input-dir downloads/ --out data/archive_events

Output uses the SAME format as the live recorder, with payloads shaped exactly like the
WebSocket messages (``<sym>@aggTrade`` and ``<sym>@bookTicker``), so replay runs them
through the same validators and handlers. Replay this data with
``market_data.depth_mode = "bbo"`` and ``scanner.static_symbols`` (see backtest/replay.py).

LIMITATIONS (be explicit about them in any conclusion):
  * L1 only: the archive has best bid/ask, not depth. Multi-level imbalance, depth
    depletion and book-walk slippage degrade to top-of-book approximations, and taker
    fills beyond the top level cannot be simulated. Treat results as indicative.
  * Binance has not published bookTicker dumps for every period/symbol; missing files
    are reported, not silently skipped.
  * Archive timestamps are exchange times; there is no local receive time (receive time
    is set equal to the exchange time).
"""
from __future__ import annotations

import argparse
import csv
import heapq
import io
import json
import logging
import os
import urllib.request
import zipfile
from dataclasses import asdict
from datetime import date, timedelta
from typing import Iterator

from data.event_store import EventWriter
from exchange.models import SymbolInfo

log = logging.getLogger("archive")
BASE = "https://data.binance.vision/data/futures/um/daily"


def archive_url(kind: str, symbol: str, day: str) -> str:
    return f"{BASE}/{kind}/{symbol}/{symbol}-{kind}-{day}.zip"


def download(url: str, dest: str) -> bool:
    if os.path.exists(dest):
        return True
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    try:
        with urllib.request.urlopen(url, timeout=60) as resp, open(dest + ".part", "wb") as fh:
            fh.write(resp.read())
        os.replace(dest + ".part", dest)
        return True
    except Exception as exc:  # noqa: BLE001
        log.warning("download failed %s: %s", url, exc)
        return False


def _csv_rows(path: str) -> Iterator[list[str]]:
    if path.endswith(".zip"):
        with zipfile.ZipFile(path) as zf:
            for name in zf.namelist():
                with zf.open(name) as fh:
                    yield from csv.reader(io.TextIOWrapper(fh, encoding="utf-8"))
    else:
        with open(path, encoding="utf-8") as fh:
            yield from csv.reader(fh)


def agg_trade_events(path: str, symbol: str) -> Iterator[tuple[int, dict]]:
    """Columns: agg_trade_id,price,quantity,first_trade_id,last_trade_id,transact_time,is_buyer_maker"""
    for row in _csv_rows(path):
        if not row or not row[0].strip().lstrip("-").isdigit():
            continue  # header
        a, p, q, f, l, t, m = row[:7]
        ts = int(t)
        yield ts, {"e": "aggTrade", "E": ts, "s": symbol, "a": int(a), "p": p, "q": q,
                   "f": int(f), "l": int(l), "T": ts, "m": m.strip().lower() == "true"}


def book_ticker_events(path: str, symbol: str) -> Iterator[tuple[int, dict]]:
    """Columns: update_id,best_bid_price,best_bid_qty,best_ask_price,best_ask_qty,transaction_time,event_time"""
    for row in _csv_rows(path):
        if not row or not row[0].strip().isdigit():
            continue
        u, b, bq, a, aq, tt, et = row[:7]
        ev = int(et) if et.strip() else int(tt)
        yield ev, {"e": "bookTicker", "u": int(u), "E": ev, "T": int(tt), "s": symbol,
                   "b": b, "B": bq, "a": a, "A": aq}


def infer_symbol_info(symbol: str, prices: list[str], qtys: list[str]) -> SymbolInfo:
    def step(vals: list[str]) -> float:
        dec = max((len(v.split(".")[1].rstrip("0")) if "." in v else 0) for v in vals) if vals else 2
        return 10 ** -dec

    return SymbolInfo(symbol, tick_size=step(prices), step_size=step(qtys), min_qty=step(qtys), min_notional=5.0)


def convert(symbol: str, agg_files: list[str], bt_files: list[str], writer: EventWriter) -> dict:
    stream_a = f"{symbol.lower()}@aggTrade"
    stream_b = f"{symbol.lower()}@bookTicker"
    prices: list[str] = []
    qtys: list[str] = []
    n = {"aggTrade": 0, "bookTicker": 0}

    def tagged(it, kind):
        for ts, d in it:
            yield ts, (kind, d)

    its = [tagged(agg_trade_events(f, symbol), "a") for f in agg_files]
    its += [tagged(book_ticker_events(f, symbol), "b") for f in bt_files]
    for ts, (kind, d) in heapq.merge(*its, key=lambda x: x[0]):
        if kind == "a":
            writer.write("detail", stream_a, d, ts)
            n["aggTrade"] += 1
            if len(prices) < 2000:
                prices.append(d["p"])
                qtys.append(d["q"])
        else:
            writer.write("detail", stream_b, d, ts)
            n["bookTicker"] += 1
    info = infer_symbol_info(symbol, prices, qtys)
    return {"counts": n, "symbol_info": asdict(info)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbols", nargs="+", required=True)
    ap.add_argument("--start", help="YYYY-MM-DD")
    ap.add_argument("--days", type=int, default=1)
    ap.add_argument("--download-dir", default="data/archive_raw")
    ap.add_argument("--convert-only", action="store_true")
    ap.add_argument("--input-dir", default=None, help="with --convert-only: dir containing the zip/csv files")
    ap.add_argument("--out", default="data/archive_events")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    days: list[str] = []
    if args.start:
        d0 = date.fromisoformat(args.start)
        days = [(d0 + timedelta(days=i)).isoformat() for i in range(args.days)]
    src = args.input_dir or args.download_dir
    writer = EventWriter(args.out)
    meta: dict = {"source": "binance_public_archive_L1", "depth_mode": "bbo", "symbol_info": {}, "missing": []}
    for sym in args.symbols:
        aggs, bts = [], []
        for day in days:
            for kind, bucket in (("aggTrades", aggs), ("bookTicker", bts)):
                fname = f"{sym}-{kind}-{day}.zip"
                dest = os.path.join(src, sym, fname)
                if not args.convert_only:
                    download(archive_url(kind, sym, day), dest)
                if os.path.exists(dest):
                    bucket.append(dest)
                else:
                    meta["missing"].append(fname)
        if args.convert_only and not days:
            folder = os.path.join(src, sym)
            files = sorted(os.listdir(folder)) if os.path.isdir(folder) else []
            aggs = [os.path.join(folder, f) for f in files if "aggTrades" in f]
            bts = [os.path.join(folder, f) for f in files if "bookTicker" in f]
        if not aggs or not bts:
            log.error("%s: need both aggTrades and bookTicker files (have %d / %d); skipping",
                      sym, len(aggs), len(bts))
            continue
        res = convert(sym, aggs, bts, writer)
        meta["symbol_info"][sym] = res["symbol_info"]
        log.info("%s converted: %s", sym, res["counts"])
    writer.close()
    writer.write_meta(meta)
    if meta["missing"]:
        log.warning("missing archive files: %s", meta["missing"])
    print(json.dumps({k: v for k, v in meta.items() if k != "symbol_info"}, indent=1))
    print(f"replay with: python -m backtest.replay --events {args.out} --depth-mode bbo "
          f"--symbols {' '.join(args.symbols)}")


if __name__ == "__main__":
    main()
