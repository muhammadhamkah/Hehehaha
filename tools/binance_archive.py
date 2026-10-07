"""Download Binance public USDT-M archive data and make it replayable.

Source: https://data.binance.vision  (futures/um/daily/{aggTrades,bookTicker})

    # download 3 days of BTCUSDT and ETHUSDT and write a replay manifest (fast, default)
    python -m tools.binance_archive --symbols BTCUSDT ETHUSDT --start 2024-03-01 --days 3 \\
        --out data/archive_btc_eth
    python -m backtest.replay --events data/archive_btc_eth --out runs/a1

    # optionally convert to the live recorder's event format (slow: ~8k events/s)
    python -m tools.binance_archive ... --convert --out data/archive_events

The default mode writes ``archive.json`` listing the zip files; replay streams them
directly (data/archive_reader.py). Converted output uses the SAME format as the live recorder, with payloads shaped exactly like the
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
import heapq
import json
import logging
import os
import urllib.request
from dataclasses import asdict
from datetime import date, timedelta
from itertools import islice

from data.archive_reader import MANIFEST, agg_trade_events, book_ticker_events
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


def sample_symbol_info(symbol: str, agg_file: str) -> SymbolInfo:
    rows = list(islice(agg_trade_events(agg_file, symbol), 2000))
    return infer_symbol_info(symbol, [d["p"] for _, d in rows], [d["q"] for _, d in rows])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbols", nargs="+", required=True)
    ap.add_argument("--start", required=True, help="YYYY-MM-DD")
    ap.add_argument("--days", type=int, default=1)
    ap.add_argument("--download-dir", default="data/archive_raw")
    ap.add_argument("--out", required=True)
    ap.add_argument("--convert", action="store_true", help="also convert to recorder event files (slow)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    d0 = date.fromisoformat(args.start)
    days = [(d0 + timedelta(days=i)).isoformat() for i in range(args.days)]
    os.makedirs(args.out, exist_ok=True)
    meta: dict = {"source": "binance_public_archive_L1", "depth_mode": "bbo", "days": days,
                  "symbol_info": {}, "files": {}, "missing": []}
    for sym in args.symbols:
        files: dict[str, list[str]] = {"aggTrades": [], "bookTicker": []}
        for day in days:
            for kind in files:
                fname = f"{sym}-{kind}-{day}.zip"
                dest = os.path.join(args.download_dir, sym, fname)
                if download(archive_url(kind, sym, day), dest):
                    files[kind].append(os.path.abspath(dest))
                else:
                    meta["missing"].append(fname)
        if not files["aggTrades"] or not files["bookTicker"]:
            log.error("%s: need both aggTrades and bookTicker files; skipping", sym)
            continue
        meta["files"][sym] = files
        meta["symbol_info"][sym] = asdict(sample_symbol_info(sym, files["aggTrades"][0]))
        log.info("%s: %d days ready", sym, len(files["bookTicker"]))
    with open(os.path.join(args.out, MANIFEST), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1)

    if args.convert:
        writer = EventWriter(args.out, blocking=True)
        expected = 0
        for sym, files in meta["files"].items():
            res = convert(sym, files["aggTrades"], files["bookTicker"], writer)
            expected += sum(res["counts"].values())
            log.info("%s converted: %s", sym, res["counts"])
        writer.close()
        if writer.dropped or writer.n_written != expected:
            raise SystemExit(f"conversion incomplete: wrote {writer.n_written}, expected {expected}, "
                             f"dropped {writer.dropped}")
        os.remove(os.path.join(args.out, MANIFEST))   # event files now take precedence
        writer.write_meta({k: v for k, v in meta.items() if k != "files"})
    if meta["missing"]:
        log.warning("missing archive files: %s", meta["missing"])
    print(json.dumps({k: v for k, v in meta.items() if k not in ("symbol_info", "files")}, indent=1))
    print(f"replay with: python -m backtest.replay --events {args.out} --out runs/archive1")


if __name__ == "__main__":
    main()
