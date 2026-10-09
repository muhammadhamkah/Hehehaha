"""Validate real Binance USDT-M connectivity and every parser against live payloads.

READ-ONLY: uses public REST + public WebSocket streams only. Never places orders.

    python -m tools.validate_binance --symbols 10 --duration 30 --out validation_report.json

Checks (each reported PASS / FAIL / WARN with details):
  rest_discovery      active USDT perpetuals discovered, filters parse
  clock_skew          local clock vs Binance server time
  market_streams      !ticker@arr and !bookTicker payloads validate
  detail_streams      <sym>@aggTrade, <sym>@depth20@100ms, <sym>@bookTicker validate
  timestamps          event-time monotonicity and receive latency
  update_ids          aggTrade id gaps, depth pu/u continuity, bookTicker u monotonic
  diff_depth_sync     <sym>@depth@100ms + REST snapshot sync per Binance rules
  resubscription      unsubscribed streams stop, resubscribed streams resume
  reconnect           forced socket close -> reconnect + automatic resubscription
  stale_detection     a silent stream is flagged stale by the RiskManager
  subscription_limit  many streams on one connection (acks, errors, delivery)
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from collections import defaultdict
from typing import Any

from config import BotConfig, load_config
from exchange.binance_client import BinanceFuturesClient
from exchange.schemas import FeedValidator
from exchange.websocket_manager import StreamConnection
from market_data.orderbook import OrderBook, SyncStatus
from risk.risk_manager import RiskManager

log = logging.getLogger("validate")


class Collector:
    def __init__(self) -> None:
        self.v = FeedValidator()
        self.counts: dict[str, int] = defaultdict(int)
        self.last_seen: dict[str, float] = {}
        self.samples: dict[str, list] = defaultdict(list)

    def on_message(self, stream: str, data: Any, ts: int) -> None:
        self.counts[stream] += 1
        self.last_seen[stream] = time.monotonic()
        sym, _, kind = stream.partition("@")
        key = kind or stream
        if len(self.samples[key]) < 2:
            self.samples[key].append(data)
        if stream == "!ticker@arr":
            self.v.ticker_arr(stream, data, ts)
        elif stream == "!bookTicker":
            self.v.book_ticker(stream, data, ts)
        elif kind == "aggTrade":
            self.v.agg_trade(stream, data, ts, sym.upper())
        elif kind.startswith("depth") and "@" in kind and kind.split("@")[0] != "depth":
            self.v.depth(stream, data, ts, sym.upper(), partial=True)
        elif kind.startswith("depth"):
            self.v.depth(stream, data, ts, sym.upper(), partial=False)
        elif kind == "bookTicker":
            self.v.book_ticker(stream, data, ts, sym.upper())
        else:
            self.v.report_unexpected(stream, data)


def result(status: str, **details: Any) -> dict[str, Any]:
    return {"status": status, **details}


async def wait_for(pred, timeout: float) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return True
        await asyncio.sleep(0.1)
    return False


async def run(cfg: BotConfig, n_symbols: int, duration: float, limit_streams: int,
              min_symbols: int = 50) -> dict[str, Any]:
    report: dict[str, Any] = {"started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "checks": {}}
    checks = report["checks"]
    client = BinanceFuturesClient(cfg.exchange, allow_trading=False)
    await client.start()
    try:
        # ---------------------------------------------------------- REST
        t0 = time.time() * 1000
        offset = await client.sync_time()
        rtt = time.time() * 1000 - t0
        checks["clock_skew"] = result("PASS" if abs(offset) < 1000 else "WARN", offset_ms=offset, rtt_ms=round(rtt))
        info = await client.perpetual_symbols("USDT")
        bad = [s for s, i in info.items() if i.tick_size <= 0 or i.step_size <= 0]
        checks["rest_discovery"] = result("PASS" if len(info) >= min_symbols and not bad else "FAIL",
                                          perpetual_usdt_trading=len(info), bad_filters=bad[:10])
        tickers = await client.ticker_24h()
        vol = sorted((t for t in tickers if t["symbol"] in info), key=lambda t: -float(t["quoteVolume"]))
        symbols = [t["symbol"] for t in vol[:n_symbols]]
        report["symbols"] = symbols

        # ---------------------------------------------------------- streams
        col = Collector()
        risk = RiskManager(cfg)
        connects: list[str] = []
        disconnects: list[str] = []

        def on_msg(stream, data, ts):
            risk.heartbeat("detail" if "@" in stream and not stream.startswith("!") else "market", ts)
            risk.heartbeat(stream, ts)
            col.on_message(stream, data, ts)

        market = StreamConnection("market", cfg.exchange.ws_url, on_msg)
        detail = StreamConnection("detail", cfg.exchange.ws_url, on_msg,
                                  on_connect=connects.append, on_disconnect=disconnects.append)
        streams = []
        for s in symbols:
            sl = s.lower()
            streams += [f"{sl}@aggTrade", f"{sl}@depth20@100ms", f"{sl}@bookTicker"]
        await market.set_streams(["!ticker@arr", "!bookTicker"])
        await detail.set_streams(streams)
        tasks = [asyncio.ensure_future(market.run()), asyncio.ensure_future(detail.run())]
        log.info("collecting %ss of data for %d symbols", duration, len(symbols))
        await asyncio.sleep(duration)

        summary = col.v.summary()
        report["feed_health"] = summary
        report["message_counts"] = dict(col.counts)
        report["samples"] = dict(col.samples)
        bk = summary["by_kind"]
        mal = summary["malformed_by_kind"]
        checks["market_streams"] = result(
            "PASS" if col.counts["!ticker@arr"] > 0 and col.counts["!bookTicker"] > 0 and not mal.get("ticker")
            and not mal.get("ticker_arr") else "FAIL",
            ticker_arr_msgs=col.counts["!ticker@arr"], all_book_ticker_msgs=col.counts["!bookTicker"])
        silent = [s for s in streams if col.counts.get(s, 0) == 0]
        checks["detail_streams"] = result(
            "PASS" if not silent and not mal else ("WARN" if not mal else "FAIL"),
            silent_streams=silent, malformed=mal, unexpected=summary["unexpected"])
        nonmono = sum(v.get("non_monotonic_ts", 0) for v in bk.values())
        checks["timestamps"] = result(
            "PASS" if nonmono == 0 else "WARN", non_monotonic=nonmono,
            latency_p50={k: v["latency_ms_p50"] for k, v in bk.items()},
            latency_p99={k: v["latency_ms_p99"] for k, v in bk.items()})
        checks["update_ids"] = result(
            "PASS" if all(v["duplicates"] == 0 for v in bk.values()) else "WARN",
            agg_missing_ids=bk.get("aggTrade", {}).get("missing_ids"),
            agg_gaps=bk.get("aggTrade", {}).get("seq_gaps"),
            depth_pu_gaps=bk.get("depth", {}).get("seq_gaps"),
            duplicates={k: v["duplicates"] for k, v in bk.items()},
            note="aggTrade id gaps can occur legitimately on reconnect; depth20 pu gaps are informational")

        # ---------------------------------------------------------- resubscription
        drop = symbols[: max(1, len(symbols) // 2)]
        drop_streams = {f"{s.lower()}@aggTrade" for s in drop}
        await detail.set_streams([s for s in streams if s not in drop_streams])
        await asyncio.sleep(3)
        mark = dict(col.counts)
        await asyncio.sleep(3)
        still = [s for s in drop_streams if col.counts.get(s, 0) > mark.get(s, 0)]
        await detail.set_streams(streams)
        resumed = await wait_for(lambda: all(col.counts.get(s, 0) > mark.get(s, 0) for s in drop_streams), 15)
        checks["resubscription"] = result("PASS" if not still and resumed else "FAIL",
                                          still_receiving_after_unsub=still, resumed=resumed)

        # ---------------------------------------------------------- stale detection
        quiet = f"{symbols[-1].lower()}@bookTicker"
        await detail.set_streams([s for s in streams if s != quiet])
        await asyncio.sleep(cfg.risk.stale_data_ms / 1000 + 1.5)
        flagged = risk.stream_stale(quiet)
        await detail.set_streams(streams)
        recovered = await wait_for(lambda: not risk.stream_stale(quiet), 10)
        checks["stale_detection"] = result("PASS" if flagged and recovered else "FAIL",
                                           flagged_when_silent=flagged, recovered=recovered)

        # ---------------------------------------------------------- reconnect
        n_conn = len(connects)
        before = dict(col.counts)
        if detail._ws is not None:
            await detail._ws.close()
        reconnected = await wait_for(lambda: len(connects) > n_conn, 30)
        flowing = await wait_for(lambda: all(col.counts.get(s, 0) > before.get(s, 0) for s in streams
                                             if s.endswith("@bookTicker")), 20)
        checks["reconnect"] = result("PASS" if reconnected and flowing else "FAIL",
                                     reconnected=reconnected, data_resumed_on_all_streams=flowing,
                                     disconnect_events=len(disconnects))

        # ---------------------------------------------------------- diff depth sync
        sym = symbols[0]
        book = OrderBook(sym)
        statuses: dict[str, int] = defaultdict(int)

        def on_diff(stream, data, ts):
            st = book.apply_diff(data, ts)
            statuses[st] += 1

        diff = StreamConnection("diff", cfg.exchange.ws_url, on_diff)
        await diff.set_streams([f"{sym.lower()}@depth@100ms"])
        dtask = asyncio.ensure_future(diff.run())
        await asyncio.sleep(1.5)
        snap = await client.depth(sym, 1000)
        st = book.sync_from_snapshot([(float(p), float(q)) for p, q in snap["bids"]],
                                     [(float(p), float(q)) for p, q in snap["asks"]],
                                     int(snap["lastUpdateId"]), int(time.time() * 1000))
        await asyncio.sleep(10)
        await diff.stop()
        dtask.cancel()
        checks["diff_depth_sync"] = result(
            "PASS" if book.synced and statuses.get(SyncStatus.RESYNC, 0) == 0 and not book.crossed() else "FAIL",
            initial=st, statuses=dict(statuses), synced=book.synced, spread_bps=round(book.spread_bps, 3))

        # ---------------------------------------------------------- subscription limit
        many_syms = [t["symbol"] for t in vol[:limit_streams // 3]]
        many = [f"{s.lower()}@{k}" for s in many_syms for k in ("aggTrade", "bookTicker", "depth20@100ms")]
        lim_col = Collector()
        lim = StreamConnection("limit", cfg.exchange.ws_url, lim_col.on_message, max_streams=1024)
        await lim.set_streams(many)
        ltask = asyncio.ensure_future(lim.run())
        await asyncio.sleep(max(15.0, len(many) / 50 * 0.25 + 10))
        delivering = sum(1 for s in many if lim_col.counts.get(s, 0) > 0)
        await lim.stop()
        ltask.cancel()
        checks["subscription_limit"] = result(
            "PASS" if lim.n_sub_errors == 0 and delivering >= 0.9 * len(many) else "WARN",
            requested=len(many), delivering=delivering, sub_errors=lim.n_sub_errors,
            unacked=len(lim.pending_acks),
            note="bookTicker of illiquid symbols may legitimately be quiet; errors indicate a limit")

        for c in (market, detail):
            await c.stop()
        for t in tasks:
            t.cancel()
    finally:
        await client.close()
    statuses = [c["status"] for c in checks.values()]
    report["overall"] = "FAIL" if "FAIL" in statuses else ("WARN" if "WARN" in statuses else "PASS")
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate Binance USDT-M connectivity and parsers (read-only)")
    ap.add_argument("--config")
    ap.add_argument("--symbols", type=int, default=10)
    ap.add_argument("--duration", type=float, default=30.0)
    ap.add_argument("--limit-streams", type=int, default=300)
    ap.add_argument("--out", default="validation_report.json")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config(args.config)
    report = asyncio.run(run(cfg, args.symbols, args.duration, args.limit_streams))
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)
    for name, c in report["checks"].items():
        print(f"{c['status']:<5} {name}")
    print(f"OVERALL: {report['overall']}  (details: {args.out})")


if __name__ == "__main__":
    main()
