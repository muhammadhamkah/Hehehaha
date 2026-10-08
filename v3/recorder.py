"""V3 standalone L2 recorder for Binance USDT-M perpetuals (public market data only).

    python -m v3.recorder --out data/l2 --symbols BTCUSDT ETHUSDT SOLUSDT [--days 14]

Must run on a machine where Binance USDT-M market data is reachable and permitted. It
never places orders and needs no API key.

Recorded per symbol (one event store per symbol: ``<out>/<SYMBOL>/<YYYYMMDD>/<HH>.jsonl.gz``,
each line ``{"r": local_receive_ms, "c": "detail", "s": stream, "d": payload}``):
  <sym>@depth@100ms        diff depth (fastest supported USDT-M interval), U/u/pu ids
  <sym>@depth20@100ms      partial top-20 book (levels 1-20), cross-checks the diff book
  <sym>@bookTicker         every best bid/ask change
  <sym>@aggTrade           aggregated trades
  __snapshot__@SYM         REST /fapi/v1/depth?limit=1000 used to (re)synchronise
  __snapshot_audit__@SYM   periodic REST snapshot (lets any replay start mid-recording)
  __gap__@SYM              continuity break (reason, ids); the book is rebuilt from a snapshot
  __resync__@SYM           result of a resynchronisation
  __check__@SYM            diff-book vs depth20 mismatch (only when they disagree)
Exchange timestamps (E/T) and update ids are kept verbatim inside the payloads.

Also written: ``<out>/v3_store.json`` (manifest), ``<out>/_system`` (server-time samples),
``<out>/_health/YYYYMMDD.jsonl`` (per-minute health: message rates, gaps, resyncs, depth20
mismatches, latency percentiles, writer drops, disk space). ``python -m v3.qa`` turns these
into a recording-quality report.

Continuity is validated live with the same rules the dataset builder and the replay use;
on any break the gap is marked and the book is rebuilt from a fresh REST snapshot -- the
recorder never continues on a corrupted depth state.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import shutil
import signal
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any

from config import ExchangeConfig
from data.event_store import EventWriter
from exchange.binance_client import BinanceFuturesClient
from exchange.websocket_manager import StreamConnection
from v3.book import GAP, L2Book

log = logging.getLogger("v3.recorder")
VERSION = 1
DEFAULT_SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")


def now_ms() -> int:
    return int(time.time() * 1000)


@dataclass
class RecorderConfig:
    out: str = "data/l2"
    symbols: tuple[str, ...] = DEFAULT_SYMBOLS
    rest_base: str = "https://fapi.binance.com"
    ws_base: str = "wss://fstream.binance.com"
    diff_stream: str = "depth@100ms"
    partial_stream: str = "depth20@100ms"      # "" to disable
    snapshot_limit: int = 1000
    audit_interval_s: float = 600.0
    time_sync_interval_s: float = 600.0
    health_interval_s: float = 60.0
    resync_delay_s: float = 0.5               # let diffs buffer before the snapshot request
    min_free_gb: float = 5.0
    duration_s: float | None = None


@dataclass
class SymState:
    book: L2Book
    writer: EventWriter
    counts: dict = field(default_factory=lambda: defaultdict(int))
    lat: dict = field(default_factory=lambda: defaultdict(list))
    resyncing: bool = False
    gaps: int = 0
    resyncs: int = 0
    resync_fail: int = 0
    d20_checked: int = 0
    d20_mismatch: int = 0
    pending_d20: dict = field(default_factory=dict)
    dropped_seen: int = 0


class L2Recorder:
    def __init__(self, cfg: RecorderConfig) -> None:
        self.cfg = cfg
        os.makedirs(cfg.out, exist_ok=True)
        self.client = BinanceFuturesClient(ExchangeConfig(rest_base=cfg.rest_base, ws_base=cfg.ws_base))
        self.sym: dict[str, SymState] = {}
        self.system = EventWriter(os.path.join(cfg.out, "_system"))
        self.conn = StreamConnection("l2", cfg.ws_base, self._on_msg, self._on_connect, self._on_disconnect,
                                     max_streams=1000)
        self._stop = asyncio.Event()
        self._tasks: list[asyncio.Task] = []
        self.started_ms = now_ms()

    # ------------------------------------------------------------------ setup
    async def start(self) -> None:
        await self.client.start()
        info = {}
        try:
            infos = await self.client.perpetual_symbols()
            info = {s: asdict(infos[s]) for s in self.cfg.symbols if s in infos}
            missing = [s for s in self.cfg.symbols if s not in infos]
            if missing:
                raise SystemExit(f"not tradable USDT-M perpetuals: {missing}")
        except SystemExit:
            raise
        except Exception as exc:  # noqa: BLE001
            log.error("exchangeInfo failed (%s); tick sizes will be inferred later", exc)
        streams = {}
        for s in self.cfg.symbols:
            sl = s.lower()
            st = [f"{sl}@{self.cfg.diff_stream}", f"{sl}@bookTicker", f"{sl}@aggTrade"]
            if self.cfg.partial_stream:
                st.append(f"{sl}@{self.cfg.partial_stream}")
            streams[s] = st
            meta = {"symbol_info": {s: info[s]} if s in info else {}, "depth_mode": "diff",
                    "depth_levels": 20, "source": "binance-usdm-live", "recorder": "v3", "version": VERSION,
                    "streams": st, "snapshot_limit": self.cfg.snapshot_limit}
            self.sym[s] = SymState(L2Book(s), EventWriter(os.path.join(self.cfg.out, s), meta=meta))
        manifest = {"format": "v3_store", "version": VERSION, "symbols": list(self.cfg.symbols),
                    "streams": streams, "created_ms": self.started_ms, "config": asdict(self.cfg)}
        path = os.path.join(self.cfg.out, "v3_store.json")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as fh:
                old = json.load(fh)
            manifest["symbols"] = sorted(set(old.get("symbols", [])) | set(manifest["symbols"]))
            manifest["streams"] = {**old.get("streams", {}), **streams}
            manifest["created_ms"] = old.get("created_ms", self.started_ms)
            manifest.setdefault("sessions", old.get("sessions", []))
        manifest.setdefault("sessions", []).append({"start_ms": self.started_ms, "symbols": list(self.cfg.symbols)})
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=1)
        write_root_meta(self.cfg.out, info)
        await self.conn.set_streams([x for s in self.cfg.symbols for x in streams[s]])

    async def run(self) -> None:
        await self.start()
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, self._stop.set)
            except (NotImplementedError, RuntimeError):
                pass
        self._tasks = [asyncio.create_task(c) for c in (self.conn.run(), self._audit_loop(), self._time_loop(),
                                                         self._health_loop())]
        try:
            if self.cfg.duration_s:
                await asyncio.wait_for(self._stop.wait(), self.cfg.duration_s)
            else:
                await self._stop.wait()
        except asyncio.TimeoutError:
            pass
        await self.shutdown()

    async def shutdown(self) -> None:
        log.info("recorder stopping")
        await self.conn.stop()
        for t in self._tasks:
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._health_tick(final=True)
        for st in self.sym.values():
            st.writer.close()
        self.system.close()
        await self.client.close()

    def request_stop(self) -> None:
        self._stop.set()

    # ------------------------------------------------------------------ messages
    def _on_connect(self, name: str) -> None:
        log.info("connected; scheduling snapshots for %d symbols", len(self.sym))
        for s in self.sym:
            self._schedule_resync(s, "connect")

    def _on_disconnect(self, name: str) -> None:
        ts = now_ms()
        for s, st in self.sym.items():
            st.writer.write("detail", f"__gap__@{s}", {"reason": "disconnect", "last_u": st.book.last_u}, ts)
            st.gaps += 1
            st.book.mark_gap("disconnect")

    def _on_msg(self, stream: str, data: Any, ts: int) -> None:
        sym_lower, _, kind = stream.partition("@")
        st = self.sym.get(sym_lower.upper())
        if st is None:
            self.system.write("detail", f"__unexpected__@{stream}", data, ts)
            return
        st.writer.write("detail", stream, data, ts)
        st.counts[kind] += 1
        if isinstance(data, dict) and isinstance(data.get("E"), int):
            lat = st.lat[kind]
            if len(lat) < 20_000:
                lat.append(ts - data["E"])
        if kind == self.cfg.diff_stream:
            self._on_diff(sym_lower.upper(), st, data, ts)
        elif self.cfg.partial_stream and kind == self.cfg.partial_stream:
            u = data.get("u") if isinstance(data, dict) else None
            if isinstance(u, int):
                st.pending_d20[u] = data
                if len(st.pending_d20) > 100:
                    st.pending_d20.pop(next(iter(st.pending_d20)))
                self._check_d20(sym_lower.upper(), st, ts)
        if st.writer.dropped > st.dropped_seen:
            st.dropped_seen = st.writer.dropped
            st.writer.write("detail", f"__gap__@{sym_lower.upper()}", {"reason": "writer_drop"}, ts)
            st.book.mark_gap("writer_drop")
            self._schedule_resync(sym_lower.upper(), "writer_drop")

    def _on_diff(self, symbol: str, st: SymState, data: Any, ts: int) -> None:
        if not isinstance(data, dict) or not all(k in data for k in ("U", "u")):
            st.writer.write("detail", f"__gap__@{symbol}", {"reason": "malformed_diff"}, ts)
            st.book.mark_gap("malformed_diff")
            self._schedule_resync(symbol, "malformed")
            return
        status, _ = st.book.on_diff(data, ts)
        if status == GAP:
            st.gaps += 1
            st.writer.write("detail", f"__gap__@{symbol}",
                            {"reason": next(reversed(st.book.stats.gap_reasons)), "last_u": st.book.last_u,
                             "U": data.get("U"), "u": data.get("u"), "pu": data.get("pu")}, ts)
            log.warning("%s depth continuity broken (%s); rebuilding from snapshot", symbol,
                        next(reversed(st.book.stats.gap_reasons)))
            self._schedule_resync(symbol, "gap")
        elif st.book.valid:
            self._check_d20(symbol, st, ts)

    def _check_d20(self, symbol: str, st: SymState, ts: int) -> None:
        if not st.book.valid:
            return
        d = st.pending_d20.pop(st.book.last_u, None)
        if d is None:
            return
        st.d20_checked += 1
        b, a = st.book.top(20)
        pb = [(float(p), float(q)) for p, q in d.get("b", [])][:20]
        pa = [(float(p), float(q)) for p, q in d.get("a", [])][:20]
        n = min(len(pb), len(b), len(pa), len(a))
        if pb[:n] != b[:n] or pa[:n] != a[:n]:
            st.d20_mismatch += 1
            first = next((i for i in range(n) if pb[i] != b[i] or pa[i] != a[i]), n)
            st.writer.write("detail", f"__check__@{symbol}", {"u": st.book.last_u, "first_mismatch_level": first,
                                                             "book_b": b[:3], "d20_b": pb[:3],
                                                             "book_a": a[:3], "d20_a": pa[:3]}, ts)

    # ------------------------------------------------------------------ snapshots
    def _schedule_resync(self, symbol: str, why: str) -> None:
        st = self.sym[symbol]
        if st.resyncing:
            return
        st.resyncing = True
        asyncio.ensure_future(self._resync(symbol, why))

    async def _resync(self, symbol: str, why: str) -> None:
        st = self.sym[symbol]
        try:
            for attempt in range(8):
                await asyncio.sleep(self.cfg.resync_delay_s * (1 + attempt))
                try:
                    snap = await self.client.depth(symbol, self.cfg.snapshot_limit)
                except Exception as exc:  # noqa: BLE001
                    log.warning("%s snapshot failed (%s), retrying", symbol, exc)
                    continue
                ts = now_ms()
                st.writer.write("detail", f"__snapshot__@{symbol}", snap, ts)
                status, _ = st.book.on_snapshot(snap, ts)
                st.resyncs += 1
                st.writer.write("detail", f"__resync__@{symbol}", {"why": why, "status": status, "attempt": attempt,
                                                                  "lastUpdateId": snap.get("lastUpdateId")}, ts)
                if status != GAP:
                    log.info("%s resynced (%s): %s", symbol, why, status)
                    return
            st.resync_fail += 1
            log.error("%s could not resynchronise after 8 attempts", symbol)
        finally:
            st.resyncing = False

    async def _audit_loop(self) -> None:
        while True:
            await asyncio.sleep(self.cfg.audit_interval_s)
            for s, st in self.sym.items():
                try:
                    snap = await self.client.depth(s, self.cfg.snapshot_limit)
                    st.writer.write("detail", f"__snapshot_audit__@{s}", snap, now_ms())
                except Exception as exc:  # noqa: BLE001
                    log.warning("%s audit snapshot failed: %s", s, exc)
                await asyncio.sleep(1.0)

    async def _time_loop(self) -> None:
        while True:
            try:
                t0 = now_ms()
                r = await self.client.request("GET", "/fapi/v1/time")
                t1 = now_ms()
                self.system.write("detail", "__time__@ALL", {"serverTime": r.get("serverTime"), "local_send": t0,
                                                            "local_recv": t1}, t1)
            except Exception as exc:  # noqa: BLE001
                log.warning("server time request failed: %s", exc)
            await asyncio.sleep(self.cfg.time_sync_interval_s)

    # ------------------------------------------------------------------ health
    async def _health_loop(self) -> None:
        while True:
            await asyncio.sleep(self.cfg.health_interval_s)
            self._health_tick()

    def _health_tick(self, final: bool = False) -> None:
        ts = now_ms()
        free_gb = shutil.disk_usage(self.cfg.out).free / 1e9
        rec: dict[str, Any] = {"ts": ts, "final": final, "free_gb": round(free_gb, 2),
                               "connected": self.conn.connected.is_set(), "reconnects": self.conn.n_reconnects,
                               "symbols": {}}
        for s, st in self.sym.items():
            lat = {}
            for k, v in st.lat.items():
                if v:
                    v = sorted(v)
                    lat[k] = {"p50": v[len(v) // 2], "p99": v[int(len(v) * 0.99)], "max": v[-1]}
            rec["symbols"][s] = {"msgs": dict(st.counts), "latency_ms": lat, "book_valid": st.book.valid,
                                 "gaps": st.gaps, "resyncs": st.resyncs, "resync_fail": st.resync_fail,
                                 "gap_reasons": dict(st.book.stats.gap_reasons),
                                 "d20_checked": st.d20_checked, "d20_mismatch": st.d20_mismatch,
                                 "writer_dropped": st.writer.dropped, "writer_written": st.writer.n_written,
                                 "levels": [len(st.book.bids), len(st.book.asks)]}
            st.counts = defaultdict(int)
            st.lat = defaultdict(list)
        d = os.path.join(self.cfg.out, "_health")
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, time.strftime("%Y%m%d", time.gmtime(ts / 1000)) + ".jsonl"), "a",
                  encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
        bad = [s for s, v in rec["symbols"].items() if not v["book_valid"]]
        log.info("health: free=%.1fGB reconnects=%d invalid_books=%s gaps=%s", free_gb, self.conn.n_reconnects, bad,
                 {s: v["gaps"] for s, v in rec["symbols"].items()})
        if free_gb < self.cfg.min_free_gb and not final:
            log.critical("free disk %.1f GB below %.1f GB; stopping the recorder", free_gb, self.cfg.min_free_gb)
            self._stop.set()


def write_root_meta(root: str, symbol_info: dict) -> None:
    """Top-level meta.json (what backtest.replay reads: symbol filters + depth mode)."""
    path = os.path.join(root, "meta.json")
    meta = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            meta = json.load(fh)
    meta.setdefault("symbol_info", {}).update(symbol_info)
    meta.update({"depth_mode": "diff", "depth_levels": 20, "source": meta.get("source", "binance-usdm-live"),
                 "recorder": "v3"})
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/l2")
    ap.add_argument("--symbols", nargs="+", default=list(DEFAULT_SYMBOLS))
    ap.add_argument("--days", type=float, help="stop after this many days (default: run until stopped)")
    ap.add_argument("--rest-base", default=RecorderConfig.rest_base)
    ap.add_argument("--ws-base", default=RecorderConfig.ws_base)
    ap.add_argument("--no-depth20", action="store_true")
    ap.add_argument("--audit-interval-s", type=float, default=600.0)
    ap.add_argument("--min-free-gb", type=float, default=5.0)
    ap.add_argument("--log-level", default="INFO")
    a = ap.parse_args(argv)
    logging.basicConfig(level=a.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = RecorderConfig(out=a.out, symbols=tuple(s.upper() for s in a.symbols), rest_base=a.rest_base,
                         ws_base=a.ws_base, partial_stream="" if a.no_depth20 else RecorderConfig.partial_stream,
                         audit_interval_s=a.audit_interval_s, min_free_gb=a.min_free_gb,
                         duration_s=a.days * 86400 if a.days else None)
    asyncio.run(L2Recorder(cfg).run())


if __name__ == "__main__":
    main()
