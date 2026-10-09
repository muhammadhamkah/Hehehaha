"""SQLite storage with a background writer thread (never blocks the event loop).

Tables
  signals          every evaluated potential signal + decision + forward labels
  trades           every completed (paper or live) trade
  orders           order lifecycle events
  scanner          periodic scanner rankings
  book_snapshots   optional raw top-of-book/depth snapshots
"""
from __future__ import annotations

import logging
import os
import queue
import sqlite3
import threading
import time
from typing import Any, Iterable

from config import HORIZONS_S

log = logging.getLogger(__name__)

_RET_COLS = ",\n    ".join(f"ret_{h}s REAL" for h in HORIZONS_S)

SCHEMA = f"""
CREATE TABLE IF NOT EXISTS signals (
    id INTEGER PRIMARY KEY,
    ts_ms INTEGER NOT NULL,
    symbol TEXT NOT NULL,
    sample_type TEXT,
    bid REAL, ask REAL, mid REAL, last_price REAL, spread_bps REAL,
    bid_depth_10bps REAL, ask_depth_10bps REAL,
    imb_l1 REAL, imb_l5 REAL, imb_weighted REAL, ofi_3s REAL,
    flow_imb_3s REAL, flow_imb_10s REAL,
    buy_notional_3s REAL, sell_notional_3s REAL,
    trades_per_s_10s REAL, rv_1s_bps REAL,
    score REAL, direction INTEGER, p_long REAL, p_short REAL, flow_confirms INTEGER,
    p_target REAL, target_bps REAL, stop_bps REAL,
    expected_net_usdt REAL, expected_hold_s REAL,
    decision TEXT, rejection_reason TEXT,
    features_json TEXT, predicted_json TEXT,
    {_RET_COLS},
    tp_long INTEGER, tp_short INTEGER, tp_pred INTEGER, tp_plan INTEGER,
    mfe_bps REAL, mae_bps REAL,
    label_target_bps REAL, label_stop_bps REAL,
    labeled INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_signals_ts ON signals(ts_ms);
CREATE INDEX IF NOT EXISTS idx_signals_sym ON signals(symbol, ts_ms);

CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_id TEXT UNIQUE,
    mode TEXT,
    symbol TEXT, direction INTEGER,
    entry_ts_ms INTEGER, exit_ts_ms INTEGER, holding_s REAL,
    signal_confidence REAL, signal_score REAL, signal_id INTEGER,
    entry_price REAL, exit_price REAL, qty REAL, notional REAL,
    gross_pnl REAL, entry_fee REAL, exit_fee REAL,
    est_slippage_usdt REAL, actual_slippage_usdt REAL,
    entry_slippage_bps REAL, exit_slippage_bps REAL,
    net_pnl REAL, exit_reason TEXT,
    mfe_bps REAL, mae_bps REAL, mfe_usdt REAL, mae_usdt REAL,
    entry_maker INTEGER, exit_maker INTEGER,
    expected_net_usdt REAL, target_bps REAL, stop_bps REAL,
    features_json TEXT
);

CREATE TABLE IF NOT EXISTS orders (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_ms INTEGER, client_id TEXT, exchange_id TEXT, symbol TEXT, side TEXT,
    type TEXT, tif TEXT, price REAL, qty REAL, status TEXT, filled_qty REAL,
    avg_price REAL, fee REAL, maker INTEGER, event TEXT, detail TEXT
);

CREATE TABLE IF NOT EXISTS scanner (
    ts_ms INTEGER, rank INTEGER, symbol TEXT, score REAL,
    quote_volume_24h REAL, spread_bps REAL, volatility_bps REAL,
    activity REAL, volume_accel REAL, tob_imbalance REAL
);

CREATE TABLE IF NOT EXISTS book_snapshots (
    ts_ms INTEGER, symbol TEXT, bid REAL, ask REAL, bids_json TEXT, asks_json TEXT,
    last_trade REAL
);
"""


class Database:
    def __init__(self, path: str, flush_interval_s: float = 1.0) -> None:
        self.path = path
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
        self.flush_interval_s = flush_interval_s
        with self._connect() as conn:
            conn.executescript(SCHEMA)
            self._migrate(conn)
            row = conn.execute("SELECT COALESCE(MAX(id), 0) FROM signals").fetchone()
            self._next_signal_id = int(row[0]) + 1
        self._q: queue.Queue[tuple[str, Any] | None] = queue.Queue()
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._writer, name="db-writer", daemon=True)
        self._running = True
        self._thread.start()

    @staticmethod
    def _migrate(conn: sqlite3.Connection) -> None:
        """Add columns introduced after a database was created."""
        have = {r[1] for r in conn.execute("PRAGMA table_info(signals)")}
        for col, typ in (("tp_plan", "INTEGER"),):
            if col not in have:
                conn.execute(f"ALTER TABLE signals ADD COLUMN {col} {typ}")

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=30)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def next_signal_id(self) -> int:
        with self._lock:
            sid = self._next_signal_id
            self._next_signal_id += 1
            return sid

    # ------------------------------------------------------------------ writes
    def execute(self, sql: str, params: Iterable[Any] = ()) -> None:
        self._q.put((sql, tuple(params)))

    def insert(self, table: str, row: dict[str, Any]) -> None:
        cols = ", ".join(row)
        ph = ", ".join("?" for _ in row)
        self.execute(f"INSERT OR REPLACE INTO {table} ({cols}) VALUES ({ph})", row.values())

    def update(self, table: str, key: str, key_value: Any, values: dict[str, Any]) -> None:
        sets = ", ".join(f"{k} = ?" for k in values)
        self.execute(f"UPDATE {table} SET {sets} WHERE {key} = ?", [*values.values(), key_value])

    def _writer(self) -> None:
        conn = self._connect()
        pending = 0
        last_commit = time.monotonic()
        while True:
            try:
                item = self._q.get(timeout=self.flush_interval_s)
            except queue.Empty:
                item = ...
            if item is None:
                break
            if isinstance(item, threading.Event):
                conn.commit()
                pending = 0
                last_commit = time.monotonic()
                item.set()
                continue
            if item is not ...:
                sql, params = item
                try:
                    conn.execute(sql, params)
                    pending += 1
                except sqlite3.Error as exc:
                    log.error("db write failed: %s | %s", exc, sql[:120])
            if pending and (time.monotonic() - last_commit >= self.flush_interval_s or pending > 2000):
                conn.commit()
                pending = 0
                last_commit = time.monotonic()
        conn.commit()
        conn.close()

    def flush(self, timeout_s: float = 10.0) -> bool:
        """Block until all previously queued writes are committed."""
        ev = threading.Event()
        self._q.put(ev)  # type: ignore[arg-type]
        return ev.wait(timeout_s)

    def close(self) -> None:
        if self._running:
            self._running = False
            self._q.put(None)
            self._thread.join(timeout=30)

    # ------------------------------------------------------------------ reads
    def query(self, sql: str, params: Iterable[Any] = ()) -> list[dict[str, Any]]:
        conn = self._connect()
        conn.row_factory = sqlite3.Row
        try:
            return [dict(r) for r in conn.execute(sql, tuple(params)).fetchall()]
        finally:
            conn.close()
