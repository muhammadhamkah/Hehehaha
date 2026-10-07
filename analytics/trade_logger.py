"""Executed-trade logging (SQLite ``trades`` table + JSONL file)."""
from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field

from data.database import Database

log = logging.getLogger(__name__)


@dataclass
class TradeRecord:
    trade_id: str
    mode: str
    symbol: str
    direction: int
    entry_ts_ms: int
    exit_ts_ms: int
    holding_s: float
    signal_confidence: float
    signal_score: float
    signal_id: int | None
    entry_price: float
    exit_price: float
    qty: float
    notional: float
    gross_pnl: float
    entry_fee: float
    exit_fee: float
    est_slippage_usdt: float
    actual_slippage_usdt: float
    entry_slippage_bps: float
    exit_slippage_bps: float
    net_pnl: float
    exit_reason: str
    mfe_bps: float
    mae_bps: float
    mfe_usdt: float
    mae_usdt: float
    entry_maker: bool
    exit_maker: bool
    expected_net_usdt: float
    target_bps: float
    stop_bps: float
    features: dict[str, float] = field(default_factory=dict)


class TradeLogger:
    def __init__(self, db: Database, jsonl_path: str | None = None) -> None:
        self.db = db
        self.jsonl_path = jsonl_path
        if jsonl_path and os.path.dirname(jsonl_path):
            os.makedirs(os.path.dirname(jsonl_path), exist_ok=True)

    def log(self, t: TradeRecord) -> None:
        row = asdict(t)
        features = row.pop("features")
        row["entry_maker"] = int(t.entry_maker)
        row["exit_maker"] = int(t.exit_maker)
        row["features_json"] = json.dumps(features)
        self.db.insert("trades", row)
        if self.jsonl_path:
            with open(self.jsonl_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(asdict(t)) + "\n")
        log.info(
            "TRADE %s %s %s qty=%.6g entry=%.8g exit=%.8g gross=%.4f fees=%.4f net=%.4f hold=%.1fs reason=%s",
            t.mode, t.symbol, "LONG" if t.direction > 0 else "SHORT", t.qty, t.entry_price,
            t.exit_price, t.gross_pnl, t.entry_fee + t.exit_fee, t.net_pnl, t.holding_s, t.exit_reason,
        )
