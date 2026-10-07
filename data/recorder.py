"""Signal recorder and forward-outcome labeller.

Every potential signal (plus a small random baseline sample, so the dataset is not
only conditioned on the model's own opinion) is stored with its features, prediction,
entry decision and rejection reason. The labeller then follows the symbol's quotes
for 60 seconds and fills in:

  * ret_{h}s       signed mid return in bps at h in HORIZONS_S
  * tp_long        +1 target hit before stop, -1 stop first, 0 neither (long)
  * tp_short       same for a short
  * tp_pred        outcome for the predicted direction
  * mfe/mae_bps    max favourable / adverse executable excursion (predicted direction)

Target/stop touches are evaluated on EXECUTABLE prices (enter at ask/exit at bid for
longs), so labels already include the spread.
"""
from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass, field
from typing import Any

from config import HORIZONS_S, BotConfig
from data.database import Database
from strategy.predictor import Prediction

log = logging.getLogger(__name__)

LABEL_SPAN_MS = max(HORIZONS_S) * 1000
HORIZON_TOLERANCE_MS = 1500
SWEEP_GRACE_MS = 30_000


@dataclass
class PendingLabel:
    signal_id: int
    symbol: str
    t0: int
    mid0: float
    bid0: float
    ask0: float
    direction: int
    target_bps: float
    stop_bps: float
    rets: dict[int, float | None] = field(default_factory=dict)
    tp_long: int = 0
    tp_short: int = 0
    mfe_bps: float = 0.0
    mae_bps: float = 0.0
    last_ts: int = 0

    def __post_init__(self) -> None:
        self.long_tp = self.ask0 * (1 + self.target_bps / 1e4)
        self.long_sl = self.ask0 * (1 - self.stop_bps / 1e4)
        self.short_tp = self.bid0 * (1 - self.target_bps / 1e4)
        self.short_sl = self.bid0 * (1 + self.stop_bps / 1e4)

    def update(self, ts: int, bid: float, ask: float) -> bool:
        """Feed a quote. Returns True once the label window is complete."""
        if ts < self.t0:
            return False
        self.last_ts = ts
        mid = 0.5 * (bid + ask)
        elapsed = ts - self.t0
        for h in HORIZONS_S:
            if h not in self.rets and elapsed >= h * 1000:
                # Only accept the quote if it arrived close to the horizon.
                self.rets[h] = (mid - self.mid0) / self.mid0 * 1e4 if elapsed <= h * 1000 + HORIZON_TOLERANCE_MS else None
        if self.tp_long == 0:
            if bid >= self.long_tp:
                self.tp_long = 1
            elif bid <= self.long_sl:
                self.tp_long = -1
        if self.tp_short == 0:
            if ask <= self.short_tp:
                self.tp_short = 1
            elif ask >= self.short_sl:
                self.tp_short = -1
        d = self.direction or 1
        exc = (bid - self.ask0) / self.ask0 * 1e4 if d > 0 else (self.bid0 - ask) / self.bid0 * 1e4
        self.mfe_bps = max(self.mfe_bps, exc)
        self.mae_bps = min(self.mae_bps, exc)
        return elapsed >= LABEL_SPAN_MS

    def result(self, complete: bool) -> dict[str, Any]:
        out: dict[str, Any] = {f"ret_{h}s": self.rets.get(h) for h in HORIZONS_S}
        tp_pred = self.tp_long if self.direction > 0 else self.tp_short if self.direction < 0 else None
        out.update(
            tp_long=self.tp_long,
            tp_short=self.tp_short,
            tp_pred=tp_pred,
            mfe_bps=self.mfe_bps,
            mae_bps=self.mae_bps,
            labeled=1 if complete else 2,
        )
        return out


class Recorder:
    def __init__(self, cfg: BotConfig, db: Database, rng: random.Random | None = None) -> None:
        self.cfg = cfg
        self.db = db
        self.rng = rng or random.Random()
        self.pending: dict[str, list[PendingLabel]] = {}
        self._last_record_ms: dict[str, int] = {}
        self.n_recorded = 0
        self.n_labeled = 0

    # ------------------------------------------------------------------ policy
    def should_record(self, symbol: str, score: float, entered: bool, now_ms: int) -> str | None:
        """Return the sample type to record, or None to skip."""
        if entered:
            return "entry"
        last = self._last_record_ms.get(symbol, 0)
        if now_ms - last < self.cfg.strategy.record_min_interval_ms:
            return None
        if abs(score) >= self.cfg.strategy.record_score_threshold:
            return "signal"
        if self.rng.random() < self.cfg.strategy.baseline_sample_prob:
            return "baseline"
        return None

    # ------------------------------------------------------------------ record
    def record_signal(
        self,
        symbol: str,
        now_ms: int,
        features: dict[str, float],
        pred: Prediction,
        decision: str,
        rejection_reason: str,
        sample_type: str,
        details: dict[str, float] | None = None,
    ) -> int:
        details = details or {}
        sid = self.db.next_signal_id()
        lt, ls = self.cfg.recorder.label_target_bps, self.cfg.recorder.label_stop_bps
        row = {
            "id": sid,
            "ts_ms": now_ms,
            "symbol": symbol,
            "sample_type": sample_type,
            "bid": features.get("bid"),
            "ask": features.get("ask"),
            "mid": features.get("mid"),
            "last_price": features.get("last_price"),
            "spread_bps": features.get("spread_bps"),
            "bid_depth_10bps": features.get("bid_depth_10bps"),
            "ask_depth_10bps": features.get("ask_depth_10bps"),
            "imb_l1": features.get("imb_l1"),
            "imb_l5": features.get("imb_l5"),
            "imb_weighted": features.get("imb_weighted"),
            "ofi_3s": features.get("ofi_3s"),
            "flow_imb_3s": features.get("flow_imb_3s"),
            "flow_imb_10s": features.get("flow_imb_10s"),
            "buy_notional_3s": features.get("buy_notional_3s"),
            "sell_notional_3s": features.get("sell_notional_3s"),
            "trades_per_s_10s": features.get("trades_per_s_10s"),
            "rv_1s_bps": features.get("rv_1s_bps"),
            "score": pred.score,
            "direction": pred.direction,
            "p_long": pred.p_long,
            "p_short": pred.p_short,
            "flow_confirms": int(pred.flow_confirms),
            "p_target": details.get("p_target"),
            "target_bps": details.get("target_bps"),
            "stop_bps": details.get("stop_bps"),
            "expected_net_usdt": details.get("expected_net"),
            "expected_hold_s": details.get("expected_hold_s"),
            "decision": decision,
            "rejection_reason": rejection_reason,
            "features_json": json.dumps({k: round(v, 8) for k, v in features.items()}),
            "predicted_json": json.dumps(
                {"moves_bps": {str(h): round(m, 4) for h, m in pred.expected_moves().items()},
                 "components": {k: round(v, 5) for k, v in pred.components.items()},
                 "drift_bps_per_s": pred.drift_bps_per_s, "sigma_1s_bps": pred.sigma_1s_bps,
                 "model": pred.model}
            ),
            "label_target_bps": lt,
            "label_stop_bps": ls,
            "labeled": 0,
        }
        self.db.insert("signals", row)
        self._last_record_ms[symbol] = now_ms
        self.n_recorded += 1
        bid, ask = features.get("bid", 0.0), features.get("ask", 0.0)
        if bid > 0 and ask > 0:
            self.pending.setdefault(symbol, []).append(
                PendingLabel(sid, symbol, now_ms, 0.5 * (bid + ask), bid, ask, pred.direction, lt, ls)
            )
        return sid

    # ------------------------------------------------------------------ label
    def on_quote(self, symbol: str, ts_ms: int, bid: float, ask: float) -> None:
        lst = self.pending.get(symbol)
        if not lst or bid <= 0 or ask <= 0:
            return
        keep = []
        for p in lst:
            if p.update(ts_ms, bid, ask):
                self._finalize(p, complete=True)
            else:
                keep.append(p)
        if keep:
            self.pending[symbol] = keep
        else:
            del self.pending[symbol]

    def sweep(self, now_ms: int) -> None:
        """Finalize labels whose symbol stopped producing quotes."""
        for symbol in list(self.pending):
            keep = []
            for p in self.pending[symbol]:
                if now_ms - p.t0 > LABEL_SPAN_MS + SWEEP_GRACE_MS:
                    self._finalize(p, complete=False)
                else:
                    keep.append(p)
            if keep:
                self.pending[symbol] = keep
            else:
                del self.pending[symbol]

    def _finalize(self, p: PendingLabel, complete: bool) -> None:
        self.db.update("signals", "id", p.signal_id, p.result(complete))
        self.n_labeled += 1

    def pending_symbols(self) -> set[str]:
        return set(self.pending)

    def flush_all(self) -> None:
        for symbol in list(self.pending):
            for p in self.pending[symbol]:
                self._finalize(p, complete=False)
        self.pending.clear()

    # ------------------------------------------------------------------ raw data
    def record_scanner(self, now_ms: int, ranking: list) -> None:
        if not self.cfg.recorder.record_scanner:
            return
        for i, r in enumerate(ranking[:50]):
            self.db.insert("scanner", {
                "ts_ms": now_ms, "rank": i + 1, "symbol": r.symbol, "score": r.score,
                "quote_volume_24h": r.quote_volume_24h, "spread_bps": r.spread_bps,
                "volatility_bps": r.volatility_bps, "activity": r.activity,
                "volume_accel": r.volume_accel, "tob_imbalance": r.tob_imbalance,
            })

    def record_book(self, now_ms: int, symbol: str, book, last_trade: float) -> None:
        b, a = book.sorted_levels(self.cfg.market_data.depth_levels)
        self.db.insert("book_snapshots", {
            "ts_ms": now_ms, "symbol": symbol, "bid": b[0][0] if b else None,
            "ask": a[0][0] if a else None, "bids_json": json.dumps(b), "asks_json": json.dumps(a),
            "last_trade": last_trade,
        })
