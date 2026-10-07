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
from array import array
from dataclasses import dataclass
from typing import Any

import numpy as np

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
    plan_target_bps: float | None = None
    plan_stop_bps: float | None = None
    start_abs: int = 0          # first quote (absolute path index) seen after creation

    def compute(self, ts: np.ndarray, bid: np.ndarray, ask: np.ndarray, complete: bool) -> dict[str, Any]:
        """Labels from the quote path observed after the signal (vectorized, computed once).

        Semantics: horizon h uses the first quote with elapsed >= h (NULL if it arrived more
        than HORIZON_TOLERANCE_MS late); barrier touches use the FIRST quote that crosses,
        target checked before stop on the same quote; executable prices throughout.
        """
        out: dict[str, Any] = {f"ret_{h}s": None for h in HORIZONS_S}
        tp_long = tp_short = tp_plan = 0
        mfe = mae = 0.0
        n = len(ts)
        if n:
            elapsed = ts - self.t0
            mid = 0.5 * (bid + ask)
            for h in HORIZONS_S:
                i = int(np.searchsorted(elapsed, h * 1000, side="left"))
                if i < n:
                    out[f"ret_{h}s"] = (float((mid[i] - self.mid0) / self.mid0 * 1e4)
                                        if elapsed[i] <= h * 1000 + HORIZON_TOLERANCE_MS else None)
            tp_long = _first_touch(bid >= self.ask0 * (1 + self.target_bps / 1e4),
                                   bid <= self.ask0 * (1 - self.stop_bps / 1e4))
            tp_short = _first_touch(ask <= self.bid0 * (1 - self.target_bps / 1e4),
                                    ask >= self.bid0 * (1 + self.stop_bps / 1e4))
            d = self.direction
            if self.plan_target_bps and self.plan_stop_bps and d:
                entry = self.ask0 if d > 0 else self.bid0
                mark = bid if d > 0 else ask
                tp_px = entry * (1 + d * self.plan_target_bps / 1e4)
                sl_px = entry * (1 - d * self.plan_stop_bps / 1e4)
                tp_plan = _first_touch((mark - tp_px) * d >= 0, (mark - sl_px) * d <= 0)
            exc = (bid - self.ask0) / self.ask0 * 1e4 if (d or 1) > 0 else (self.bid0 - ask) / self.bid0 * 1e4
            mfe = max(0.0, float(exc.max()))
            mae = min(0.0, float(exc.min()))
        has_plan = bool(self.plan_target_bps and self.plan_stop_bps and self.direction)
        out.update(
            tp_long=tp_long,
            tp_short=tp_short,
            tp_pred=tp_long if self.direction > 0 else tp_short if self.direction < 0 else None,
            tp_plan=tp_plan if has_plan else None,
            mfe_bps=mfe,
            mae_bps=mae,
            labeled=1 if complete else 2,
        )
        return out


def _first_touch(hit_tp: np.ndarray, hit_sl: np.ndarray) -> int:
    i_tp = int(np.argmax(hit_tp)) if hit_tp.any() else None
    i_sl = int(np.argmax(hit_sl)) if hit_sl.any() else None
    if i_tp is None and i_sl is None:
        return 0
    if i_sl is None or (i_tp is not None and i_tp <= i_sl):
        return 1
    return -1


class _QuotePath:
    """Per-symbol quote history since the oldest pending label (absolute indexing).

    Typed buffers (array.array) so a label's window is a zero-copy numpy view instead of
    a list -> ndarray conversion of ~10k quotes on busy symbols.
    """

    __slots__ = ("base", "ts", "bid", "ask")

    def __init__(self) -> None:
        self.base = 0
        self.ts = array("q")
        self.bid = array("d")
        self.ask = array("d")

    @property
    def end(self) -> int:
        return self.base + len(self.ts)

    def arrays(self, start_abs: int, end_abs: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        i, j = start_abs - self.base, end_abs - self.base
        # Copies are cheap (contiguous memcpy) and avoid holding buffer exports that would
        # block later in-place pruning of the arrays.
        return (np.frombuffer(self.ts, dtype=np.int64)[i:j].copy(),
                np.frombuffer(self.bid, dtype=np.float64)[i:j].copy(),
                np.frombuffer(self.ask, dtype=np.float64)[i:j].copy())

    def prune(self, keep_from_abs: int) -> None:
        k = keep_from_abs - self.base
        if k > 4096 and k > len(self.ts) // 2:
            del self.ts[:k]
            del self.bid[:k]
            del self.ask[:k]
            self.base = keep_from_abs


class Recorder:
    def __init__(self, cfg: BotConfig, db: Database, rng: random.Random | None = None) -> None:
        self.cfg = cfg
        self.db = db
        self.rng = rng or random.Random()
        self.pending: dict[str, list[PendingLabel]] = {}
        self.paths: dict[str, _QuotePath] = {}
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
            path = self.paths.setdefault(symbol, _QuotePath())
            self.pending.setdefault(symbol, []).append(
                PendingLabel(sid, symbol, now_ms, 0.5 * (bid + ask), bid, ask, pred.direction, lt, ls,
                             details.get("target_bps"), details.get("stop_bps"), start_abs=path.end)
            )
        return sid

    # ------------------------------------------------------------------ label
    def on_quote(self, symbol: str, ts_ms: int, bid: float, ask: float) -> None:
        lst = self.pending.get(symbol)
        if not lst or bid <= 0 or ask <= 0:
            return
        path = self.paths[symbol]
        path.ts.append(ts_ms)
        path.bid.append(bid)
        path.ask.append(ask)
        # Labels are created in time order, so the ready ones form a prefix.
        while lst and ts_ms - lst[0].t0 >= LABEL_SPAN_MS:
            p = lst.pop(0)
            self._finalize(p, path, path.end, complete=True)
        if lst:
            path.prune(lst[0].start_abs)
        else:
            del self.pending[symbol]
            del self.paths[symbol]

    def sweep(self, now_ms: int) -> None:
        """Finalize labels whose symbol stopped producing quotes."""
        for symbol in list(self.pending):
            lst = self.pending[symbol]
            path = self.paths[symbol]
            while lst and now_ms - lst[0].t0 > LABEL_SPAN_MS + SWEEP_GRACE_MS:
                self._finalize(lst.pop(0), path, path.end, complete=False)
            if not lst:
                del self.pending[symbol]
                del self.paths[symbol]

    def _finalize(self, p: PendingLabel, path: _QuotePath, end_abs: int, complete: bool) -> None:
        ts, bid, ask = path.arrays(p.start_abs, end_abs)
        self.db.update("signals", "id", p.signal_id, p.compute(ts, bid, ask, complete))
        self.n_labeled += 1

    def pending_symbols(self) -> set[str]:
        return set(self.pending)

    def flush_all(self) -> None:
        for symbol in list(self.pending):
            path = self.paths[symbol]
            for p in self.pending[symbol]:
                self._finalize(p, path, path.end, complete=False)
        self.pending.clear()
        self.paths.clear()

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
