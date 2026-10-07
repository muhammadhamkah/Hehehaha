"""Per-symbol evaluation pipeline:

    features -> directional prediction -> entry gate (net edge) -> risk check
             -> record (signal + decision + rejection reason) -> enter / skip
"""
from __future__ import annotations

from dataclasses import dataclass

from config import BotConfig
from data.recorder import Recorder
from features.microstructure_features import compute_features
from market_data.orderbook import OrderBook
from market_data.tradeflow import TradeFlow
from risk.risk_manager import RiskManager
from strategy.entry_filter import EntryDecision, EntryFilter
from strategy.predictor import BasePredictor, Prediction


@dataclass
class SignalResult:
    symbol: str
    ts_ms: int
    features: dict[str, float]
    prediction: Prediction | None
    decision: EntryDecision
    action: str                # "enter" | "would_enter" | "reject"
    signal_id: int | None = None


class SignalEngine:
    def __init__(self, cfg: BotConfig, predictor: BasePredictor, entry_filter: EntryFilter,
                 recorder: Recorder | None, risk: RiskManager) -> None:
        self.cfg = cfg
        self.predictor = predictor
        self.entry_filter = entry_filter
        self.recorder = recorder
        self.risk = risk
        self.rejections: dict[str, int] = {}
        # Every evaluation's outcome: "enter" / "would_enter" or the rejection reason.
        self.decisions: dict[str, int] = {}

    def features(self, book: OrderBook, flow: TradeFlow, now_ms: int) -> dict[str, float]:
        f = compute_features(book, flow, self.cfg.features, now_ms)
        if f:
            f["last_price"] = flow.last_price()
        return f

    def evaluate(self, symbol: str, book: OrderBook, flow: TradeFlow, now_ms: int,
                 trading_enabled: bool, record: bool = True) -> SignalResult:
        f = self.features(book, flow, now_ms)
        if not f:
            return SignalResult(symbol, now_ms, f, None, EntryDecision(False, "no_features"), "reject")
        pred = self.predictor.predict(f)
        decision = self.entry_filter.evaluate(symbol, pred, f, book)
        action = "reject"
        if decision.ok and decision.plan is not None:
            plan = decision.plan
            ok, reason = self.risk.can_open(
                symbol, plan.notional, self.cfg.sizing.leverage, f["spread_bps"], plan.costs.exit_slippage_bps
            )
            if not ok:
                decision = EntryDecision(False, f"risk:{reason}", plan=plan, details=decision.details)
            else:
                action = "enter" if trading_enabled else "would_enter"
        if not decision.ok:
            self.rejections[decision.reason] = self.rejections.get(decision.reason, 0) + 1
        if record:   # not counted during replay warm-up
            key = action if action != "reject" else decision.reason
            self.decisions[key] = self.decisions.get(key, 0) + 1

        result = SignalResult(symbol, now_ms, f, pred, decision, action)
        if record and self.recorder is not None:
            sample = self.recorder.should_record(symbol, pred.score, action != "reject", now_ms)
            if sample is not None:
                details = dict(decision.details)
                if decision.plan is not None:
                    details["expected_hold_s"] = decision.plan.outcome.expected_hold_s
                result.signal_id = self.recorder.record_signal(
                    symbol, now_ms, f, pred, action,
                    "" if action != "reject" else decision.reason, sample, details,
                )
        return result

    def recheck_taker(self, symbol: str, book: OrderBook, flow: TradeFlow, now_ms: int):
        """Fresh re-evaluation assuming TAKER entry; returns an EntryPlan or None."""
        f = self.features(book, flow, now_ms)
        if not f:
            return None
        pred = self.predictor.predict(f)
        d = self.entry_filter.evaluate(symbol, pred, f, book, force_taker=True)
        if not d.ok or d.plan is None:
            return None
        if d.plan.costs.entry_slippage_bps > self.cfg.execution.taker_max_slippage_bps:
            return None
        return d.plan
