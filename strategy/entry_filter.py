"""Entry gate: every important condition must pass, and EXPECTED NET PnL is decisive.

    Expected Net Profit = E[gross move value]  (probability-weighted, barrier model)
                          - entry fee - exit fee
                          - entry slippage - exit slippage (incl. latency/buffer)

Two net checks are applied:
  * net_if_target  >= min_net_profit_usdt  (the target itself must clear costs)
  * expected_net   >= min_net_profit_usdt  (probability-weighted expectation)

A trade is NEVER taken just because the gross target looks attractive.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from config import BotConfig
from market_data.orderbook import OrderBook
from strategy.barrier import BarrierOutcome
from strategy.costs import CostEstimate, CostModel
from strategy.predictor import Prediction


@dataclass
class EntryPlan:
    symbol: str
    direction: int
    notional: float
    entry_maker: bool
    target_bps: float
    stop_bps: float
    horizon_s: float
    outcome: BarrierOutcome
    costs: CostEstimate
    expected_gross_usdt: float
    expected_net_usdt: float
    net_if_target_usdt: float
    loss_if_stop_usdt: float
    ref_mid: float
    confidence: float
    score: float


@dataclass
class EntryDecision:
    ok: bool
    reason: str = ""
    plan: EntryPlan | None = None
    details: dict[str, float] = field(default_factory=dict)


class EntryFilter:
    def __init__(self, cfg: BotConfig, costs: CostModel) -> None:
        self.cfg = cfg
        self.costs = costs

    # ------------------------------------------------------------------ sizing
    def size_notional(self, stop_bps: float, exit_slip_bps: float) -> float:
        s, r = self.cfg.sizing, self.cfg.risk
        notional = min(s.position_notional_usdt, r.max_position_notional_usdt)
        notional = min(notional, s.account_equity_usdt * min(s.leverage, r.max_leverage) * 0.9)
        # Worst-case loss at the stop: move + exit slippage + both fees (taker).
        loss_frac = (stop_bps + exit_slip_bps) / 1e4 + self.costs.maker_fee + self.costs.taker_fee
        if loss_frac > 0:
            notional = min(notional, s.max_risk_per_trade_usdt / loss_frac)
        return max(notional, 0.0)

    # ------------------------------------------------------------------ gate
    def evaluate(self, symbol: str, pred: Prediction, f: dict[str, float], book: OrderBook,
                 force_taker: bool = False) -> EntryDecision:
        e = self.cfg.entry
        md = self.cfg.market_data
        if not f:
            return EntryDecision(False, "no_features")
        if f["book_age_ms"] > md.stale_after_ms or f["flow_age_ms"] > md.stale_after_ms * 5:
            return EntryDecision(False, "stale_data")
        if f["spread_bps"] > e.max_spread_bps:
            return EntryDecision(False, "spread_too_wide", details={"spread_bps": f["spread_bps"]})
        if abs(f.get("imb_l10", 0.0)) > e.max_book_imbalance_abs:
            return EntryDecision(False, "abnormal_book_one_sided")
        if f.get("rv_1s_bps", 0.0) > e.max_realized_vol_bps_1s:
            return EntryDecision(False, "abnormal_volatility")
        if f.get("liquidity_change", 0.0) < -0.5:
            return EntryDecision(False, "liquidity_withdrawal")
        if f.get("trades_per_s_10s", 0.0) < e.min_trades_per_s:
            return EntryDecision(False, "thin_tape")

        direction = pred.direction
        if direction == 0 or pred.confidence < e.min_confidence:
            return EntryDecision(False, "low_confidence", details={"confidence": pred.confidence})
        if e.require_flow_confirmation and not pred.flow_confirms:
            return EntryDecision(False, "flow_not_confirming")
        if direction * f.get("imb_weighted", 0.0) < e.min_book_imbalance:
            return EntryDecision(False, "book_imbalance_below_min")
        if direction * f.get("flow_imb_3s", 0.0) < e.min_flow_imbalance:
            return EntryDecision(False, "flow_imbalance_below_min")

        bd10, ad10 = f.get("bid_depth_10bps", 0.0), f.get("ask_depth_10bps", 0.0)
        if min(bd10, ad10) < e.min_depth_usdt_within_10bps:
            return EntryDecision(False, "insufficient_depth", details={"min_depth": min(bd10, ad10)})

        # Stop distance (optionally volatility-scaled).
        horizon = self.cfg.strategy.max_hold_s
        stop_bps = e.stop_bps
        if e.stop_vol_mult > 0:
            stop_bps = max(stop_bps, e.stop_vol_mult * pred.sigma_1s_bps * (horizon ** 0.5) / 3.0)

        exit_side = "SELL" if direction > 0 else "BUY"
        notional_probe = min(self.cfg.sizing.position_notional_usdt, self.cfg.risk.max_position_notional_usdt)
        exit_slip_probe = self.costs.taker_slippage_bps(book, exit_side, notional_probe)
        notional = self.size_notional(stop_bps, exit_slip_probe)
        if notional < 5.0:
            return EntryDecision(False, "size_below_minimum", details={"notional": notional})

        entry_maker = (
            not force_taker
            and self.cfg.execution.entry_mode == "maker_first"
            and self.cfg.costs.maker_entry_expected
        )
        costs = self.costs.estimate(book, direction, notional, entry_maker)
        if costs.exit_slippage_bps > e.max_expected_slippage_bps or (
            not entry_maker and costs.entry_slippage_bps > e.max_expected_slippage_bps
        ):
            return EntryDecision(False, "slippage_too_high",
                                 details={"exit_slip_bps": costs.exit_slippage_bps,
                                          "entry_slip_bps": costs.entry_slippage_bps})

        required = self.costs.required_move_bps(costs, e.min_net_profit_usdt)
        base_target = max(e.min_target_bps, required)
        if base_target > e.max_target_bps:
            return EntryDecision(False, "required_move_too_large", details={"required_bps": required})
        # Pick the target that maximizes expected net SUBJECT TO the target-before-stop
        # probability constraint; if no candidate satisfies it, keep the most likely one
        # so the rejection reason is reported accurately.
        best = None
        most_likely = None
        for mult in e.target_multipliers:
            tgt = min(base_target * mult, e.max_target_bps)
            oc = pred.outcome(direction, tgt, stop_bps, horizon)
            exp_net = notional * oc.expected_value_bps / 1e4 - costs.total_usdt
            cand = (tgt, oc, exp_net)
            if most_likely is None or oc.p_target > most_likely[1].p_target:
                most_likely = cand
            if oc.p_target >= e.min_p_target_before_stop and (best is None or exp_net > best[2]):
                best = cand
            if tgt >= e.max_target_bps:
                break
        target_bps, outcome, expected_net = best or most_likely
        expected_gross = notional * outcome.expected_value_bps / 1e4
        net_if_target = notional * target_bps / 1e4 - costs.total_usdt
        loss_if_stop = notional * stop_bps / 1e4 + costs.total_usdt
        details = {
            "target_bps": target_bps,
            "stop_bps": stop_bps,
            "p_target": outcome.p_target,
            "p_stop": outcome.p_stop,
            "expected_net": expected_net,
            "net_if_target": net_if_target,
            "costs_usdt": costs.total_usdt,
            "notional": notional,
        }
        if loss_if_stop > self.cfg.sizing.max_risk_per_trade_usdt * 1.05:
            return EntryDecision(False, "risk_per_trade_exceeded", details=details)
        if net_if_target < e.min_net_profit_usdt - 1e-9:
            return EntryDecision(False, "target_net_below_min", details=details)
        if outcome.p_target < e.min_p_target_before_stop:
            return EntryDecision(False, "p_target_too_low", details=details)
        if expected_net < e.min_net_profit_usdt:
            return EntryDecision(False, "expected_net_below_min", details=details)

        plan = EntryPlan(
            symbol=symbol,
            direction=direction,
            notional=notional,
            entry_maker=entry_maker,
            target_bps=target_bps,
            stop_bps=stop_bps,
            horizon_s=horizon,
            outcome=outcome,
            costs=costs,
            expected_gross_usdt=expected_gross,
            expected_net_usdt=expected_net,
            net_if_target_usdt=net_if_target,
            loss_if_stop_usdt=loss_if_stop,
            ref_mid=f["mid"],
            confidence=pred.confidence,
            score=pred.score,
        )
        return EntryDecision(True, "ok", plan=plan, details=details)
