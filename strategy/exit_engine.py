"""Exit logic for an open position.

0.10 USDT is the minimum edge to justify ENTRY, not a fixed take-profit. Exits:

  stop_loss          executable PnL hits the (initial) stop
  break_even_stop    stop moved to net break-even after break_even_trigger_bps MFE
  trailing_stop      trailing stop after trail_activate_bps MFE (lets winners run)
  target_profit      target reached and momentum is NOT continuing
                     (if momentum continues, the target arms a trailing stop instead)
  flow_reversal      model score flips strongly against the position
  imbalance_lost     book + tape against the position for N consecutive evaluations
  momentum_exhaustion in net profit, tape velocity collapsed and score neutral
  time_stop          held beyond the expected holding time multiple / max hold
  emergency          stale data, spread blow-out (risk-level emergencies are injected
                     by the bot via ``force_exit``)

PnL used for exit decisions is measured on the EXECUTABLE side of the book
(bid for longs, ask for shorts), never on mid.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from config import BotConfig
from strategy.entry_filter import EntryPlan
from strategy.predictor import Prediction


@dataclass
class Position:
    symbol: str
    direction: int
    qty: float
    entry_price: float
    entry_ts_ms: int
    plan: EntryPlan
    entry_fee: float
    entry_maker: bool
    entry_slippage_bps: float            # actual, vs decision mid (positive == cost)
    entry_features: dict[str, float] = field(default_factory=dict)
    entry_trades_per_s: float = 0.0
    # dynamic state
    stop_level_bps: float = 0.0          # exit when executable pnl_bps <= this
    break_even_armed: bool = False
    trailing_active: bool = False
    target_hit: bool = False
    mfe_bps: float = 0.0
    mae_bps: float = 0.0
    against_evals: int = 0
    last_mark: float = 0.0
    closing: bool = False
    signal_id: int | None = None
    # exit accumulation (an exit may need several orders)
    orig_qty: float = 0.0
    exit_qty: float = 0.0
    exit_notional: float = 0.0
    exit_fee: float = 0.0
    exit_maker_qty: float = 0.0
    exit_reason: str = ""
    exit_ref_mid: float = 0.0

    def __post_init__(self) -> None:
        if not self.orig_qty:
            self.orig_qty = self.qty

    @property
    def notional(self) -> float:
        return self.orig_qty * self.entry_price

    def pnl_bps(self, exit_price: float) -> float:
        return self.direction * (exit_price - self.entry_price) / self.entry_price * 1e4

    def gross_pnl(self, exit_price: float) -> float:
        return self.direction * (exit_price - self.entry_price) * self.qty


@dataclass
class ExitSignal:
    reason: str
    emergency: bool = False


class ExitEngine:
    def __init__(self, cfg: BotConfig, maker_fee: float, taker_fee: float) -> None:
        self.cfg = cfg
        self.maker_fee = maker_fee
        self.taker_fee = taker_fee

    def init_position(self, pos: Position) -> None:
        pos.stop_level_bps = -pos.plan.stop_bps

    def break_even_bps(self, pos: Position) -> float:
        """Executable move needed for NET zero: entry fee + exit fee (taker), in bps."""
        exit_fee_bps = self.taker_fee * 1e4
        entry_fee_bps = pos.entry_fee / pos.notional * 1e4 if pos.notional else 0.0
        return entry_fee_bps + exit_fee_bps

    def time_stop_s(self, pos: Position) -> float:
        s = self.cfg.strategy
        expected = pos.plan.outcome.expected_hold_s
        return min(max(expected * self.cfg.exit.time_stop_mult, 3 * s.min_hold_s), s.max_hold_s)

    def update(self, pos: Position, bid: float, ask: float, f: dict[str, float],
               pred: Prediction | None, now_ms: int) -> ExitSignal | None:
        x = self.cfg.exit
        if bid <= 0 or ask <= 0:
            return ExitSignal("emergency_no_quotes", emergency=True)
        mid = 0.5 * (bid + ask)
        spread_bps = (ask - bid) / mid * 1e4
        if spread_bps > x.emergency_spread_bps:
            return ExitSignal("emergency_spread", emergency=True)
        if f and f.get("book_age_ms", 0.0) > self.cfg.risk.stale_data_ms:
            return ExitSignal("emergency_stale_data", emergency=True)

        mark = bid if pos.direction > 0 else ask
        pos.last_mark = mark
        pnl = pos.pnl_bps(mark)
        pos.mfe_bps = max(pos.mfe_bps, pnl)
        pos.mae_bps = min(pos.mae_bps, pnl)
        held_s = (now_ms - pos.entry_ts_ms) / 1000.0
        if x.mode == "barrier":
            if pnl >= pos.plan.target_bps:
                return ExitSignal("target_profit")
            if pnl <= -pos.plan.stop_bps:
                return ExitSignal("stop_loss")
            if held_s >= pos.plan.horizon_s:
                return ExitSignal("time_stop")
            return None
        be = self.break_even_bps(pos)

        # --- ratchet the stop (never loosens)
        if not pos.break_even_armed and pos.mfe_bps >= max(x.break_even_trigger_bps, be + x.break_even_buffer_bps):
            pos.break_even_armed = True
            pos.stop_level_bps = max(pos.stop_level_bps, be)
        if pos.mfe_bps >= x.trail_activate_bps:
            pos.trailing_active = True
        if pos.trailing_active:
            pos.stop_level_bps = max(pos.stop_level_bps, pos.mfe_bps - x.trail_distance_bps)

        # --- target: take profit, or arm a tighter trail if momentum continues
        momentum_continues = bool(
            pred is not None
            and pred.direction == pos.direction
            and abs(pred.score) >= self.cfg.exit.reversal_score * 0.5
            and pred.flow_confirms
        )
        if pnl >= pos.plan.target_bps and not pos.target_hit:
            pos.target_hit = True
            if momentum_continues or not x.take_profit_on_target_if_no_momentum:
                pos.trailing_active = True
                pos.stop_level_bps = max(pos.stop_level_bps, pos.plan.target_bps - x.trail_distance_bps, be)
            else:
                return ExitSignal("target_profit")

        if pnl <= pos.stop_level_bps:
            if pos.trailing_active:
                return ExitSignal("trailing_stop")
            if pos.break_even_armed:
                return ExitSignal("break_even_stop")
            return ExitSignal("stop_loss")

        if held_s >= self.cfg.strategy.min_hold_s and pred is not None:
            if pred.score * pos.direction <= -x.reversal_score:
                return ExitSignal("flow_reversal")

        if f:
            book_against = f.get("imb_weighted", 0.0) * pos.direction < 0
            tape_against = f.get("flow_imb_3s", 0.0) * pos.direction < 0
            pos.against_evals = pos.against_evals + 1 if (book_against and tape_against) else 0
            if pos.against_evals >= x.imbalance_loss_evals:
                return ExitSignal("imbalance_lost")

            if pnl > be and pos.entry_trades_per_s > 0 and pred is not None:
                tps = f.get("trades_per_s_3s", 0.0)
                if tps < pos.entry_trades_per_s * x.exhaustion_velocity_ratio and abs(pred.score) < 0.1:
                    return ExitSignal("momentum_exhaustion")

        if held_s >= self.time_stop_s(pos):
            return ExitSignal("time_stop")
        return None
