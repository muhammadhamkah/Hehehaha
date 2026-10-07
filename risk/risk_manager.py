"""Stateful risk controls. Every entry must pass :meth:`RiskManager.can_open`.

Safeguards: daily loss limit, max consecutive losses, max trades/hour, per-symbol
cooldown after a loss, max position size / leverage / simultaneous positions,
max spread / slippage, stale WebSocket data, exchange disconnects, API error bursts,
and a kill switch (file on disk or programmatic).

Hard halts (daily loss, consecutive losses, kill switch, API errors, disconnect) stop
new entries AND request flattening of open positions via ``emergency_exit_required``.
"""
from __future__ import annotations

import logging
import os
from collections import deque
from datetime import datetime, timezone

from config import BotConfig
from utils.clock import SYSTEM_CLOCK, Clock

log = logging.getLogger(__name__)


class RiskManager:
    def __init__(self, cfg: BotConfig, clock: Clock = SYSTEM_CLOCK) -> None:
        self.cfg = cfg
        self.rc = cfg.risk
        self.clock = clock
        self.day = self._utc_day(clock.now_ms())
        self.daily_pnl = 0.0
        self.consecutive_losses = 0
        self.trade_times: deque[int] = deque()
        self.cooldowns: dict[str, int] = {}
        self.api_errors: deque[int] = deque()
        self.stream_heartbeats: dict[str, int] = {}
        self.disconnected_since: dict[str, int] = {}
        self.open_positions: dict[str, float] = {}     # symbol -> notional
        self.halted: bool = False
        self.halt_reason: str = ""
        self._manual_kill = False

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _utc_day(ms: int) -> str:
        return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")

    def _roll_day(self, now_ms: int) -> None:
        day = self._utc_day(now_ms)
        if day != self.day:
            log.info("risk: new UTC day %s, resetting daily PnL (was %.4f)", day, self.daily_pnl)
            self.day = day
            self.daily_pnl = 0.0
            if self.halted and self.halt_reason == "daily_loss_limit":
                self.resume()

    def halt(self, reason: str) -> None:
        if not self.halted:
            log.error("RISK HALT: %s", reason)
        self.halted = True
        self.halt_reason = reason

    def resume(self) -> None:
        log.warning("risk: resuming after halt (%s)", self.halt_reason)
        self.halted = False
        self.halt_reason = ""

    def kill(self) -> None:
        self._manual_kill = True
        self.halt("kill_switch")

    def kill_switch_active(self) -> bool:
        return self._manual_kill or (bool(self.rc.kill_switch_file) and os.path.exists(self.rc.kill_switch_file))

    # ------------------------------------------------------------------ events
    def on_trade_opened(self, symbol: str, notional: float) -> None:
        now = self.clock.now_ms()
        self.trade_times.append(now)
        self.open_positions[symbol] = notional

    def on_trade_closed(self, symbol: str, net_pnl: float) -> None:
        now = self.clock.now_ms()
        self._roll_day(now)
        self.open_positions.pop(symbol, None)
        self.daily_pnl += net_pnl
        if net_pnl < 0:
            self.consecutive_losses += 1
            self.cooldowns[symbol] = now + int(self.rc.symbol_cooldown_after_loss_s * 1000)
        else:
            self.consecutive_losses = 0
        if self.daily_pnl <= -abs(self.rc.daily_loss_limit_usdt):
            self.halt("daily_loss_limit")
        if self.consecutive_losses >= self.rc.max_consecutive_losses:
            self.halt("max_consecutive_losses")

    def on_api_error(self, msg: str = "") -> None:
        now = self.clock.now_ms()
        self.api_errors.append(now)
        cutoff = now - int(self.rc.api_error_window_s * 1000)
        while self.api_errors and self.api_errors[0] < cutoff:
            self.api_errors.popleft()
        log.warning("api error (%d in window): %s", len(self.api_errors), msg)
        if len(self.api_errors) >= self.rc.max_api_errors:
            self.halt("api_error_burst")

    def heartbeat(self, stream: str, now_ms: int | None = None) -> None:
        self.stream_heartbeats[stream] = now_ms if now_ms is not None else self.clock.now_ms()
        self.disconnected_since.pop(stream, None)

    def on_disconnect(self, stream: str) -> None:
        self.disconnected_since.setdefault(stream, self.clock.now_ms())

    # ------------------------------------------------------------------ checks
    def periodic_check(self) -> None:
        """Call frequently: evaluates kill switch and disconnect timeouts."""
        now = self.clock.now_ms()
        self._roll_day(now)
        if self.kill_switch_active():
            self.halt("kill_switch")
        for stream, since in self.disconnected_since.items():
            if now - since > self.rc.disconnect_halt_s * 1000:
                self.halt(f"disconnected:{stream}")
                break
        else:
            if self.halted and self.halt_reason.startswith("disconnected:"):
                self.resume()

    def emergency_exit_required(self) -> bool:
        if not self.halted:
            return False
        return self.halt_reason in ("kill_switch", "daily_loss_limit", "api_error_burst") or \
            self.halt_reason.startswith("disconnected:")

    def stream_stale(self, stream: str, now_ms: int | None = None) -> bool:
        now = now_ms if now_ms is not None else self.clock.now_ms()
        ts = self.stream_heartbeats.get(stream)
        return ts is None or now - ts > self.rc.stale_data_ms

    def can_open(self, symbol: str, notional: float, leverage: int, spread_bps: float,
                 expected_slippage_bps: float) -> tuple[bool, str]:
        now = self.clock.now_ms()
        self._roll_day(now)
        if self.kill_switch_active():
            self.halt("kill_switch")
        if self.halted:
            return False, f"halted:{self.halt_reason}"
        if self.daily_pnl <= -abs(self.rc.daily_loss_limit_usdt):
            return False, "daily_loss_limit"
        if self.consecutive_losses >= self.rc.max_consecutive_losses:
            return False, "max_consecutive_losses"
        cutoff = now - 3_600_000
        while self.trade_times and self.trade_times[0] < cutoff:
            self.trade_times.popleft()
        if len(self.trade_times) >= self.rc.max_trades_per_hour:
            return False, "max_trades_per_hour"
        if self.cooldowns.get(symbol, 0) > now:
            return False, "symbol_cooldown"
        if symbol in self.open_positions:
            return False, "position_already_open"
        if len(self.open_positions) >= self.cfg.sizing.max_simultaneous_positions:
            return False, "max_simultaneous_positions"
        if notional > self.rc.max_position_notional_usdt:
            return False, "max_position_size"
        if leverage > self.rc.max_leverage:
            return False, "max_leverage"
        if spread_bps > self.rc.max_spread_bps:
            return False, "risk_max_spread"
        if expected_slippage_bps > self.rc.max_slippage_bps:
            return False, "risk_max_slippage"
        if self.disconnected_since:
            return False, f"disconnected:{next(iter(self.disconnected_since))}"
        return True, "ok"

    def status(self) -> dict:
        return {
            "halted": self.halted,
            "halt_reason": self.halt_reason,
            "daily_pnl": round(self.daily_pnl, 4),
            "consecutive_losses": self.consecutive_losses,
            "trades_last_hour": len(self.trade_times),
            "open_positions": dict(self.open_positions),
            "api_errors_in_window": len(self.api_errors),
        }
