import os

from config import BotConfig
from risk.risk_manager import RiskManager
from utils.clock import ManualClock


def _rm(**risk):
    cfg = BotConfig()
    cfg.risk.kill_switch_file = ""
    for k, v in risk.items():
        setattr(cfg.risk, k, v)
    clock = ManualClock(1_700_000_000_000)
    return RiskManager(cfg, clock), clock


def ok(rm, sym="AAA", notional=100, lev=5, spread=1.0, slip=1.0):
    return rm.can_open(sym, notional, lev, spread, slip)


def test_daily_loss_limit_halts_and_requests_flatten():
    rm, _ = _rm(daily_loss_limit_usdt=1.0, max_consecutive_losses=99)
    rm.on_trade_closed("AAA", -0.6)
    assert ok(rm, "ZZZ")[0]
    rm.on_trade_closed("BBB", -0.5)
    allowed, reason = ok(rm, "CCC")
    assert not allowed and "daily_loss_limit" in reason
    assert rm.emergency_exit_required()


def test_daily_reset_resumes():
    rm, clock = _rm(daily_loss_limit_usdt=1.0)
    rm.on_trade_closed("AAA", -2.0)
    assert not ok(rm)[0]
    clock.advance(86_400_000)
    assert ok(rm)[0]


def test_consecutive_losses_and_reset_on_win():
    rm, _ = _rm(max_consecutive_losses=3, daily_loss_limit_usdt=100)
    rm.on_trade_closed("A", -0.1)
    rm.on_trade_closed("B", -0.1)
    rm.on_trade_closed("C", 0.2)
    assert rm.consecutive_losses == 0
    for s in ("D", "E", "F"):
        rm.on_trade_closed(s, -0.1)
    assert not ok(rm, "Z")[0]


def test_symbol_cooldown():
    rm, clock = _rm(symbol_cooldown_after_loss_s=60)
    rm.on_trade_closed("AAA", -0.1)
    assert ok(rm, "AAA") == (False, "symbol_cooldown")
    assert ok(rm, "BBB")[0]
    clock.advance(61_000)
    assert ok(rm, "AAA")[0]


def test_trades_per_hour_and_positions():
    rm, clock = _rm(max_trades_per_hour=2)
    rm.on_trade_opened("A", 100)
    assert ok(rm, "B") == (False, "max_simultaneous_positions")
    assert ok(rm, "A") == (False, "position_already_open")
    rm.on_trade_closed("A", 0.1)
    rm.on_trade_opened("B", 100)
    rm.on_trade_closed("B", 0.1)
    assert ok(rm, "C") == (False, "max_trades_per_hour")
    clock.advance(3_600_001)
    assert ok(rm, "C")[0]


def test_size_leverage_spread_slippage_limits():
    rm, _ = _rm(max_position_notional_usdt=200, max_leverage=10, max_spread_bps=4, max_slippage_bps=3)
    assert ok(rm, notional=500) == (False, "max_position_size")
    assert ok(rm, lev=20) == (False, "max_leverage")
    assert ok(rm, spread=5) == (False, "risk_max_spread")
    assert ok(rm, slip=5) == (False, "risk_max_slippage")


def test_api_error_burst():
    rm, clock = _rm(max_api_errors=3, api_error_window_s=10)
    rm.on_api_error("x")
    rm.on_api_error("x")
    clock.advance(11_000)
    rm.on_api_error("x")
    assert not rm.halted          # old errors expired
    rm.on_api_error("x")
    rm.on_api_error("x")
    assert rm.halted and rm.halt_reason == "api_error_burst"


def test_disconnect_and_stale_streams():
    rm, clock = _rm(disconnect_halt_s=5, stale_data_ms=2000)
    rm.heartbeat("detail")
    assert not rm.stream_stale("detail")
    clock.advance(2500)
    assert rm.stream_stale("detail")
    rm.on_disconnect("detail")
    assert not ok(rm)[0]
    clock.advance(6000)
    rm.periodic_check()
    assert rm.halted and rm.emergency_exit_required()
    rm.heartbeat("detail")
    rm.periodic_check()
    assert not rm.halted


def test_kill_switch_file(tmp_path):
    rm, _ = _rm()
    path = tmp_path / "KILL"
    rm.rc.kill_switch_file = str(path)
    assert ok(rm)[0]
    path.write_text("stop")
    assert ok(rm) == (False, "halted:kill_switch")
    assert rm.emergency_exit_required()
    os.remove(path)


def test_manual_kill():
    rm, _ = _rm()
    rm.kill()
    assert not ok(rm)[0]
