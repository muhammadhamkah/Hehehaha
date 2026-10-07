from config import BotConfig
from strategy.barrier import BarrierOutcome
from strategy.costs import CostEstimate
from strategy.entry_filter import EntryPlan
from strategy.exit_engine import ExitEngine, Position
from strategy.predictor import Prediction

T0 = 1_000_000


def _plan(target=15.0, stop=8.0, hold=20.0):
    costs = CostEstimate(150, True, 0.03, 0.075, 0.5, 1.3, 0.0075, 0.0195)
    return EntryPlan("T", 1, 150, True, target, stop, 60, BarrierOutcome(0.6, 0.3, 0.1, 5, hold), costs,
                     0.3, 0.17, 0.2, 0.25, 100.0, 0.7, 0.5)


def _pos(direction=1, **kw):
    p = Position("T", direction, 1.5, 100.0, T0, _plan(**kw), entry_fee=0.03, entry_maker=True,
                 entry_slippage_bps=0.0, entry_trades_per_s=10.0)
    return p


def _engine():
    cfg = BotConfig()
    return cfg, ExitEngine(cfg, cfg.costs.maker_fee, cfg.costs.taker_fee)


def _pred(score):
    p = 0.5 + score / 2
    return Prediction(score, p, 1 - p, score, 2.0, flow_confirms=abs(score) > 0.2)


def _quote(pnl_bps, direction=1, spread=0.01):
    """Quotes such that the executable side gives pnl_bps vs entry 100."""
    mark = 100.0 * (1 + direction * pnl_bps / 1e4)
    return (mark, mark + spread) if direction > 0 else (mark - spread, mark)


F = {"book_age_ms": 0.0, "imb_weighted": 0.2, "flow_imb_3s": 0.2, "trades_per_s_3s": 10.0}


def test_stop_loss_long_and_short():
    cfg, ee = _engine()
    for d in (1, -1):
        pos = _pos(direction=d)
        ee.init_position(pos)
        f = dict(F, imb_weighted=0.2 * d, flow_imb_3s=0.2 * d)
        assert ee.update(pos, *_quote(-3, d), f, _pred(0.3 * d), T0 + 500) is None
        sig = ee.update(pos, *_quote(-8.5, d), f, _pred(0.3 * d), T0 + 600)
        assert sig is not None and sig.reason == "stop_loss"


def test_break_even_then_stop():
    cfg, ee = _engine()
    pos = _pos()
    ee.init_position(pos)
    assert ee.update(pos, *_quote(9), F, _pred(0.3), T0 + 500) is None   # below trigger (10)
    assert not pos.break_even_armed
    assert ee.update(pos, *_quote(11), F, _pred(0.3), T0 + 550) is None  # arms, no instant exit
    assert pos.break_even_armed and pos.stop_level_bps == ee.break_even_bps(pos)
    sig = ee.update(pos, *_quote(ee.break_even_bps(pos) - 0.5), F, _pred(0.3), T0 + 600)
    assert sig.reason == "break_even_stop"


def test_target_takes_profit_without_momentum():
    cfg, ee = _engine()
    pos = _pos(target=10)
    ee.init_position(pos)
    sig = ee.update(pos, *_quote(10.5), F, _pred(0.0), T0 + 2000)
    assert sig.reason == "target_profit"


def test_target_with_momentum_trails_and_captures_more():
    cfg, ee = _engine()
    pos = _pos(target=10)
    ee.init_position(pos)
    assert ee.update(pos, *_quote(10.5), F, _pred(0.6), T0 + 2000) is None   # momentum -> keep
    assert pos.trailing_active
    assert ee.update(pos, *_quote(25), F, _pred(0.6), T0 + 3000) is None      # runs further
    sig = ee.update(pos, *_quote(19), F, _pred(0.6), T0 + 4000)               # 25 - 5 trail = 20
    assert sig.reason == "trailing_stop"
    assert pos.mfe_bps >= 25


def test_flow_reversal_and_imbalance_loss():
    cfg, ee = _engine()
    pos = _pos()
    ee.init_position(pos)
    sig = ee.update(pos, *_quote(1), F, _pred(-0.5), T0 + 2000)
    assert sig.reason == "flow_reversal"

    pos = _pos()
    ee.init_position(pos)
    against = dict(F, imb_weighted=-0.3, flow_imb_3s=-0.3)
    sig = None
    for i in range(cfg.exit.imbalance_loss_evals):
        sig = ee.update(pos, *_quote(1), against, _pred(-0.1), T0 + 2000 + i * 100)
    assert sig is not None and sig.reason == "imbalance_lost"


def test_momentum_exhaustion():
    cfg, ee = _engine()
    pos = _pos()
    ee.init_position(pos)
    f = dict(F, trades_per_s_3s=1.0)
    sig = ee.update(pos, *_quote(8), f, _pred(0.05), T0 + 2000)   # net positive (> ~7bps fees)
    assert sig.reason == "momentum_exhaustion"


def test_time_stop_and_emergency():
    cfg, ee = _engine()
    pos = _pos(hold=5)
    ee.init_position(pos)
    limit = ee.time_stop_s(pos)
    assert ee.update(pos, *_quote(1), F, _pred(0.3), T0 + int(limit * 1000) - 100) is None
    assert ee.update(pos, *_quote(1), F, _pred(0.3), T0 + int(limit * 1000) + 100).reason == "time_stop"
    pos = _pos()
    ee.init_position(pos)
    sig = ee.update(pos, 99.0, 101.0, F, _pred(0.3), T0 + 100)
    assert sig.emergency and sig.reason == "emergency_spread"
    sig = ee.update(pos, *_quote(1), dict(F, book_age_ms=10_000), _pred(0.3), T0 + 100)
    assert sig.emergency and sig.reason == "emergency_stale_data"


def test_stop_never_loosens():
    cfg, ee = _engine()
    pos = _pos()
    ee.init_position(pos)
    ee.update(pos, *_quote(12), F, _pred(0.6), T0 + 1000)
    lvl = pos.stop_level_bps
    ee.update(pos, *_quote(9), F, _pred(0.6), T0 + 1100)
    assert pos.stop_level_bps >= lvl
