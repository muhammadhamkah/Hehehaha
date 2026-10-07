from config import BotConfig
from strategy.costs import CostModel
from strategy.entry_filter import EntryFilter
from strategy.predictor import Prediction
from tests.helpers import make_book


def _features(book, **kw):
    bd, ad = book.depth_within_bps(10)
    f = {
        "book_age_ms": 0.0, "flow_age_ms": 0.0, "spread_bps": book.spread_bps, "imb_l10": 0.3,
        "rv_1s_bps": 2.0, "liquidity_change": 0.0, "trades_per_s_10s": 5.0,
        "bid_depth_10bps": bd, "ask_depth_10bps": ad, "mid": book.mid,
    }
    f.update(kw)
    return f


def _pred(p_long=0.85, drift=0.6, sigma=2.0, confirms=True):
    return Prediction(score=0.6, p_long=p_long, p_short=1 - p_long, drift_bps_per_s=drift,
                      sigma_1s_bps=sigma, flow_confirms=confirms)


def _setup(**cfg_over):
    cfg = BotConfig()
    for k, v in cfg_over.items():
        section, key = k.split("__")
        setattr(getattr(cfg, section), key, v)
    book = make_book(bid=100.0, tick=0.01, levels=20, bid_qty=500, ask_qty=500)
    return cfg, EntryFilter(cfg, CostModel(cfg.costs)), book


def test_strong_signal_passes_with_positive_net_edge():
    cfg, ef, book = _setup()
    d = ef.evaluate("TESTUSDT", _pred(), _features(book), book)
    assert d.ok, d.reason
    p = d.plan
    assert p.direction == 1
    assert p.expected_net_usdt >= cfg.entry.min_net_profit_usdt
    assert p.net_if_target_usdt >= cfg.entry.min_net_profit_usdt
    # Net = probability-weighted gross minus ALL costs
    assert abs(p.expected_net_usdt - (p.expected_gross_usdt - p.costs.total_usdt)) < 1e-9
    assert p.loss_if_stop_usdt <= cfg.sizing.max_risk_per_trade_usdt * 1.05


def test_attractive_gross_but_insufficient_net_is_rejected():
    # A moderate signal passes with normal fees ...
    cfg, ef, book = _setup()
    assert ef.evaluate("TESTUSDT", _pred(drift=0.5), _features(book), book).ok
    # ... but the identical signal is rejected when fees eat the edge, even though the
    # gross target itself still looks attractive.
    cfg, ef, book = _setup(costs__maker_fee=0.0006, costs__taker_fee=0.0010)
    d = ef.evaluate("TESTUSDT", _pred(drift=0.5), _features(book), book)
    assert not d.ok
    assert d.reason == "expected_net_below_min"
    assert d.details["net_if_target"] >= 0.10


def test_small_notional_cannot_reach_min_net():
    cfg, ef, book = _setup(sizing__position_notional_usdt=10.0)
    d = ef.evaluate("TESTUSDT", _pred(), _features(book), book)
    assert not d.ok and d.reason == "required_move_too_large"


def test_weak_drift_rejected_on_expected_net():
    cfg, ef, book = _setup()
    d = ef.evaluate("TESTUSDT", _pred(p_long=0.7, drift=0.05), _features(book), book)
    assert not d.ok and d.reason in ("p_target_too_low", "expected_net_below_min")


def test_gate_conditions():
    cfg, ef, book = _setup()
    assert ef.evaluate("T", _pred(confirms=False), _features(book), book).reason == "flow_not_confirming"
    assert ef.evaluate("T", _pred(p_long=0.55), _features(book), book).reason == "low_confidence"
    assert ef.evaluate("T", _pred(), _features(book, spread_bps=10.0), book).reason == "spread_too_wide"
    assert ef.evaluate("T", _pred(), _features(book, book_age_ms=10_000), book).reason == "stale_data"
    assert ef.evaluate("T", _pred(), _features(book, imb_l10=0.99), book).reason == "abnormal_book_one_sided"
    assert ef.evaluate("T", _pred(), _features(book, rv_1s_bps=50), book).reason == "abnormal_volatility"
    assert ef.evaluate("T", _pred(), _features(book, trades_per_s_10s=0.1), book).reason == "thin_tape"
    assert ef.evaluate("T", _pred(), _features(book, bid_depth_10bps=10), book).reason == "insufficient_depth"


def test_thin_book_slippage_rejected():
    cfg, ef, _ = _setup()
    thin = make_book(bid=100.0, tick=0.01, levels=20, bid_qty=0.2, ask_qty=0.2)
    f = _features(thin, bid_depth_10bps=1e6, ask_depth_10bps=1e6)
    assert ef.evaluate("T", _pred(), f, thin).reason == "slippage_too_high"


def test_force_taker_costs_more():
    cfg, ef, book = _setup()
    maker = ef.evaluate("T", _pred(), _features(book), book)
    taker = ef.evaluate("T", _pred(), _features(book), book, force_taker=True)
    assert maker.ok and maker.plan.entry_maker
    if taker.ok:
        assert not taker.plan.entry_maker
        assert taker.plan.costs.total_usdt > maker.plan.costs.total_usdt


def test_sizing_respects_risk_cap():
    cfg, ef, _ = _setup(sizing__max_risk_per_trade_usdt=0.1)
    n = ef.size_notional(stop_bps=10, exit_slip_bps=2)
    loss = n * (12 / 1e4 + cfg.costs.maker_fee + cfg.costs.taker_fee)
    assert abs(loss - 0.1) < 1e-9
