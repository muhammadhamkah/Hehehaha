from config import FeatureConfig
from features import orderbook_features as obf
from features.microstructure_features import compute_features
from market_data.tradeflow import TradeFlow
from tests.helpers import feed_trades, make_book


def test_imbalance_signs():
    b = make_book(bid_qty=90, ask_qty=10)
    assert obf.level_imbalance(b, 5) > 0.7
    assert obf.weighted_imbalance(b, 10, 0.7) > 0.7
    _, off, tilt = obf.microprice_features(b)
    assert off > 0 and tilt > 0


def test_ofi_positive_when_bids_build():
    b = make_book(bid_qty=10, ask_qty=10, ts=1000)
    for i, q in enumerate([20, 30, 40, 50]):
        make_book(bid_qty=q, ask_qty=10, ts=1100 + i * 100, history=b)
    assert obf.order_flow_imbalance(b.history, 1500, 1.0) > 0
    _, _, asym = obf.depletion(b.history, 1500, 1.0)
    assert asym > 0  # bids thickened (negative bid depletion) -> bullish asymmetry
    assert obf.pressure_persistence(b.history, 10) > 0


def test_momentum_and_vol():
    b = make_book(bid=100.0, ts=0)
    for i in range(1, 31):
        make_book(bid=100.0 + 0.01 * i, ts=i * 1000, history=b)
    assert obf.mid_momentum_bps(b.history, 30_000, 10.0) > 0
    assert obf.realized_vol_bps_1s(b.history, 30_000, 30.0) > 0


def test_trade_flow_window_and_features():
    flow = TradeFlow("X")
    end = feed_trades(flow, 0, 50, 100.0, 1.0, buy_ratio=0.8, step_ms=100)
    w = flow.window(end - 100, 5.0)  # last trade at end-100; window is (t-5s, t]
    assert w.count == 50
    assert w.buy_notional > w.sell_notional
    b = make_book(ts=end)
    f = compute_features(b, flow, FeatureConfig(), end)
    assert f["flow_imb_10s"] > 0.5
    assert f["trades_per_s_10s"] > 0
    for key in ("imb_weighted", "ofi_3s", "micro_tilt", "rv_1s_bps", "vol_accel", "spread_bps"):
        assert key in f


def test_duplicate_trade_ids_ignored():
    from market_data.tradeflow import Trade

    flow = TradeFlow("X")
    flow.add(Trade(1, 1.0, 1.0, False), 1, trade_id=5)
    flow.add(Trade(2, 1.0, 1.0, False), 2, trade_id=5)
    assert len(flow.trades) == 1
