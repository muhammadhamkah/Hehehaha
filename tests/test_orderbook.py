import math

from market_data.orderbook import OrderBook, SyncStatus
from tests.helpers import make_book


def test_snapshot_basics():
    b = make_book(bid=100.0, tick=0.01, bid_qty=10, ask_qty=30)
    assert b.best_bid == 100.0 and b.best_ask == 100.01
    assert math.isclose(b.mid, 100.005)
    assert 0.9 < b.spread_bps < 1.1
    # microprice leans toward the thinner (bid) side's opposite: more ask qty -> closer to bid
    assert b.best_bid < b.microprice() < b.mid


def test_walk_and_slippage():
    b = make_book(bid=100.0, tick=0.01, levels=5, ask_qty=1.0)
    avg, qty, full = b.walk("BUY", 100.01 * 1.0 + 100.02 * 0.5)
    assert full and math.isclose(qty, 1.5, rel_tol=1e-9)
    assert 100.01 < avg < 100.02
    # More notional than the book holds -> not fully filled -> infinite slippage
    assert b.slippage_bps("BUY", 1e9) == float("inf")
    assert b.slippage_bps("BUY", 10) > 0


def test_depth_within_bps():
    b = make_book(bid=100.0, tick=0.01, levels=20, bid_qty=1, ask_qty=1)
    bd, ad = b.depth_within_bps(5)  # 5bps of 100 = 0.05 -> ~5 levels each side
    assert 4 * 99 < bd < 7 * 101
    assert 4 * 99 < ad < 7 * 101


def test_diff_sync_sequence_and_gap():
    b = OrderBook("X")
    # event before snapshot is buffered
    assert b.apply_diff({"U": 95, "u": 100, "pu": 94, "b": [], "a": []}) == SyncStatus.BUFFERING
    b.buffer_diff({"U": 101, "u": 105, "pu": 100, "b": [["99.0", "5"]], "a": []})
    status = b.sync_from_snapshot([(99.0, 1.0), (98.0, 1.0)], [(100.0, 1.0)], last_update_id=103, local_ts_ms=1)
    assert status == SyncStatus.OK
    assert b.bids[99.0] == 5.0 and b.last_update_id == 105
    # in-sequence event
    assert b.apply_diff({"U": 106, "u": 110, "pu": 105, "b": [["98.0", "0"]], "a": []}, 2) == SyncStatus.OK
    assert 98.0 not in b.bids
    # stale event ignored
    assert b.apply_diff({"U": 1, "u": 2, "pu": 0, "b": [], "a": []}, 3) == SyncStatus.IGNORED
    # gap -> resync
    assert b.apply_diff({"U": 120, "u": 125, "pu": 119, "b": [], "a": []}, 4) == SyncStatus.RESYNC
    assert not b.synced


def test_crossed_book_triggers_resync():
    b = OrderBook("X")
    b.sync_from_snapshot([(99.0, 1.0)], [(100.0, 1.0)], 10, 1)
    b.apply_diff({"U": 9, "u": 11, "pu": 8, "b": [], "a": []}, 1)
    assert b.synced
    assert b.apply_diff({"U": 12, "u": 12, "pu": 11, "b": [["101.0", "1"]], "a": []}, 2) == SyncStatus.RESYNC


def test_bbo_overlay_removes_stale_levels():
    b = make_book(bid=100.0, tick=0.01)
    b.update_bbo(100.03, 2.0, 100.04, 3.0, 2_000_000)
    assert b.best_bid == 100.03 and b.best_ask == 100.04
    assert not b.crossed()


def test_staleness():
    b = make_book(ts=1000)
    assert not b.is_stale(1500, 2000)
    assert b.is_stale(5000, 2000)


def test_history_conflation_for_tick_level_feeds():
    b = OrderBook("X", history_len=600, history_interval_ms=100)
    for i in range(2000):                       # 2000 updates over 10s (200/s)
        b.apply_snapshot([(100.0 + (i % 3) * 0.01, 1.0)], [(100.05, 1.0)], i, local_ts_ms=i * 5)
    assert len(b.history) == 100               # one state per 100ms bucket
    assert b.history[-1].ts_ms == 1999 * 5     # latest state kept in its bucket
    assert b.history[0].ts_ms // 100 == 0
