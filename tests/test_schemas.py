"""Validator tests using payload shapes from the Binance USDT-M WebSocket docs."""
import logging

from exchange.schemas import FeedValidator

AGG = {"e": "aggTrade", "E": 123456789, "s": "BTCUSDT", "a": 5933014, "p": "0.001", "q": "100",
       "f": 100, "l": 105, "T": 123456785, "m": True}
BT = {"e": "bookTicker", "u": 400900217, "E": 1568014460893, "T": 1568014460891, "s": "BNBUSDT",
      "b": "25.35190000", "B": "31.21000000", "a": "25.36520000", "A": "40.66000000"}
DEPTH = {"e": "depthUpdate", "E": 1571889248277, "T": 1571889248276, "s": "BTCUSDT", "U": 390497796,
         "u": 390497878, "pu": 390497794, "b": [["7403.89", "0.002"], ["7403.80", "3.906"]],
         "a": [["7405.96", "3.340"], ["7406.63", "4.525"]]}
TICK = {"e": "24hrTicker", "E": 123456789, "s": "BTCUSDT", "p": "0.0015", "P": "250.00", "w": "0.0018",
        "c": "0.0025", "Q": "10", "o": "0.0010", "h": "0.0025", "l": "0.0010", "v": "10000", "q": "18",
        "O": 0, "C": 86400000, "F": 0, "L": 18150, "n": 18151}


def test_documented_payloads_parse():
    v = FeedValidator()
    t = v.agg_trade("btcusdt@aggTrade", AGG, 123456800, "BTCUSDT")
    assert t and t.price == 0.001 and t.qty == 100 and t.is_buyer_maker and t.agg_id == 5933014
    b = v.book_ticker("bnbusdt@bookTicker", BT, 1568014460900, "BNBUSDT")
    assert b and b.bid < b.ask and b.update_id == 400900217
    d = v.depth("btcusdt@depth20@100ms", DEPTH, 1571889248300, "BTCUSDT")
    assert d and d.bids[0][0] == 7403.89 and d.final_id == 390497878 and d.prev_final_id == 390497794
    assert len(v.ticker_arr("!ticker@arr", [TICK, TICK], 123456800)) == 2
    assert v.summary()["malformed_by_kind"] == {}
    lat = v.summary()["by_kind"]["aggTrade"]["latency_ms_p50"]
    assert lat == 123456800 - 123456789


def test_malformed_messages_are_counted_and_logged(caplog):
    v = FeedValidator()
    with caplog.at_level(logging.WARNING, logger="feed"):
        assert v.agg_trade("x@aggTrade", {**AGG, "p": "abc"}, 0, "BTCUSDT") is None
        assert v.agg_trade("x@aggTrade", {k: x for k, x in AGG.items() if k != "T"}, 0, "BTCUSDT") is None
        assert v.agg_trade("x@aggTrade", {**AGG, "m": "true"}, 0, "BTCUSDT") is None
        assert v.agg_trade("x@aggTrade", {**AGG, "s": "ETHUSDT"}, 0, "BTCUSDT") is None
        assert v.book_ticker("x@bookTicker", {**BT, "b": "30", "a": "25"}, 0, "BNBUSDT") is None
        assert v.depth("x@depth20@100ms", {**DEPTH, "b": [["7403.0", "1"], ["7404.0", "1"]]}, 0, "BTCUSDT") is None
        assert v.depth("x@depth20@100ms", {**DEPTH, "U": 10, "u": 5}, 0, "BTCUSDT") is None
        assert v.depth("x@depth20@100ms", {**DEPTH, "b": "nope"}, 0, "BTCUSDT") is None
        assert v.ticker_arr("!ticker@arr", {"not": "a list"}, 0) == []
        v.report_unexpected("weird@stream", {"x": 1})
    s = v.summary()
    assert s["malformed_by_kind"] == {"aggTrade": 4, "bookTicker": 1, "depth": 3, "ticker_arr": 1}
    assert s["unexpected"] == {"weird@stream": 1}
    assert "malformed aggTrade" in caplog.text and "unexpected message" in caplog.text


def test_sequence_health_tracking():
    v = FeedValidator()
    for a in (1, 2, 5, 5, 6):        # gap of 2 ids (3,4), one duplicate
        v.agg_trade("x@aggTrade", {**AGG, "a": a, "E": 100 + a, "T": 99 + a}, 200, "BTCUSDT")
    h = v.summary()["by_kind"]["aggTrade"]
    assert h["missing_ids"] == 2 and h["seq_gaps"] == 1 and h["duplicates"] == 1
    d1 = {**DEPTH, "U": 1, "u": 10, "pu": 0}
    d2 = {**DEPTH, "U": 11, "u": 20, "pu": 10}      # continuous
    d3 = {**DEPTH, "U": 31, "u": 40, "pu": 30}      # pu != previous u -> gap
    for d in (d1, d2, d3):
        v.depth("x@depth20@100ms", d, 0, "BTCUSDT")
    assert v.summary()["by_kind"]["depth"]["seq_gaps"] == 1
    v.book_ticker("x@bookTicker", {**BT, "E": 10}, 20, "BNBUSDT")
    v.book_ticker("x@bookTicker", {**BT, "E": 5}, 20, "BNBUSDT")
    assert v.summary()["by_kind"]["bookTicker"]["non_monotonic_ts"] == 1
