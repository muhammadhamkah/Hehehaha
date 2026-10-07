import zipfile

from data.event_store import EventReader, EventWriter
from tools.binance_archive import convert


def test_archive_conversion_to_ws_shaped_events(tmp_path):
    agg = tmp_path / "BTCUSDT-aggTrades-2024-03-01.zip"
    with zipfile.ZipFile(agg, "w") as zf:
        zf.writestr("a.csv", "agg_trade_id,price,quantity,first_trade_id,last_trade_id,transact_time,is_buyer_maker\n"
                             "1,62000.10,0.005,10,11,1709251200100,true\n2,62000.20,0.010,12,12,1709251200300,false\n")
    bt = tmp_path / "BTCUSDT-bookTicker-2024-03-01.zip"
    with zipfile.ZipFile(bt, "w") as zf:
        zf.writestr("b.csv", "update_id,best_bid_price,best_bid_qty,best_ask_price,best_ask_qty,transaction_time,event_time\n"
                             "100,62000.0,1.2,62000.1,0.8,1709251200050,1709251200051\n"
                             "101,62000.1,1.0,62000.2,0.9,1709251200200,1709251200201\n")
    out = tmp_path / "events"
    w = EventWriter(str(out))
    res = convert("BTCUSDT", [str(agg)], [str(bt)], w)
    w.close()
    assert res["counts"] == {"aggTrade": 2, "bookTicker": 2}
    assert res["symbol_info"]["tick_size"] == 0.1 and res["symbol_info"]["step_size"] == 0.001
    evs = list(EventReader(str(out)))
    assert [e.stream for e in evs] == ["btcusdt@bookTicker", "btcusdt@aggTrade", "btcusdt@bookTicker",
                                       "btcusdt@aggTrade"]
    assert evs[1].data["m"] is True and evs[1].data["a"] == 1
    # payloads pass the same validator the live bot uses
    from exchange.schemas import FeedValidator
    v = FeedValidator()
    assert v.agg_trade(evs[1].stream, evs[1].data, evs[1].ts, "BTCUSDT") is not None
    assert v.book_ticker(evs[0].stream, evs[0].data, evs[0].ts, "BTCUSDT") is not None
