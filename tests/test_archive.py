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


def test_archive_reader_streams_zips_in_time_order(tmp_path):
    import json as _json

    from data.archive_reader import ArchiveReader
    from data.event_store import open_reader

    def z(path, header, rows):
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr("x.csv", header + "\n" + "\n".join(rows) + "\n")
    day0 = 1709251200000      # 2024-03-01T00:00Z
    a = tmp_path / "BTCUSDT-aggTrades-2024-03-01.zip"
    z(a, "agg_trade_id,price,quantity,first_trade_id,last_trade_id,transact_time,is_buyer_maker",
      [f"{i},62000.1,0.01,{i},{i},{day0 + i * 10},false" for i in range(1, 6)])
    b = tmp_path / "BTCUSDT-bookTicker-2024-03-01.zip"
    z(b, "update_id,best_bid_price,best_bid_qty,best_ask_price,best_ask_qty,transaction_time,event_time",
      [f"{100 + i},62000.0,1,62000.1,1,{day0 + i * 10 + 4},{day0 + i * 10 + 5}" for i in range(6)])
    (tmp_path / "archive.json").write_text(_json.dumps({
        "files": {"BTCUSDT": {"aggTrades": [str(a)], "bookTicker": [str(b)]}},
        "symbol_info": {}, "depth_mode": "bbo"}))
    r = open_reader(str(tmp_path), start_ms=day0 + 10, end_ms=day0 + 50)
    assert isinstance(r, ArchiveReader)
    evs = list(r)
    ts = [e.ts for e in evs]
    assert ts == sorted(ts) and ts[0] >= day0 + 10 and ts[-1] < day0 + 50
    assert {e.stream for e in evs} == {"btcusdt@aggTrade", "btcusdt@bookTicker"}
    assert r.time_range() == (day0, day0 + 86_400_000 - 1)
