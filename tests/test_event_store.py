import gzip
import json
import os

from data.event_store import EventReader, EventWriter, exchange_time, load_meta


def _write_raw(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with gzip.open(path, "wt") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def test_writer_reader_roundtrip(tmp_path):
    w = EventWriter(str(tmp_path), meta={"symbol_info": {"A": 1}})
    for i in range(100):
        w.write("detail", "a@aggTrade", {"E": 1_700_000_000_000 + i, "x": i}, 1_700_000_000_005 + i)
    w.close()
    evs = list(EventReader(str(tmp_path)))
    assert len(evs) == 100 and [e.data["x"] for e in evs] == list(range(100))
    assert evs[0].ts == 1_700_000_000_000 and evs[0].local_ts == 1_700_000_000_005
    assert load_meta(str(tmp_path))["symbol_info"] == {"A": 1}


def test_exchange_time_sorting_and_clamping(tmp_path):
    rows = [
        {"r": 1000, "c": "detail", "s": "a@aggTrade", "d": {"E": 990}},
        {"r": 1001, "c": "detail", "s": "b@aggTrade", "d": {"E": 980}},   # earlier exchange time
        {"r": 1002, "c": "market", "s": "!ticker@arr", "d": [{"E": 995}, {"E": 999}]},
        {"r": 9000, "c": "detail", "s": "a@aggTrade", "d": {"E": 8990}},
        {"r": 9001, "c": "detail", "s": "a@aggTrade", "d": {"E": 100}},   # hopelessly late
        "garbage-line",
    ]
    p = tmp_path / "20240101" / "00.jsonl.gz"
    _write_raw(str(p), rows[:-1])
    with gzip.open(p, "at") as fh:
        fh.write("not json\n")
    r = EventReader(str(tmp_path), reorder_window_ms=100)
    evs = list(r)
    # the 8.9s-late message is replayed at its receive time, not its stale exchange time
    assert [e.ts for e in evs] == [980, 990, 999, 8990, 9001]
    assert r.n_late == 1 and r.n_bad_lines == 1
    loc = [e.ts for e in EventReader(str(tmp_path), time_source="local")]
    assert loc == sorted(loc)
    lat = [e.ts for e in EventReader(str(tmp_path), feed_latency_ms=50, reorder_window_ms=100)]
    assert lat[0] == 1030
    win = [e.ts for e in EventReader(str(tmp_path), start_ms=985, end_ms=8990, reorder_window_ms=100)]
    assert win == [990, 999]


def test_exchange_time_extraction():
    assert exchange_time("x", {"E": 5, "T": 4}, 9) == 5
    assert exchange_time("x", {"T": 4}, 9) == 4
    assert exchange_time("x", {"bids": []}, 9) == 9
    assert exchange_time("x", [{"E": 3}, {"E": 7}], 9) == 7


def test_blocking_writer_never_drops(tmp_path):
    w = EventWriter(str(tmp_path), blocking=True)
    w._q.maxsize = 100          # tiny queue: a non-blocking writer would drop most of these
    for i in range(20_000):
        w.write("detail", "a@aggTrade", {"E": 1_700_000_000_000 + i}, 1_700_000_000_000 + i)
    w.close()
    assert w.dropped == 0 and w.n_written == 20_000
    assert sum(1 for _ in EventReader(str(tmp_path))) == 20_000
