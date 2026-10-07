"""Replay backtester: determinism, no look-ahead, latency modelling, segment isolation."""
import gzip
import json
import os
import shutil

import pytest

from backtest.replay import ReplaySpec, run_replay
from backtest.synthetic import generate
from config import BotConfig
from data.database import Database
from data.event_store import list_event_files


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    d = tmp_path_factory.mktemp("events")
    info = generate(str(d), n_symbols=2, minutes=4, edge=0.8, seed=11)
    return str(d), info


def _cfg(latency_ms=50):
    cfg = BotConfig()
    cfg.execution.sim_latency_ms = latency_ms
    # loosen gates so the short synthetic sample produces trades to compare
    cfg.entry.min_net_profit_usdt = 0.01
    cfg.entry.min_p_target_before_stop = 0.3
    cfg.entry.require_flow_confirmation = False
    cfg.risk.symbol_cooldown_after_loss_s = 5
    cfg.strategy.record_min_interval_ms = 250
    return cfg


def _rows(path, sql):
    db = Database(path)
    try:
        return db.query(sql)
    finally:
        db.close()


def test_replay_is_deterministic(dataset, tmp_path):
    d, _ = dataset
    r1 = run_replay(_cfg(), ReplaySpec(d, str(tmp_path / "a")))
    r2 = run_replay(_cfg(), ReplaySpec(d, str(tmp_path / "b")))
    assert r1["performance"] == r2["performance"]
    assert r1["decisions"] == r2["decisions"]
    t1 = _rows(str(tmp_path / "a" / "replay.sqlite"), "SELECT symbol, entry_ts_ms, exit_ts_ms, net_pnl FROM trades")
    t2 = _rows(str(tmp_path / "b" / "replay.sqlite"), "SELECT symbol, entry_ts_ms, exit_ts_ms, net_pnl FROM trades")
    assert t1 == t2
    assert r1["evaluations"] > 100 and r1["performance"]["trades"] >= 1


def test_no_lookahead_future_tampering_does_not_change_past(dataset, tmp_path):
    """Replace every event after T with garbage-shifted prices: nothing before T may change."""
    d, info = dataset
    T = info["start_ms"] + 150_000
    tampered = tmp_path / "tampered"
    shutil.copytree(d, tampered)
    for path in list_event_files(str(tampered)):
        out = []
        with gzip.open(path, "rt") as fh:
            for line in fh:
                o = json.loads(line)
                data = o["d"]
                ex = data.get("E") if isinstance(data, dict) else max(x["E"] for x in data)
                if ex >= T and isinstance(data, dict):
                    for k in ("p", "b", "a"):
                        if k in data and isinstance(data[k], str):
                            data[k] = f"{float(data[k]) * 1.05:.2f}"       # +5% jump in the future
                        elif k in data and isinstance(data[k], list):
                            data[k] = [[f"{float(p) * 1.05:.2f}", q] for p, q in data[k]]
                out.append(json.dumps(o))
        with gzip.open(path, "wt") as fh:
            fh.write("\n".join(out) + "\n")
    run_replay(_cfg(), ReplaySpec(d, str(tmp_path / "orig")))
    run_replay(_cfg(), ReplaySpec(str(tampered), str(tmp_path / "tamp")))
    q_sig = f"SELECT ts_ms, symbol, features_json, score, decision, rejection_reason FROM signals WHERE ts_ms < {T} ORDER BY id"
    q_ord = f"SELECT ts_ms, client_id, event, status, filled_qty, avg_price FROM orders WHERE ts_ms < {T} ORDER BY id"
    a = _rows(str(tmp_path / "orig" / "replay.sqlite"), q_sig)
    b = _rows(str(tmp_path / "tamp" / "replay.sqlite"), q_sig)
    assert len(a) > 50 and a == b
    oa = [dict(r, client_id=None) for r in _rows(str(tmp_path / "orig" / "replay.sqlite"), q_ord)]
    ob = [dict(r, client_id=None) for r in _rows(str(tmp_path / "tamp" / "replay.sqlite"), q_ord)]
    assert oa == ob
    q_tr = f"SELECT symbol, entry_ts_ms, entry_price, qty FROM trades WHERE entry_ts_ms < {T - 70_000}"
    assert (_rows(str(tmp_path / "orig" / "replay.sqlite"), q_tr) ==
            _rows(str(tmp_path / "tamp" / "replay.sqlite"), q_tr))


@pytest.mark.parametrize("latency", [20, 250])
def test_order_latency_is_modelled(dataset, tmp_path, latency):
    d, _ = dataset
    out = str(tmp_path / f"lat{latency}")
    run_replay(_cfg(latency), ReplaySpec(d, out))
    rows = _rows(os.path.join(out, "replay.sqlite"), "SELECT client_id, ts_ms, event FROM orders ORDER BY id")
    sub = {r["client_id"]: r["ts_ms"] for r in rows if r["event"] == "submitted"}
    ack = {r["client_id"]: r["ts_ms"] for r in rows if r["event"] in ("ack", "expired", "rejected")}
    gaps = [ack[c] - sub[c] for c in ack if c in sub]
    assert gaps and all(g == latency for g in gaps)


def test_segment_end_is_respected(dataset, tmp_path):
    d, info = dataset
    end = info["start_ms"] + 180_000
    res = run_replay(_cfg(), ReplaySpec(d, str(tmp_path / "seg"), start_ms=info["start_ms"] + 60_000, end_ms=end))
    rows = _rows(str(tmp_path / "seg" / "replay.sqlite"), "SELECT MAX(ts_ms) AS m, MIN(ts_ms) AS n FROM signals")
    assert rows[0]["m"] < end and rows[0]["n"] >= info["start_ms"] + 60_000
    assert res["replay"]["last_ts"] < end
