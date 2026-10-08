"""V3: L2 book sync, recorder (mock exchange), features (no look-ahead), labels, dataset,
research, serving parity, replay integration, and V1/V2 invariance."""
import asyncio
import json
import os

import numpy as np
import pandas as pd
import pytest

from v3.book import BUFFERING, GAP, IGNORED, OK, L2Book

DAY0 = 1_717_200_000_000          # 2024-06-01 00:00 UTC


def _snap(L, bids, asks):
    return {"lastUpdateId": L, "bids": [[str(p), str(q)] for p, q in bids], "asks": [[str(p), str(q)] for p, q in asks]}


def _diff(U, u, pu, b=(), a=(), E=1):
    return {"U": U, "u": u, "pu": pu, "E": E, "T": E, "b": [[str(p), str(q)] for p, q in b],
            "a": [[str(p), str(q)] for p, q in a]}


# ---------------------------------------------------------------------- book
def test_book_sync_rules_and_gaps():
    bk = L2Book("X")
    assert bk.on_diff(_diff(95, 99, 94, b=[(10, 1)]), 0)[0] == BUFFERING
    assert bk.on_diff(_diff(100, 104, 99, b=[(10, 2)]), 0)[0] == BUFFERING
    status, ch = bk.on_snapshot(_snap(102, [(10, 5), (9, 1)], [(11, 1), (12, 1)]), 0)
    assert status == OK and bk.valid and bk.bids[10] == 2.0           # u<L dropped, 100<=102<=104 bridged
    assert [(c.price, c.old, c.new) for c in ch] == [(10.0, 5.0, 2.0)]
    assert bk.on_diff(_diff(105, 106, 104, a=[(11, 0)], b=[(10.5, 1)]), 0)[0] == OK
    assert 11.0 not in bk.asks and bk.best()[0] == 10.5
    assert bk.on_diff(_diff(101, 103, 100), 0)[0] == IGNORED          # stale
    st, _ = bk.on_diff(_diff(108, 110, 107), 0)                      # pu != 106 -> gap
    assert st == GAP and not bk.valid and bk.stats.gap_reasons == {"pu_mismatch": 1}
    assert bk.on_diff(_diff(111, 112, 110), 0)[0] == BUFFERING         # never continues silently
    st, _ = bk.on_snapshot(_snap(200, [(10, 1)], [(11, 1)]), 0)        # snapshot newer than buffered
    assert st == BUFFERING
    assert bk.on_diff(_diff(250, 260, 249), 0)[0] == GAP               # U > L: snapshot not bridged
    assert bk.stats.gap_reasons["snapshot_not_bridged"] == 1
    bk2 = L2Book("Y")
    bk2.on_diff(_diff(1, 5, 0), 0)
    bk2.on_snapshot(_snap(3, [(10, 1)], [(11, 1)]), 0)
    assert bk2.on_diff(_diff(6, 7, 5, b=[(12, 1)]), 0)[0] == GAP and bk2.stats.crossed == 1


def test_walk_bps_uses_depth_beyond_touch():
    bk = L2Book("X")
    bk.on_snapshot(_snap(1, [(100, 1), (99, 10)], [(101, 1), (102, 10)]), 0)
    bk.on_diff(_diff(1, 2, 0), 0)
    assert bk.walk_bps("BUY", 50) == 0.0                               # fits in the top level
    assert bk.walk_bps("BUY", 303) > 0 and bk.walk_bps("SELL", 10_000) == float("inf")


# ---------------------------------------------------------------------- synthetic store
@pytest.fixture(scope="module")
def syn(tmp_path_factory):
    from v3.synthetic_l2 import generate
    root = tmp_path_factory.mktemp("v3")
    store = str(root / "store")
    for i in range(4):
        generate(store, minutes=9, edge=0.3, gap_prob=0.002, seed=21 + i, start_ms=DAY0 + i * 86_400_000)
    return root, store


def test_store_reader_and_depth20_agreement(syn):
    from data.event_store import open_reader
    from v3.store import V3StoreReader, is_v3_store

    _, store = syn
    assert is_v3_store(store)
    r = open_reader(store, symbols=["SYNAUSDT"])
    assert isinstance(r, V3StoreReader)
    bk, ok, bad, gaps, last = L2Book("SYNAUSDT"), 0, 0, 0, None
    for ev in r:
        assert last is None or ev.ts >= last
        last = ev.ts
        head, _, kind = ev.stream.partition("@")
        if head.startswith("__"):
            if head.startswith("__snapshot") and not bk.valid:
                bk.on_snapshot(ev.data, ev.ts)
            continue
        assert head == "synausdt"
        if kind == "depth@100ms":
            gaps += bk.on_diff(ev.data, ev.ts)[0] == GAP
        elif kind == "depth20@100ms" and bk.valid and bk.last_u == ev.data["u"]:
            b, a = bk.top(20)
            same = b == [(float(p), float(q)) for p, q in ev.data["b"]] and a == [(float(p), float(q)) for p, q in ev.data["a"]]
            ok, bad = ok + same, bad + (not same)
    assert ok > 1000 and bad == 0 and gaps >= 1


def test_features_have_no_lookahead(syn):
    from data.event_store import open_reader
    from v3.state import SymbolState

    _, store = syn
    evs = list(open_reader(store, symbols=["SYNBUSDT"], start_ms=DAY0, end_ms=DAY0 + 120_000))
    t = DAY0 + 90_000
    full = SymbolState("SYNBUSDT")
    cut = SymbolState("SYNBUSDT")
    for ev in evs:
        full.on_message(ev.stream, ev.data, ev.ts)
        if ev.ts < t:
            cut.on_message(ev.stream, ev.data, ev.ts)
    cut.advance(t)
    rows_full = {}
    full2 = SymbolState("SYNBUSDT", want_row=lambda x: x == t)
    full2.collect = True
    for ev in evs:
        full2.on_message(ev.stream, ev.data, ev.ts)
    rows_full = {r["ts_ms"]: r for r in full2.rows}
    a, b = rows_full[t], cut.last_row
    assert cut.last_eval_ts == t and b is not None
    for k, v in a.items():
        assert (np.isnan(v) and np.isnan(b[k])) or v == b[k], k


def test_feature_groups_cover_ablation_sets():
    from v3.features import feature_group
    assert feature_group("v3d_imb_5") == "l2" and feature_group("v3d_imb_1") == "l1"
    assert feature_group("v3d_dep_a_1s") == "l1" and feature_group("v3d_can_asym_3s") == "l2"
    assert feature_group("v3f_sweepimb_3s") == "flow" and feature_group("flow_imb_3s") == "flow"
    assert feature_group("rmean_imb_l1_1000ms") == "l1" and feature_group("v3x_absorb_net") == "l2flow"
    assert feature_group("v3t_sflow_d1000") == "flow" and feature_group("v3t_mlofi_d500") == "l2"


# ---------------------------------------------------------------------- labels
def test_labels_first_touch_pairs_and_excursions():
    from v3.labels import INF, labels, pair_outcome, up_first
    t0 = 1_000_000
    ts = np.arange(t0, t0 + 200_000, 50)
    bid = np.full(len(ts), 100.0)
    ask = np.full(len(ts), 100.01)
    up_at = (ts >= t0 + 5_000) & (ts < t0 + 9_000)           # +25 bps rally, later a -40 bps drop
    bid[up_at], ask[up_at] = 100.26, 100.27
    dn = ts >= t0 + 30_000
    bid[dn], ask[dn] = 99.60, 99.61
    df = labels(np.array([t0]), np.array([100.005]), ts, bid, ask, latency_ms=100)
    r = df.iloc[0]
    assert r["entry_ask"] == 100.01 and r["entry_bid"] == 100.0
    assert 4_800 <= r["t_up_20"] <= 5_000 and r["t_up_30"] == INF          # 100.26 vs 100.01*(1.003)
    assert 29_800 <= r["t_dn_30"] <= 30_000 and up_first(df, 20)[0] == 1 and up_first(df, 30)[0] == 0
    win, gross = pair_outcome(df, "long", 20, 12)
    assert win[0] == 1 and gross[0] == 20
    win, gross = pair_outcome(df, "long", 30, 20)
    assert win[0] == 0 and gross[0] == -20                                  # stop at -40 bps drop
    assert r["mfe_long"] > 24 and r["mae_long"] < -39


# ---------------------------------------------------------------------- recorder vs mock exchange
def test_recorder_marks_gap_and_resyncs(tmp_path):
    from tests.mock_binance import MockBinanceServer
    from v3.recorder import L2Recorder, RecorderConfig
    from v3.store import is_v3_store

    class GapServer(MockBinanceServer):
        n = 0

        def payload(self, stream):
            if stream.endswith("@depth@100ms"):
                self.n += 1
                if self.n == 12:
                    s = stream.split("@")[0].upper()
                    self.seq[s] = self.seq.get(s, 1000) + 3            # one diff never sent
            return super().payload(stream)

    async def go():
        srv = GapServer()
        runner, host = await srv.start()
        cfg = RecorderConfig(out=str(tmp_path / "l2"), symbols=("BTCUSDT", "ETHUSDT"), rest_base=f"http://{host}",
                             ws_base=f"ws://{host}", duration_s=3.0, health_interval_s=1.0, audit_interval_s=1.0,
                             min_free_gb=0.0, resync_delay_s=0.1)
        await L2Recorder(cfg).run()
        await runner.cleanup()

    asyncio.run(go())
    from data.event_store import open_reader
    store = str(tmp_path / "l2")
    assert is_v3_store(store)
    kinds = {}
    for ev in open_reader(store):
        k = ev.stream.split("@")[0] if ev.stream.startswith("__") else ev.stream.split("@", 1)[1]
        kinds[k] = kinds.get(k, 0) + 1
    assert kinds["depth@100ms"] > 10 and kinds["aggTrade"] > 10 and kinds["bookTicker"] > 10
    assert kinds["depth20@100ms"] > 10 and kinds["__snapshot__"] >= 3 and kinds["__gap__"] >= 1
    assert kinds["__resync__"] >= 3
    health = [json.loads(x) for f in os.listdir(os.path.join(store, "_health"))
              for x in open(os.path.join(store, "_health", f))]
    last = health[-1]["symbols"]
    assert sum(v["gaps"] for v in last.values()) >= 1
    assert sum(v["gap_reasons"].get("pu_mismatch", 0) for v in last.values()) >= 1
    assert all(v["book_valid"] or v["resyncs"] for v in last.values())
    meta = json.load(open(os.path.join(store, "meta.json")))
    assert meta["depth_mode"] == "diff" and "BTCUSDT" in meta["symbol_info"]


# ---------------------------------------------------------------------- dataset / research / serving / replay
@pytest.fixture(scope="module")
def built(syn):
    from v3.dataset import build_shard
    root, store = syn
    ds = str(root / "ds")
    days = ["2024-06-01", "2024-06-02", "2024-06-03", "2024-06-04"]
    for d in days:
        for s in ("SYNAUSDT", "SYNBUSDT"):
            st = build_shard((store, ds, s, d, (0, 1), 1000, 250, 100))
            assert st["rows"] > 0
    return root, store, ds, days


def test_dataset_rows_costs_labels(built):
    from v3.labels import PAIRS, XS
    _, _, ds, _ = built
    df = pd.read_parquet(os.path.join(ds, "SYNAUSDT_2024-06-01.parquet"))
    assert len(df) > 200 and df["ts_ms"].is_monotonic_increasing
    for c in ["imb_l1", "v3d_imb_20", "v3d_can_asym_3s", "v3x_absorb_net", "v3t_mid_d100", "cost_long_1000",
              "costm_short_150", "mfe_long", "quote_gap_ms"] + [f"t_up_{x}" for x in XS] + \
            [f"tp_long_{t}_{s}" for t, s in PAIRS]:
        assert c in df, c
    assert (df["cost_long_1000"] >= df["cost_long_100"] - 1e-6).all()
    assert (df["quote_gap_ms"] <= 3000).all()
    assert os.path.exists(os.path.join(ds, "timeline", "SYNAUSDT_2024-06-01.parquet"))


def test_research_refuses_test_days(built, tmp_path, monkeypatch):
    import sys
    from v3 import research
    _, _, ds, days = built
    monkeypatch.setattr(sys, "argv", ["x", "--data", ds, "--out", str(tmp_path), "--train-days", days[0],
                                      "--cal-days", days[1], "--sel-days", days[2], "--forbid-days", days[2]])
    with pytest.raises(SystemExit, match="TEST days"):
        research.main()


def test_research_export_and_serving_parity(built, tmp_path, monkeypatch):
    import sys
    from v3 import research
    from v3.dataset import load
    from data.event_store import open_reader
    from config import BotConfig
    from strategy.v3 import SignalEngineV3

    _, store, ds, days = built
    out, models = str(tmp_path / "r"), str(tmp_path / "m")
    monkeypatch.setattr(sys, "argv", ["x", "--data", ds, "--out", out, "--export", models, "--train-days", days[0],
                                      days[1], "--cal-days", days[2], "--sel-days", days[3], "--threads", "2",
                                      "--min-trades", "1"])
    research.main()
    rep = json.load(open(os.path.join(out, "v3_research.json")))
    assert rep["split"]["train"] == days[:2] and "classification" in rep
    assert {"L1", "L2", "FLOW", "L1+L2", "ALL"} <= set(rep["feature_sets"])
    spec_dir = os.path.join(models, "v3_all")
    spec = json.load(open(os.path.join(spec_dir, "spec.json")))
    assert spec["kind"] == "v3" and spec["models"]
    # serving computes the same features as the dataset (bot receives only subscribed streams)
    cfg = BotConfig()
    eng = SignalEngineV3(cfg, None, None, None, spec_dir)
    se = load(ds, [days[3]], ["SYNAUSDT"]).set_index("ts_ms")
    subscribed = ("synausdt@depth@100ms", "synausdt@aggTrade", "synausdt@bookTicker")
    checked = 0
    for ev in open_reader(store, symbols=["SYNAUSDT"], start_ms=DAY0 + 3 * 86_400_000 - 900_000,
                          end_ms=DAY0 + 3 * 86_400_000 + 400_000):
        if ev.stream in subscribed or ev.stream.startswith("__snapshot"):
            eng.on_detail(ev.stream, ev.data, ev.ts)
            st = eng.states.get("SYNAUSDT")
            r = st.last_row if st else None
            if r is not None and r["ts_ms"] in se.index and r.get("_chk") is None:
                r["_chk"] = 1
                ref = se.loc[r["ts_ms"]]
                for c in spec["features"]:
                    a, b = float(ref[c]), float(np.float32(r[c]))
                    assert (np.isnan(a) and np.isnan(b)) or abs(a - b) <= 1e-5 * max(1, abs(a)), c
                checked += 1
    assert checked > 50


def test_v3_replay_through_bot(built, tmp_path):
    from backtest.replay import ReplaySpec, run_replay
    from v3.evaluate import v3_config
    from v3 import research

    root, store, ds, days = built
    models = str(tmp_path / "m")
    import sys
    argv = sys.argv
    sys.argv = ["x", "--data", ds, "--out", str(tmp_path / "r"), "--export", models, "--train-days", days[0], days[1],
                "--cal-days", days[2], "--sel-days", days[3], "--threads", "2", "--min-trades", "1"]
    try:
        research.main()
    finally:
        sys.argv = argv
    cfg = v3_config(os.path.join(models, "v3_all"))
    cfg.risk.research_mode = True
    cfg.v3.threshold = 0.0
    cfg.entry.min_net_profit_usdt = 0.001
    start = DAY0 + 3 * 86_400_000
    res = run_replay(cfg, ReplaySpec(store, str(tmp_path / "rep"), start_ms=start + 60_000, end_ms=start + 480_000,
                                     symbols=["SYNAUSDT", "SYNBUSDT"], warmup_ms=60_000))
    assert res["research_only"] is True and res["evaluations"] > 100
    from data.database import Database
    db = Database(str(tmp_path / "rep" / "replay.sqlite"))
    try:
        sig = db.query("SELECT predicted_json, rejection_reason FROM signals LIMIT 200")
    finally:
        db.close()
    assert sig and any('"model": "v3:v3_all"' in s["predicted_json"] for s in sig)
    assert not all(s["rejection_reason"] == "v3_features_unavailable" for s in sig)


# ---------------------------------------------------------------------- invariance / safety
def test_v1_v2_fingerprints_unchanged():
    from config import BotConfig, _merge, load_config
    assert BotConfig().fingerprint() == "b7776ffa19edc30b"
    c = load_config("configs/l1_adapted.json")
    assert c.fingerprint() == "1f363c0ac9cdd77f"
    path = "runs/real_l1_protocol/chosen_config.json"
    if os.path.exists(path):
        d = json.load(open(path))
        for k in ("exchange", "recorder", "mode", "dry_run", "log_level"):
            d.pop(k, None)
        _merge(c, d)
        assert c.fingerprint() == "09b8b8a39f5c637b"          # V1 locked test hash


def test_v3_refused_live():
    from config import BotConfig
    c = BotConfig()
    c.mode = "live"
    c.strategy.predictor = "v3"
    assert any("V3" in p for p in c.validate())
