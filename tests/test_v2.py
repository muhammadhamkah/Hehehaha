"""V2: dataset labels, train/serve parity, bot integration, research-only safety."""
import os

import numpy as np
import pandas as pd
import pytest

from backtest.synthetic import generate
from config import BotConfig
from research.v2_dataset import barrier_labels, build_shard, check_days, tp_first
from research.v2_models import feature_columns, matrix, run

DAY = 86_400_000
D1 = 1717200000000          # 2024-06-01T00:00Z
DAYS = ["2024-06-01", "2024-06-02", "2024-06-03"]


def test_barrier_labels_first_touch_and_ties():
    # quotes every 100 ms; mid rises 1 bp per quote after entry
    q_ts = np.arange(0, 70_000, 100, dtype=np.int64)
    mid = 100.0 * (1 + np.maximum(q_ts - 1000, 0) / 100 / 1e4)
    bid, ask = mid - 0.005, mid + 0.005
    lab = barrier_labels(np.array([1000], np.int64), np.array([100.0]), q_ts, bid, ask, 60_000, 0)
    # long entry at ask(1000)=100.005; TP level 100.005*1.0008=100.085004; bid=100+0.01k-0.005 first
    # exceeds it at k=10 -> 1000 ms after entry
    assert lab["ttp_long_8"].iat[0] == 1000
    assert lab["tsl_long_8"].iat[0] == np.iinfo(np.int64).max          # never hit
    assert lab["ttp_short_8"].iat[0] == np.iinfo(np.int64).max
    assert lab["tsl_short_6"].iat[0] > 0
    df = pd.DataFrame({"ttp_long_8": [500, 900, 900, 70_000], "tsl_long_8": [900, 500, 900, 80_000]})
    assert tp_first(df, "long", 8, 8).tolist() == [1, 0, 0, 0]        # tie -> stop; beyond horizon -> 0


def test_forbidden_test_days_are_refused():
    with pytest.raises(SystemExit):
        check_days(["2024-03-28", "2024-03-29"], ["2024-03-29", "2024-03-30"])
    check_days(["2024-03-28"], ["2024-03-29"])


@pytest.fixture(scope="module")
def v2_setup(tmp_path_factory):
    root = tmp_path_factory.mktemp("v2")
    ev = str(root / "events")
    for i in range(3):
        generate(ev, n_symbols=2, minutes=7, edge=1.0, seed=31 + i, start_ms=D1 + i * DAY)
    cfg = BotConfig()
    ds = str(root / "ds")
    for day in DAYS:
        for sym in ("SYN0USDT", "SYN1USDT"):
            build_shard((ev, ds, sym, day, (0, 1), cfg, 1000, 250, 60_000, 100, 0.10))
    out = str(root / "models")
    rep = run(ds, out, [DAYS[0]], DAYS[1], DAYS[2], 8, 150.0, 0.01, ["logistic", "lightgbm"], [8, 12], [8], 2,
              [0.3, 0.5])
    return ev, ds, out, rep


def test_dataset_has_no_future_in_features_and_labels(v2_setup):
    ev, ds, out, rep = v2_setup
    df = pd.read_parquet(os.path.join(ds, "SYN0USDT_2024-06-01.parquet"))
    cols = feature_columns(df)
    assert len(cols) > 300 and not any(c.startswith(("ttp_", "tsl_", "fret_", "hret_")) for c in cols)
    assert {"ttp_long_8", "tsl_short_15", "fret_60s", "req_bps_long_150", "cost_bps_short_100"} <= set(df.columns)
    assert df["ts_ms"].is_monotonic_increasing and (df["ts_ms"].diff().dropna() == 1000).mean() > 0.9


def test_serving_model_matches_research_predictions(v2_setup):
    from strategy.v2 import V2Model

    ev, ds, out, rep = v2_setup
    se = pd.read_parquet(os.path.join(ds, "SYN1USDT_2024-06-03.parquet"))
    cols = feature_columns(se)
    for kind in ("logistic", "lightgbm"):
        served = V2Model(os.path.join(out, kind))
        assert served.features == cols
        art = rep["_artifacts"][kind]
        m, cals, _, _ = art["arts"][("long", 8)]
        cal = cals.get(art["method"], cals["raw"])
        x = matrix(se.iloc[:25], cols)
        research = cal(m.predict(x))
        serving = [served.predict(r)[("long", 8)] for r in se.iloc[:25][cols].to_dict("records")]
        assert np.allclose(research, serving, atol=1e-6)


def test_v2_runs_through_the_same_replay_framework(v2_setup, tmp_path):
    from backtest.replay import ReplaySpec, run_replay

    ev, ds, out, rep = v2_setup
    cfg = BotConfig()
    cfg.strategy.predictor = "v2"
    cfg.strategy.v2_model_dir = os.path.join(out, "lightgbm")
    cfg.strategy.v2_threshold = 0.0
    cfg.entry.min_net_profit_usdt = 0.01
    cfg.execution.entry_mode = "taker"
    cfg.exit.mode = "barrier"
    cfg.risk.research_mode = True
    res = run_replay(cfg, ReplaySpec(ev, str(tmp_path / "r"), start_ms=D1 + 2 * DAY + 60_000,
                                     end_ms=D1 + 2 * DAY + 400_000))
    assert res["research_only"] is True
    assert res["evaluations"] > 100
    from data.database import Database
    db = Database(str(tmp_path / "r" / "replay.sqlite"))
    try:
        sig = db.query("SELECT predicted_json, target_bps, p_target FROM signals LIMIT 50")
        trades = db.query("SELECT exit_reason FROM trades")
    finally:
        db.close()
    assert sig and '"model": "v2:lightgbm"' in sig[0]["predicted_json"]
    assert all(t["exit_reason"] in ("target_profit", "stop_loss", "time_stop", "shutdown") or
               t["exit_reason"].startswith("emergency") for t in trades)


def test_research_mode_and_v2_live_are_refused():
    import asyncio

    from main import TradingBot
    from risk.risk_manager import RiskManager

    cfg = BotConfig()
    cfg.risk.research_mode = True
    cfg.risk.kill_switch_file = ""
    rm = RiskManager(cfg)
    for _ in range(10):
        rm.on_trade_closed("X", -0.1)
    assert not rm.halted and rm.can_open("Y", 100, 5, 1, 1)[0]      # research mode ignores the streak
    bot = TradingBot(cfg)
    with pytest.raises(SystemExit):
        asyncio.run(bot.start())                                      # never online with research mode
    bot.db.close()
    live = BotConfig(mode="live", dry_run=False)
    live.strategy.predictor = "v2"
    assert any("V2" in p for p in live.validate())


def test_barrier_exit_mode():
    from strategy.barrier import BarrierOutcome
    from strategy.costs import CostEstimate
    from strategy.entry_filter import EntryPlan
    from strategy.exit_engine import ExitEngine, Position

    cfg = BotConfig()
    cfg.exit.mode = "barrier"
    ee = ExitEngine(cfg, cfg.costs.maker_fee, cfg.costs.taker_fee)
    plan = EntryPlan("T", 1, 150, False, 12, 8, 30, BarrierOutcome(0.6, 0.4, 0, 1, 10),
                     CostEstimate(150, False, 0.075, 0.075, 1, 1, 0.015, 0.015), 0.2, 0.1, 0.1, 0.3, 100, 0.6, 0.6)
    f = {"book_age_ms": 0.0}

    def pos():
        p = Position("T", 1, 1.5, 100.0, 0, plan, 0.075, False, 0.0)
        ee.init_position(p)
        return p
    assert ee.update(pos(), 100.13, 100.14, f, None, 1000).reason == "target_profit"
    assert ee.update(pos(), 99.91, 99.92, f, None, 1000).reason == "stop_loss"
    assert ee.update(pos(), 100.05, 100.06, f, None, 10_000) is None    # no V1 reversal/trailing exits
    assert ee.update(pos(), 100.05, 100.06, f, None, 31_000).reason == "time_stop"


def test_compare_runner_and_test_lock(v2_setup, tmp_path, monkeypatch):
    import json
    import sys

    from research import compare_v1_v2 as C

    ev, ds, out, rep = v2_setup
    o = str(tmp_path / "cmp")
    base = tmp_path / "base.json"
    base.write_text(json.dumps({"entry": {"min_net_profit_usdt": 0.01, "min_depth_usdt_within_10bps": 0.0}}))
    argv = ["x", "--events", ev, "--out", o, "--base-config", str(base), "--v2-models", out,
            "--kinds", "lightgbm", "--symbols", "SYN0USDT", "SYN1USDT", "--val-days", DAYS[1],
            "--test-days", DAYS[2], "--window", "0", "1", "--workers", "1", "--final-test"]
    monkeypatch.setattr(sys, "argv", argv)
    C.main()
    r = json.load(open(os.path.join(o, "comparison.json")))
    assert set(r["test"]) == {"V1 rule-based", "V2 lightgbm"}
    assert "V2 lightgbm" in r["validation_research_only"]
    assert os.path.exists(os.path.join(o, "V1_VS_V2.md"))
    lock = json.load(open(os.path.join(o, "test_lock.json")))
    assert set(lock) == {"V1 rule-based", "V2 lightgbm"}
    # a CHANGED frozen config may not be evaluated on the same test period
    base.write_text(json.dumps({"entry": {"min_net_profit_usdt": 0.02, "min_depth_usdt_within_10bps": 0.0}}))
    with pytest.raises(SystemExit):
        C.main()


def test_twostage_serving_matches_research(v2_setup, tmp_path):
    """Stage A -> Stage B -> combined calibration: served predictions equal the research ones."""
    from research import v2_twostage as TS
    from research.v2_dataset import load_dataset
    from strategy.v2 import TwoStageModel, load_v2_model

    ev, ds, out, rep = v2_setup
    tr, ca, se = (load_dataset(ds, [d]) for d in DAYS)
    cols = feature_columns(tr)
    xtr, xca, xse = matrix(tr, cols), matrix(ca, cols), matrix(se, cols)
    T = 8
    a = TS.stage_a("lightgbm", tr, ca, se, xtr, xca, xse, T, 60, 2)
    b = TS.stage_b("lightgbm", tr, ca, se, xtr, xca, xse, T, 60, 2, min_pop=(20, 5, 5))
    assert a is not None and b is not None and 0 <= a["metrics"]["base_rate"] <= 1
    A, B = {T: a}, {T: b}
    calib = TS.fit_combo_calib(A, B, ca, xca, [T], 8)
    a_thr = {T: float(np.quantile(a["p_ca"], 0.5))}
    sel = {"targets": [T], "b_thr": 0.5, "a_thr": a_thr, "calib": calib, "selected": {"trades": 0}}
    d = TS.export_twostage(str(tmp_path), "lightgbm", cols, A, B, sel, 8, 150.0, 0.1)
    served = load_v2_model(d)
    assert isinstance(served, TwoStageModel)
    research = TS.combined_probs(A, B, xse, a_thr, 0.5, {T: calib[T].predict}, [T])
    for i, r in enumerate(se[cols].to_dict("records")[:60]):
        got = served.predict(r)
        for side in ("long", "short"):
            assert abs(got.get((side, T), 0.0) - research[(side, T)][i]) < 1e-6
        if not got:
            assert served.last_reject in ("stage_a_no_large_move", "stage_b_direction_weak")
    assert TS.feature_kind("rmean_ofi_3s_1000ms") == "signed"
    assert TS.feature_kind("rv_1s_bps") == "intensity"
    assert TS.feature_kind("rmax_spread_bps_3000ms") == "intensity"
    assert TS.feature_kind("bid_depth_5bps") == "side_pair"
