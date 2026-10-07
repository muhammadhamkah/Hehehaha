import random

from config import BotConfig
from data.database import Database
from data.recorder import Recorder
from strategy.predictor import Prediction

T0 = 1_700_000_000_000


def _pred(direction=1):
    p = 0.8 if direction > 0 else 0.2
    return Prediction(0.5 * direction, p, 1 - p, 0.5 * direction, 2.0, True)


def _features(bid=100.0, ask=100.01):
    return {"bid": bid, "ask": ask, "mid": (bid + ask) / 2, "spread_bps": 1.0, "imb_l1": 0.2}


def _setup(tmp_path):
    cfg = BotConfig()
    cfg.recorder.label_target_bps = 6.0
    cfg.recorder.label_stop_bps = 8.0
    db = Database(str(tmp_path / "t.sqlite"), flush_interval_s=0.05)
    return cfg, db, Recorder(cfg, db, random.Random(0))


def test_forward_labels_and_target_before_stop(tmp_path):
    cfg, db, rec = _setup(tmp_path)
    sid = rec.record_signal("AAA", T0, _features(), _pred(1), "reject", "low_confidence", "signal",
                            {"p_target": 0.6})
    # price rises steadily: +1bp per second for 61 s
    for i in range(1, 62):
        mid = 100.005 * (1 + i / 1e4)
        rec.on_quote("AAA", T0 + i * 1000, mid - 0.005, mid + 0.005)
    assert "AAA" not in rec.pending
    db.flush()
    row = db.query("SELECT * FROM signals WHERE id = ?", (sid,))[0]
    db.close()
    assert row["labeled"] == 1
    assert abs(row["ret_1s"] - 1.0) < 0.05 and abs(row["ret_10s"] - 10.0) < 0.1
    assert row["tp_long"] == 1 and row["tp_short"] == -1 and row["tp_pred"] == 1
    assert row["mfe_bps"] > 50 and row["rejection_reason"] == "low_confidence"
    assert row["features_json"] and row["predicted_json"]


def test_stop_first_and_sweep_of_dead_symbols(tmp_path):
    cfg, db, rec = _setup(tmp_path)
    a = rec.record_signal("AAA", T0, _features(), _pred(1), "reject", "x", "signal")
    rec.on_quote("AAA", T0 + 500, 99.90, 99.91)     # -9bps bid vs ask0 -> stop first
    rec.on_quote("AAA", T0 + 900, 100.2, 100.21)    # later spike doesn't change outcome
    b = rec.record_signal("BBB", T0, _features(), _pred(-1), "reject", "x", "signal")
    rec.sweep(T0 + 200_000)                          # no more quotes ever arrive
    assert not rec.pending
    db.flush()
    ra = db.query("SELECT * FROM signals WHERE id = ?", (a,))[0]
    rb = db.query("SELECT * FROM signals WHERE id = ?", (b,))[0]
    db.close()
    assert ra["tp_long"] == -1 and ra["labeled"] == 2
    assert rb["labeled"] == 2 and rb["ret_1s"] is None


def test_recording_policy(tmp_path):
    cfg, db, rec = _setup(tmp_path)
    cfg.strategy.record_score_threshold = 0.2
    cfg.strategy.baseline_sample_prob = 0.0
    assert rec.should_record("A", 0.1, False, T0) is None
    assert rec.should_record("A", 0.3, False, T0) == "signal"
    assert rec.should_record("A", 0.0, True, T0) == "entry"
    rec._last_record_ms["A"] = T0
    assert rec.should_record("A", 0.9, False, T0 + 10) is None    # rate limited
    db.close()


def test_horizon_gap_yields_null(tmp_path):
    cfg, db, rec = _setup(tmp_path)
    sid = rec.record_signal("AAA", T0, _features(), _pred(1), "reject", "x", "signal")
    rec.on_quote("AAA", T0 + 4000, 100.0, 100.01)    # first quote 4s later: 1s/3s too stale
    rec.on_quote("AAA", T0 + 61_000, 100.0, 100.01)
    db.flush()
    row = db.query("SELECT ret_1s, ret_3s, ret_5s FROM signals WHERE id = ?", (sid,))[0]
    db.close()
    # 1s: first quote 3s late -> NULL; 3s: quote 1s late (within tolerance) -> labelled;
    # 5s: next quote arrives at 61s -> NULL rather than a misleading stale value.
    assert row["ret_1s"] is None and row["ret_3s"] is not None and row["ret_5s"] is None
