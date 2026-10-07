import json
import os

import pytest

from backtest.synthetic import generate
from config import BotConfig
from research.validate import apply, combos, config_hash, nested, split_bounds, validate


def test_split_is_chronological_and_disjoint():
    b = split_bounds(0, 1000, (0.6, 0.2, 0.2))
    assert b["train"] == (0, 600) and b["validation"] == (600, 800) and b["test"][0] == 800
    with pytest.raises(ValueError):
        split_bounds(0, 10, (0.5, 0.5, 0.5))


def test_param_mapping_and_hash():
    assert nested({"stop_bps": 7, "latency_ms": 20}) == {"entry": {"stop_bps": 7}, "execution": {"sim_latency_ms": 20}}
    c = apply(BotConfig(), {"stop_bps": 7.0, "taker_fallback": "skip"})
    assert c.entry.stop_bps == 7.0 and c.execution.ttl_fallback == "skip"
    assert config_hash(c) != config_hash(BotConfig())
    assert len(combos({"a": [1, 2], "b": [3, 4, 5]}, None)) == 6
    assert len(combos({"a": [1, 2], "b": [3, 4, 5]}, 4)) == 4


def test_validation_protocol_end_to_end_and_test_lock(tmp_path):
    ev = tmp_path / "ev"
    generate(str(ev), n_symbols=2, minutes=7, edge=0.0, seed=21)
    out = str(tmp_path / "out")
    grid = {"stop_bps": [8.0], "min_probability": [0.3, 0.55]}
    cfg = BotConfig()
    cfg.entry.min_net_profit_usdt = 0.01            # tiny sample: let some trades happen
    cfg.entry.require_flow_confirmation = False
    rep = validate(cfg, str(ev), out, grid, None, 1, [], min_trades=1, final_test=True,
                   latencies=[20, 100, 250], warmup_ms=60_000)
    assert os.path.exists(os.path.join(out, "VALIDATION_REPORT.md"))
    assert set(rep["bounds"]) == {"train", "validation", "test"}
    assert len(rep["sweep_train"]) == 2
    assert "verdict" in rep and rep["verdict"]
    if rep["chosen"] is not None:
        lock = json.load(open(os.path.join(out, "test_lock.json")))
        assert lock["config_hash"] == rep["chosen_config_hash"]
        # A DIFFERENT config on the same test period must be refused.
        with pytest.raises(SystemExit):
            validate(cfg, str(ev), out, {"stop_bps": [12.0], "min_probability": [0.3]}, None, 1, [],
                     min_trades=1, final_test=True, latencies=[100], warmup_ms=60_000)
    md = open(os.path.join(out, "VALIDATION_REPORT.md")).read()
    for section in ("Verdict", "Opportunity decisions", "Feature analysis", "calibration", "Minimum tradable edge",
                    "Per-symbol", "Parameter sweep", "TEST"):
        assert section in md


def test_ineffective_params_and_insufficient_test_verdict():
    from research.validate import PRIMARY_LATENCY, ineffective_params, verdict

    m1 = {"trades": 40, "expectancy": 0.2}
    m2 = {"trades": 35, "expectancy": -0.1}
    sweep = [
        {"params": {"a": 1, "b": 1}, "train": m1}, {"params": {"a": 2, "b": 1}, "train": m1},
        {"params": {"a": 1, "b": 2}, "train": m2}, {"params": {"a": 2, "b": 2}, "train": m2},
    ]
    assert ineffective_params(sweep) == ["a"]
    rep = {"chosen": {"validation": {}}, "test": {PRIMARY_LATENCY: {"trades": 5, "expectancy": 0.3, "t_stat": 1.6}}}
    assert verdict(rep, 30).startswith("INSUFFICIENT TEST SAMPLE")
