import json

import pytest

from analytics.performance import compute_performance, format_report, max_drawdown
from analytics.trade_logger import TradeLogger, TradeRecord
from config import LIVE_CONFIRM_PHRASE, BotConfig, load_config
from data.database import Database


def _trade(net, gross=None, sym="AAA", conf=0.7, maker=True, ts=0):
    gross = net + 0.1 if gross is None else gross
    return {"symbol": sym, "net_pnl": net, "gross_pnl": gross, "entry_fee": 0.03, "exit_fee": 0.07,
            "actual_slippage_usdt": 0.02, "est_slippage_usdt": 0.03, "holding_s": 5.0,
            "entry_ts_ms": ts, "exit_reason": "target_profit" if net > 0 else "stop_loss",
            "signal_confidence": conf, "entry_maker": maker, "exit_maker": False}


def test_performance_metrics():
    trades = [_trade(0.2, ts=1), _trade(-0.1, ts=2), _trade(0.3, ts=3, sym="BBB"), _trade(-0.2, ts=4)]
    p = compute_performance(trades)
    assert p["trades"] == 4 and p["win_rate"] == 0.5
    assert abs(p["net_pnl"] - 0.2) < 1e-9
    assert abs(p["gross_pnl"] - 0.6) < 1e-9
    assert abs(p["fees_paid"] - 0.4) < 1e-9
    assert abs(p["profit_factor"] - 0.5 / 0.3) < 1e-3
    assert abs(p["max_drawdown"] - 0.2) < 1e-9
    assert p["maker_entry_pct"] == 1.0
    assert set(p["by_symbol"]) == {"AAA", "BBB"}
    assert p["verdict"].startswith("INSUFFICIENT")
    assert "VERDICT" in format_report(p)


def test_verdicts():
    losing = [_trade(-0.05, ts=i) for i in range(40)]
    assert compute_performance(losing)["verdict"] == "NOT PROFITABLE AFTER COSTS"
    winning = [_trade(0.1 + 0.01 * (i % 3), ts=i) for i in range(40)]
    assert compute_performance(winning)["verdict"].startswith("PROFITABLE AFTER COSTS")
    assert compute_performance([])["trades"] == 0


def test_max_drawdown():
    assert max_drawdown([1, -2, 1, -3, 5]) == 4


def test_trade_logger_roundtrip(tmp_path):
    db = Database(str(tmp_path / "x.sqlite"), 0.05)
    tl = TradeLogger(db, str(tmp_path / "trades.jsonl"))
    rec = TradeRecord("t1", "paper", "AAA", 1, 0, 5000, 5.0, 0.7, 0.5, 1, 100.0, 100.2, 1.5, 150.0,
                      0.3, 0.03, 0.075, 0.03, 0.02, 0.5, 0.8, 0.195, "target_profit", 25, -3, 0.375,
                      -0.045, True, False, 0.17, 15, 8, {"imb_l1": 0.3})
    tl.log(rec)
    db.flush()
    rows = db.query("SELECT * FROM trades")
    db.close()
    assert len(rows) == 1 and rows[0]["net_pnl"] == 0.195
    assert json.loads(rows[0]["features_json"]) == {"imb_l1": 0.3}
    assert (tmp_path / "trades.jsonl").read_text().strip()


def test_defaults_are_safe(monkeypatch):
    monkeypatch.delenv("LIVE_TRADING_CONFIRM", raising=False)
    cfg = BotConfig()
    assert cfg.dry_run is True
    assert not cfg.live_trading_enabled()
    assert cfg.sizing.max_simultaneous_positions == 1
    assert cfg.entry.min_net_profit_usdt == 0.10
    assert cfg.validate() == []


def test_live_requires_every_switch(monkeypatch):
    cfg = BotConfig(mode="live", dry_run=False)
    cfg.exchange.api_key, cfg.exchange.api_secret = "k", "s"
    monkeypatch.delenv("LIVE_TRADING_CONFIRM", raising=False)
    assert not cfg.live_trading_enabled()
    assert cfg.validate()      # flagged as a problem
    monkeypatch.setenv("LIVE_TRADING_CONFIRM", LIVE_CONFIRM_PHRASE)
    assert cfg.live_trading_enabled()
    cfg.dry_run = True
    assert not cfg.live_trading_enabled()


def test_load_config_overrides(tmp_path, monkeypatch):
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"mode": "record", "entry": {"min_net_profit_usdt": 0.2},
                             "features": {"flow_windows_s": [1, 5]}}))
    monkeypatch.setenv("DRY_RUN", "true")
    cfg = load_config(str(p))
    assert cfg.mode == "record" and cfg.entry.min_net_profit_usdt == 0.2
    assert cfg.features.flow_windows_s == (1, 5)
    with pytest.raises(KeyError):
        load_config(overrides={"entry": {"nope": 1}})


def test_validate_catches_oversized_margin():
    cfg = BotConfig()
    cfg.sizing.position_notional_usdt = 300
    cfg.sizing.leverage = 2
    assert any("margin" in p for p in cfg.validate())
