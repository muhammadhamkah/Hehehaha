"""Run the live-connectivity validator end-to-end against the local mock Binance."""
from config import BotConfig
from tests.mock_binance import MockBinanceServer
from tools.validate_binance import run


async def test_validator_against_mock(monkeypatch):
    import tools.validate_binance as vb

    real_sleep = vb.asyncio.sleep

    async def fast_sleep(s):          # compress the tool's fixed waits
        await real_sleep(min(s, 0.6))

    monkeypatch.setattr(vb.asyncio, "sleep", fast_sleep)
    server = MockBinanceServer()
    runner, host = await server.start()
    try:
        cfg = BotConfig()
        cfg.exchange.rest_base = f"http://{host}"
        cfg.exchange.ws_base = f"ws://{host}"
        cfg.risk.stale_data_ms = 300
        report = await run(cfg, n_symbols=3, duration=1.0, limit_streams=9, min_symbols=4)
    finally:
        await runner.cleanup()
    checks = report["checks"]
    for name in ("rest_discovery", "market_streams", "detail_streams", "resubscription",
                 "reconnect", "stale_detection", "diff_depth_sync", "subscription_limit"):
        assert checks[name]["status"] == "PASS", (name, checks[name])
    assert report["feed_health"]["malformed_by_kind"] == {}
