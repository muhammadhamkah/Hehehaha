"""Offline end-to-end test of the orchestrator glue using synthetic WS messages."""
import asyncio

from config import BotConfig
from exchange.models import SymbolInfo
from main import TradingBot
from market_data.orderbook import OrderBook
from market_data.tradeflow import TradeFlow
from strategy.barrier import BarrierOutcome
from strategy.costs import CostEstimate
from strategy.entry_filter import EntryDecision, EntryPlan
from strategy.signal_engine import SignalResult

SYM = "TESTUSDT"


def _bot(tmp_path):
    cfg = BotConfig(mode="paper")
    cfg.recorder.db_path = str(tmp_path / "bot.sqlite")
    cfg.recorder.trades_jsonl = str(tmp_path / "trades.jsonl")
    cfg.recorder.flush_interval_s = 0.05
    cfg.risk.kill_switch_file = ""
    cfg.execution.sim_latency_ms = 1
    cfg.execution.maker_ttl_ms = 300
    cfg.execution.sim_queue_factor = 0.0
    bot = TradingBot(cfg)
    bot.symbol_info[SYM] = SymbolInfo(SYM, 0.01, 0.001, 0.001, 5)
    bot.books[SYM] = OrderBook(SYM)
    bot.flows[SYM] = TradeFlow(SYM)
    bot.scanner.selected = [SYM]
    return bot


def _depth(bid, ts):
    return {"E": ts, "u": ts, "b": [[f"{bid - i * 0.01:.2f}", "50"] for i in range(20)],
            "a": [[f"{bid + 0.01 + i * 0.01:.2f}", "50"] for i in range(20)]}


def _plan():
    costs = CostEstimate(150, True, 0.03, 0.075, 0.5, 1.3, 0.0075, 0.0195)
    return EntryPlan(SYM, 1, 150, True, 15, 8, 60, BarrierOutcome(0.6, 0.3, 0.1, 5, 10), costs,
                     0.3, 0.17, 0.2, 0.25, 100.005, 0.7, 0.5)


async def test_message_handling_entry_exit_and_trade_log(tmp_path):
    bot = _bot(tmp_path)
    ts = bot.clock.now_ms()
    for i in range(30):
        bot._on_detail_msg(f"{SYM.lower()}@depth20@100ms", _depth(100.0, ts + i), ts + i)
        bot._on_detail_msg(f"{SYM.lower()}@aggTrade",
                           {"T": ts + i, "p": "100.01", "q": "1", "m": False, "a": i + 1}, ts + i)
    book = bot.books[SYM]
    assert book.synced and book.best_bid == 100.0 and len(book.history) == 30
    assert len(bot.flows[SYM].trades) == 30

    # Signal evaluation runs end-to-end on the live structures (records a signal or not).
    res = bot.signals.evaluate(SYM, book, bot.flows[SYM], ts + 30, trading_enabled=True)
    assert res.features["mid"] > 0 and res.prediction is not None

    # Force an entry with a known plan; a sell-aggressor print at our bid fills the maker order.
    sig = SignalResult(SYM, ts, res.features, res.prediction, EntryDecision(True, "ok", plan=_plan()), "enter")
    bot._entering.add(SYM)

    async def fill_soon():
        await asyncio.sleep(0.05)
        bot._on_detail_msg(f"{SYM.lower()}@aggTrade",
                           {"T": ts + 100, "p": "99.99", "q": "10", "m": True, "a": 1000}, ts + 100)

    asyncio.ensure_future(fill_soon())
    await bot._enter(sig)
    pos = bot.positions[SYM]
    assert pos.entry_maker and abs(pos.qty - 1.5) < 1e-9 and pos.entry_price == 100.0
    assert bot.risk.open_positions == {SYM: pos.notional}
    assert SYM in bot.scanner.pinned

    # Price rallies; exit via taker into the bids.
    bot._on_detail_msg(f"{SYM.lower()}@depth20@100ms", _depth(100.20, ts + 200), ts + 200)
    bot._start_exit(pos, "target_profit", False)
    for _ in range(100):
        if SYM not in bot.positions:
            break
        await asyncio.sleep(0.01)
    assert SYM not in bot.positions
    assert SYM not in bot.risk.open_positions
    bot.db.flush()
    rows = bot.db.query("SELECT * FROM trades")
    orders = bot.db.query("SELECT DISTINCT event FROM orders")
    bot.db.close()
    await bot.client.close()
    assert len(rows) == 1
    t = rows[0]
    assert t["exit_reason"] == "target_profit" and t["direction"] == 1
    assert abs(t["net_pnl"] - (t["gross_pnl"] - t["entry_fee"] - t["exit_fee"])) < 1e-9
    assert t["gross_pnl"] > 0 and t["entry_maker"] == 1 and t["exit_maker"] == 0
    assert {"submitted", "ack", "fill"} <= {o["event"] for o in orders}


async def test_risk_halt_flattens_positions(tmp_path):
    bot = _bot(tmp_path)
    ts = bot.clock.now_ms()
    for i in range(25):
        bot._on_detail_msg(f"{SYM.lower()}@depth20@100ms", _depth(100.0, ts + i), ts + i)
    from strategy.exit_engine import Position

    pos = Position(SYM, 1, 1.5, 100.0, ts, _plan(), 0.03, True, 0.0)
    bot.exit_engine.init_position(pos)
    bot.positions[SYM] = pos
    bot.risk.on_trade_opened(SYM, pos.notional)
    bot.risk.kill()
    task = asyncio.ensure_future(bot._position_loop())
    for _ in range(100):
        if SYM not in bot.positions:
            break
        await asyncio.sleep(0.01)
    bot._stop.set()
    await task
    bot.db.flush()
    rows = bot.db.query("SELECT exit_reason FROM trades")
    bot.db.close()
    await bot.client.close()
    assert rows and rows[0]["exit_reason"] == "emergency_kill_switch"
