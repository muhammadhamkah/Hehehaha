import asyncio

from config import BotConfig
from exchange.execution import PaperExchange, TradeExecutor
from exchange.models import Order, OrderStatus, SymbolInfo
from market_data.tradeflow import Trade
from strategy.barrier import BarrierOutcome
from strategy.costs import CostEstimate, CostModel
from strategy.entry_filter import EntryPlan
from tests.helpers import make_book

SYM = "TESTUSDT"

def _setup(ttl_ms=200, fallback="skip", latency_ms=1, queue_factor=1.0, bid_qty=5.0):
    cfg = BotConfig()
    cfg.execution.maker_ttl_ms = ttl_ms
    cfg.execution.ttl_fallback = fallback
    cfg.execution.sim_latency_ms = latency_ms
    cfg.execution.sim_queue_factor = queue_factor
    book = make_book(SYM, bid=100.0, tick=0.01, levels=20, bid_qty=bid_qty, ask_qty=5.0)
    books = {SYM: book}
    costs = CostModel(cfg.costs)
    paper = PaperExchange(cfg, books, costs)
    info = {SYM: SymbolInfo(SYM, tick_size=0.01, step_size=0.001, min_qty=0.001, min_notional=5)}
    return cfg, book, paper, TradeExecutor(cfg, paper, books, info), costs

def _plan(direction=1, notional=150.0, maker=True):
    costs = CostEstimate(notional, maker, 0.03, 0.075, 0.5, 1.3, 0.0075, 0.0195)
    return EntryPlan(SYM, direction, notional, maker, 15, 8, 60, BarrierOutcome(0.6, 0.3, 0.1, 5, 10),
                     costs, 0.3, 0.17, 0.2, 0.25, 100.005, 0.7, 0.5)

async def _later(delay, fn):
    await asyncio.sleep(delay)
    fn()

async def test_post_only_rejected_if_it_would_take():
    _, book, paper, _, _ = _setup()
    o = Order(SYM, "BUY", "LIMIT", 1.0, price=100.01, tif="GTX")   # at the ask
    await paper.submit(o)
    await asyncio.sleep(0.02)
    assert o.status == OrderStatus.EXPIRED and o.reject_reason == "post_only_would_take"

async def test_maker_fill_respects_queue_and_partial_fills():
    _, book, paper, _, _ = _setup(queue_factor=1.0, bid_qty=5.0)
    o = Order(SYM, "BUY", "LIMIT", 1.0, price=100.0, tif="GTX")
    await paper.submit(o)
    await asyncio.sleep(0.02)
    assert o.status == OrderStatus.NEW
    paper.on_trade(SYM, Trade(1, 100.0, 4.0, is_buyer_maker=True))   # consumes queue (5 ahead)
    assert o.filled_qty == 0
    paper.on_trade(SYM, Trade(2, 100.0, 1.4, is_buyer_maker=True))   # 1 left ahead, 0.4 to us
    assert o.status == OrderStatus.PARTIALLY_FILLED and abs(o.filled_qty - 0.4) < 1e-9
    paper.on_trade(SYM, Trade(3, 100.0, 5.0, is_buyer_maker=False))  # buyer aggressor: no fill
    assert abs(o.filled_qty - 0.4) < 1e-9
    paper.on_trade(SYM, Trade(4, 99.99, 0.1, is_buyer_maker=True))   # traded through -> full
    assert o.status == OrderStatus.FILLED
    assert all(f.maker for f in o.fills) and o.avg_price == 100.0
    assert o.fee > 0

async def test_ioc_taker_is_price_capped():
    _, book, paper, _, _ = _setup()
    o = Order(SYM, "BUY", "LIMIT", 12.0, price=100.02, tif="IOC")   # two levels of 5 within cap
    await paper.submit(o)
    await asyncio.sleep(0.02)
    assert o.status == OrderStatus.EXPIRED          # remainder expired, not chased
    assert abs(o.filled_qty - 10.0) < 1e-9
    assert all(not f.maker for f in o.fills)

async def test_entry_maker_filled_within_ttl():
    _, book, paper, ex, _ = _setup(ttl_ms=500, queue_factor=0.0)
    asyncio.ensure_future(_later(0.05, lambda: paper.on_trade(SYM, Trade(1, 99.99, 100, True))))
    res = await ex.enter(_plan(), recheck=lambda: None)
    assert res.success and res.reason == "maker_filled" and res.maker
    assert abs(res.qty - 1.5) < 1e-9 and res.avg_price == 100.0

async def test_entry_ttl_skip_when_configured():
    _, book, paper, ex, _ = _setup(ttl_ms=100, fallback="skip")
    res = await ex.enter(_plan(), recheck=lambda: _plan(maker=False))
    assert not res.success and res.reason == "maker_ttl_skip"
    assert res.orders[0].status == OrderStatus.CANCELED

async def test_entry_ttl_no_taker_without_edge():
    _, book, paper, ex, _ = _setup(ttl_ms=100, fallback="taker_if_edge")
    res = await ex.enter(_plan(), recheck=lambda: None)          # edge gone -> never chase
    assert not res.success and res.reason == "maker_ttl_no_taker_edge"
    assert len(res.orders) == 1

async def test_entry_ttl_taker_when_edge_remains():
    _, book, paper, ex, _ = _setup(ttl_ms=100, fallback="taker_if_edge")
    res = await ex.enter(_plan(), recheck=lambda: _plan(maker=False))
    assert res.success and res.used_taker_fallback
    taker = res.orders[-1]
    assert taker.tif == "IOC" and taker.type == "LIMIT"           # capped, not MARKET
    assert not res.maker

async def test_small_partial_is_unwound_on_skip():
    _, book, paper, ex, _ = _setup(ttl_ms=150, fallback="skip", queue_factor=0.0)
    asyncio.ensure_future(_later(0.03, lambda: paper.on_trade(SYM, Trade(1, 100.0, 0.1, True))))
    res = await ex.enter(_plan(), recheck=lambda: None)
    assert not res.success
    unwind = res.orders[-1]
    assert unwind.reduce_only and abs(unwind.filled_qty - 0.1) < 1e-9

async def test_exit_flattens_position():
    _, book, paper, ex, _ = _setup()
    res = await ex.exit(SYM, 1, 1.5)
    assert res.success and abs(res.qty - 1.5) < 1e-9
    assert res.orders[0].reduce_only and res.orders[0].side == "SELL"
    assert res.avg_price <= 100.0

async def test_exit_falls_back_to_market_when_book_too_thin():
    cfg, book, paper, ex, _ = _setup()
    book.apply_snapshot([(100.0, 0.5)], [(100.01, 5.0)], 1, local_ts_ms=1)
    res = await ex.exit(SYM, 1, 1.5, emergency=True)
    assert res.orders[-1].type == "MARKET"
