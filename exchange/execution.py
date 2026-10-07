"""Order execution.

Layers
  ExchangeAdapter   submit / cancel primitives with a common Order lifecycle
    PaperExchange   realistic simulator: latency, post-only rejection, queue position,
                    partial fills from real trade prints, book-walk taker fills, fees
    LiveExchange    Binance REST + user-data-stream fills (actual fees / maker flag)
  TradeExecutor     entry/exit protocol shared by paper and live:
                    * maker-first post-only (GTX) entry with TTL
                    * on TTL: cancel and skip, OR taker IOC only if a fresh re-evaluation
                      with taker costs still clears the net-profit requirement
                    * never a blind market chase: taker entries are price-capped IOC
                    * exits: price-capped reduce-only IOC with widening cap; MARKET
                      reduce-only only as the final emergency step
"""
from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable

from config import BotConfig
from exchange.binance_client import BinanceAPIError, BinanceFuturesClient
from exchange.models import Fill, Order, OrderStatus, SymbolInfo
from market_data.orderbook import OrderBook
from market_data.tradeflow import Trade
from strategy.costs import CostModel
from strategy.entry_filter import EntryPlan
from utils.clock import SYSTEM_CLOCK, Clock

log = logging.getLogger(__name__)

OrderCallback = Callable[[Order, str], None]


class ExchangeAdapter(ABC):
    def __init__(self, clock: Clock = SYSTEM_CLOCK, on_order: OrderCallback | None = None) -> None:
        self.clock = clock
        self.on_order = on_order

    def _notify(self, order: Order, event: str) -> None:
        if self.on_order:
            try:
                self.on_order(order, event)
            except Exception:  # noqa: BLE001
                log.exception("order callback failed")

    @abstractmethod
    async def submit(self, order: Order) -> Order: ...

    @abstractmethod
    async def cancel(self, order: Order) -> None: ...

    async def reconcile(self, order: Order) -> None:
        """Refresh order state from the source of truth (no-op for the simulator)."""


# ====================================================================== paper
@dataclass
class _Resting:
    order: Order
    queue_ahead: float


class PaperExchange(ExchangeAdapter):
    def __init__(self, cfg: BotConfig, books: dict[str, OrderBook], costs: CostModel,
                 clock: Clock = SYSTEM_CLOCK, on_order: OrderCallback | None = None) -> None:
        super().__init__(clock, on_order)
        self.cfg = cfg
        self.books = books
        self.costs = costs
        self.resting: dict[str, _Resting] = {}
        self.latency_s = cfg.execution.sim_latency_ms / 1000.0

    async def submit(self, order: Order) -> Order:
        order.submit_ts_ms = self.clock.now_ms()
        order.events.append((order.submit_ts_ms, "submitted"))
        self._notify(order, "submitted")
        asyncio.ensure_future(self._activate(order))
        return order

    async def _activate(self, order: Order) -> None:
        await asyncio.sleep(self.latency_s)
        now = self.clock.now_ms()
        if order.is_done:
            return
        book = self.books.get(order.symbol)
        if book is None or not book.synced:
            order.reject_reason = "no_market_data"
            order.mark(OrderStatus.REJECTED, now)
            self._notify(order, "rejected")
            return
        bid, _, ask, _ = book.best()
        if order.type == "LIMIT" and order.tif == "GTX":
            if (order.side == "BUY" and order.price >= ask) or (order.side == "SELL" and order.price <= bid):
                order.reject_reason = "post_only_would_take"
                order.mark(OrderStatus.EXPIRED, now, "expired: post-only would take")
                self._notify(order, "expired")
                return
            order.mark(OrderStatus.NEW, now, "ack")
            self._notify(order, "ack")
            self.resting[order.client_id] = _Resting(order, self._queue_ahead(book, order))
            return
        # Taker: MARKET or LIMIT IOC (price-capped)
        order.mark(OrderStatus.NEW, now, "ack")
        self._notify(order, "ack")
        self._taker_fill(book, order, now)
        if not order.is_done:
            order.mark(OrderStatus.EXPIRED, now, "expired: IOC remainder")
        self._notify(order, order.status.value.lower())

    def _queue_ahead(self, book: OrderBook, order: Order) -> float:
        side = book.bids if order.side == "BUY" else book.asks
        return side.get(order.price, 0.0) * self.cfg.execution.sim_queue_factor

    def _taker_fill(self, book: OrderBook, order: Order, now: int) -> None:
        b, a = book.sorted_levels()
        levels = a if order.side == "BUY" else b
        extra = self.cfg.execution.sim_extra_taker_slippage_bps / 1e4
        for price, qty in levels:
            if order.remaining <= 1e-12:
                break
            if order.type == "LIMIT":
                if (order.side == "BUY" and price > order.price) or (order.side == "SELL" and price < order.price):
                    break
            take = min(qty, order.remaining)
            px = price * (1 + extra) if order.side == "BUY" else price * (1 - extra)
            order.add_fill(Fill(now, px, take, self.costs.fee(px * take, maker=False), maker=False))

    async def cancel(self, order: Order) -> None:
        await asyncio.sleep(self.latency_s)   # fills can still happen while the cancel is in flight
        self.resting.pop(order.client_id, None)
        if not order.is_done:
            order.mark(OrderStatus.CANCELED, self.clock.now_ms(), "canceled")
            self._notify(order, "canceled")

    # --- market data hooks driving maker fills
    def on_trade(self, symbol: str, t: Trade) -> None:
        for cid, r in list(self.resting.items()):
            o = r.order
            if o.symbol != symbol or o.is_done:
                if o.is_done:
                    self.resting.pop(cid, None)
                continue
            fill_qty = 0.0
            if o.side == "BUY" and t.is_buyer_maker and t.price <= o.price:
                if t.price < o.price:
                    fill_qty = o.remaining          # traded through our level
                else:
                    avail = t.qty - r.queue_ahead
                    r.queue_ahead = max(0.0, r.queue_ahead - t.qty)
                    fill_qty = max(0.0, avail)
            elif o.side == "SELL" and not t.is_buyer_maker and t.price >= o.price:
                if t.price > o.price:
                    fill_qty = o.remaining
                else:
                    avail = t.qty - r.queue_ahead
                    r.queue_ahead = max(0.0, r.queue_ahead - t.qty)
                    fill_qty = max(0.0, avail)
            if fill_qty > 0:
                self._maker_fill(o, min(fill_qty, o.remaining), t.ts_ms)
                if o.is_done:
                    self.resting.pop(cid, None)

    def on_book(self, symbol: str) -> None:
        book = self.books.get(symbol)
        if book is None:
            return
        bid, _, ask, _ = book.best()
        for cid, r in list(self.resting.items()):
            o = r.order
            if o.symbol != symbol:
                continue
            if o.is_done:
                self.resting.pop(cid, None)
                continue
            side = book.bids if o.side == "BUY" else book.asks
            r.queue_ahead = min(r.queue_ahead, side.get(o.price, 0.0))
            crossed = (o.side == "BUY" and ask and ask <= o.price) or (o.side == "SELL" and bid and bid >= o.price)
            if crossed:
                self._maker_fill(o, o.remaining, self.clock.now_ms())
                self.resting.pop(cid, None)

    def _maker_fill(self, o: Order, qty: float, ts: int) -> None:
        o.add_fill(Fill(ts, o.price, qty, self.costs.fee(o.price * qty, maker=True), maker=True))
        self._notify(o, "fill")


# ====================================================================== live
class LiveExchange(ExchangeAdapter):
    def __init__(self, client: BinanceFuturesClient, clock: Clock = SYSTEM_CLOCK,
                 on_order: OrderCallback | None = None) -> None:
        super().__init__(clock, on_order)
        self.client = client
        self.orders: dict[str, Order] = {}

    async def submit(self, order: Order) -> Order:
        order.submit_ts_ms = self.clock.now_ms()
        order.events.append((order.submit_ts_ms, "submitted"))
        self.orders[order.client_id] = order
        self._notify(order, "submitted")
        try:
            resp = await self.client.new_order(
                order.symbol, order.side, order.type, order.qty,
                price=order.price if order.type == "LIMIT" else None,
                time_in_force=order.tif if order.type == "LIMIT" else None,
                reduce_only=order.reduce_only, client_id=order.client_id,
            )
        except BinanceAPIError as exc:
            order.reject_reason = f"{exc.code}:{exc.msg}"
            status = OrderStatus.EXPIRED if exc.code == -5022 else OrderStatus.REJECTED
            order.mark(status, self.clock.now_ms(), f"rejected: {exc.msg}")
            self._notify(order, "rejected")
            return order
        except Exception as exc:  # noqa: BLE001 - network: state unknown, reconcile later
            order.reject_reason = f"submit_error:{exc}"
            log.warning("order submit uncertain (%s); reconciling", exc)
            await self.reconcile(order)
            return order
        self._apply_rest(order, resp)
        return order

    def _apply_rest(self, order: Order, resp: dict) -> None:
        order.exchange_id = str(resp.get("orderId", order.exchange_id))
        status = resp.get("status", "NEW")
        now = self.clock.now_ms()
        if status == "NEW" and order.status == OrderStatus.PENDING_NEW:
            order.mark(OrderStatus.NEW, now, "ack")
            self._notify(order, "ack")
        elif status in ("CANCELED", "EXPIRED", "REJECTED") and not order.is_done:
            # Fills (if any) arrive through the user stream / reconcile.
            if float(resp.get("executedQty", 0) or 0) <= order.filled_qty + 1e-12:
                order.mark(OrderStatus(status), now, status.lower())
                self._notify(order, status.lower())
        elif order.status == OrderStatus.PENDING_NEW:
            order.mark(OrderStatus.NEW, now, "ack")

    def on_user_event(self, ev: dict) -> None:
        if ev.get("e") != "ORDER_TRADE_UPDATE":
            return
        o = ev["o"]
        order = self.orders.get(o.get("c", ""))
        if order is None:
            return
        order.exchange_id = str(o.get("i", order.exchange_id))
        ts = int(o.get("T", ev.get("E", self.clock.now_ms())))
        x, status = o.get("x"), o.get("X")
        if x == "TRADE":
            qty = float(o.get("l", 0))
            fee = float(o.get("n", 0) or 0)
            if o.get("N") not in (None, "USDT"):
                log.warning("commission paid in %s; fee accounting may be off", o.get("N"))
            order.add_fill(Fill(ts, float(o.get("L", 0)), qty, fee, bool(o.get("m"))))
            self._notify(order, "fill")
        elif status == "NEW" and order.status == OrderStatus.PENDING_NEW:
            order.mark(OrderStatus.NEW, ts, "ack")
            self._notify(order, "ack")
        if status in ("CANCELED", "EXPIRED", "REJECTED") and not order.is_done:
            order.mark(OrderStatus(status), ts, status.lower())
            self._notify(order, status.lower())

    async def cancel(self, order: Order) -> None:
        if order.is_done:
            return
        try:
            resp = await self.client.cancel_order(order.symbol, order.client_id)
            self._apply_rest(order, resp)
        except BinanceAPIError as exc:
            if exc.code == -2011:   # unknown order: already filled / canceled
                await self.reconcile(order)
            else:
                raise

    async def reconcile(self, order: Order) -> None:
        try:
            d = await self.client.get_order(order.symbol, order.client_id)
        except BinanceAPIError as exc:
            if exc.code == -2013 and order.status == OrderStatus.PENDING_NEW:  # never created
                order.mark(OrderStatus.REJECTED, self.clock.now_ms(), "not found")
            return
        order.exchange_id = str(d.get("orderId", order.exchange_id))
        executed = float(d.get("executedQty", 0) or 0)
        if executed > order.filled_qty + 1e-12 and order.exchange_id:
            trades = await self.client.user_trades(order.symbol, order.exchange_id)
            order.fills = [
                Fill(int(t["time"]), float(t["price"]), float(t["qty"]), float(t["commission"]), bool(t["maker"]))
                for t in trades
            ]
        status = OrderStatus(d.get("status", order.status.value))
        if status != order.status:
            order.mark(status, self.clock.now_ms(), f"reconciled {status.value}")


# ====================================================================== protocol
@dataclass
class ExecResult:
    success: bool
    reason: str
    qty: float = 0.0
    avg_price: float = 0.0
    fee: float = 0.0
    maker_qty: float = 0.0
    orders: list[Order] = field(default_factory=list)
    used_taker_fallback: bool = False
    plan: EntryPlan | None = None

    @property
    def maker(self) -> bool:
        return self.qty > 0 and self.maker_qty >= 0.5 * self.qty


def _aggregate(orders: list[Order]) -> tuple[float, float, float, float]:
    qty = sum(o.filled_qty for o in orders)
    notional = sum(o.avg_price * o.filled_qty for o in orders)
    fee = sum(o.fee for o in orders)
    maker = sum(o.maker_qty for o in orders)
    return qty, (notional / qty if qty else 0.0), fee, maker


class TradeExecutor:
    def __init__(self, cfg: BotConfig, adapter: ExchangeAdapter, books: dict[str, OrderBook],
                 symbol_info: dict[str, SymbolInfo], clock: Clock = SYSTEM_CLOCK) -> None:
        self.cfg = cfg
        self.x = cfg.execution
        self.adapter = adapter
        self.books = books
        self.symbol_info = symbol_info
        self.clock = clock

    def info(self, symbol: str) -> SymbolInfo:
        si = self.symbol_info.get(symbol)
        if si is None:
            si = SymbolInfo(symbol, tick_size=1e-8, step_size=1e-8, min_qty=0.0)
        return si

    async def _wait(self, order: Order, timeout_s: float) -> bool:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_s
        while True:
            order.changed.clear()
            if order.is_done:
                return True
            remaining = deadline - loop.time()
            if remaining <= 0:
                return False
            try:
                await asyncio.wait_for(order.changed.wait(), remaining)
            except asyncio.TimeoutError:
                return order.is_done

    async def _cancel_and_settle(self, order: Order) -> None:
        if order.is_done:
            return
        try:
            await self.adapter.cancel(order)
        except Exception as exc:  # noqa: BLE001
            log.warning("cancel failed for %s: %s", order.client_id, exc)
        if not await self._wait(order, 3.0):
            await self.adapter.reconcile(order)

    async def _taker_ioc(self, symbol: str, side: str, qty: float, cap_bps: float,
                         reduce_only: bool = False) -> Order | None:
        book = self.books[symbol]
        bid, _, ask, _ = book.best()
        if not bid or not ask:
            return None
        si = self.info(symbol)
        if side == "BUY":
            price = si.round_price(ask * (1 + cap_bps / 1e4), "down")
            price = max(price, ask)
        else:
            price = si.round_price(bid * (1 - cap_bps / 1e4), "up")
            price = min(price, bid)
        order = Order(symbol, side, "LIMIT", qty, price=price, tif="IOC", reduce_only=reduce_only)
        await self.adapter.submit(order)
        if not await self._wait(order, 5.0):
            await self.adapter.reconcile(order)
        return order

    # ------------------------------------------------------------------ entry
    async def enter(self, plan: EntryPlan, recheck: Callable[[], EntryPlan | None]) -> ExecResult:
        symbol = plan.symbol
        side = "BUY" if plan.direction > 0 else "SELL"
        book = self.books[symbol]
        bid, _, ask, _ = book.best()
        if not bid or not ask:
            return ExecResult(False, "no_quotes")
        si = self.info(symbol)
        ref_price = bid if side == "BUY" else ask
        qty = si.round_qty(plan.notional / ref_price)
        if qty <= 0 or qty < si.min_qty or qty * ref_price < si.min_notional:
            return ExecResult(False, "qty_below_exchange_minimum")
        orders: list[Order] = []

        if plan.entry_maker and self.x.entry_mode == "maker_first":
            maker = Order(symbol, side, "LIMIT", qty, price=ref_price, tif="GTX")
            orders.append(maker)
            await self.adapter.submit(maker)
            filled_in_time = await self._wait(maker, self.x.maker_ttl_ms / 1000.0)
            if not filled_in_time or maker.status != OrderStatus.FILLED:
                await self._cancel_and_settle(maker)
            if maker.status == OrderStatus.FILLED:
                return self._result(True, "maker_filled", orders, plan)
            filled = maker.filled_qty
            if filled >= qty * self.x.min_partial_fill_ratio and filled * ref_price >= si.min_notional:
                return self._result(True, "maker_partial_accepted", orders, plan)
            # Unfilled (or too small a partial) at TTL.
            if self.x.ttl_fallback != "taker_if_edge":
                return await self._abort(symbol, side, orders, plan, "maker_ttl_skip")
            fresh = recheck()
            if fresh is None:
                return await self._abort(symbol, side, orders, plan, "maker_ttl_no_taker_edge")
            remaining = si.round_qty(qty - filled)
            if remaining <= 0:
                return self._result(True, "maker_partial_accepted", orders, plan)
            taker = await self._taker_ioc(symbol, side, remaining, self.x.taker_max_slippage_bps)
            if taker is not None:
                orders.append(taker)
            res = self._result(True, "taker_fallback_filled", orders, fresh)
            res.used_taker_fallback = True
            if res.qty <= 0:
                return ExecResult(False, "taker_fallback_unfilled", orders=orders, plan=plan)
            return res

        taker = await self._taker_ioc(symbol, side, qty, self.x.taker_max_slippage_bps)
        if taker is None:
            return ExecResult(False, "no_quotes")
        orders.append(taker)
        res = self._result(True, "taker_filled", orders, plan)
        if res.qty <= 0:
            return ExecResult(False, "taker_unfilled", orders=orders, plan=plan)
        return res

    async def _abort(self, symbol: str, side: str, orders: list[Order], plan: EntryPlan, reason: str) -> ExecResult:
        """Skip the trade; unwind any small partial fill so nothing is left behind."""
        qty, _, _, _ = _aggregate(orders)
        if qty > 0:
            close_side = "SELL" if side == "BUY" else "BUY"
            unwind = await self._taker_ioc(symbol, close_side, qty, self.x.exit_max_slippage_bps, reduce_only=True)
            if unwind is not None:
                orders.append(unwind)
            if unwind is None or unwind.filled_qty + 1e-12 < qty:
                # Could not flatten: report as a filled position so it is managed by exits.
                log.error("%s: could not unwind partial entry; managing as position", symbol)
                entry_orders = [o for o in orders if not o.reduce_only]
                res = self._result(True, reason + "_partial_kept", entry_orders, plan)
                res.qty -= unwind.filled_qty if unwind else 0.0
                return res
        return ExecResult(False, reason, orders=orders, plan=plan)

    def _result(self, ok: bool, reason: str, orders: list[Order], plan: EntryPlan) -> ExecResult:
        qty, avg, fee, maker = _aggregate([o for o in orders if not o.reduce_only])
        return ExecResult(ok and qty > 0, reason, qty, avg, fee, maker, orders, plan=plan)

    # ------------------------------------------------------------------ exit
    async def exit(self, symbol: str, direction: int, qty: float, emergency: bool = False) -> ExecResult:
        side = "SELL" if direction > 0 else "BUY"
        si = self.info(symbol)
        orders: list[Order] = []
        remaining = qty
        attempts = 1 if emergency else 3
        for i in range(attempts):
            cap = self.x.exit_max_slippage_bps * (i + 1)
            o = await self._taker_ioc(symbol, side, si.round_qty(remaining) or remaining, cap, reduce_only=True)
            if o is None:
                await asyncio.sleep(0.2)
                continue
            orders.append(o)
            remaining = qty - sum(x.filled_qty for x in orders)
            if remaining <= max(si.step_size * 0.5, 1e-12):
                break
        remaining = qty - sum(x.filled_qty for x in orders)
        if remaining > max(si.step_size * 0.5, 1e-12) and self.x.emergency_market_exit:
            mkt = Order(symbol, side, "MARKET", si.round_qty(remaining) or remaining, reduce_only=True)
            orders.append(mkt)
            await self.adapter.submit(mkt)
            if not await self._wait(mkt, 5.0):
                await self.adapter.reconcile(mkt)
        filled, avg, fee, maker = _aggregate(orders)
        ok = filled >= qty - max(si.step_size * 0.5, 1e-12)
        return ExecResult(ok, "closed" if ok else "exit_incomplete", filled, avg, fee, maker, orders)
