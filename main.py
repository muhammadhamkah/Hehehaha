"""Binance USDT-M microstructure scalping bot — orchestrator.

Modes
  record  Phases 1-5: scanner + microstructure features + signals + forward labels.
          No orders, no simulated trades. Builds the research dataset.
  paper   Phases 6-7: everything in ``record`` plus simulated execution (latency,
          queue position, partial fills, fees, slippage). DRY RUN.
  live    Phase 8: real orders. Requires dry_run=False, API keys and
          LIVE_TRADING_CONFIRM=I_UNDERSTAND_THE_RISK. Otherwise falls back to paper.

Pipeline
  universe -> liquidity/spread filter -> rank -> detailed microstructure
  -> directional probability -> expected move -> fees/slippage -> expected net PnL
  -> risk check -> execute or skip
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import signal
import sys
import uuid

from analytics.performance import compute_performance, format_report
from analytics.trade_logger import TradeLogger, TradeRecord
from config import BotConfig, load_config
from data.database import Database
from data.recorder import Recorder
from exchange.binance_client import BinanceFuturesClient
from exchange.execution import ExchangeAdapter, LiveExchange, PaperExchange, TradeExecutor
from exchange.models import Order, SymbolInfo
from exchange.schemas import FeedValidator
from exchange.websocket_manager import StreamConnection, UserDataStream
from market_data.orderbook import OrderBook, SyncStatus
from market_data.scanner import MarketScanner
from market_data.tradeflow import Trade, TradeFlow
from risk.risk_manager import RiskManager
from strategy.costs import CostModel
from strategy.entry_filter import EntryFilter
from strategy.exit_engine import ExitEngine, Position
from strategy.predictor import build_predictor
from strategy.signal_engine import SignalEngine, SignalResult
from utils.clock import SYSTEM_CLOCK, Clock

log = logging.getLogger("bot")


class OfflineStreams:
    """Stand-in for a StreamConnection during replay: tracks the subscribed stream set so
    the replay driver delivers only events the live bot would actually have received."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.streams: set[str] = set()

    async def set_streams(self, streams) -> None:
        self.streams = set(streams)

    async def run(self) -> None:
        return None

    async def stop(self) -> None:
        return None


class TradingBot:
    def __init__(self, cfg: BotConfig, clock: Clock = SYSTEM_CLOCK, offline: bool = False,
                 event_writer=None, rng_seed: int | None = None) -> None:
        """offline=True: no REST/WebSocket I/O; market events are pushed in by a replay
        driver through the same handlers. Live trading is never enabled offline."""
        self.cfg = cfg
        self.clock = clock
        self.offline = offline
        self.event_writer = event_writer
        self.live = cfg.live_trading_enabled() and not offline
        self.trading = cfg.trading_enabled()
        self.mode_label = "live" if self.live else "paper"

        self.risk = RiskManager(cfg, clock)
        self.client = BinanceFuturesClient(cfg.exchange, allow_trading=self.live, on_error=self.risk.on_api_error)
        self.scanner = MarketScanner(cfg.scanner)
        self.books: dict[str, OrderBook] = {}
        self.flows: dict[str, TradeFlow] = {}
        self.symbol_info: dict[str, SymbolInfo] = {}

        self.db = Database(cfg.recorder.db_path, cfg.recorder.flush_interval_s)
        import random as _random
        self.recorder = Recorder(cfg, self.db, _random.Random(rng_seed) if rng_seed is not None else None)
        self.feed = FeedValidator()
        self.trade_logger = TradeLogger(self.db, cfg.recorder.trades_jsonl)

        self.costs = CostModel(cfg.costs)
        self.predictor = build_predictor(cfg.strategy)
        self.entry_filter = EntryFilter(cfg, self.costs)
        self.signals = SignalEngine(cfg, self.predictor, self.entry_filter, self.recorder, self.risk)
        self.exit_engine = ExitEngine(cfg, self.costs.maker_fee, self.costs.taker_fee)

        self.adapter: ExchangeAdapter
        if self.live:
            self.adapter = LiveExchange(self.client, clock, on_order=self._on_order)
        else:
            self.adapter = PaperExchange(cfg, self.books, self.costs, clock, on_order=self._on_order)
        self.executor = TradeExecutor(cfg, self.adapter, self.books, self.symbol_info, clock)

        self.market_ws: StreamConnection | OfflineStreams
        self.detail_ws: StreamConnection | OfflineStreams
        if offline:
            self.market_ws, self.detail_ws = OfflineStreams("market"), OfflineStreams("detail")
        else:
            self._make_ws()
        self.user_ws: UserDataStream | None = None

        self.positions: dict[str, Position] = {}
        self._entry_lock = asyncio.Lock()
        self._entering: set[str] = set()
        self._leverage_set: set[str] = set()
        self._resyncing: set[str] = set()
        self._tasks: list[asyncio.Task] = []
        self._stop = asyncio.Event()
        self._last_scanner_record = 0
        self.n_evals = 0
        self.n_entries = 0
        # Replay warm-up: market state is built from earlier events, but no trades are
        # opened and no signals recorded before this timestamp.
        self.active_from_ms = 0
        self.active_until_ms = float("inf")   # replay segment end: no evaluations at/after it

    def _make_ws(self) -> None:
        cfg = self.cfg
        self.market_ws = StreamConnection(
            "market", cfg.exchange.ws_url, self._on_market_msg,
            on_connect=self.risk.heartbeat, on_disconnect=self.risk.on_disconnect,
            max_backoff_s=cfg.exchange.ws_reconnect_max_backoff_s,
            max_age_s=cfg.exchange.ws_max_connection_age_s,
            max_streams=cfg.exchange.ws_max_streams_per_connection,
            silence_timeout_s=cfg.exchange.ws_silence_timeout_s,
        )
        self.detail_ws = StreamConnection(
            "detail", cfg.exchange.ws_url, self._on_detail_msg,
            on_connect=self.risk.heartbeat, on_disconnect=self.risk.on_disconnect,
            max_backoff_s=cfg.exchange.ws_reconnect_max_backoff_s,
            max_age_s=cfg.exchange.ws_max_connection_age_s,
            max_streams=cfg.exchange.ws_max_streams_per_connection,
            silence_timeout_s=cfg.exchange.ws_silence_timeout_s,
        )

    # ================================================================== startup
    async def start(self) -> None:
        problems = self.cfg.validate()
        if problems:
            raise SystemExit("config problems:\n  " + "\n  ".join(problems))
        log.info("mode=%s dry_run=%s live_orders=%s trading=%s", self.cfg.mode, self.cfg.dry_run,
                 self.live, self.trading)
        if self.cfg.mode == "live" and not self.live:
            log.warning("live mode requested but live trading NOT enabled -> running as PAPER (dry run)")
        await self.client.start()
        try:
            await self.client.sync_time()
        except Exception as exc:  # noqa: BLE001
            log.warning("time sync failed: %s", exc)
        self.symbol_info.update(await self.client.perpetual_symbols(self.cfg.scanner.quote_asset))
        self.scanner.set_universe(set(self.symbol_info))
        log.info("universe: %d USDT-M perpetuals", len(self.symbol_info))
        if self.event_writer is not None:
            from dataclasses import asdict
            self.event_writer.write_meta({
                "symbol_info": {k: asdict(v) for k, v in self.symbol_info.items()},
                "depth_mode": self.cfg.market_data.depth_mode,
                "depth_levels": self.cfg.market_data.depth_levels,
                "source": "binance_ws_live",
            })
            log.info("recording raw events to %s", self.cfg.recorder.events_dir)

        if self.live:
            await self._live_preflight()

        await self.market_ws.set_streams(["!ticker@arr", "!bookTicker"])
        self._spawn(self.market_ws.run(), "market_ws")
        self._spawn(self.detail_ws.run(), "detail_ws")
        if self.user_ws is not None:
            self._spawn(self.user_ws.run(), "user_ws")
        await self.start_core()

    async def start_core(self, symbol_info: dict[str, SymbolInfo] | None = None) -> None:
        """Start the strategy loops. Used directly by the offline replay driver."""
        if symbol_info is not None:
            self.symbol_info.update(symbol_info)
            self.scanner.set_universe(set(self.symbol_info))
        if self.offline:
            await self.market_ws.set_streams(["!ticker@arr", "!bookTicker"])
        self._spawn(self._scanner_loop(), "scanner")
        self._spawn(self._eval_loop(), "eval")
        self._spawn(self._position_loop(), "positions")
        self._spawn(self._risk_loop(), "risk")
        self._spawn(self._housekeeping_loop(), "housekeeping")

    async def _live_preflight(self) -> None:
        positions = await self.client.position_risk()
        open_pos = [p for p in positions if abs(float(p.get("positionAmt", 0))) > 0]
        if open_pos:
            raise SystemExit(f"refusing to start: account has open positions {[p['symbol'] for p in open_pos]}")
        assert isinstance(self.adapter, LiveExchange)
        self.user_ws = UserDataStream(
            self.cfg.exchange.ws_url, self.client.new_listen_key, self.client.keepalive_listen_key,
            self.adapter.on_user_event, on_disconnect=self.risk.on_disconnect, on_connect=self.risk.heartbeat,
        )
        log.warning("LIVE TRADING ENABLED: notional=%.2f leverage=%d max_risk/trade=%.2f",
                    self.cfg.sizing.position_notional_usdt, self.cfg.sizing.leverage,
                    self.cfg.sizing.max_risk_per_trade_usdt)

    def _spawn(self, coro, name: str) -> None:
        task = asyncio.ensure_future(coro)
        task.set_name(name)
        task.add_done_callback(self._task_done)
        self._tasks.append(task)

    def _task_done(self, task: asyncio.Task) -> None:
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            log.error("task %s crashed: %r", task.get_name(), exc, exc_info=exc)
            self.risk.halt(f"task_crash:{task.get_name()}")

    # ================================================================== market data
    def _on_market_msg(self, stream: str, data, ts: int) -> None:
        if self.event_writer is not None:
            self.event_writer.write("market", stream, data, ts)
        self.risk.heartbeat("market", ts)
        if stream == "!ticker@arr":
            for t in self.feed.ticker_arr(stream, data, ts):
                self.scanner.on_ticker(t, ts)
        elif stream == "!bookTicker":
            bt = self.feed.book_ticker(stream, data, ts)
            if bt is not None:
                self.scanner.on_book_ticker(data, ts)
        else:
            self.feed.report_unexpected(stream, data)

    def _on_detail_msg(self, stream: str, data, ts: int) -> None:
        if self.event_writer is not None:
            self.event_writer.write("detail", stream, data, ts)
        self.risk.heartbeat("detail", ts)
        sym_lower, _, kind = stream.partition("@")
        symbol = sym_lower.upper()
        if sym_lower == "__snapshot__":
            self._apply_recorded_snapshot(kind, data, ts)
            return
        book = self.books.get(symbol)
        flow = self.flows.get(symbol)
        if book is None or flow is None:
            return  # late message for a symbol we just unsubscribed from
        mode = self.cfg.market_data.depth_mode
        if kind == "aggTrade":
            pt = self.feed.agg_trade(stream, data, ts, symbol)
            if pt is None:
                return
            t = Trade(pt.trade_ts, pt.price, pt.qty, pt.is_buyer_maker)
            flow.add(t, ts, pt.agg_id)
            if isinstance(self.adapter, PaperExchange):
                self.adapter.on_trade(symbol, t)
            return
        if kind.startswith("depth"):
            pd_ = self.feed.depth(stream, data, ts, symbol, partial=(mode == "partial"))
            if pd_ is None:
                return
            if mode == "partial":
                book.apply_snapshot(pd_.bids, pd_.asks, pd_.final_id, pd_.event_ts, ts)
            elif mode == "diff":
                status = book.apply_diff(data, ts)
                if status == SyncStatus.RESYNC or status == SyncStatus.BUFFERING:
                    if status == SyncStatus.RESYNC:
                        book.last_update_id = 0
                        book.buffer_diff(data)
                    self._schedule_resync(symbol)
            else:
                self.feed.report_unexpected(stream, "depth message in bbo mode")
                return
        elif kind == "bookTicker":
            bt = self.feed.book_ticker(stream, data, ts, symbol)
            if bt is None:
                return
            if mode == "bbo":
                # L1-only data (e.g. Binance public archive): top of book IS the book.
                book.apply_snapshot([(bt.bid, bt.bid_qty)], [(bt.ask, bt.ask_qty)], bt.update_id, bt.event_ts, ts)
            else:
                book.update_bbo(bt.bid, bt.bid_qty, bt.ask, bt.ask_qty, ts)
        else:
            self.feed.report_unexpected(stream, data)
            return
        bid, _, ask, _ = book.best()
        self.recorder.on_quote(symbol, ts, bid, ask)
        if isinstance(self.adapter, PaperExchange):
            self.adapter.on_book(symbol)

    def _apply_recorded_snapshot(self, symbol: str, snap: dict, ts: int) -> None:
        book = self.books.get(symbol)
        if book is None:
            return
        status = book.sync_from_snapshot(
            [(float(p), float(q)) for p, q in snap["bids"]],
            [(float(p), float(q)) for p, q in snap["asks"]],
            int(snap["lastUpdateId"]), ts,
        )
        self._resyncing.discard(symbol)
        log.info("%s diff book resync (recorded snapshot): %s", symbol, status)

    def _schedule_resync(self, symbol: str) -> None:
        if symbol in self._resyncing:
            return
        self._resyncing.add(symbol)
        if self.offline:
            return  # replay: the recorded REST snapshot arrives as a "__snapshot__" event

        async def resync() -> None:
            try:
                await asyncio.sleep(0.5)  # let a few diffs buffer first
                snap = await self.client.depth(symbol, self.cfg.market_data.diff_snapshot_limit)
                now = self.clock.now_ms()
                if self.event_writer is not None:
                    self.event_writer.write("detail", f"__snapshot__@{symbol}", snap, now)
                self._apply_recorded_snapshot(symbol, snap, now)
            except Exception as exc:  # noqa: BLE001
                log.warning("%s resync failed: %s", symbol, exc)
            finally:
                self._resyncing.discard(symbol)

        asyncio.ensure_future(resync())

    def _detail_streams(self, symbols: list[str]) -> list[str]:
        mode = self.cfg.market_data.depth_mode
        depth = {"partial": f"@depth{self.cfg.market_data.depth_levels}@100ms", "diff": "@depth@100ms"}.get(mode)
        out = []
        for s in symbols:
            sl = s.lower()
            out += [f"{sl}@aggTrade", f"{sl}@bookTicker"]
            if depth:
                out.append(f"{sl}{depth}")
        return out

    # ================================================================== loops
    async def _scanner_loop(self) -> None:
        while not self._stop.is_set():
            now = self.clock.now_ms()
            for s in self.recorder.pending_symbols():
                self.scanner.pin(s, "labels")
            for s in list(self.scanner.pinned):
                if s not in self.recorder.pending and "labels" in self.scanner.pinned.get(s, set()):
                    self.scanner.unpin(s, "labels")
            active = self.scanner.select(now)
            for s in active:
                if s not in self.books:
                    self.books[s] = OrderBook(s, self.cfg.market_data.book_history_len)
                    self.flows[s] = TradeFlow(s, self.cfg.market_data.trade_history_s)
            for s in list(self.books):
                if s not in active and s not in self.positions:
                    del self.books[s]
                    del self.flows[s]
            await self.detail_ws.set_streams(self._detail_streams(active))
            if now - self._last_scanner_record > 30_000 and self.scanner.last_ranking:
                self.recorder.record_scanner(now, self.scanner.last_ranking)
                self._last_scanner_record = now
                log.info("shortlist: %s", ", ".join(self.scanner.selected[:30]))
            await asyncio.sleep(self.cfg.scanner.rescan_interval_s)

    async def _eval_loop(self) -> None:
        interval = self.cfg.strategy.eval_interval_ms / 1000.0
        stale = self.cfg.market_data.stale_after_ms
        while not self._stop.is_set():
            now = self.clock.now_ms()
            for symbol in list(self.scanner.selected):
                book, flow = self.books.get(symbol), self.flows.get(symbol)
                if book is None or flow is None or book.is_stale(now, stale) or len(book.history) < 20:
                    continue
                if symbol in self.positions or symbol in self._entering:
                    continue
                self.n_evals += 1
                active = self.active_from_ms <= now < self.active_until_ms
                res = self.signals.evaluate(symbol, book, flow, now, trading_enabled=self.trading and active,
                                            record=active)
                if res.action == "enter" and not self._entering and \
                        len(self.positions) < self.cfg.sizing.max_simultaneous_positions:
                    self._entering.add(symbol)
                    asyncio.ensure_future(self._enter(res))
                elif res.action == "would_enter" and active:
                    log.info("WOULD ENTER %s dir=%+d conf=%.3f exp_net=%.4f", symbol,
                             res.decision.plan.direction, res.decision.plan.confidence,
                             res.decision.plan.expected_net_usdt)
            await asyncio.sleep(interval)

    async def _position_loop(self) -> None:
        while not self._stop.is_set():
            now = self.clock.now_ms()
            emergency = self.risk.emergency_exit_required()
            for symbol, pos in list(self.positions.items()):
                if pos.closing:
                    continue
                if emergency:
                    self._start_exit(pos, f"emergency_{self.risk.halt_reason}", True)
                    continue
                if pos.exit_reason:
                    # A previous exit attempt failed or was incomplete: retry aggressively.
                    self._start_exit(pos, pos.exit_reason, True)
                    continue
                book, flow = self.books.get(symbol), self.flows.get(symbol)
                if book is None or flow is None:
                    self._start_exit(pos, "emergency_no_data", True)
                    continue
                f = self.signals.features(book, flow, now)
                pred = self.predictor.predict(f) if f else None
                bid, _, ask, _ = book.best()
                sig = self.exit_engine.update(pos, bid, ask, f, pred, now)
                if sig is not None:
                    self._start_exit(pos, sig.reason, sig.emergency)
            await asyncio.sleep(0.1)

    async def _risk_loop(self) -> None:
        while not self._stop.is_set():
            self.risk.periodic_check()
            await asyncio.sleep(0.5)

    async def _housekeeping_loop(self) -> None:
        last_status = 0
        while not self._stop.is_set():
            now = self.clock.now_ms()
            self.recorder.sweep(now)
            if self.cfg.recorder.record_raw_book:
                for s in self.scanner.selected:
                    if s in self.books and self.books[s].synced:
                        self.recorder.record_book(now, s, self.books[s], self.flows[s].last_price())
            if now - last_status > 30_000:
                last_status = now
                top_rej = sorted(self.signals.rejections.items(), key=lambda kv: -kv[1])[:6]
                fs = self.feed.summary()
                log.info("status evals=%d recorded=%d labeled=%d entries=%d open=%d risk=%s rejections=%s "
                         "malformed=%s unexpected=%d",
                         self.n_evals, self.recorder.n_recorded, self.recorder.n_labeled, self.n_entries,
                         len(self.positions), json.dumps(self.risk.status()), top_rej,
                         fs["malformed_by_kind"], sum(fs["unexpected"].values()))
            await asyncio.sleep(max(self.cfg.recorder.raw_book_interval_ms / 1000.0, 1.0))

    # ================================================================== trading
    async def _ensure_leverage(self, symbol: str) -> None:
        if not self.live or symbol in self._leverage_set:
            return
        await self.client.set_margin_type(symbol, self.cfg.sizing.margin_type)
        await self.client.set_leverage(symbol, self.cfg.sizing.leverage)
        if self.cfg.costs.use_exchange_commission:
            maker, taker = await self.client.commission_rate(symbol)
            self.costs.set_commission(maker, taker)
            self.exit_engine.maker_fee, self.exit_engine.taker_fee = maker, taker
        self._leverage_set.add(symbol)

    async def _enter(self, res: SignalResult) -> None:
        symbol = res.symbol
        plan = res.decision.plan
        assert plan is not None
        try:
            async with self._entry_lock:
                if symbol in self.positions or self.risk.halted:
                    return
                await self._ensure_leverage(symbol)
                book, flow = self.books[symbol], self.flows[symbol]

                def recheck():
                    if self.risk.halted:
                        return None
                    fresh = self.signals.recheck_taker(symbol, book, flow, self.clock.now_ms())
                    if fresh is None or fresh.direction != plan.direction:
                        return None
                    return fresh

                result = await self.executor.enter(plan, recheck)
                if not result.success:
                    log.info("entry skipped %s: %s", symbol, result.reason)
                    if res.signal_id:
                        self.db.update("signals", "id", res.signal_id, {"decision": f"entry_failed:{result.reason}"})
                    return
                used_plan = result.plan or plan
                entry_slip_bps = plan.direction * (result.avg_price - plan.ref_mid) / plan.ref_mid * 1e4
                pos = Position(
                    symbol=symbol, direction=plan.direction, qty=result.qty, entry_price=result.avg_price,
                    entry_ts_ms=self.clock.now_ms(), plan=used_plan, entry_fee=result.fee,
                    entry_maker=result.maker, entry_slippage_bps=entry_slip_bps,
                    entry_features=res.features, entry_trades_per_s=res.features.get("trades_per_s_3s", 0.0),
                    signal_id=res.signal_id,
                )
                self.exit_engine.init_position(pos)
                self.positions[symbol] = pos
                self.scanner.pin(symbol, "position")
                self.risk.on_trade_opened(symbol, pos.notional)
                self.n_entries += 1
                if res.signal_id:
                    self.db.update("signals", "id", res.signal_id, {"decision": f"entered:{result.reason}"})
                log.info("ENTER %s %s qty=%.6g @ %.8g (%s) target=%.1fbps stop=%.1fbps exp_net=%.4f",
                         symbol, "LONG" if pos.direction > 0 else "SHORT", pos.qty, pos.entry_price,
                         result.reason, used_plan.target_bps, used_plan.stop_bps, used_plan.expected_net_usdt)
        except Exception:  # noqa: BLE001
            log.exception("entry failed for %s", symbol)
            self.risk.on_api_error(f"entry exception {symbol}")
        finally:
            self._entering.discard(symbol)

    def _start_exit(self, pos: Position, reason: str, emergency: bool) -> None:
        pos.closing = True
        if not pos.exit_reason:
            pos.exit_reason = reason
        book = self.books.get(pos.symbol)
        if book is not None and not pos.exit_ref_mid:
            pos.exit_ref_mid = book.mid
        asyncio.ensure_future(self._exit(pos, emergency))

    async def _exit(self, pos: Position, emergency: bool) -> None:
        try:
            res = await self.executor.exit(pos.symbol, pos.direction, pos.qty, emergency)
        except Exception:  # noqa: BLE001
            log.exception("exit failed for %s", pos.symbol)
            self.risk.on_api_error(f"exit exception {pos.symbol}")
            pos.closing = False
            return
        pos.exit_qty += res.qty
        pos.exit_notional += res.avg_price * res.qty
        pos.exit_fee += res.fee
        pos.exit_maker_qty += res.maker_qty
        pos.qty = max(pos.orig_qty - pos.exit_qty, 0.0)
        si = self.executor.info(pos.symbol)
        if pos.qty > max(si.step_size * 0.5, 1e-12):
            log.error("%s exit incomplete, %.6g remaining; will retry as emergency", pos.symbol, pos.qty)
            pos.closing = False
            pos.exit_reason = pos.exit_reason or "exit_incomplete"
            return
        self._finalize_trade(pos)

    def _finalize_trade(self, pos: Position) -> None:
        now = self.clock.now_ms()
        qty = pos.orig_qty
        exit_price = pos.exit_notional / pos.exit_qty if pos.exit_qty else pos.last_mark
        gross = pos.direction * (exit_price - pos.entry_price) * qty
        net = gross - pos.entry_fee - pos.exit_fee
        ref_mid = pos.plan.ref_mid
        entry_slip_usdt = pos.direction * (pos.entry_price - ref_mid) * qty
        exit_ref = pos.exit_ref_mid or exit_price
        exit_slip_usdt = pos.direction * (exit_ref - exit_price) * qty
        exit_slip_bps = exit_slip_usdt / (exit_ref * qty) * 1e4 if exit_ref and qty else 0.0
        notional = pos.notional
        scale = notional / pos.plan.notional if pos.plan.notional else 1.0
        rec = TradeRecord(
            trade_id=uuid.uuid4().hex[:16], mode=self.mode_label, symbol=pos.symbol, direction=pos.direction,
            entry_ts_ms=pos.entry_ts_ms, exit_ts_ms=now, holding_s=(now - pos.entry_ts_ms) / 1000.0,
            signal_confidence=pos.plan.confidence, signal_score=pos.plan.score, signal_id=pos.signal_id,
            entry_price=pos.entry_price, exit_price=exit_price, qty=qty, notional=notional,
            gross_pnl=gross, entry_fee=pos.entry_fee, exit_fee=pos.exit_fee,
            est_slippage_usdt=pos.plan.costs.slippage_usdt * scale,
            actual_slippage_usdt=entry_slip_usdt + exit_slip_usdt,
            entry_slippage_bps=pos.entry_slippage_bps, exit_slippage_bps=exit_slip_bps,
            net_pnl=net, exit_reason=pos.exit_reason, mfe_bps=pos.mfe_bps, mae_bps=pos.mae_bps,
            mfe_usdt=pos.mfe_bps / 1e4 * notional, mae_usdt=pos.mae_bps / 1e4 * notional,
            entry_maker=pos.entry_maker, exit_maker=pos.exit_maker_qty >= 0.5 * qty,
            expected_net_usdt=pos.plan.expected_net_usdt, target_bps=pos.plan.target_bps,
            stop_bps=pos.plan.stop_bps, features=pos.entry_features,
        )
        self.trade_logger.log(rec)
        self.risk.on_trade_closed(pos.symbol, net)
        self.positions.pop(pos.symbol, None)
        self.scanner.unpin(pos.symbol, "position")

    def _on_order(self, order: Order, event: str) -> None:
        self.db.insert("orders", {
            "ts_ms": self.clock.now_ms(), "client_id": order.client_id, "exchange_id": order.exchange_id,
            "symbol": order.symbol, "side": order.side, "type": order.type, "tif": order.tif,
            "price": order.price, "qty": order.qty, "status": order.status.value,
            "filled_qty": order.filled_qty, "avg_price": order.avg_price, "fee": order.fee,
            "maker": int(order.maker_qty > 0), "event": event, "detail": order.reject_reason,
        })

    # ================================================================== shutdown
    async def run(self, duration_s: float | None = None) -> None:
        await self.start()
        try:
            if duration_s:
                try:
                    await asyncio.wait_for(self._stop.wait(), duration_s)
                except asyncio.TimeoutError:
                    pass
            else:
                await self._stop.wait()
        finally:
            await self.shutdown()

    def request_stop(self) -> None:
        self._stop.set()

    async def shutdown(self) -> None:
        log.info("shutting down: flattening %d open position(s)", len(self.positions))
        self._stop.set()
        for _ in range(3):
            if not self.positions:
                break
            await asyncio.gather(*(self._exit_now(p) for p in list(self.positions.values())),
                                 return_exceptions=True)
        if self.positions:
            log.critical("POSITIONS STILL OPEN AT SHUTDOWN: %s", list(self.positions))
        await self.market_ws.stop()
        await self.detail_ws.stop()
        if self.user_ws:
            self.user_ws.stop()
        for t in self._tasks:
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self.recorder.flush_all()
        if self.event_writer is not None:
            self.event_writer.close()
        log.info("feed health: %s", json.dumps(self.feed.summary()["by_kind"]))
        self.db.flush()
        trades = self.db.query("SELECT * FROM trades WHERE mode = ?", (self.mode_label,))
        if trades:
            log.info("\n%s", format_report(compute_performance(trades)))
        await self.client.close()
        self.db.close()

    async def _exit_now(self, pos: Position) -> None:
        if not pos.exit_reason:
            pos.exit_reason = "shutdown"
        book = self.books.get(pos.symbol)
        if book is not None and not pos.exit_ref_mid:
            pos.exit_ref_mid = book.mid
        pos.closing = True
        await self._exit(pos, emergency=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Binance USDT-M microstructure scalper")
    ap.add_argument("--config", help="JSON config overrides")
    ap.add_argument("--mode", choices=["record", "paper", "live"], help="override mode")
    ap.add_argument("--duration", type=float, default=None, help="run for N seconds then stop")
    ap.add_argument("--top-n", type=int, default=None, help="override scanner.top_n")
    ap.add_argument("--log-level", default=None)
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    cfg = load_config(args.config)
    if args.mode:
        cfg.mode = args.mode
    if args.top_n:
        cfg.scanner.top_n = args.top_n
    logging.basicConfig(
        level=getattr(logging, (args.log_level or cfg.log_level).upper()),
        format="%(asctime)s.%(msecs)03d %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )
    writer = None
    if cfg.recorder.record_events:
        from data.event_store import EventWriter
        writer = EventWriter(cfg.recorder.events_dir)
    bot = TradingBot(cfg, event_writer=writer)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, bot.request_stop)
        except NotImplementedError:  # Windows
            pass
    try:
        loop.run_until_complete(bot.run(args.duration))
    finally:
        loop.close()


if __name__ == "__main__":
    main()
