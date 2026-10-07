# Binance USDT-M Microstructure Scalper

A short-horizon research and trading system for Binance USDT-M perpetual futures. Its edge
is meant to come from **real-time market microstructure**: order-book shape and dynamics,
plus executed aggressor flow. It does **not** use candle indicators.

The system only enters when the **expected NET profit after fees and slippage is at least
0.10 USDT**. It would rather skip a trade than take a marginal one.

> ⚠️ **Status:** Phases 1–7 are implemented, along with an event-driven tick replay
> backtester and a train/validation/test research protocol. All of it is tested offline:
> against mock Binance servers and synthetic data. **None of it has run against real
> Binance data yet**, because the build sandbox cannot reach Binance. Follow
> [`docs/VALIDATION_RUNBOOK.md`](docs/VALIDATION_RUNBOOK.md) on a machine that can. Live
> trading stays disabled. The rule-based weights are **uncalibrated placeholders**.

## Decision pipeline

```
Binance USDT-M universe (exchangeInfo: TRADING, PERPETUAL, USDT)
 → liquidity / spread filter               market_data/scanner.py   (!ticker@arr, !bookTicker)
 → rank symbols, shortlist top 10–30       (percentile ranks + hysteresis)
 → detailed microstructure analysis        market_data/orderbook.py, tradeflow.py, features/*
 → directional probability                 strategy/predictor.py    (rule-based → ML-ready)
 → expected move & target-before-stop      strategy/barrier.py      (finite-horizon barrier model)
 → expected fees / slippage                strategy/costs.py        (book-walk + fees + buffers)
 → expected NET PnL ≥ 0.10 USDT gate       strategy/entry_filter.py
 → risk check                              risk/risk_manager.py
 → execute (maker-first) or skip           exchange/execution.py
```

### The net-profit gate

```
Expected Net Profit = notional × E[exit value in bps]      # probability-weighted (barrier model)
                    − entry fee − exit fee
                    − entry slippage − exit slippage       # book walk + latency + safety buffer
```

To enter, a trade must pass all of the following:
* `net_if_target ≥ 0.10`: the profit target itself must clear all costs.
* `expected_net ≥ 0.10`: the probability-weighted expectation must too.
* `P(target before stop) ≥ min_p_target_before_stop`.

The target is picked from a set of candidates. The chosen one maximises expected net while
still satisfying the probability constraint. Exits are always costed as **taker**, which is
the worst case. A maker entry is charged an adverse-selection allowance rather than credited
with half the spread.

## Replay backtester and validation

Recorded Binance events are replayed through the **same** `TradingBot`: the same order
book, trade flow, features, predictor, entry filter, risk manager, exit engine, and
paper execution simulator. There is no separate backtest strategy.

```
recorded events → chronological queue (EventReader, exchange or local timestamps)
  → virtual-time asyncio loop (max-speed deterministic, 1×, or N×)
  → bot WebSocket handlers (only streams the bot was subscribed to at that moment)
  → signal engine → PaperExchange (latency, queue position, partial fills, TTL, fees)
  → exits → trade log → performance / decisions / calibration / features / min-edge reports
```

* **No look-ahead.** A test changes every event after time T and asserts that every
  signal, order, and entry before T is identical. Messages that arrived abnormally late
  are replayed at their receive time.
* **Deterministic.** The same input always produces the same trades.
* **Latency is configurable.** Order acknowledgement lags submission by exactly the
  configured latency (tested at 20 and 250 ms); the protocol re-runs at 20/50/100/250/500 ms.
* **Research protocol (`research.validate`).** Chronological 60/20/20 split. Sweep on
  TRAIN, select on VALIDATION, then evaluate **once** on TEST (enforced by
  `test_lock.json`). The primary metric is expected net PnL per trade after all costs.

```bash
python -m tools.validate_binance                     # live connectivity + parser validation (read-only)
python main.py --mode record                         # records raw events to data/events/
python -m backtest.replay --events data/events --out runs/r1 --latency-ms 100
python -m research.validate --events data/events --out runs/v1 --workers 4   # TEST untouched
python -m research.validate --events data/events --out runs/v1 --final-test  # once, frozen config
```

## Project layout

```
config.py                  all tunables (dataclasses), JSON/env overrides, live-trading gate
main.py                    orchestrator (asyncio): WS → books → signals → execution → logs
exchange/
  binance_client.py        async REST (signed endpoints, weight/backoff handling, -1021 resync)
  websocket_manager.py     combined streams, dynamic SUBSCRIBE (ack/error tracking), reconnect/backoff,
                           silence watchdog, stream cap, 24h recycle, user-data stream
  schemas.py               strict payload validation + feed health (id gaps, pu continuity, latency)
  execution.py             PaperExchange (simulator), LiveExchange, TradeExecutor (entry/exit protocol)
  models.py                Order / Fill / SymbolInfo (tick/step rounding)
market_data/
  scanner.py               stage-1 market-wide ranking, hysteresis, symbol pinning
  orderbook.py             L2 book: partial-depth snapshots or diff-depth with Binance sequencing
  tradeflow.py             rolling aggTrade windows
features/
  orderbook_features.py    multi-level/decayed imbalance, microprice, OFI, depletion,
                           replenishment, persistence, liquidity change, momentum, realised vol
  trade_features.py        aggressor imbalance, trade velocity, volume acceleration, large trades
  microstructure_features.py  flat feature dict (what the model, recorder and ML read)
strategy/
  predictor.py             Prediction interface, RuleBasedPredictor, LinearModelPredictor
  barrier.py               P(target)/P(stop)/P(timeout), E[value], E[hold] on a trinomial lattice
  costs.py                 fee + slippage model
  entry_filter.py          all entry conditions + sizing + net-profit gate
  exit_engine.py           stop / break-even / trailing / target / reversal / exhaustion / time / emergency
  signal_engine.py         features → prediction → gate → risk → record
risk/risk_manager.py       daily loss, consecutive losses, trades/hour, cooldowns, limits, kill switch
data/
  database.py              SQLite (WAL) with background writer thread
  event_store.py           raw event writer/reader for replay (gzip JSONL, exchange-time ordering)
  recorder.py              signal recorder + forward labeller (1/3/5/10/30/60 s, TP-before-SL, MFE/MAE)
analytics/
  trade_logger.py          per-trade record (SQLite + JSONL)
  performance.py           stats + "profitable after costs?" verdict (CLI)
backtest/
  virtual_loop.py          virtual-time asyncio event loop (deterministic simulated clock)
  replay.py                event-driven replay through the real TradingBot (CLI)
  synthetic.py             Binance-shaped synthetic events (pipeline tests ONLY)
  report.py                replay summary formatting
tools/
  validate_binance.py      read-only live connectivity / parser validation
  binance_archive.py       data.binance.vision aggTrades + bookTicker → replayable events (L1)
research/
  validate.py              train/validation/test protocol, sweep, latency robustness, test lock
  analysis.py              per-symbol, calibration, feature IC analysis, minimum tradable edge
  report.py                VALIDATION_REPORT.md writer
  analyze_signals.py       Phase 5: edge by signal bucket, in-sample vs out-of-sample, IC, calibration
  train_model.py           logistic model → models/linear_model.json (plug into predictor)
docs/VALIDATION_RUNBOOK.md step-by-step validation on real data
tests/                     unit, mock-Binance, replay determinism / look-ahead / latency tests
```

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest
```

Requires Python 3.11 or newer (developed on 3.13).

## Development sequence: how to use it

| Phase | What | Command |
|---|---|---|
| 1–4 | Ingestion, scanner, features, signals + forward labels | `python main.py --mode record --config configs/example_record.json` |
| 5 | Does an edge exist **after costs**, out of sample? | `python -m research.analyze_signals --db data/microstructure.sqlite` |
| 5b | Optional: train a model | `python -m research.train_model --horizon 10` then set `strategy.predictor="linear"` |
| 6–7 | Paper trading / dry run with the realistic simulator | `python main.py --mode paper --config configs/example_paper.json` |
| — | Performance report | `python -m analytics.performance --mode paper` |
| 8 | Tiny live positions, **only after positive out-of-sample results** | see "Enabling live trading" below |

Add `--duration 3600` to run for a fixed time. `Ctrl-C` shuts down cleanly: open positions
are flattened, pending labels are flushed, and a report is printed.

### Reading the Phase 5 output

`analyze_signals` groups signals by |score| and by confidence. For each bucket it reports:
* the direction-adjusted mid return at each horizon;
* the realised target-before-stop rate (labels use *executable* prices, so the spread is included);
* `net_bps` / `net_usdt`: the barrier outcome minus a conservative cost estimate;
* a t-statistic.

Treat a bucket as a real edge only if all three hold:
1. `net_usdt` ≥ the 0.10 threshold;
2. t > 2;
3. it holds up **out of sample**.

A high hit rate with negative `net_usdt` is the common failure mode. During development, a
synthetic feed with a 70% directional hit rate still came out net-negative after about 8.8 bps
of round-trip costs.

## Execution behaviour

* **Entry:** post-only (GTX) limit order at the touch, with a configurable TTL (`execution.maker_ttl_ms`).
* **When the TTL expires**, behaviour depends on `ttl_fallback`:
  * `skip`: cancel the order. Any small partial fill is unwound.
  * `taker_if_edge`: run a **fresh** re-evaluation assuming taker costs. If the net-profit
    gate still passes in the same direction, send a **price-capped IOC** order
    (`taker_max_slippage_bps`). Otherwise skip. The bot never sends a blind market order on entry.
* **Exit:** reduce-only IOC orders, with the price cap widening on each retry. A reduce-only
  MARKET order is used only as the last emergency step.
* **Tracked per order:** submitted, ack, partial fill, full fill, cancel, expired/TTL, average
  fill price, actual fees, and maker/taker flag. In live trading these come from
  `ORDER_TRADE_UPDATE` and are reconciled through REST when needed.
* **Paper simulator:**
  * order latency;
  * post-only rejection if the order would take liquidity;
  * **queue position** (visible quantity ahead of you is consumed by real trade prints first);
  * partial fills from real prints;
  * book-walk taker fills with extra slippage;
  * maker/taker fees.

## Exit logic

The 0.10 USDT figure is the minimum edge required to **enter**. It is not a fixed take-profit.
* **Stop loss.**
* **Break-even:** arms at `max(trigger, net break-even + buffer)`.
* **Trailing stop:** the stop only ever tightens.
* **Target:** when the target is reached, the bot takes profit **unless momentum is still
  going**. If it is, the target instead arms a tighter trailing stop so the bot can catch a
  bigger move.
* **Order-flow reversal.**
* **Loss of book and tape imbalance** for N consecutive evaluations.
* **Momentum exhaustion.**
* **Time stop.**
* **Emergencies:** stale data, spread blow-out, risk halt, kill switch.

## Risk controls

Daily loss limit, maximum consecutive losses, maximum trades per hour, per-symbol cooldown
after a loss, maximum position notional, maximum leverage, one position at a time (default),
maximum spread, maximum expected slippage, stale-data detection, disconnect timeout, API-error
burst halt, and a kill switch.

* **Kill switch:** create a file named `KILL_SWITCH` in the working directory, or call
  `RiskManager.kill()`.
* **Hard halts** stop new entries **and** flatten open positions. These are: daily loss limit,
  kill switch, API-error burst, and disconnect.
* **Live start-up check:** the bot refuses to start live if the account already has open positions.

## Enabling live trading

The default is `dry_run=True`. Live orders need **all** of the following:

```bash
export BINANCE_API_KEY=...            # futures-enabled key; restrict to your IP, no withdrawals
export BINANCE_API_SECRET=...
export DRY_RUN=false
export LIVE_TRADING_CONFIRM=I_UNDERSTAND_THE_RISK
python main.py --mode live --config configs/your_tiny_live.json
```

If any one of these is missing, the bot runs as paper (dry run) and logs a warning. Set
`BINANCE_TESTNET=true` to use the futures testnet endpoints. On first use of a symbol, the
bot sets the margin type and leverage and fetches your real commission rates
(`/fapi/v1/commissionRate`) for the cost model.

## Data recorded

* **`signals`:** every potential signal, plus a small random baseline sample so the dataset
  isn't conditioned only on the model's own opinions. Each row has:
  * a millisecond timestamp and symbol;
  * bid/ask, spread, depth;
  * imbalance, OFI, and aggressor-flow features (all features are also stored as JSON);
  * score, direction, `p_long`/`p_short`, `p_target`;
  * expected net, the entry decision, and the **rejection reason**;
  * forward returns at 1/3/5/10/30/60 s;
  * `tp_long`/`tp_short`/`tp_pred` (target before stop), and MFE/MAE.
* **`trades`:** every field listed in the spec. This includes gross PnL, fees, estimated vs
  actual slippage, net PnL, MFE/MAE, exit reason, and maker flags.
  `net_pnl = gross_pnl − fees`, because slippage is already inside the actual fill prices.
  It is reported separately for diagnostics.
* **`orders`:** the full order lifecycle.
* **`scanner`:** periodic rankings.
* **`book_snapshots`:** optional raw depth.

## Replacing the predictor

Every predictor returns a `Prediction` containing a score, `p_long`/`p_short`, a signed drift
(bps/s), volatility, and flow confirmation. The entry filter, barrier model, execution, and
exits only ever use that object. To switch models, implement `BasePredictor.predict(features)`
or train with `research/train_model.py`, then register it in `build_predictor`. Nothing in
execution has to change.

## Known limitations / next steps

* **Not yet verified against live Binance.** Run `tools.validate_binance` first.
* **WebSocket URL changes:** Binance has been reorganising its futures WebSocket URLs. If the
  connection fails, update `exchange.ws_base` in your config.
* **Uncalibrated defaults:** `logistic_k`, `drift_per_score`, and the component weights are
  heuristic defaults. Calibrate them with Phase 5 data.
* **Replay speed:** pure Python, roughly 15–20× real time per 3 symbols on synthetic data.
  Real BTC-class feeds are busier, so sweeps over many symbol-days need several workers
  and time.
* **Simulation limits:** the maker queue is approximated from visible size and trade
  prints. Hidden liquidity, queue jumping, and exchange-side latency variance are not
  modelled.
* **Geo-restrictions:** Binance futures is not available in every jurisdiction. Make sure you
  are allowed to use it.

*Not financial advice. High-frequency futures trading with leverage can lose money quickly.*
