# Validation Runbook: real Binance data → out-of-sample verdict

Live trading stays **disabled** throughout. Every step below is read-only towards
Binance (public market data only) and all execution is simulated.

## 0. Environment

The machine needs outbound access to `fapi.binance.com`, `fstream.binance.com`, and
optionally `data.binance.vision` (historical archive). Binance futures is not available
in every jurisdiction.

```bash
pip install -r requirements.txt
pytest                      # all tests must pass (~2 min)
```

## 1. Validate connectivity and parsers (≈2 min)

```bash
python -m tools.validate_binance --symbols 10 --duration 60 --out validation_report.json
```

Every check should be `PASS`. The JSON report holds feed-health counters, latency
percentiles, and sample payloads for each stream type.

| Check | What it proves |
|---|---|
| `rest_discovery` | Active USDT perpetuals are discovered and their filters parse |
| `clock_skew` | The local clock is close to Binance server time |
| `market_streams` / `detail_streams` | Every payload passes the strict schema validator |
| `timestamps` | Event-time monotonicity and receive latency (p50/p99) |
| `update_ids` | aggTrade id gaps, depth `pu`/`u` continuity, bookTicker `u` ordering |
| `diff_depth_sync` | A full order book built from diff depth plus a REST snapshot stays in sync |
| `resubscription` | Unsubscribed streams stop delivering and resubscribed ones resume |
| `reconnect` | A forced socket close reconnects and resubscribes automatically |
| `stale_detection` | A silent stream is flagged stale by the risk manager |
| `subscription_limit` | Many streams on one connection are acked and deliver data |

If anything fails, fix it before recording. Malformed messages are logged with their
payload (logger `feed`).

## 2. Record real microstructure data

```bash
python main.py --mode record --config configs/example_record.json --duration 86400
```

* Raw WebSocket events go to `data/events/YYYYMMDD/HH.jsonl.gz`. Expect about
  3–6 GB/day at 30 symbols.
* Signals and decisions go to `data/microstructure.sqlite`.
* The status line every 30 s reports evaluations, rejections, and malformed/unexpected
  message counts.
* For a fixed universe, which is easier to compare across days, set
  `"scanner": {"static_symbols": ["BTCUSDT", "ETHUSDT", ...]}` in the config.
* Aim for **at least several days** that include different volatility regimes. A few
  hours is not enough for a trustworthy out-of-sample test.

Optional: build longer history from the public archive. It only has top of book (L1),
so treat the results as indicative.

```bash
python -m tools.binance_archive --symbols BTCUSDT ETHUSDT --start 2024-03-01 --days 7 --out data/archive_events
```

## 3. Single replay (sanity check)

```bash
python -m backtest.replay --events data/events --out runs/replay1 --latency-ms 100
python -m research.analysis --db runs/replay1/replay.sqlite
```

* `--speed max` (the default) is deterministic. `--speed 1` replays in real time and
  `--speed 10` at 10×.
* `--time-source local` replays on receive timestamps instead of exchange timestamps.
  This is the most conservative option. Run both and compare.

## 4. Research protocol (TRAIN / VALIDATION; TEST untouched)

```bash
python -m research.validate --events data/events --out runs/v1 --workers 4
```

This produces `runs/v1/VALIDATION_REPORT.md` and `validation_report.json` with:
* opportunity decisions (ENTERED vs. rejection categories)
* feature analysis: information coefficient (IC), non-overlapping t-stats,
  first-half vs. second-half stability, quintile net returns
* confidence calibration with Wilson confidence intervals
* minimum tradable edge by notional (50–250 USDT, maker vs. taker entry)
* per-symbol results
* parameter sweep on TRAIN, ranked by expectancy per trade after all costs
* validation of the top-K configurations, plus latency robustness at 20/50/100/250/500 ms

Customise the grid with `--grid grid.json`. Valid keys are `target_bps`, `stop_bps`,
`min_probability`, `min_confidence`, `max_spread_bps`, `imbalance_threshold`,
`flow_threshold`, `max_hold_s`, `maker_ttl_ms`, `taker_fallback`, `latency_ms` and
`notional`.

## 5. Final test (run ONCE)

Freeze the configuration, then:

```bash
python -m research.validate --events data/events --out runs/v1 --workers 4 --final-test
```

* The TEST period is evaluated once, for the configuration chosen on VALIDATION, at
  every latency scenario.
* `test_lock.json` records that configuration. Any later attempt to evaluate a
  **different** configuration on the same test period is refused, because the test
  data would no longer be unseen.
* To iterate after this point, record **new** data and use it as the next test set.

## 6. Criteria before considering Phase 8

All of the following must hold:

1. Feature analysis finds at least one directional feature that is significant
   (|t| ≥ 3), has a stable sign across both halves, and is net-positive in its top
   quintile.
2. On TEST, at 100 ms latency: at least 30 trades, expectancy > 0 USDT/trade after
   fees and slippage, and t > 2.
3. Expectancy is still positive at 250 ms latency.
4. Calibration is not overconfident in the buckets the strategy actually trades.
5. The result does not depend on a single symbol or a single day.

If these aren't met, the report says so plainly (for example, "NO EDGE ON UNSEEN DATA").
Do **not** loosen thresholds to manufacture trades.
