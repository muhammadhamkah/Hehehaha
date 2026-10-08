# V3 runbook — L2 recording and directional research

V3 is a separate, research-only path. V1 (rule-based baseline) and V2 (L1 ML baseline), their
experiments, reports and locked test results are unchanged. Adding V3 changes no V1/V2 config
fingerprint, including the V1 locked hash `09b8b8a39f5c637b` (a test enforces this). Live trading
stays disabled: the bot refuses `strategy.predictor = "v3"` in live mode.

The research question is: **does full L2 order-book and queue-dynamics data contain directional
information that the L1 archive does not?**

---

## 1. Where to record

The recorder needs a machine where Binance USDT-M market data is reachable **and permitted** for
you: Binance restricts access from some jurisdictions. The Claude cloud container used to build V3
is blocked (HTTP 451 / 403), so recording has to run on your own machine or a VPS.

- **Machine:** 1–2 vCPU and 2 GB RAM are enough. Use a stable connection, NTP-synced time, and a
  region close to Binance's matching engine (Tokyo or Singapore is typical).
- **Disk:** about 4–6 GB/day for the 5 symbols with depth20 enabled (gzip JSONL), so 14 days ≈
  80 GB and 30 days ≈ 170 GB. `--no-depth20` saves about 30%, but you lose the independent
  cross-check of the diff book.
- **No API key is needed.** The recorder uses public streams only and has no order code path.

## 2. Check connectivity first

```bash
python -m tools.validate_binance --symbols 5 --duration 60 --out validation_report.json
```

Every check should PASS, especially `diff_depth_sync`. If Binance has changed its websocket base
URL or stream paths, pass the new ones to the recorder with `--ws-base` / `--rest-base`.

## 3. Record

Start with the three most liquid contracts, then add DOGE and LINK after a day of clean QA:

```bash
python -m v3.recorder --out /data/l2 --symbols BTCUSDT ETHUSDT SOLUSDT
# after QA looks clean (restarting appends to the same store):
python -m v3.recorder --out /data/l2 --symbols BTCUSDT ETHUSDT SOLUSDT DOGEUSDT LINKUSDT
```

For unattended operation use `deploy/v3-recorder.service` (systemd) or `deploy/Dockerfile.recorder`.

**What is recorded, per symbol (`/data/l2/<SYMBOL>/<YYYYMMDD>/<HH>.jsonl.gz`):**

- **Streams:** `depth@100ms` diff depth (U/u/pu update ids; 100 ms is the fastest USDT-M diff
  interval), `depth20@100ms` (levels 1–20), `bookTicker`, `aggTrade`.
- **Timestamps:** local receive time on every line; exchange `E`/`T` timestamps inside the payloads.
- **Snapshots:**
  - `__snapshot__`: the REST snapshot (`limit=1000`) used for every (re)synchronisation.
  - `__snapshot_audit__`: taken every 10 min, so any replay or dataset can start mid-recording.
- **Markers:**
  - `__gap__`: a continuity break, with its reason (`pu_mismatch`, `snapshot_not_bridged`,
    `crossed`, `disconnect`, `writer_drop`) and update ids.
  - `__resync__`: the result of the rebuild.
  - `__check__`: the diff book disagreed with depth20.
- **Store files:** `_system/` (server-time samples), `_health/` (per-minute health),
  `v3_store.json` (manifest), `meta.json` (symbol filters).

**Continuity rules (identical in the recorder, the dataset builder, QA and the replay):**

1. Buffer diffs.
2. Fetch the snapshot (L).
3. Drop events with u < L.
4. The first applied event must satisfy U ≤ L ≤ u.
5. After that, every event must satisfy `pu` = previous `u`.

Any violation, a crossed book, a disconnect or a writer drop is a GAP. The book is marked invalid
and rebuilt from a fresh snapshot. Nothing continues on a corrupted depth state.

## 4. Monitor quality (daily)

```bash
python -m v3.qa --store /data/l2 --deep
```

A day is **usable** for a symbol only if the book was valid ≥ 95% of the day, the writer dropped
nothing, and no resync failed. `--deep` re-validates every diff offline and reports:

- gaps by reason;
- the share of time the book was valid;
- agreement between the diff book and depth20 (should be ≈ 1.0);
- the longest silence per stream.

Copy the store to the research machine with `rsync -a /data/l2/ research:/data/l2/`. Files are
append-only.

## 5. Plan the split before looking at results

- **Minimum:** 7 full days. **Preferred:** 14–30.
- Example for 21 days: TRAIN days 1–12, CALIBRATION days 13–14, SELECTION days 15–17, TEST days 18–21.
- Write the split down **before** building the dataset. The test days are never passed to
  `v3.research`; it refuses them through `--forbid-days`.

## 6. Build the dataset

```bash
python -m v3.dataset --store /data/l2 --out data/v3/ds --days 2025-01-06 ... 2025-01-26 --workers 4
```

**Each row (1 s sampling, 250 ms evaluation grid) contains:**

- **V2 features**, computed from an L1-only view of the same recording (exactly V2's definition;
  used for Stage A and the L1 baseline).
- **About 230 V3 features:**
  - `v3d_*` multi-level depth (1/3/5/10/20): totals, weighted imbalance, ratio, cumulative,
    slope, convexity.
  - Queue depletion and replenishment, replenishment velocity and ratio, queue persistence and
    lifetime.
  - Cancellation intensity and asymmetry, liquidity pulling, rapid removals.
  - L1 and multi-level microprice, displacement and divergence.
  - Book shape: concentration, cliffs, empty levels, distance to large resting liquidity.
  - `v3f_*` aggressive flow, trade-count imbalance, large-trade imbalance, sweeps.
  - `v3t_*` changes over 100 ms, 250 ms, 500 ms, 1 s, 2 s, 5 s and 10 s, plus persistence,
    acceleration, slope and sign flips.
  - `v3x_*` the explicit directional events (formulas below).
- **Costs:** taker and maker round-trip costs for 100/150/250/500/1000 USDT, from walking the
  actual L2 book.
- **Labels (executable prices, entry after 100 ms latency):**
  - first touch of ±10/15/20/25/30 bps within 60 s (horizons 5/10/30/60 s are derived);
  - MFE, MAE, time to barrier, and which barrier was touched first;
  - TP/SL pairs 20/12, 25/15, 30/15, 30/20 and 40/20 within 120 s.

Rows are dropped and counted when the L2 book was not continuously valid (gap, resync, or 10 s
warm-up after a resync), or when the label path crosses a disconnect.

**Explicit directional events (quantitative definitions):**

- **Queue depletion (up):** (ask best-queue depletion over 3 s ÷ ask queue size) ×
  max(1 − ask replenishment ratio, 0) × share of 500 ms buckets with net aggressive buying.
  "Down" is the mirror image.
- **Liquidity pulling (down):** bid cancellations in the top 10 over 1 s ÷ bid depth10 × sell
  share of aggressive flow. "Up" is the mirror image.
- **Absorption (bullish):** aggressive selling over 3 s ÷ bid depth5 × [best bid held] ×
  bid replenishment ratio. "Bearish" is the mirror image.

## 7. Research (train / calibration / selection only)

```bash
python -m v3.research --data data/v3/ds --out runs/v3 \
    --train-days <days 1-12> --cal-days <13 14> --sel-days <15 16 17> --forbid-days <18 19 20 21> \
    --notional 150 --min-net 0.10 --export models/v3
```

The study fits logistic regression and LightGBM only. It reports:

- **Stage A:** P(|move| ≥ X within H), using V2's L1 definition and using all V3 features.
- **Direction P(UP first | large move)** for each feature set — L1 only / L2 depth only /
  trade flow only / L1+L2 / L1+L2+flow — tested two ways:
  - a **direct** model, trained on all realised large moves;
  - a **gated** model, trained only inside Stage A's top 20%.
- **Direction AUC and accuracy inside Stage A's top 100/20/10/5/2/1%**, with a cluster-bootstrap
  (symbol × minute) 95% CI of every feature set's AUC gain over L1-only.
- The explicit events' univariate direction AUC.
- **Entry timing:** signed feature and price paths from −5 s to +5 s around confident signals, to
  tell pressure that precedes price from detection after the move has started.
- **Economics by notional** with depth-based costs.
- **Execution-aware selection:** pair models, EV gate, optional Stage-A gate. The chosen configs
  are exported to `models/v3/v3_all` and `models/v3/v3_l1` (the L1 baseline).

**Barrier pairs are chosen on the selection days only.** Touch-only quantities computed from the
L2 book are counted as L1: level-1 depth, spread, L1 microprice, best-queue depletion and
lifetime. Any gain attributed to L2 therefore comes from depth beyond the touch.

## 8. Execution-aware evaluation (same replay simulator)

```bash
python -m v3.evaluate --store /data/l2 --out runs/v3_replay --models models/v3 --kinds v3_all v3_l1 \
    --symbols BTCUSDT ETHUSDT SOLUSDT DOGEUSDT LINKUSDT --val-days <15 16 17> --test-days <18 19 20 21> \
    --latencies 50 100 250
# then, exactly once, with the frozen configs:
python -m v3.evaluate ... --final-test
```

- The bot replays the raw store with a diff-depth book and V3 serving (`strategy/v3.py`; same
  `v3.state.SymbolState` as the dataset).
- Fills and exits run through the existing paper execution simulator: taker fills walk the real
  L2 book, latency is configurable, barrier exits apply.
- **Primary metric:** net expectancy per trade.
- **Also reported:** PF, max DD, win rate, target-before-stop rate, calibration, trade count,
  hold time and per-symbol results.
- `test_lock.json` refuses a changed configuration on the same test period.

## 9. Classification (`V3_L2_DIRECTIONAL_REPORT.md`)

| Classification | Criteria |
|---|---|
| 1 STRONG L2 DIRECTIONAL EDGE | material AUC gain over L1 (≥ 0.03, CI > 0) inside Stage-A top buckets, direction accuracy ≥ 0.56, AND locked-test net expectancy > 0 with t ≥ 2, ≥ 100 trades, positive on more than one day/regime |
| 2 WEAK BUT PROMISING | material gain and accuracy, test expectancy ≈ break-even or positive but not significant |
| 3 L2 IMPROVES DIRECTION BUT NOT ENOUGH AFTER COSTS | material gain, test expectancy negative |
| 4 NO USEFUL DIRECTIONAL EDGE | no material gain over L1, or direction 50–55% |

If L2 direction stays around 50–55% inside the strong large-move states, the conclusion is that
the current strategy class is unlikely to work. No XGBoost, MLP or Laya models are used unless
LightGBM first shows real directional discrimination.

Before any paper trading, all of these are required:

- statistically credible positive out-of-sample expectancy;
- realistic execution costs;
- enough trades;
- positive results on more than one day or regime.
