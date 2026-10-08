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
- **Disk:** see §3a. For a machine with about 130 GB in total, give the recorder **`--max-gb 95`**
  (90–100 GB at most). Keep the remaining 30–35 GB for the OS, temporary files, dataset building
  and research outputs (at least 20–30 GB must stay free).
- **No API key is needed.** The recorder uses public streams only and has no order code path.

## 2. Check connectivity first

```bash
python -m tools.validate_binance --symbols 5 --duration 60 --out validation_report.json
```

Every check should PASS, especially `diff_depth_sync`. Then confirm which websocket address carries
each stream the recorder needs:

```bash
python -m tools.probe_ws
```

Binance has been moving futures streams to separate paths, for example `/public` for order-book
streams and `/market` for trades and tickers. On the first real setup, `aggTrade` was silent on
the legacy base URL. The probe prints the exact flags to add to the recorder command:
`--ws-depth-base`, `--ws-bookticker-base` and `--ws-trades-base`.

The recorder also protects itself:

- **A stream never arrives:** if any stream (diff depth, bookTicker or aggTrade) has delivered
  nothing for any symbol after 2 minutes, it stops with reason `stream_never_arrived`.
- **A stream goes quiet later:** a silent minute is logged and marked in the store
  (`__gap__`, reason `stream_silent`).

## 3. Record — recommended initial run (130 GB machine)

```bash
python -m v3.recorder --out /data/l2 --symbols BTCUSDT ETHUSDT SOLUSDT \
    --no-depth20 --max-gb 95 --min-free-gb 10 --days 14
```

- **Symbols and duration:** BTCUSDT, ETHUSDT and SOLUSDT only, with a 14-day target.
- **`--no-depth20`** drops only the redundant top-20 snapshot stream. The full book is rebuilt from
  the diff stream, and the sequence checks and gap handling are unchanged.
- **Compression:** finished hours are recompressed to zstd automatically (level 19, lossless,
  verified before the original is deleted).
- **Budgets:** the recorder stops cleanly before 95 GB, or when the drive has less than 10 GB
  free.

**External SSD:**

- Point `--out` at the SSD, for example `/Volumes/<SSD>/l2` on macOS or `D:\l2` on Windows. The
  code can live anywhere.
- `--max-gb` and `--min-free-gb` both measure the drive that `--out` is on.
- Keep the SSD plugged in and disable disk sleep (macOS: Energy → untick "Put hard disks to
  sleep"; Windows: turn off USB selective suspend). Use exFAT, NTFS or APFS.
- If the drive disappears, fills or turns read-only mid-recording, the recorder stops cleanly
  (`write_error`) instead of losing data silently. Start it again with the same command once the
  drive is back; it continues in the same store.

For unattended operation, `deploy/v3-recorder.service` (systemd) and `deploy/Dockerfile.recorder`
already contain these flags.

### 3a. Storage: what is kept and what is saved

| Kept (never removed or thinned) | Saved |
|---|---|
| Diff-depth updates (`depth@100ms`) with U/u/pu update ids | `depth20@100ms` not recorded (`--no-depth20`), about 30% of raw bytes |
| REST snapshots for sync (`__snapshot__`) and audit snapshots | Closed hours recompressed gzip → zstd 19, about 2× smaller on real Binance JSON |
| `bookTicker`, `aggTrade` |  |
| Exchange timestamps (E/T), local receive timestamps |  |
| Gap / resync markers |  |

**Compression is lossless and safe:**

- A file is recompressed only after its hour has ended (plus a 3-minute grace period) **and** no
  writer still holds it.
- The `.zst` file is decompressed again and checked against the original content (SHA-256 and
  line count). Only then is the original removed.
- Original size, compressed size and ratio go to `_health/compaction.jsonl`.
- Readers handle `.gz`, `.zst` and `.xz` transparently.
- `python -m v3.compact --store /data/l2` does the same by hand, for example for hours left
  open by a restart.

**Hard budget (`--max-gb`):**

- The whole store size is checked every 15 s.
- The recorder stops *before* reaching the budget, keeping a margin of at least 0.5 GB or five
  minutes of current growth, whichever is larger.
- Already-written data is untouched, and the reason is written to `_health/stops.jsonl`, the
  health log and `_system` (`__stop__`).
- If the store is already over budget, the recorder refuses to start and exits with status 3.
  The systemd unit does not restart on that status.
- `--min-free-gb` stays on as a second safeguard for the drive itself.

### 3b. After the first 1–3 hours: measured storage projection

```bash
python -m v3.storage --store /data/l2 --max-gb 95 --hours 3 --days 14 21
```

This uses actual recorded bytes (an hour still in gzip is measured by compressing it at the
recorder's zstd level). It reports:

- compressed and raw MB/hour per symbol;
- compressed MB/hour per stream type;
- projected GB/day;
- projected GB for 14 and 21 days;
- the store size now and the headroom under `--max-gb`.

It also writes `<store>/storage_projection.json`. It needs at least one hour of data per symbol
and refuses to guess with less.

**Decision rule:**

- If 14 days fits with a 10% margin, keep recording.
- If it does **not** fit, stop and look at the projection before changing anything. Never reduce
  or thin the core diff-depth feed to make it fit. The acceptable options are a shorter duration,
  fewer symbols, or a larger disk.

**What is recorded, per symbol (`/data/l2/<SYMBOL>/<YYYYMMDD>/<HH>.jsonl.gz`, `.jsonl.zst` once the hour is closed):**

- **Streams:** `depth@100ms` diff depth (U/u/pu update ids; 100 ms is the fastest USDT-M diff
  interval), `bookTicker`, `aggTrade`; plus `depth20@100ms` (levels 1–20) unless `--no-depth20`.
- **Timestamps:** local receive time on every line; exchange `E`/`T` timestamps inside the payloads.
- **Snapshots:**
  - `__snapshot__`: the REST snapshot (`limit=1000`) used for every (re)synchronisation.
  - `__snapshot_audit__`: taken every 10 min, so any replay or dataset can start mid-recording.
- **Markers:**
  - `__gap__`: a continuity break, with its reason (`pu_mismatch`, `snapshot_not_bridged`,
    `crossed`, `disconnect`, `writer_drop`) and update ids.
  - `__resync__`: the result of the rebuild.
  - `__check__`: the diff book disagreed with depth20 (only when depth20 is recorded).
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
- agreement between the diff book and depth20 (≈ 1.0; only when depth20 is recorded);
- the longest silence per stream.

Copy the store to the research machine with `rsync -a /data/l2/ research:/data/l2/`. Files are
append-only.

## 5. Plan the split before looking at results

- **Minimum:** 7 full days. **Preferred:** 14–30.
- Example for 14 days: TRAIN days 1–8, CALIBRATION day 9, SELECTION days 10–11, TEST days 12–14.
- Example for 21 days: TRAIN days 1–12, CALIBRATION days 13–14, SELECTION days 15–17, TEST days 18–21.
- Write the split down **before** building the dataset. The test days are never passed to
  `v3.research`; it refuses them through `--forbid-days`.

## 6. Build the dataset

```bash
python -m v3.dataset --store /data/l2 --out data/v3/ds --days 2025-01-06 ... --sample-ms 2000 --workers 4
```

`--sample-ms` only sets the spacing of the generated training rows. The raw L2 stream and the
features are computed at full recorded fidelity either way: every message is processed, features
are evaluated on a 250 ms grid, and labels use every bookTicker update.

| | 1000 ms rows | 2000 ms rows |
|---|---|---|
| Rows per symbol per day | 86,400 | 43,200 |
| Dataset disk (parquet, ~800 columns) | ≈ 0.3–0.5 GB / symbol-day | ≈ 0.15–0.25 GB / symbol-day |
| RAM / model fitting time | 1× | ≈ 0.5× |
| Information | adjacent rows overlap heavily: features share 1–10 s windows and labels share 5–120 s paths | almost the same; still dozens of rows per 60 s label horizon |
| Effective independent samples | dominated by the number of distinct minutes and regimes, not row count | about the same |
| When to choose | if disk and RAM allow | **recommended on the 130 GB machine** |

Build time is the same for both, because the full stream is processed either way.

`dataset_manifest.json` records, per (symbol, day):

- rows and expected rows;
- the parquet SHA-256;
- the feature schema version and column hash;
- the label configuration;
- sample/eval spacing, window and latency.

**Each row (250 ms evaluation grid, spaced by `--sample-ms`) contains:**

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

## 6a. Optional: delete raw TRAINING days to reclaim space (never automatic)

```bash
python -m v3.qa --store /data/l2 --deep                       # writes /data/l2/QA.json
python -m v3.cleanup_training_raw --store /data/l2 --dataset data/v3/ds --split split.json   # dry run
python -m v3.cleanup_training_raw ... --delete --confirm DELETE-TRAINING-RAW
```

`split.json` is the split you fixed in §5:
`{"train": [...], "calibration": [...], "selection": [...], "test": [...]}`.

A training day's raw files are deleted only if **every** symbol that day passes all of these:

- the dataset shard exists and its SHA-256 matches the manifest;
- the row count is valid and at least 50% of expected;
- the feature schema version, column hash and label configuration are recorded;
- deep QA passed (book valid ≥ 95% of the day; health marked the day usable);
- symbol/date coverage is complete.

Other rules:

- It prints every file it would delete.
- Validation and test raw data are **never** deleted, because the execution replay needs them.
  Nor is hour 23 of the day before a protected day, which is warm-up data.
- Deletions are logged to `_health/cleanup.jsonl`.
- Do this only if space is actually short. Raw training days are what allow rebuilding the
  dataset if a feature definition changes later.

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
    --symbols BTCUSDT ETHUSDT SOLUSDT --val-days <15 16 17> --test-days <18 19 20 21> \
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
