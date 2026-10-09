# V3 — L2 directional research: status

**Research question:** does full L2 order-book and queue-dynamics data provide directional
information that the L1 archive does not?

**Status: the full pipeline is built and validated end-to-end on synthetic L2 data. No real L2
data has been recorded yet, so there is no classification yet.**

The Claude cloud environment that built V3 cannot reach Binance market data (REST HTTP 451,
websocket 403), and its container is ephemeral. The 7–30 day recording therefore has to run on a
machine where Binance is reachable and permitted — see `docs/V3_RUNBOOK.md`.
`V3_L2_DIRECTIONAL_REPORT.md` with the final classification will be written from that data.

V1 and V2, their experiments, reports and locked results are untouched. The V1 locked config hash
is still `09b8b8a39f5c637b`, and `tests/test_v3.py::test_v1_v2_fingerprints_unchanged` enforces
this. V3 is refused in live mode.

## What is built

| Component | File | What it does |
|---|---|---|
| L2 book | `v3/book.py` | Binance diff-depth sync rules. Any continuity break (pu mismatch, unbridged snapshot, crossed book) is a GAP: the book is invalid until rebuilt from a snapshot. Per-level change attribution. Multi-level book walk for slippage. |
| Recorder | `v3/recorder.py` | Records `depth@100ms`, `depth20@100ms`, `bookTicker` and `aggTrade` for each symbol. Keeps local and exchange timestamps and update ids. Writes the REST snapshot used for every resync, plus an audit snapshot every 10 min. Live continuity validation: on a gap it writes a `__gap__` marker and rebuilds from a snapshot. Also: depth20 cross-check, per-minute health log, server-time samples, writer-drop detection, disk guard. Public data only; no keys, no orders. |
| Deployment | `deploy/` | systemd unit and Dockerfile for unattended recording. |
| Storage | `v3/compact.py`, `v3/storage.py`, `v3/cleanup_training_raw.py` | Lossless zstd recompression of closed hours, verified before the original is deleted. Hard `--max-gb` budget with a clean, logged stop. First-hours storage projection from actual bytes. Manual, guarded deletion of raw training days (dry run by default; never deletes validation/test data). |
| QA | `v3/qa.py` | Per-day, per-symbol recording quality, plus `--deep` offline re-validation of every diff. Marks which days are usable. |
| Store reader | `v3/store.py` | Merges the per-symbol stores, so the existing replay simulator reads V3 recordings directly. |
| Features | `v3/features.py` | About 230 L2, queue, cancellation, flow, temporal and event features (list in the runbook §6). Includes explicit queue-depletion, liquidity-pulling and absorption features. |
| Shared state | `v3/state.py` | Same per-symbol computation for the dataset and for serving, including V2's L1 features from an L1-only view of the same recording. |
| Labels | `v3/labels.py` | First touch of ±10/15/20/25/30 bps within 5/10/30/60 s; which side was touched first; MFE/MAE; TP/SL pairs 20/12, 25/15, 30/15, 30/20 and 40/20 over 120 s; all on executable prices after latency. |
| Dataset | `v3/dataset.py` | Builds rows with depth-based taker and maker costs for 100–1000 USDT. Drops rows touched by gaps, warm-up or label disconnects, and counts them. Writes a 250 ms timeline for the timing study. |
| Research | `v3/research.py` | See the list below this table. |
| Serving | `strategy/v3.py` | `predictor="v3"`: L2 features → calibrated P(TP first) per pair → EV gate using the same depth-based cost function as the research. Optional Stage-A gate. |
| Evaluation | `v3/evaluate.py` | Validation replays (production risk and research-only), latency sensitivity, and `--final-test` once per frozen config with `test_lock.json`. |

**`v3/research.py` covers:**

- Stage A, using V2's L1 definition and using all features.
- Direction models per feature set — L1 / L2 / FLOW / L1+L2 / ALL — with logistic regression and
  LightGBM.
- Both a direct direction model and one trained only inside the Stage-A gate.
- Direction AUC and accuracy inside Stage-A top 100/20/10/5/2/1%, with cluster-bootstrap CIs of
  the gain over L1.
- Explicit event study, entry timing (−5 s … +5 s), and economics by notional.
- Execution-aware pair/threshold/gate selection on the selection days, exporting frozen configs.
- A mechanical classification suggestion.

## Validation on synthetic L2 data (pipeline check only — says nothing about real markets)

`v3/synthetic_l2.py` writes the recorder's exact format, with real U/u/pu sequencing and random
lost messages. It can plant a direction signal that is visible only in deeper-book behaviour
(ask/bid pulling at levels 2–10).

| Check | Result |
|---|---|
| Diff book vs depth20 | 5,966 / 5,966 identical (synthetic). Injected lost messages were all detected (`pu_mismatch`) and resynced from snapshots. |
| Recorder vs local mock exchange | Lost diff detected, `__gap__` written, book rebuilt from a REST snapshot. Health log and audit snapshots written. |
| No look-ahead | Features at time t are identical whether or not later events exist. |
| Train/serve parity | Bot serving features equal the dataset rows (> 50 timestamps checked). |
| Replay | The bot replays the V3 store with a diff-depth book and V3 serving, through the existing simulator. |
| **Planted L2 signal (edge 0.3)** | Direction AUC on 3,646 large moves (20 bps / 60 s): L1-only 0.706 → ALL 0.771. Gain +0.065, CI [0.041, 0.099]. Inside Stage-A top 10%: gain +0.139, CI [0.075, 0.205]. Suggested class "L2 improves direction". |
| **Null control (edge 0)** | Gains not significant (CIs include 0). Suggested class "4 NO USEFUL DIRECTIONAL EDGE". The pipeline does not invent direction. |

All 114 existing tests and the 13 new V3 tests pass.

## Next steps (in order)

1. Run `python -m tools.validate_binance` on the recording machine. All checks must pass,
   including `diff_depth_sync`.
2. Record BTCUSDT, ETHUSDT and SOLUSDT for 14 days, configured for the 130 GB machine:
   `python -m v3.recorder --out /data/l2 --symbols BTCUSDT ETHUSDT SOLUSDT --no-depth20 --max-gb 95 --min-free-gb 10 --days 14`.
   After 1–3 hours run `python -m v3.storage --store /data/l2 --max-gb 95`. If 14 days does not fit,
   stop and review; the diff-depth feed is never reduced. Run `python -m v3.qa --store /data/l2 --deep` daily.
3. After at least 7 (preferably 14–30) usable days, fix the TRAIN / CALIBRATION / SELECTION / TEST
   split **in writing before looking at any results**.
4. Run `v3.dataset` → `v3.research` (test days forbidden) → `v3.evaluate` (validation), then
   `v3.evaluate --final-test` once.
5. Write `V3_L2_DIRECTIONAL_REPORT.md` with one of the four classifications (criteria in the
   runbook §9).

**Dataset build cost:** synthetic data takes about 11 s per symbol per 30 min. Real BTCUSDT carries
roughly 3–5× the message rate, so expect about 30–60 CPU-minutes per symbol-day. For example,
70 symbol-days on 4 cores take about 10–15 h.
