# V2 Directional Edge Report

## Classification: **2 — LARGE-MOVE EDGE ONLY, DIRECTION WEAK**

> **The L1 feature set predicts volatility / move magnitude but not direction strongly enough.**

- **Stage A (magnitude):** works. AUC is 0.78–0.93 across every target and horizon. The top 1% of predicted 20 bps moves within 60 s realise one 92% of the time, against a 24% base rate.
- **Stage B (direction, given that a large move happened):** stays at chance on unseen data. AUC is 0.49–0.52 at the 60 s trade horizon, and accuracy is 0.50 against a 0.50 majority baseline.
- **Volatility check:** the earlier single-stage AUC is ~80–88% unsigned intensity (realised volatility, spread, activity). A model built from intensity features alone matches the full model.
- **Even with perfect direction** the trades do not pay at the current stop. Large-move states are two-sided: the average adverse excursion (MAE) is −25 to −31 bps, against a +22 to +24 bps average favourable excursion (MFE). An 8 bps stop is hit first.
- **The combined Stage A → Stage B → EV rule never trades.** 150 threshold/target configurations were tried per model on the selection day; none produced a single entry.
- **Replay and test days:** per your instruction 10 (proceed to the combined replay only if Stage B materially improves direction), the combined rule was **not** replayed. The **29–30 March test days remain untouched**. No V2 configuration has been evaluated on them, and the V1 locked result was not re-run.

Live trading stays disabled (DRY_RUN=True; the bot refuses `predictor="v2"` and `risk.research_mode` in live mode).

---

## 1. Setup and discipline

| Role | Days (2024, 13:00–17:00 UTC) | Used for |
|---|---|---|
| TRAIN | 21–26 March | fitting |
| CALIBRATION | 27 March | early stopping, Platt/isotonic calibration, Stage-A quantile thresholds |
| SELECTION | 28 March | every number in this report, every choice |
| TEST | 29–30 March | **untouched** (refused by `check_days`) |

- **Data:** BTC, ETH, SOL, DOGE, LINK USDT-M; Binance public archive (aggTrades + bookTicker; L1 only). 575,962 rows sampled at 1 s, 462 features, 0 malformed messages.
- **Labels:** taken on **executable** prices 100 ms after the decision. Long enters at the ask and exits on the bid; short enters at the bid and exits on the ask.
  - *large(X, H):* a ±X bps barrier is touched within H seconds.
  - *up(X):* the long barrier is touched first.
- **Stage B population:** trained only on rows where large(X, H) actually happened. At inference it runs only after Stage A passes, with no conditioning on the future.
- **Models:** LightGBM (grid over all X × H) and XGBoost (X ∈ 15/20/25/30, H = 60 s). Same small hyper-parameters as the V2 study.
- **Code:** `research/v2_twostage.py`. Raw output is in `reports/v2_twostage/` (`twostage.json`, `top_buckets_raw_direction.json`, `forced_top1.json`; the forced-trade diagnostic is `research/v2_forced_top.py`).

## 2. Stage A — P(|move| ≥ X bps within H) (selection day)

AUC, with the base rate in brackets:

| X \ H | 5 s | 10 s | 30 s | 60 s |
|---|---|---|---|---|
| 10 bps | 0.911 (5.1%) | 0.875 (13.3%) | 0.800 (38.9%) | 0.783 (60.7%) |
| 15 bps | 0.920 (1.6%) | 0.905 (5.0%) | 0.844 (19.4%) | 0.799 (37.3%) |
| 20 bps | 0.926 (0.5%) | 0.911 (2.2%) | 0.870 (10.7%) | 0.822 (23.5%) |
| 25 bps | 0.931 (0.2%) | 0.926 (1.0%) | 0.883 (6.3%) | 0.838 (15.3%) |
| 30 bps | 0.915 (0.1%) | 0.929 (0.5%) | 0.902 (3.8%) | 0.852 (10.2%) |

Hit rate inside the top-k% of predictions (LightGBM, H = 60 s):

| X | base | top 20% | 10% | 5% | 2% | 1% | 0.5% |
|---|---|---|---|---|---|---|---|
| 15 | 37.3% | 82.4% | 92.2% | 96.8% | 98.4% | 98.6% | 99.4% |
| 20 | 23.5% | 65.3% | 78.9% | 87.0% | 89.1% | 92.4% | 91.7% |
| 25 | 15.3% | 49.4% | 62.3% | 74.3% | 76.0% | 80.0% | 83.9% |
| 30 | 10.2% | 36.3% | 47.9% | 59.3% | 61.9% | 64.6% | 68.3% |

- **XGBoost is indistinguishable:** AUC 0.799 / 0.822 / 0.840 / 0.852 for X = 15 / 20 / 25 / 30.
- **Calibration (isotonic, fitted on the 27 March calibration day):** ECE is 0.04–0.09 at H = 60 s. Calibration is decent but drifts between days. The selection day was calmer than the calibration day, so calibration-day quantile thresholds flag far fewer rows on 28 March (e.g. only 0.1% instead of 1%).

**Verdict:** move magnitude is strongly and robustly predictable from L1 state.

## 3. Stage B — P(UP | large move) (selection day, realised large moves only)

| X | H | n (sel) | up-rate | majority acc. | LightGBM AUC | LightGBM acc. | acc. in most-confident 10% | XGBoost AUC | sign(ofi_3s) acc. | sign(mom_3s) acc. |
|---|---|---|---|---|---|---|---|---|---|---|
| 10 | 10 s | 9,538 | 49.8% | 50.2% | 0.546 | 0.529 | 0.622 | – | 0.529 | 0.527 |
| 15 | 10 s | 3,589 | 48.2% | 51.8% | 0.565 | 0.550 | 0.638 | – | 0.562 | 0.550 |
| 20 | 10 s | 1,553 | 42.7% | 57.3% | 0.579 | 0.567 | 0.712 | – | 0.567 | 0.563 |
| 25 | 10 s | 715 | 44.2% | 55.8% | 0.547 | 0.544 | 0.639 | – | 0.576 | 0.577 |
| 10 | 60 s | 43,697 | 50.4% | 50.4% | 0.522 | 0.496 | 0.523 | – | 0.517 | 0.512 |
| 15 | 60 s | 26,823 | 50.2% | 50.2% | 0.521 | 0.498 | 0.553 | 0.508 | 0.529 | 0.521 |
| 20 | 60 s | 16,926 | 49.9% | 50.1% | 0.518 | 0.501 | 0.542 | 0.501 | 0.529 | 0.525 |
| 25 | 60 s | 11,010 | 50.2% | 50.2% | 0.508 | 0.497 | 0.491 | 0.524 | 0.538 | 0.531 |
| 30 | 60 s | 7,316 | 49.7% | 50.3% | 0.491 | 0.503 | 0.495 | 0.501 | 0.550 | 0.534 |

**At the 60 s trade horizon, direction is at chance.** The boosted Stage B models are also nearly constant: raw P(UP) averages 0.494–0.499 on the selection day.

The one-line rules sign(OFI 3 s) and sign(momentum 3 s) do slightly better than the trained models, at 0.52–0.55. So weak, consistent directional information exists in the signed flow features. The boosted model does not extract it out of sample; it fits day-specific drift instead.

At **10 s** there is a little short-horizon continuation (AUC 0.55–0.58; 0.62–0.71 accuracy in the most-confident decile). It does not survive to 60 s, and it is far below what is needed (§6). For example, a 15 bps / 10 s trade needs P(TP) ≥ 0.85 at taker costs. In the best bucket the joint rate is P(large) 0.53 × P(correct) 0.64 ≈ 0.34.

**Verdict:** Stage B does **not** materially improve direction.

## 4. Is the earlier V2 AUC just volatility? — **Yes, mostly**

Feature taxonomy (462 features):
- **intensity (130):** unsigned. Realised volatility, spread, trade/notional rate, volume acceleration, liquidity change, and their rolling stats.
- **signed (284):** directional. Order-flow imbalance (OFI), book imbalance, microprice tilt and offset, aggressive buy/sell imbalance, momentum, mid changes, and their rolling stats.
- **side-pair (48):** raw bid-side vs ask-side levels.

**SHAP share by group (mean |SHAP|, selection day):**

| Model | intensity | signed | side-pair |
|---|---|---|---|
| Direct LightGBM long T20 | **83%** | 15% | 2% |
| Direct LightGBM short T20 | **79%** | 17% | 4% |
| Direct LightGBM long T30 | **88%** | 10% | 2% |
| Stage A X20/60 s | **80%** | 18% | 2% |
| Stage B X20/60 s | 23% | 55% | 21% |

The top features of the direct long *and* short models are the same: `rmax/rmean/rmin_rv_1s_bps` over 3–10 s, then spread.

**The direct LONG and SHORT models predict the same thing:**

| T (bps) | 10 | 14 | 16 | 18 | 20 | 25 | 30 |
|---|---|---|---|---|---|---|---|
| corr(p_long, p_short) | 0.83 | 0.83 | 0.85 | 0.91 | 0.94 | 0.96 | 0.95 |
| long model's AUC on **long** TP | 0.653 | 0.700 | 0.722 | 0.748 | 0.767 | 0.809 | 0.836 |
| long model's AUC on **short** TP | 0.609 | 0.652 | 0.675 | 0.695 | 0.703 | 0.741 | 0.766 |
| direction AUC of (p_long − p_short) when exactly one side wins | 0.491 | 0.501 | 0.497 | 0.495 | 0.486 | 0.485 | 0.455 |

The long model predicts the *short* trade's success almost as well as its own. Once a move has happened, the difference between the two models carries no directional information.

**Ablation (LightGBM, X = 20, H = 60 s):**

| Feature set | n | Stage A AUC | Stage A top-1% hit | Stage B AUC |
|---|---|---|---|---|
| intensity only | 130 | **0.823** | **92.1%** | 0.525 |
| signed + side-pair only | 332 | 0.815 | 89.4% | 0.507 |
| all | 462 | 0.822 | 92.4% | 0.518 |

Intensity alone is enough for magnitude, and nothing is enough for direction. (The signed features reach 0.815 on magnitude only because their rolling extremes — e.g. the max/min of 1 s momentum — encode |movement|.)

**Univariate (Spearman, selection day):**
- With the large-move label: intensity features up to |ρ| 0.47 (realised volatility); mean |ρ| 0.097 for intensity vs 0.034 for signed.
- With direction given a move: every feature has |ρ| ≤ 0.085.

## 5. Inside the top predicted large-move buckets (selection day)

Stage A rank, direction from raw Stage B. Rows overlap, so t-stats use (symbol, minute) clusters. Net is at 150 USDT notional, taker.

**Two-stage LightGBM, X = 20 bps, H = 60 s:**

| top | rows | actual large-move rate | direction acc. given move | avg MFE (bps) | avg MAE (bps) | P(TP), S=8 | net/trade S=8 (USDT) | t | P(TP), S=12 | net, S=12 | net with **perfect direction**, S=8 / S=12 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 20% | 14,399 | 65.3% | 0.489 | 17.5 | −19.0 | 0.225 | −0.185 | −31.6 | 0.273 | −0.188 | −0.080 / −0.047 |
| 10% | 7,200 | 78.9% | 0.483 | 20.0 | −22.0 | 0.260 | −0.183 | −22.7 | 0.318 | −0.188 | −0.061 / −0.021 |
| 5% | 3,600 | 87.0% | 0.501 | 22.0 | −24.9 | 0.275 | −0.182 | −16.3 | 0.353 | −0.182 | −0.058 / −0.009 |
| 2% | 1,440 | 89.1% | 0.479 | 22.7 | −27.8 | 0.269 | −0.186 | −12.3 | 0.340 | −0.190 | −0.065 / −0.012 |
| 1% | 720 | 92.4% | 0.448 | 22.4 | −31.0 | 0.268 | −0.189 | −10.4 | 0.342 | −0.193 | −0.055 / +0.003 |

**Two-stage LightGBM, X = 25 / 30 bps:**
- Same shape: top-1% move rate 80% / 65%; direction 0.51 / 0.59; net −0.187 / −0.183 USDT per trade.
- The 0.59 at X = 30 is 720 overlapping rows that are 89% long on one day. It is a day-level drift, not a model skill (the Stage B AUC at X = 30 is 0.49).

**Direct LightGBM T25, ranked by max(p_long, p_short):**
- Top 1%: move rate 82%, direction 0.56, net −0.154 USDT (t −8.3).
- 98% of those picks are long. On a single day that is a drift bet, not direction.

**What this says:**
1. The model finds the right states: the large move occurs 80–98% of the time in the top buckets.
2. Direction inside them is a coin flip.
3. **Perfect knowledge of the first-touched direction still loses with an 8 bps stop** (−0.05 to −0.08 USDT per trade), and is only ~break-even with 12 bps. High-volatility states move both ways first; MAE grows *faster* than MFE as the bucket narrows. Even a perfect direction call is worth almost nothing at these costs with a tight stop.

## 6. Economics by notional (research only; the default notional is unchanged)

Fees are taker 0.05% and maker 0.02%; the latency and slippage buffer is 0.8 bps per side.
- **Book walk:** median ≈ 0 for ≤ 250 USDT (L1 size usually exceeds the order). Above that it can't be measured from L1 data, so 500 USDT is extrapolated and is a lower bound on cost.
- **Round-trip cost:** taker in + taker out ≈ **11.65 bps**; maker in (if filled) + taker out ≈ **8.33 bps**.
- **Formulas:** break-even P(TP) = (S + c)/(T + S). P(TP) for +0.10 USDT EV = (1000/N + S + c)/(T + S). Stop S = 8 bps.

| Notional | entry | T=15 BE / +0.10 | T=20 BE / +0.10 | T=25 BE / +0.10 | T=30 BE / +0.10 |
|---|---|---|---|---|---|
| 100 | taker | 0.854 / >1 | 0.702 / >1 | 0.596 / 0.899 | 0.517 / 0.780 |
| 100 | maker | 0.710 / >1 | 0.583 / 0.940 | 0.495 / 0.798 | 0.430 / 0.693 |
| 150 | taker | 0.854 / >1 | 0.702 / 0.940 | 0.596 / 0.798 | 0.517 / 0.693 |
| 150 | maker | 0.710 / 1.000 | 0.583 / 0.821 | 0.495 / 0.697 | 0.430 / 0.605 |
| 200 | taker | 0.854 / >1 | 0.702 / 0.880 | 0.596 / 0.747 | 0.517 / 0.649 |
| 200 | maker | 0.710 / 0.927 | 0.583 / 0.762 | 0.495 / 0.646 | 0.430 / 0.561 |
| 250 | taker | 0.854 / >1 | 0.702 / 0.845 | 0.596 / 0.717 | 0.517 / 0.622 |
| 250 | maker | 0.710 / 0.884 | 0.583 / 0.726 | 0.495 / 0.616 | 0.430 / 0.535 |
| 500\* | taker | 0.854 / 0.942 | 0.702 / 0.773 | 0.596 / 0.656 | 0.517 / 0.570 |
| 500\* | maker | 0.710 / 0.797 | 0.583 / 0.654 | 0.495 / 0.555 | 0.430 / 0.482 |

\* Book walk extrapolated (L1 data).

**Observed P(TP) in the predicted direction, top 1%, S = 8:** 0.34 (T15), 0.27 (T20), 0.23 (T25), 0.19 (T30). Every value is below even the cheapest break-even in the table (0.43–0.71). A larger notional shrinks only the +0.10 margin; break-even is notional-independent.

## 7. Dynamic target study (selection day only)

- **Strength measure:** Stage A P(|move| ≥ 20 bps in 60 s).
- **Buckets:** cut at calibration-day quantiles p90 / p97 / p99 — weak, moderate, strong, very strong.
- **Target:** the best of T ∈ {15, 20, 25, 30} per bucket, chosen on the selection day.

| bucket | rows | T15 net | T20 net | T25 net | T30 net | chosen |
|---|---|---|---|---|---|---|
| weak (< p90) | 69,478 | −0.177 | −0.176 | −0.175 | −0.174 | **skip** |
| moderate (p90–p97) | 2,404 | −0.184 | −0.176 | −0.169 | −0.166 | **skip** |
| strong (p97–p99) | 106 | −0.218 | −0.210 | −0.216 | −0.223 | **skip** |
| very strong (≥ p99) | 5 | −0.205 | −0.320 | −0.320 | −0.320 | **skip** |

Every bucket and target is negative, so the validation-chosen mapping is "skip" everywhere. (Larger targets lose slightly less in the moderate bucket, consistent with "bigger move, same coin flip, fixed cost".)

## 8. Combined rule (Stage A thr → Stage B thr → cost/EV check)

- **Grid on the selection day:**
  - Stage A threshold at the calibration-day top 20 / 10 / 5 / 2 / 1 / 0.5%.
  - Stage B confidence 0.50 / 0.55 / 0.60 / 0.65 / 0.70.
  - Target set: dynamic {15, 20, 25, 30} or each single target.
  - That is 150 configurations per model. P(TP) of the predicted side is isotonic-calibrated on the calibration day; the EV gate is ≥ 0.10 USDT at 150 USDT taker.
- **Result:** **0 trades for every configuration**, two-stage LightGBM and two-stage XGBoost alike. The calibrated P(TP) never reaches the EV requirement, consistent with §5–6.
- The frozen configurations are exported (`models/v2/twostage_{lightgbm,xgboost}`) and can be served by the bot (`strategy/v2.TwoStageModel`; reject reasons `stage_a_no_large_move` / `stage_b_direction_weak`).
- **They were not replayed:** with 0 trades on validation and Stage B at chance, a test replay would only spend the untouched test period.

## 9. Comparison table (selection day; TEST untouched)

"Gated" is the production EV rule. "Forced top 1%" bypasses the EV gate and takes non-overlapping trades in each model's top 1% of predicted large moves, using the calibration-day threshold. It is a diagnostic only (target T25, S 8 bps, 150 USDT taker).

| Model | gated trades | forced trades | target hit rate (P(TP)) | large-move rate | direction acc. given move | net / trade (USDT) | PF | max DD (USDT) | total net (USDT) | inference median / p99 per decision |
|---|---|---|---|---|---|---|---|---|---|---|
| Direct LightGBM (18 models) | 0 | 12 | 0.17 | 0.83 | 0.60 (n=10) | −0.218 | 0.13 | 2.62 | −2.62 | 1.23 ms / 2.59 ms |
| Direct XGBoost (18 models) | 0 | 9 | 0.11 | 1.00 | 0.44 (n=9) | −0.246 | 0.08 | 2.41 | −2.21 | 4.54 ms / 8.04 ms |
| Direct MLP 64→32 (8 models) | 0 | 106 | 0.25 | 0.65 | 0.58 (n=69) | −0.148 | 0.26 | 15.71 | −15.71 | 0.47 ms / 0.71 ms |
| Two-stage LightGBM (8 models) | 0 | 3 | 0.33 | 1.00 | 1.00 (n=3) | −0.136 | 0.32 | 0.60 | −0.41 | 0.45 ms / 1.38 ms |
| Two-stage XGBoost (8 models) | 0 | 3 | 0.33 | 1.00 | 1.00 (n=3) | −0.136 | 0.32 | 0.60 | −0.41 | 1.29 ms / 3.78 ms |
| (V1 rule-based, locked test) | 10 on test | – | 0.00 | – | – | −0.141 | 0 | – | −1.41 | – |

- **T20 forced:** direct LightGBM 28 trades, −0.162/trade, PF 0.18. Direct XGBoost 10 trades, −0.137. Direct MLP 59 trades, −0.158 (t −6.2). Two-stage LightGBM 1 trade, −0.315. Two-stage XGBoost 6 trades, −0.161.
- **Direct-model selection-day AUC (long/short):** T20: LightGBM 0.767/0.707, XGBoost 0.768/0.708, MLP 0.751/0.696. T25: LightGBM 0.809/0.740, XGBoost 0.809/0.742, MLP 0.779/0.734. All three direct families converge on the same volatility signal.
- **Direct MLP picks:** 90% long. Its 0.58 direction accuracy is the selection day's upward drift inside those trades, not skill; it still loses −0.148 per trade.
- **Sample sizes:** the forced samples are tiny because the selection day was calmer than the calibration day. Larger slices (§5, thousands of rows) give the same answer with |t| > 8.
- **Latency:** none of the models meets the < 100 µs preference. The fastest full decisions are two-stage LightGBM (~0.45 ms; Stage B runs only when Stage A passes) and direct MLP (~0.47 ms). Direct XGBoost is slowest (~4.5 ms for 18 boosters).

## 10. Conclusions and recommendation

1. **Large-move edge: present and robust** (AUC 0.80–0.93; top-1% move rate ~92% at 20 bps / 60 s). It is volatility clustering, measured from realised volatility, spread and activity.
2. **Directional edge: absent at the trade horizon.** Stage B AUC is 0.49–0.52 at 60 s. A faint short-horizon (≤ 10 s) continuation from OFI/momentum (0.53–0.58) is far below the 0.70–0.85 P(TP) that break-even needs.
3. **Even perfect direction is not enough with an 8 bps stop**, because large-move states are two-sided (MAE > MFE). At these costs, a directional 60 s taker scalp on L1 data has no viable configuration.
4. Per instruction 10: **no Laya-MLX, no new model families, and no combined replay. The test days remain untouched for a future hypothesis.**

What could change the picture (proposals, none implemented):
- **True depth / queue data (L2 diff stream).** Direction in microstructure literature comes mostly from deeper-book and queue dynamics, which L1 cannot see. The L2 feature hooks (`features/l2_features.py`) and diff-depth recorder are ready. This needs a live recording period because the public archive has no depth for these dates.
- **Stops and targets that suit volatile states.** Wider symmetric stops (≥ 12–15 bps) with targets scaled to predicted volatility. Perfect-direction trades are break-even at S = 12, so this only helps alongside a real direction signal.
- **Maker-side structures that are paid for volatility** (both-side quoting with inventory limits) rather than for direction. This is a different strategy class with different risk; it would need its own design and replay study before any consideration.

---
*DRY_RUN=True throughout. No order was or can be placed by any V2 model.*
