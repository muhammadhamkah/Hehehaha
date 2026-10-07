"""Markdown rendering of the validation report."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

METRIC_COLS = ["trades", "win_rate", "net_pnl", "expectancy", "median_net", "profit_factor", "max_drawdown",
               "sharpe_per_trade", "t_stat", "fees", "slippage", "maker_fill_rate", "target_before_stop_rate"]


def _iso(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _table(rows: list[dict], cols: list[str]) -> str:
    if not rows:
        return "_none_\n"
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        out.append("| " + " | ".join("" if r.get(c) is None else str(r.get(c)) for c in cols) + " |")
    return "\n".join(out) + "\n"


def write_markdown(rep: dict[str, Any], path: str) -> None:
    L: list[str] = []
    w = L.append
    w("# Strategy Validation Report\n")
    src = rep.get("events")
    w(f"Dataset: `{src}`  ·  symbols: {', '.join(rep.get('symbols') or []) or 'scanner-selected'}\n")
    for seg, (s, e) in rep["bounds"].items():
        w(f"- **{seg.upper()}**: {_iso(s)} → {_iso(e)} ({(e - s) / 3.6e6:.2f} h)")
    w("")
    w(f"## Verdict\n\n> **{rep['verdict']}**\n")
    t = rep.get("test", {})
    if "status" in t:
        w(f"Test period: {t['status']}\n")

    b = rep.get("baseline_train", {})
    w("## 1. Data quality (TRAIN replay)\n")
    w(f"Malformed messages by kind: `{b.get('feed_health') or 'none'}`\n")

    w("## 2. Opportunity decisions (baseline config, TRAIN)\n")
    w(f"Evaluations: {b.get('evaluations')}\n")
    w(_table(b.get("decisions", []), ["category", "reason", "count", "share"]))

    w("## 3. Baseline performance (TRAIN, default config, 100 ms latency)\n")
    w(_table([b.get("metrics", {})], METRIC_COLS))

    f = b.get("features", {})
    w("## 4. Feature analysis — is there predictive structure? (TRAIN)\n")
    w(f"**{f.get('verdict')}**\n\n{f.get('note', '')}\n")
    rows = sorted((r for r in f.get("rows", []) if r.get("t") is not None),
                  key=lambda r: -abs(r["t"]))[:30]
    w(_table(rows, ["family", "feature", "horizon_s", "directional", "n_thinned", "ic", "t", "ic_first_half",
                    "ic_second_half", "significant", "aligned_net_bps_mean", "top_quintile_net_bps",
                    "aligned_tp_rate"]))

    c = b.get("calibration", {})
    w("## 5. Confidence calibration (TRAIN)\n")
    w(f"Verdict: **{c.get('verdict')}**  ·  Brier (target-before-stop): {c.get('brier_target_before_stop')}\n")
    w("Directional confidence vs realised direction (10 s, thinned):\n")
    w(_table(c.get("direction", []), ["bucket", "n", "predicted", "realised", "ci95", "consistent"]))
    w("Predicted P(target before stop) vs realised (each opportunity's own target/stop):\n")
    w(_table(c.get("target_before_stop", []), ["bucket", "n", "predicted", "realised", "ci95", "consistent"]))

    m = b.get("min_edge", {})
    w("## 6. Minimum tradable edge (mid move needed for ≥ min net profit, recorded books)\n")
    w(_table(m.get("by_notional", []), ["notional", "entry", "samples", "required_bps_p25", "required_bps_median",
                                         "required_bps_p75", "required_bps_p90", "unfillable_share",
                                         "avg_spread_bps", "avg_exit_slip_bps",
                                         "share_of_60s_moves_exceeding_median_required"]))

    w("## 7. Per-symbol results (baseline, TRAIN)\n")
    w(_table(b.get("per_symbol", []), ["symbol", "trades", "net_pnl", "expectancy", "median_net", "win_rate",
                                       "profit_factor", "avg_entry_spread_bps", "avg_market_spread_bps",
                                       "avg_slippage_bps", "avg_hold_s"]))

    w("## 8. Parameter sweep (TRAIN) — ranked by expectancy per trade after costs\n")
    sw = rep.get("sweep_train", [])
    w(f"{len(sw)} configurations; eligible = at least {rep.get('min_trades')} trades.\n")
    if rep.get("ineffective_params"):
        w(f"⚠️ Parameters with **no effect** in the tested range (another constraint binds first): "
          f"`{rep['ineffective_params']}`\n")
    w(_table([{"params": s["params"], "eligible": s["eligible"], **s["train"]} for s in sw[:15]],
             ["params", "eligible"] + METRIC_COLS))

    w("## 9. Validation of top TRAIN configurations\n")
    w(_table([{"params": s["params"], **s.get("validation", {})} for s in rep.get("validation", [])],
             ["params"] + METRIC_COLS))
    ch = rep.get("chosen")
    w(f"Chosen: `{ch['params'] if ch else None}`\n")
    if rep.get("validation_latency"):
        w("Latency robustness of the chosen config (VALIDATION):\n")
        w(_table([{"latency_ms": k, **v} for k, v in rep["validation_latency"].items()], ["latency_ms"] + METRIC_COLS))

    w("## 10. Final out-of-sample TEST\n")
    if isinstance(t, dict) and any(isinstance(k, int) or str(k).isdigit() for k in t):
        w(_table([{"latency_ms": k, **v} for k, v in t.items()], ["latency_ms"] + METRIC_COLS))
        w("Per-symbol (TEST, 100 ms):\n")
        w(_table(rep.get("test_per_symbol", []), ["symbol", "trades", "net_pnl", "expectancy", "win_rate",
                                                  "profit_factor", "avg_entry_spread_bps", "avg_slippage_bps",
                                                  "avg_hold_s"]))
        tc = rep.get("test_calibration", {})
        w(f"Calibration on TEST: **{tc.get('verdict')}**\n")
        w(_table(tc.get("target_before_stop", []), ["bucket", "n", "predicted", "realised", "ci95", "consistent"]))
    else:
        w(f"{t.get('status') or t.get('skipped') or 'not run'}\n")

    w("## Caveats\n")
    w("- Simulated execution: queue position is approximated from visible size and trade prints; "
      "hidden liquidity, queue jumping and exchange-side latency variance are not modelled.")
    w("- Results apply only to the recorded period, symbols and market regime.")
    w("- L1 (archive) datasets cannot model depth-based slippage beyond the top level.")
    w("- Positive train/validation results that fail on TEST indicate overfitting, not an edge.")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
