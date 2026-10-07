"""Performance statistics. The headline question: is the strategy profitable AFTER
fees and slippage?

CLI:  python -m analytics.performance --db data/microstructure.sqlite [--mode paper]
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from typing import Any, Iterable

from utils.mathx import mean, stdev

CONF_BUCKETS = (0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9, 1.01)


def _group_stats(trades: list[dict[str, Any]]) -> dict[str, float]:
    nets = [t["net_pnl"] for t in trades]
    return {
        "trades": len(trades),
        "net_pnl": round(sum(nets), 6),
        "avg_net": round(mean(nets), 6),
        "win_rate": round(sum(1 for x in nets if x > 0) / len(nets), 4) if nets else 0.0,
    }


def max_drawdown(pnls: Iterable[float]) -> float:
    peak = cum = dd = 0.0
    for p in pnls:
        cum += p
        peak = max(peak, cum)
        dd = max(dd, peak - cum)
    return dd


def compute_performance(trades: list[dict[str, Any]]) -> dict[str, Any]:
    trades = sorted(trades, key=lambda t: t["entry_ts_ms"])
    n = len(trades)
    if n == 0:
        return {"trades": 0, "verdict": "NO TRADES"}
    nets = [t["net_pnl"] for t in trades]
    gross = [t["gross_pnl"] for t in trades]
    fees = sum(t["entry_fee"] + t["exit_fee"] for t in trades)
    slippage = sum(t.get("actual_slippage_usdt") or 0.0 for t in trades)
    est_slip = sum(t.get("est_slippage_usdt") or 0.0 for t in trades)
    winners = [x for x in nets if x > 0]
    losers = [x for x in nets if x <= 0]
    gross_win = sum(winners)
    gross_loss = -sum(losers)
    sd = stdev(nets)
    avg = mean(nets)
    t_stat = avg / (sd / math.sqrt(n)) if sd > 0 and n > 1 else 0.0
    maker_entries = sum(1 for t in trades if t.get("entry_maker"))
    maker_exits = sum(1 for t in trades if t.get("exit_maker"))

    by_symbol: dict[str, list] = defaultdict(list)
    by_reason: dict[str, list] = defaultdict(list)
    by_conf: dict[str, list] = defaultdict(list)
    for t in trades:
        by_symbol[t["symbol"]].append(t)
        by_reason[t["exit_reason"]].append(t)
        c = t.get("signal_confidence") or 0.0
        for lo, hi in zip(CONF_BUCKETS, CONF_BUCKETS[1:]):
            if lo <= c < hi:
                by_conf[f"{lo:.2f}-{min(hi, 1.0):.2f}"].append(t)
                break

    net_total = sum(nets)
    if n < 30:
        verdict = "INSUFFICIENT SAMPLE (<30 trades)"
    elif net_total > 0 and t_stat > 2.0:
        verdict = "PROFITABLE AFTER COSTS (t>2)"
    elif net_total > 0:
        verdict = "NET POSITIVE BUT NOT STATISTICALLY SIGNIFICANT"
    else:
        verdict = "NOT PROFITABLE AFTER COSTS"

    return {
        "trades": n,
        "win_rate": round(len(winners) / n, 4),
        "gross_pnl": round(sum(gross), 6),
        "net_pnl": round(net_total, 6),
        "fees_paid": round(fees, 6),
        "slippage_cost_actual": round(slippage, 6),
        "slippage_cost_estimated": round(est_slip, 6),
        "avg_net_per_trade": round(avg, 6),
        "avg_winner": round(mean(winners), 6) if winners else 0.0,
        "avg_loser": round(mean(losers), 6) if losers else 0.0,
        "profit_factor": round(gross_win / gross_loss, 4) if gross_loss > 0 else float("inf"),
        "expectancy": round(avg, 6),
        "max_drawdown": round(max_drawdown(nets), 6),
        "sharpe_per_trade": round(avg / sd, 4) if sd > 0 else 0.0,
        "t_stat": round(t_stat, 3),
        "avg_holding_s": round(mean([t["holding_s"] for t in trades]), 3),
        "maker_entry_pct": round(maker_entries / n, 4),
        "maker_exit_pct": round(maker_exits / n, 4),
        "gross_minus_costs_check": round(sum(gross) - fees, 6),
        "profitable_after_costs": net_total > 0,
        "by_symbol": {k: _group_stats(v) for k, v in sorted(by_symbol.items())},
        "by_exit_reason": {k: _group_stats(v) for k, v in sorted(by_reason.items())},
        "by_confidence": {k: _group_stats(v) for k, v in sorted(by_conf.items())},
        "verdict": verdict,
    }


def format_report(perf: dict[str, Any]) -> str:
    if perf.get("trades", 0) == 0:
        return "No trades recorded."
    lines = ["=" * 64, "PERFORMANCE REPORT", "=" * 64]
    for k, v in perf.items():
        if isinstance(v, dict):
            continue
        lines.append(f"{k:<28} {v}")
    for section in ("by_symbol", "by_exit_reason", "by_confidence"):
        lines.append("-" * 64)
        lines.append(section)
        for key, st in perf[section].items():
            lines.append(f"  {key:<22} n={st['trades']:<5} net={st['net_pnl']:<10} "
                         f"avg={st['avg_net']:<10} win={st['win_rate']}")
    lines.append("=" * 64)
    lines.append(f"VERDICT: {perf['verdict']}")
    return "\n".join(lines)


def load_trades(db_path: str, mode: str | None = None) -> list[dict[str, Any]]:
    from data.database import Database

    db = Database(db_path)
    try:
        sql = "SELECT * FROM trades"
        params: tuple = ()
        if mode:
            sql += " WHERE mode = ?"
            params = (mode,)
        return db.query(sql, params)
    finally:
        db.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="Trade performance report")
    ap.add_argument("--db", default="data/microstructure.sqlite")
    ap.add_argument("--mode", default=None, help="filter: paper | live")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    perf = compute_performance(load_trades(args.db, args.mode))
    print(json.dumps(perf, indent=2) if args.json else format_report(perf))


if __name__ == "__main__":
    main()
