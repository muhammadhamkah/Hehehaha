"""Human-readable formatting of replay / research results."""
from __future__ import annotations

from typing import Any


def format_replay(res: dict[str, Any]) -> str:
    if "error" in res:
        return f"replay error: {res['error']}"
    p = res["performance"]
    r = res["replay"]
    e = res["execution"]
    lines = [
        "=" * 72,
        f"REPLAY {res.get('label') or ''}  {res['period']['from']} -> {res['period']['to']} "
        f"({res['period']['hours']} h)",
        f"events={r['events']} delivered={r['delivered']} skipped_unsubscribed={r['skipped_unsubscribed']} "
        f"late={r['reader_late']} clamped={r['reader_clamped']} wall={r['wall_s']}s speedup={r['speedup']}x",
        "-" * 72,
    ]
    if p.get("trades", 0) == 0:
        lines.append("NO TRADES")
    else:
        keys = ["trades", "win_rate", "net_pnl", "gross_pnl", "fees_paid", "slippage_cost_actual",
                "avg_net_per_trade", "median_net", "profit_factor", "max_drawdown", "sharpe_per_trade",
                "t_stat", "avg_holding_s", "maker_entry_pct", "target_before_stop_rate"]
        for k in keys:
            lines.append(f"  {k:<26} {p.get(k)}")
    lines.append(f"  maker fill rate (any/full) {e['maker_fill_rate_any']} / {e['maker_fill_rate_full']} "
                 f"of {e['maker_orders']} maker orders; post-only rejects {e['maker_post_only_rejected']}")
    lines.append("-" * 72)
    lines.append(f"DECISIONS over {res['evaluations']} evaluations (top 12):")
    for d in res["decisions"][:12]:
        lines.append(f"  {d['count']:>9}  {d['share']:>8.2%}  {d['category']}")
    mal = res["feed_health"]["malformed_by_kind"]
    lines.append(f"feed malformed: {mal or 'none'}")
    lines.append(f"VERDICT: {p.get('verdict')}")
    lines.append("=" * 72)
    return "\n".join(lines)
