"""Opportunity decision taxonomy: maps internal reasons to the categories used in reports."""
from __future__ import annotations

CATEGORIES = {
    "enter": "ENTERED",
    "would_enter": "ENTERED (record mode)",
    "low_confidence": "probability too low",
    "p_target_too_low": "target-before-stop probability too low",
    "expected_net_below_min": "expected net < minimum",
    "target_net_below_min": "target too small (net if target < minimum)",
    "required_move_too_large": "required move exceeds max target",
    "spread_too_wide": "spread too high",
    "insufficient_depth": "insufficient depth",
    "slippage_too_high": "excessive estimated slippage",
    "stale_data": "stale market data",
    "flow_not_confirming": "order-flow disagreement",
    "book_imbalance_below_min": "book imbalance below threshold",
    "flow_imbalance_below_min": "flow imbalance below threshold",
    "thin_tape": "thin tape (too few trades)",
    "abnormal_book_one_sided": "abnormal liquidity (one-sided book)",
    "abnormal_volatility": "abnormal volatility",
    "liquidity_withdrawal": "abnormal liquidity (withdrawal)",
    "size_below_minimum": "size below minimum",
    "risk_per_trade_exceeded": "risk per trade exceeded",
    "no_features": "no features (warming up)",
    "risk:symbol_cooldown": "cooldown",
}


def categorize(reason: str) -> str:
    if reason in CATEGORIES:
        return CATEGORIES[reason]
    if reason.startswith("risk:halted"):
        return "risk halt"
    if reason.startswith("risk:"):
        return f"risk limit ({reason[5:]})"
    if reason.startswith("entered") or reason.startswith("entry_failed"):
        return reason
    return reason


def decision_table(decisions: dict[str, int]) -> list[dict]:
    total = sum(decisions.values()) or 1
    rows = [{"reason": k, "category": categorize(k), "count": v, "share": round(v / total, 5)}
            for k, v in decisions.items()]
    return sorted(rows, key=lambda r: -r["count"])
