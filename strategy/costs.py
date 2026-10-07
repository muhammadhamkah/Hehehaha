"""Execution-cost model: fees + slippage, in USDT and bps.

All price moves in the strategy are measured on the MID price. Costs are therefore
expressed relative to mid:

  * taker entry/exit: book-walk average price vs mid (half-spread + impact)
    + latency slippage + safety buffer
  * maker entry: adverse-selection allowance (the half-spread we nominally capture is
    NOT credited, which keeps the estimate conservative)

Exits are always assumed to be taker (worst case).
"""
from __future__ import annotations

from dataclasses import dataclass

from config import CostConfig
from market_data.orderbook import OrderBook


@dataclass(frozen=True)
class CostEstimate:
    notional: float
    entry_maker: bool
    entry_fee: float
    exit_fee: float
    entry_slippage_bps: float
    exit_slippage_bps: float
    entry_slippage_usdt: float
    exit_slippage_usdt: float

    @property
    def fees(self) -> float:
        return self.entry_fee + self.exit_fee

    @property
    def slippage_usdt(self) -> float:
        return self.entry_slippage_usdt + self.exit_slippage_usdt

    @property
    def total_usdt(self) -> float:
        return self.fees + self.slippage_usdt

    @property
    def total_bps(self) -> float:
        return self.total_usdt / self.notional * 1e4 if self.notional else float("inf")


class CostModel:
    def __init__(self, cfg: CostConfig) -> None:
        self.cfg = cfg
        self.maker_fee = cfg.maker_fee
        self.taker_fee = cfg.taker_fee

    def set_commission(self, maker: float, taker: float) -> None:
        """Override with the account's actual commission rates (live)."""
        self.maker_fee = maker
        self.taker_fee = taker

    def fee(self, notional: float, maker: bool) -> float:
        return abs(notional) * (self.maker_fee if maker else self.taker_fee)

    def taker_slippage_bps(self, book: OrderBook, side: str, notional: float) -> float:
        walk = book.slippage_bps(side, notional)
        return walk + self.cfg.latency_slippage_bps + self.cfg.extra_slippage_bps

    def estimate(self, book: OrderBook, direction: int, notional: float, entry_maker: bool) -> CostEstimate:
        entry_side = "BUY" if direction > 0 else "SELL"
        exit_side = "SELL" if direction > 0 else "BUY"
        if entry_maker:
            entry_bps = self.cfg.maker_adverse_selection_bps
        else:
            entry_bps = self.taker_slippage_bps(book, entry_side, notional)
        exit_bps = self.taker_slippage_bps(book, exit_side, notional)
        return CostEstimate(
            notional=notional,
            entry_maker=entry_maker,
            entry_fee=self.fee(notional, entry_maker),
            exit_fee=self.fee(notional, False),
            entry_slippage_bps=entry_bps,
            exit_slippage_bps=exit_bps,
            entry_slippage_usdt=notional * entry_bps / 1e4,
            exit_slippage_usdt=notional * exit_bps / 1e4,
        )

    def required_move_bps(self, costs: CostEstimate, min_net_usdt: float) -> float:
        """Mid move (bps) needed so that the trade nets ``min_net_usdt`` after costs."""
        if costs.notional <= 0:
            return float("inf")
        return (min_net_usdt + costs.total_usdt) / costs.notional * 1e4
