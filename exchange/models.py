"""Exchange-agnostic order models shared by the paper simulator and live execution."""
from __future__ import annotations

import asyncio
import itertools
import math
import time
from dataclasses import dataclass, field
from enum import Enum

_seq = itertools.count(1)


def new_client_id(prefix: str = "msb") -> str:
    # Binance limits newClientOrderId to 36 chars matching ^[.A-Z:/a-z0-9_-]{1,36}$
    return f"{prefix}_{int(time.time() * 1000) % 10**10}_{next(_seq)}"


class OrderStatus(str, Enum):
    PENDING_NEW = "PENDING_NEW"     # created locally, not yet acknowledged
    NEW = "NEW"                     # acknowledged / resting
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELED = "CANCELED"
    EXPIRED = "EXPIRED"             # IOC remainder, GTX that would have taken, TTL
    REJECTED = "REJECTED"

    @property
    def done(self) -> bool:
        return self in (OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.EXPIRED, OrderStatus.REJECTED)


@dataclass
class Fill:
    ts_ms: int
    price: float
    qty: float
    fee: float
    maker: bool


@dataclass
class Order:
    symbol: str
    side: str                        # "BUY" | "SELL"
    type: str                        # "LIMIT" | "MARKET"
    qty: float
    price: float = 0.0
    tif: str = "GTC"                 # "GTX" (post-only) | "IOC" | "GTC"
    reduce_only: bool = False
    client_id: str = field(default_factory=new_client_id)
    exchange_id: str = ""
    status: OrderStatus = OrderStatus.PENDING_NEW
    fills: list[Fill] = field(default_factory=list)
    submit_ts_ms: int = 0
    ack_ts_ms: int = 0
    done_ts_ms: int = 0
    reject_reason: str = ""
    events: list[tuple[int, str]] = field(default_factory=list)
    _changed: asyncio.Event | None = field(default=None, repr=False)

    # ------------------------------------------------------------------ derived
    @property
    def filled_qty(self) -> float:
        return sum(f.qty for f in self.fills)

    @property
    def remaining(self) -> float:
        return max(self.qty - self.filled_qty, 0.0)

    @property
    def avg_price(self) -> float:
        q = self.filled_qty
        return sum(f.price * f.qty for f in self.fills) / q if q else 0.0

    @property
    def fee(self) -> float:
        return sum(f.fee for f in self.fills)

    @property
    def maker_qty(self) -> float:
        return sum(f.qty for f in self.fills if f.maker)

    @property
    def is_done(self) -> bool:
        return self.status.done

    # ------------------------------------------------------------------ events
    @property
    def changed(self) -> asyncio.Event:
        if self._changed is None:
            self._changed = asyncio.Event()
        return self._changed

    def mark(self, status: OrderStatus, ts_ms: int, event: str = "") -> None:
        self.status = status
        self.events.append((ts_ms, event or status.value))
        if status == OrderStatus.NEW and not self.ack_ts_ms:
            self.ack_ts_ms = ts_ms
        if status.done:
            self.done_ts_ms = ts_ms
        self.changed.set()

    def add_fill(self, fill: Fill) -> None:
        fill.qty = min(fill.qty, self.remaining)
        if fill.qty <= 0:
            return
        self.fills.append(fill)
        status = OrderStatus.FILLED if self.remaining <= 1e-12 else OrderStatus.PARTIALLY_FILLED
        self.mark(status, fill.ts_ms, f"fill {fill.qty}@{fill.price}{' M' if fill.maker else ' T'}")


@dataclass
class SymbolInfo:
    symbol: str
    tick_size: float
    step_size: float
    min_qty: float
    min_notional: float = 5.0
    max_qty: float = 1e12

    def round_price(self, price: float, mode: str = "nearest") -> float:
        ticks = price / self.tick_size
        if mode == "down":
            n = math.floor(ticks + 1e-9)
        elif mode == "up":
            n = math.ceil(ticks - 1e-9)
        else:
            n = round(ticks)
        return round(n * self.tick_size, _decimals(self.tick_size))

    def round_qty(self, qty: float) -> float:
        n = math.floor(qty / self.step_size + 1e-9)
        return round(min(n * self.step_size, self.max_qty), _decimals(self.step_size))


def _decimals(step: float) -> int:
    s = f"{step:.12f}".rstrip("0")
    return len(s.split(".")[1]) if "." in s else 0
