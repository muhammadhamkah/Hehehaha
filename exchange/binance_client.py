"""Async Binance USDT-M futures REST client (aiohttp).

Public endpoints are always available. Signed / order endpoints require API keys and
order placement additionally refuses to run unless ``allow_trading`` is True, which
the bot only sets when :meth:`BotConfig.live_trading_enabled` is True.
"""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import logging
import time
from typing import Any, Callable
from urllib.parse import urlencode

import aiohttp

from config import ExchangeConfig
from exchange.models import SymbolInfo

log = logging.getLogger(__name__)


class BinanceAPIError(Exception):
    def __init__(self, status: int, code: int | None, msg: str) -> None:
        super().__init__(f"HTTP {status} code={code} {msg}")
        self.status = status
        self.code = code
        self.msg = msg


class TradingDisabledError(RuntimeError):
    pass


class BinanceFuturesClient:
    def __init__(self, cfg: ExchangeConfig, allow_trading: bool = False,
                 on_error: Callable[[str], None] | None = None) -> None:
        self.cfg = cfg
        self.base = cfg.rest_url
        self.allow_trading = allow_trading
        self.on_error = on_error
        self._session: aiohttp.ClientSession | None = None
        self._time_offset_ms = 0
        self.used_weight_1m = 0
        self._backoff_until = 0.0

    async def __aenter__(self) -> "BinanceFuturesClient":
        await self.start()
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()

    async def start(self) -> None:
        if self._session is None:
            timeout = aiohttp.ClientTimeout(total=self.cfg.request_timeout_s)
            headers = {"X-MBX-APIKEY": self.cfg.api_key} if self.cfg.api_key else {}
            self._session = aiohttp.ClientSession(timeout=timeout, headers=headers, trust_env=True)

    async def close(self) -> None:
        if self._session is not None:
            await self._session.close()
            self._session = None

    # ------------------------------------------------------------------ core
    def _sign(self, params: dict[str, Any]) -> str:
        params["timestamp"] = int(time.time() * 1000) + self._time_offset_ms
        params["recvWindow"] = self.cfg.recv_window_ms
        query = urlencode(params)
        sig = hmac.new(self.cfg.api_secret.encode(), query.encode(), hashlib.sha256).hexdigest()
        return f"{query}&signature={sig}"

    async def request(self, method: str, path: str, params: dict[str, Any] | None = None,
                      signed: bool = False) -> Any:
        await self.start()
        assert self._session is not None
        now = time.monotonic()
        if now < self._backoff_until:
            await asyncio.sleep(self._backoff_until - now)
        params = {k: v for k, v in (params or {}).items() if v is not None}
        if signed:
            if not (self.cfg.api_key and self.cfg.api_secret):
                raise TradingDisabledError("signed endpoint requires API key/secret")
            query = self._sign(params)
        else:
            query = urlencode(params)
        url = f"{self.base}{path}" + (f"?{query}" if query else "")
        try:
            async with self._session.request(method, url) as resp:
                w = resp.headers.get("X-MBX-USED-WEIGHT-1M")
                if w is not None:
                    self.used_weight_1m = int(w)
                    if self.used_weight_1m > self.cfg.max_weight_per_min * 0.9:
                        self._backoff_until = time.monotonic() + 5.0
                data = await resp.json(content_type=None)
                if resp.status >= 400:
                    code = data.get("code") if isinstance(data, dict) else None
                    msg = data.get("msg", str(data)) if isinstance(data, dict) else str(data)
                    if resp.status in (418, 429):
                        retry = float(resp.headers.get("Retry-After", "10"))
                        self._backoff_until = time.monotonic() + retry
                    if code == -1021:  # timestamp outside recvWindow
                        asyncio.ensure_future(self.sync_time())
                    raise BinanceAPIError(resp.status, code, msg)
                return data
        except BinanceAPIError as exc:
            if self.on_error:
                self.on_error(str(exc))
            raise
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            if self.on_error:
                self.on_error(f"{type(exc).__name__}: {exc}")
            raise

    # ------------------------------------------------------------------ public
    async def sync_time(self) -> int:
        data = await self.request("GET", "/fapi/v1/time")
        self._time_offset_ms = int(data["serverTime"]) - int(time.time() * 1000)
        return self._time_offset_ms

    async def exchange_info(self) -> dict:
        return await self.request("GET", "/fapi/v1/exchangeInfo")

    async def perpetual_symbols(self, quote: str = "USDT") -> dict[str, SymbolInfo]:
        info = await self.exchange_info()
        out: dict[str, SymbolInfo] = {}
        for s in info["symbols"]:
            if s.get("contractType") != "PERPETUAL" or s.get("quoteAsset") != quote or s.get("status") != "TRADING":
                continue
            filters = {f["filterType"]: f for f in s["filters"]}
            lot = filters.get("LOT_SIZE", {})
            out[s["symbol"]] = SymbolInfo(
                symbol=s["symbol"],
                tick_size=float(filters["PRICE_FILTER"]["tickSize"]),
                step_size=float(lot.get("stepSize", 0.001)),
                min_qty=float(lot.get("minQty", 0.0)),
                max_qty=float(filters.get("MARKET_LOT_SIZE", lot).get("maxQty", 1e12)),
                min_notional=float(filters.get("MIN_NOTIONAL", {}).get("notional", 5.0)),
            )
        return out

    async def ticker_24h(self) -> list[dict]:
        return await self.request("GET", "/fapi/v1/ticker/24hr")

    async def depth(self, symbol: str, limit: int = 1000) -> dict:
        return await self.request("GET", "/fapi/v1/depth", {"symbol": symbol, "limit": limit})

    async def book_ticker(self, symbol: str | None = None) -> Any:
        return await self.request("GET", "/fapi/v1/ticker/bookTicker", {"symbol": symbol})

    # ------------------------------------------------------------------ account
    async def account(self) -> dict:
        return await self.request("GET", "/fapi/v2/account", signed=True)

    async def position_risk(self, symbol: str | None = None) -> list[dict]:
        return await self.request("GET", "/fapi/v2/positionRisk", {"symbol": symbol}, signed=True)

    async def commission_rate(self, symbol: str) -> tuple[float, float]:
        d = await self.request("GET", "/fapi/v1/commissionRate", {"symbol": symbol}, signed=True)
        return float(d["makerCommissionRate"]), float(d["takerCommissionRate"])

    async def set_leverage(self, symbol: str, leverage: int) -> dict:
        self._require_trading()
        return await self.request("POST", "/fapi/v1/leverage", {"symbol": symbol, "leverage": leverage}, signed=True)

    async def set_margin_type(self, symbol: str, margin_type: str) -> Any:
        self._require_trading()
        try:
            return await self.request("POST", "/fapi/v1/marginType",
                                      {"symbol": symbol, "marginType": margin_type}, signed=True)
        except BinanceAPIError as exc:
            if exc.code == -4046:  # no need to change margin type
                return None
            raise

    # ------------------------------------------------------------------ orders
    def _require_trading(self) -> None:
        if not self.allow_trading:
            raise TradingDisabledError("live trading is disabled (dry run)")

    async def new_order(self, symbol: str, side: str, type_: str, quantity: float,
                        price: float | None = None, time_in_force: str | None = None,
                        reduce_only: bool = False, client_id: str | None = None) -> dict:
        self._require_trading()
        params: dict[str, Any] = {
            "symbol": symbol,
            "side": side,
            "type": type_,
            "quantity": _fmt(quantity),
            "newClientOrderId": client_id,
            "newOrderRespType": "RESULT",
        }
        if type_ == "LIMIT":
            params["price"] = _fmt(price or 0.0)
            params["timeInForce"] = time_in_force or "GTC"
        if reduce_only:
            params["reduceOnly"] = "true"
        return await self.request("POST", "/fapi/v1/order", params, signed=True)

    async def cancel_order(self, symbol: str, client_id: str) -> dict:
        self._require_trading()
        return await self.request("DELETE", "/fapi/v1/order",
                                  {"symbol": symbol, "origClientOrderId": client_id}, signed=True)

    async def get_order(self, symbol: str, client_id: str) -> dict:
        return await self.request("GET", "/fapi/v1/order",
                                  {"symbol": symbol, "origClientOrderId": client_id}, signed=True)

    async def user_trades(self, symbol: str, order_id: str) -> list[dict]:
        return await self.request("GET", "/fapi/v1/userTrades",
                                  {"symbol": symbol, "orderId": order_id}, signed=True)

    async def cancel_all(self, symbol: str) -> Any:
        self._require_trading()
        return await self.request("DELETE", "/fapi/v1/allOpenOrders", {"symbol": symbol}, signed=True)

    # ------------------------------------------------------------------ user data stream
    async def new_listen_key(self) -> str:
        d = await self.request("POST", "/fapi/v1/listenKey")
        return d["listenKey"]

    async def keepalive_listen_key(self) -> None:
        await self.request("PUT", "/fapi/v1/listenKey")


def _fmt(x: float) -> str:
    s = f"{x:.10f}".rstrip("0").rstrip(".")
    return s or "0"
