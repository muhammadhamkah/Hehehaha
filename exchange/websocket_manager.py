"""Binance USDT-M WebSocket connections.

``StreamConnection`` maintains one combined-stream connection whose stream set can be
changed at runtime (SUBSCRIBE / UNSUBSCRIBE control messages). It reconnects with
exponential backoff, re-subscribes everything after reconnect, proactively recycles
the connection before Binance's 24h limit and reports connect/disconnect/heartbeat
events so the risk manager can detect stale or broken feeds.

``UserDataStream`` handles the private listenKey stream for live order updates.
"""
from __future__ import annotations

import asyncio
import itertools
import json
import logging
import random
import time
from typing import Awaitable, Callable, Iterable

import aiohttp

log = logging.getLogger(__name__)

MessageHandler = Callable[[str, object, int], None]
StatusHandler = Callable[[str], None]

MAX_STREAMS_PER_MSG = 50
CONTROL_MSG_INTERVAL_S = 0.25   # stay well under 10 incoming msgs/s


def now_ms() -> int:
    return int(time.time() * 1000)


class StreamConnection:
    def __init__(
        self,
        name: str,
        ws_base: str,
        on_message: MessageHandler,
        on_connect: StatusHandler | None = None,
        on_disconnect: StatusHandler | None = None,
        max_backoff_s: float = 30.0,
        max_age_s: float = 23 * 3600,
    ) -> None:
        self.name = name
        self.url = f"{ws_base.rstrip('/')}/stream"
        self.on_message = on_message
        self.on_connect = on_connect
        self.on_disconnect = on_disconnect
        self.max_backoff_s = max_backoff_s
        self.max_age_s = max_age_s
        self.streams: set[str] = set()
        self._ws: aiohttp.ClientWebSocketResponse | None = None
        self._ids = itertools.count(1)
        self._running = False
        self._ctrl_lock = asyncio.Lock()
        self.connected = asyncio.Event()
        self.last_msg_ms = 0
        self.n_messages = 0
        self.n_reconnects = 0

    # ------------------------------------------------------------------ control
    async def set_streams(self, streams: Iterable[str]) -> None:
        new = set(streams)
        add = sorted(new - self.streams)
        remove = sorted(self.streams - new)
        self.streams = new
        if self._ws is not None and not self._ws.closed:
            if remove:
                await self._control("UNSUBSCRIBE", remove)
            if add:
                await self._control("SUBSCRIBE", add)

    async def _control(self, method: str, streams: list[str]) -> None:
        async with self._ctrl_lock:
            for i in range(0, len(streams), MAX_STREAMS_PER_MSG):
                chunk = streams[i : i + MAX_STREAMS_PER_MSG]
                if self._ws is None or self._ws.closed:
                    return
                await self._ws.send_str(json.dumps({"method": method, "params": chunk, "id": next(self._ids)}))
                await asyncio.sleep(CONTROL_MSG_INTERVAL_S)

    # ------------------------------------------------------------------ loop
    async def run(self) -> None:
        self._running = True
        backoff = 1.0
        async with aiohttp.ClientSession(trust_env=True) as session:
            while self._running:
                started = time.monotonic()
                try:
                    async with session.ws_connect(self.url, heartbeat=20, max_msg_size=0,
                                                  autoping=True) as ws:
                        self._ws = ws
                        self.connected.set()
                        log.info("ws[%s] connected (%d streams)", self.name, len(self.streams))
                        if self.on_connect:
                            self.on_connect(self.name)
                        if self.streams:
                            asyncio.ensure_future(self._control("SUBSCRIBE", sorted(self.streams)))
                        backoff = 1.0
                        await self._read_loop(ws, started)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # noqa: BLE001 - reconnect on anything
                    log.warning("ws[%s] error: %s: %s", self.name, type(exc).__name__, exc)
                finally:
                    self._ws = None
                    self.connected.clear()
                    if self.on_disconnect and self._running:
                        self.on_disconnect(self.name)
                if not self._running:
                    break
                self.n_reconnects += 1
                delay = min(backoff, self.max_backoff_s) * (0.8 + 0.4 * random.random())
                log.info("ws[%s] reconnecting in %.1fs", self.name, delay)
                await asyncio.sleep(delay)
                backoff *= 2

    async def _read_loop(self, ws: aiohttp.ClientWebSocketResponse, started: float) -> None:
        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                ts = now_ms()
                self.last_msg_ms = ts
                self.n_messages += 1
                try:
                    payload = json.loads(msg.data)
                except json.JSONDecodeError:
                    continue
                if "stream" in payload and "data" in payload:
                    try:
                        self.on_message(payload["stream"], payload["data"], ts)
                    except Exception:  # noqa: BLE001 - never kill the socket on a handler bug
                        log.exception("ws[%s] handler error for %s", self.name, payload.get("stream"))
                elif payload.get("result") is None and "id" in payload:
                    pass  # subscription ack
                elif "error" in payload:
                    log.error("ws[%s] control error: %s", self.name, payload["error"])
            elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                break
            if time.monotonic() - started > self.max_age_s:
                log.info("ws[%s] recycling connection (age limit)", self.name)
                await ws.close()
                break

    async def stop(self) -> None:
        self._running = False
        if self._ws is not None:
            await self._ws.close()


class UserDataStream:
    """Private stream: ORDER_TRADE_UPDATE / ACCOUNT_UPDATE events for live trading."""

    def __init__(self, ws_base: str, get_listen_key: Callable[[], Awaitable[str]],
                 keepalive: Callable[[], Awaitable[None]], on_event: Callable[[dict], None],
                 on_disconnect: StatusHandler | None = None, on_connect: StatusHandler | None = None) -> None:
        self.ws_base = ws_base.rstrip("/")
        self.get_listen_key = get_listen_key
        self.keepalive = keepalive
        self.on_event = on_event
        self.on_disconnect = on_disconnect
        self.on_connect = on_connect
        self._running = False

    async def run(self) -> None:
        self._running = True
        backoff = 1.0
        async with aiohttp.ClientSession(trust_env=True) as session:
            while self._running:
                ka_task = None
                try:
                    key = await self.get_listen_key()
                    async with session.ws_connect(f"{self.ws_base}/ws/{key}", heartbeat=20) as ws:
                        if self.on_connect:
                            self.on_connect("user")
                        ka_task = asyncio.ensure_future(self._keepalive_loop())
                        backoff = 1.0
                        async for msg in ws:
                            if msg.type != aiohttp.WSMsgType.TEXT:
                                break
                            ev = json.loads(msg.data)
                            if ev.get("e") == "listenKeyExpired":
                                break
                            self.on_event(ev)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # noqa: BLE001
                    log.warning("user stream error: %s", exc)
                finally:
                    if ka_task:
                        ka_task.cancel()
                    if self.on_disconnect and self._running:
                        self.on_disconnect("user")
                if self._running:
                    await asyncio.sleep(min(backoff, 30.0))
                    backoff *= 2

    async def _keepalive_loop(self) -> None:
        while True:
            await asyncio.sleep(30 * 60)
            try:
                await self.keepalive()
            except Exception as exc:  # noqa: BLE001
                log.warning("listenKey keepalive failed: %s", exc)

    def stop(self) -> None:
        self._running = False
