"""Local mock of Binance USDT-M public REST + combined WebSocket streams (tests only)."""
from __future__ import annotations

import asyncio
import json
import time

from aiohttp import WSMsgType, web

SYMS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]


class MockBinanceServer:
    def __init__(self) -> None:
        self.conns: list = []
        self.connections_total = 0
        self.uid = 1000
        self.agg = {s: 1 for s in SYMS}
        self.seq: dict[str, int] = {}     # per-symbol order-book update id

    # ------------------------------------------------------------------ REST
    async def time(self, request):
        return web.json_response({"serverTime": int(time.time() * 1000)})

    async def exchange_info(self, request):
        return web.json_response({"symbols": [{
            "symbol": s, "contractType": "PERPETUAL", "quoteAsset": "USDT", "status": "TRADING",
            "filters": [{"filterType": "PRICE_FILTER", "tickSize": "0.10"},
                        {"filterType": "LOT_SIZE", "stepSize": "0.001", "minQty": "0.001"},
                        {"filterType": "MIN_NOTIONAL", "notional": "5"}]} for s in SYMS] * 20})

    async def ticker(self, request):
        return web.json_response([{"symbol": s, "quoteVolume": str(1e9 - i)} for i, s in enumerate(SYMS)])

    async def depth(self, request):
        sym = request.query.get("symbol", "BTCUSDT")
        return web.json_response({"lastUpdateId": self.seq.get(sym, 1000), "bids": [["100.0", "5"], ["99.9", "5"]],
                                  "asks": [["100.1", "5"], ["100.2", "5"]]})

    # ------------------------------------------------------------------ WS
    def payload(self, stream: str) -> dict | list:
        now = int(time.time() * 1000)
        if stream == "!ticker@arr":
            return [{"e": "24hrTicker", "E": now, "s": s, "c": "100.0", "q": "1000000", "n": 5} for s in SYMS]
        if stream == "!bookTicker":
            return {"e": "bookTicker", "u": self.uid, "E": now, "T": now, "s": "BTCUSDT",
                    "b": "100.0", "B": "1", "a": "100.1", "A": "1"}
        sym, _, kind = stream.partition("@")
        S = sym.upper()
        if kind == "aggTrade":
            self.agg[S] = self.agg.get(S, 0) + 1
            a = self.agg[S]
            return {"e": "aggTrade", "E": now, "s": S, "a": a, "p": "100.0", "q": "0.5", "f": a, "l": a,
                    "T": now - 1, "m": a % 2 == 0}
        if kind == "bookTicker":
            return {"e": "bookTicker", "u": self.uid, "E": now, "T": now, "s": S,
                    "b": "100.0", "B": "1", "a": "100.1", "A": "1"}
        prev = self.seq.get(S, 1000)
        if kind.split("@")[0] != "depth":
            # partial depth snapshot reflects the current book; it does not create updates
            return {"e": "depthUpdate", "E": now, "T": now, "s": S, "U": prev - 2, "u": prev, "pu": prev - 3,
                    "b": [["100.0", "5"], ["99.9", "4"]], "a": [["100.1", "5"], ["100.2", "4"]]}
        self.seq[S] = prev + 3
        return {"e": "depthUpdate", "E": now, "T": now, "s": S, "U": prev + 1, "u": prev + 3, "pu": prev,
                "b": [["100.0", "5"], ["99.9", "4"]], "a": [["100.1", "5"], ["100.2", "4"]]}

    async def ws(self, request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        subs: set = set()
        entry = (ws, subs)
        self.conns.append(entry)
        self.connections_total += 1
        sender = asyncio.ensure_future(self._send(ws, subs))
        try:
            async for msg in ws:
                if msg.type != WSMsgType.TEXT:
                    continue
                req = json.loads(msg.data)
                if req["method"] == "SUBSCRIBE":
                    subs.update(req["params"])
                elif req["method"] == "UNSUBSCRIBE":
                    subs.difference_update(req["params"])
                await ws.send_str(json.dumps({"result": None, "id": req["id"]}))
        finally:
            sender.cancel()
            if entry in self.conns:
                self.conns.remove(entry)
        return ws

    async def _send(self, ws, subs):
        while not ws.closed:
            for s in sorted(subs):
                await ws.send_str(json.dumps({"stream": s, "data": self.payload(s)}))
            await asyncio.sleep(0.05)

    async def start(self) -> tuple[web.AppRunner, str]:
        app = web.Application()
        app.router.add_get("/fapi/v1/time", self.time)
        app.router.add_get("/fapi/v1/exchangeInfo", self.exchange_info)
        app.router.add_get("/fapi/v1/ticker/24hr", self.ticker)
        app.router.add_get("/fapi/v1/depth", self.depth)
        app.router.add_get("/stream", self.ws)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        return runner, f"127.0.0.1:{port}"
