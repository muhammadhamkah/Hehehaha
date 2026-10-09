"""StreamConnection against a local mock of the Binance combined-stream endpoint."""
import asyncio
import json

from aiohttp import WSMsgType, web

from exchange.websocket_manager import StreamConnection


class MockBinance:
    def __init__(self) -> None:
        self.conns: list[tuple[web.WebSocketResponse, set]] = []
        self.silent = False
        self.connections_total = 0
        self.sub_log: list[tuple[str, list]] = []

    async def handler(self, request):
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
                self.sub_log.append((req["method"], req["params"]))
                if any("bad" in p for p in req["params"]):
                    await ws.send_str(json.dumps({"error": {"code": 2, "msg": "Invalid request"}, "id": req["id"]}))
                    continue
                if req["method"] == "SUBSCRIBE":
                    subs.update(req["params"])
                elif req["method"] == "UNSUBSCRIBE":
                    subs.difference_update(req["params"])
                await ws.send_str(json.dumps({"result": None, "id": req["id"]}))
        finally:
            sender.cancel()
            self.conns.remove(entry)
        return ws

    async def _send(self, ws, subs):
        n = 0
        while not ws.closed:
            if not self.silent:
                for s in sorted(subs):
                    n += 1
                    await ws.send_str(json.dumps({"stream": s, "data": {"n": n}}))
            await asyncio.sleep(0.01)

    async def kill_all(self):
        for ws, _ in list(self.conns):
            await ws.close()


async def _start():
    mock = MockBinance()
    app = web.Application()
    app.router.add_get("/stream", mock.handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    return mock, runner, f"ws://127.0.0.1:{port}"


async def _until(pred, timeout=5.0):
    loop = asyncio.get_running_loop()
    end = loop.time() + timeout
    while loop.time() < end:
        if pred():
            return True
        await asyncio.sleep(0.02)
    return False


async def test_subscribe_resubscribe_reconnect_and_silence():
    mock, runner, url = await _start()
    seen: list[str] = []
    events: list[str] = []
    conn = StreamConnection("t", url, lambda s, d, ts: seen.append(s),
                            on_connect=lambda n: events.append("connect"),
                            on_disconnect=lambda n: events.append("disconnect"),
                            silence_timeout_s=0.4, initial_backoff_s=0.05)
    await conn.set_streams(["a@aggTrade", "b@aggTrade"])
    task = asyncio.ensure_future(conn.run())
    try:
        assert await _until(lambda: {"a@aggTrade", "b@aggTrade"} <= set(seen))
        # dynamic change: a out, c in
        await conn.set_streams(["b@aggTrade", "c@aggTrade"])
        assert await _until(lambda: "c@aggTrade" in seen)
        await asyncio.sleep(0.1)
        seen.clear()
        await asyncio.sleep(0.2)
        assert "a@aggTrade" not in seen and {"b@aggTrade", "c@aggTrade"} <= set(seen)

        # server drops the connection -> reconnect + automatic resubscription
        await mock.kill_all()
        assert await _until(lambda: mock.connections_total >= 2 and mock.conns and
                            mock.conns[-1][1] == {"b@aggTrade", "c@aggTrade"})
        seen.clear()
        assert await _until(lambda: {"b@aggTrade", "c@aggTrade"} <= set(seen))
        assert conn.n_reconnects >= 1 and events.count("disconnect") >= 1 and events.count("connect") >= 2

        # silent-but-open socket -> watchdog forces a reconnect
        before = mock.connections_total
        mock.silent = True
        assert await _until(lambda: conn.n_silence_reconnects >= 1, timeout=3)
        mock.silent = False
        assert await _until(lambda: mock.connections_total > before)
        seen.clear()
        assert await _until(lambda: "b@aggTrade" in seen)

        # subscription error is reported, not ignored
        await conn.set_streams(["b@aggTrade", "c@aggTrade", "bad@stream"])
        assert await _until(lambda: conn.n_sub_errors >= 1)
    finally:
        await conn.stop()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await runner.cleanup()


async def test_stream_cap_keeps_highest_priority():
    conn = StreamConnection("t", "ws://unused", lambda *a: None, max_streams=2)
    await conn.set_streams(["p1", "p2", "p3"])
    assert conn.streams == {"p1", "p2"}
