"""Find which Binance USDT-M websocket address delivers each stream the V3 recorder needs.

    python -m tools.probe_ws [--symbol BTCUSDT] [--seconds 15]

Binance has been splitting futures websocket traffic across separate paths (e.g. ``/public``
for order-book streams, ``/market`` for trades/tickers). This connects to each candidate base
URL, subscribes to the test streams and counts what actually arrives, then prints the
recorder flags to use. Read-only, public data, no API key.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time

import aiohttp

CANDIDATES = ("wss://fstream.binance.com", "wss://fstream.binance.com/public", "wss://fstream.binance.com/market")
KINDS = {"depth": "depth@100ms", "bookTicker": "bookTicker", "aggTrade": "aggTrade"}


async def probe(base: str, streams: list[str], seconds: float) -> dict[str, int]:
    counts = {s: 0 for s in streams}
    try:
        async with aiohttp.ClientSession(trust_env=True) as session:
            async with session.ws_connect(f"{base}/stream", heartbeat=20, timeout=10, max_msg_size=0) as ws:
                await ws.send_str(json.dumps({"method": "SUBSCRIBE", "params": streams, "id": 1}))
                end = time.monotonic() + seconds
                while time.monotonic() < end:
                    try:
                        msg = await asyncio.wait_for(ws.receive(), max(end - time.monotonic(), 0.1))
                    except asyncio.TimeoutError:
                        break
                    if msg.type != aiohttp.WSMsgType.TEXT:
                        if msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                            break
                        continue
                    d = json.loads(msg.data)
                    s = d.get("stream")
                    if s in counts:
                        counts[s] += 1
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}
    return counts


async def main_async(symbol: str, seconds: float) -> None:
    sl = symbol.lower()
    streams = [f"{sl}@{k}" for k in KINDS.values()] + ["!ticker@arr"]
    results = await asyncio.gather(*(probe(b, streams, seconds) for b in CANDIDATES))
    best: dict[str, str] = {}
    print(f"messages received in {seconds:.0f}s per base URL:")
    for base, res in zip(CANDIDATES, results):
        print(f"  {base}")
        if "error" in res:
            print(f"      error: {res['error']}")
            continue
        for s, n in res.items():
            print(f"      {s:28s} {n}")
        for kind, suffix in KINDS.items():
            if res.get(f"{sl}@{suffix}", 0) > 0 and kind not in best:
                best[kind] = base
    missing = [k for k in KINDS if k not in best]
    if missing:
        print(f"\nNO base delivered: {missing}. Do not start recording; send this output.")
        return
    print("\nRecorder flags:")
    print(f"  --ws-depth-base {best['depth']} --ws-bookticker-base {best['bookTicker']} "
          f"--ws-trades-base {best['aggTrade']}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--seconds", type=float, default=15.0)
    a = ap.parse_args()
    asyncio.run(main_async(a.symbol, a.seconds))


if __name__ == "__main__":
    main()
