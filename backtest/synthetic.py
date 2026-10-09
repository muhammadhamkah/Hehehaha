"""Synthetic Binance-shaped event generator — FOR TESTING THE PIPELINE ONLY.

Produces event files in the exact recorder format (depth20@100ms, aggTrade, bookTicker,
!ticker@arr, !bookTicker) so the replay backtester can be exercised end-to-end without
network access. Results on synthetic data say NOTHING about real-market profitability.

A latent order-flow state x(t) (Ornstein-Uhlenbeck) tilts book imbalance and aggressor
mix. ``edge`` controls whether x(t) also drives the FUTURE price drift:
    edge = 0   -> microstructure looks informative but has zero predictive power
    edge > 0   -> drift(t) = edge * x(t) bps/s, i.e. a genuine (planted) edge
"""
from __future__ import annotations

import math
import random
from dataclasses import asdict

from data.event_store import EventWriter
from exchange.models import SymbolInfo


def generate(out_dir: str, n_symbols: int = 3, minutes: float = 20.0, edge: float = 0.0, seed: int = 1,
             start_ms: int = 1_717_200_000_000, vol_bps_per_sqrt_s: float = 2.0,
             trades_per_s: float = 8.0, feed_latency_ms: tuple[int, int] = (5, 40)) -> dict:
    rng = random.Random(seed)
    symbols = [f"SYN{i}USDT" for i in range(n_symbols)]
    tick = 0.01
    infos = {s: SymbolInfo(s, tick_size=tick, step_size=0.001, min_qty=0.001, min_notional=5.0) for s in symbols}
    w = EventWriter(out_dir, meta={"symbol_info": {k: asdict(v) for k, v in infos.items()},
                                   "depth_mode": "partial", "depth_levels": 20,
                                   "source": f"SYNTHETIC(edge={edge}, seed={seed})"})
    state = {s: {"mid": 100.0 + 10 * i, "x": 0.0, "u": 1000, "agg": 1, "qv": 5e8, "n": 1_000_000}
             for i, s in enumerate(symbols)}
    dt = 0.1
    steps = int(minutes * 60 / dt)
    tau = 20.0            # seconds, persistence of the latent flow state
    sig_x = math.sqrt(2 / tau)
    for k in range(steps):
        t = start_ms + int(k * dt * 1000)
        for s in symbols:
            st = state[s]
            st["x"] += -st["x"] * dt / tau + sig_x * math.sqrt(dt) * rng.gauss(0, 1)
            x = st["x"]
            drift_bps_s = edge * x
            ret_bps = drift_bps_s * dt + vol_bps_per_sqrt_s * math.sqrt(dt) * rng.gauss(0, 1)
            st["mid"] *= 1 + ret_bps / 1e4
            bid = math.floor(st["mid"] / tick) * tick
            spread_ticks = 1 if rng.random() < 0.9 else 2
            ask = bid + spread_ticks * tick
            tilt = math.tanh(x)
            base = 40.0 * (1 + 0.3 * rng.random())
            bids = [[f"{bid - i * tick:.2f}", f"{base * (1 + 0.6 * tilt) * (1 + 0.15 * i) * (0.7 + 0.6 * rng.random()):.3f}"]
                    for i in range(20)]
            asks = [[f"{ask + i * tick:.2f}", f"{base * (1 - 0.6 * tilt) * (1 + 0.15 * i) * (0.7 + 0.6 * rng.random()):.3f}"]
                    for i in range(20)]
            prev = st["u"]
            st["u"] += rng.randint(1, 20)
            lat = rng.randint(*feed_latency_ms)
            sl = s.lower()
            w.write("detail", f"{sl}@depth20@100ms", {"e": "depthUpdate", "E": t, "T": t - 1, "s": s,
                    "U": prev + 1, "u": st["u"], "pu": prev, "b": bids, "a": asks}, t + lat)
            w.write("detail", f"{sl}@bookTicker", {"e": "bookTicker", "u": st["u"], "E": t, "T": t - 1, "s": s,
                    "b": bids[0][0], "B": bids[0][1], "a": asks[0][0], "A": asks[0][1]}, t + lat)
            n_tr = _poisson(rng, trades_per_s * (1 + abs(x)) * dt)
            for j in range(n_tr):
                buy = rng.random() < 0.5 + 0.35 * tilt
                px = ask if buy else bid
                q = max(rng.expovariate(1 / 2.0), 0.001)
                st["agg"] += 1
                tt = t + int(j * 100 / max(n_tr, 1))
                w.write("detail", f"{sl}@aggTrade", {"e": "aggTrade", "E": tt + 1, "s": s, "a": st["agg"],
                        "p": f"{px:.2f}", "q": f"{q:.3f}", "f": st["agg"], "l": st["agg"], "T": tt,
                        "m": not buy}, tt + 1 + lat)
                st["qv"] += px * q
                st["n"] += 1
        if k % 10 == 0:
            t = start_ms + int(k * dt * 1000)
            w.write("market", "!ticker@arr", [{"e": "24hrTicker", "E": t, "s": s, "c": f"{state[s]['mid']:.2f}",
                                              "q": f"{state[s]['qv']:.2f}", "n": state[s]["n"]} for s in symbols],
                    t + 20)
            for s in symbols:
                m = state[s]["mid"]
                w.write("market", "!bookTicker", {"e": "bookTicker", "u": state[s]["u"], "E": t, "T": t, "s": s,
                        "b": f"{math.floor(m / tick) * tick:.2f}", "B": "10",
                        "a": f"{math.floor(m / tick) * tick + tick:.2f}", "A": "10"}, t + 20)
    w.close()
    return {"symbols": symbols, "start_ms": start_ms, "end_ms": start_ms + int(minutes * 60_000)}


def _poisson(rng: random.Random, lam: float) -> int:
    l_, k, p = math.exp(-lam), 0, 1.0
    while True:
        p *= rng.random()
        if p <= l_:
            return k
        k += 1
