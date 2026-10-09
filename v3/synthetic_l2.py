"""Synthetic V3 L2 store -- FOR TESTING THE PIPELINE ONLY (says nothing about real markets).

Writes exactly what ``v3.recorder`` writes (per-symbol stores, diff depth with real
U/u/pu sequencing, depth20, bookTicker, aggTrade, REST snapshots, audit snapshots, gap
markers) so recorder-format parsing, book sync, gap handling, features, labels, research
and replay can all be exercised end-to-end without network access.

A hidden pressure z(t) (Ornstein-Uhlenbeck) drives the price drift with strength ``edge``
and shows up ONLY in deeper-book behaviour: positive z pulls ask liquidity at levels 2-10
and replenishes the bid queue (negative z the mirror image). L1 sizes and the trade
aggressor mix carry no information about z. So:
    edge = 0  -> no feature set should predict direction (AUC ~ 0.5)
    edge > 0  -> L2 queue/cancellation features should, L1-only features should not
``gap_prob`` drops diff messages at random, exactly like a lost websocket message; the
store then contains the recorder's gap marker and resynchronisation snapshot.
"""
from __future__ import annotations

import json
import math
import os
import random
from dataclasses import asdict

from data.event_store import EventWriter
from exchange.models import SymbolInfo
from v3.recorder import write_root_meta


def _fmt(x: float, nd: int) -> str:
    return f"{x:.{nd}f}"


def generate(out: str, symbols: tuple[str, ...] = ("SYNAUSDT", "SYNBUSDT"), minutes: float = 20.0,
             edge: float = 0.0, seed: int = 1, start_ms: int = 1_717_200_000_000, gap_prob: float = 0.0,
             audit_interval_s: float = 120.0, move_prob: float = 0.30, levels: int = 60) -> dict:
    rng = random.Random(seed)
    tick, nd = 0.01, 2
    os.makedirs(out, exist_ok=True)
    infos = {s: asdict(SymbolInfo(s, tick_size=tick, step_size=0.001, min_qty=0.001, min_notional=5.0))
             for s in symbols}
    writers = {s: EventWriter(os.path.join(out, s), meta={"symbol_info": {s: infos[s]}, "depth_mode": "diff",
                                                          "source": f"SYNTHETIC_L2(edge={edge}, seed={seed})",
                                                          "recorder": "v3"}, blocking=True)
               for s in symbols}
    with open(os.path.join(out, "v3_store.json"), "w", encoding="utf-8") as fh:
        json.dump({"format": "v3_store", "version": 1, "symbols": list(symbols), "synthetic": True,
                   "created_ms": start_ms}, fh)
    write_root_meta(out, infos)
    st = {}
    for i, s in enumerate(symbols):
        bb = round(100.0 + 7 * i, nd)
        st[s] = {"z": 0.0, "u": 10_000, "agg": 1, "bb": bb,
                 "bids": {round(bb - k * tick, nd): rng.uniform(20, 60) for k in range(levels)},
                 "asks": {round(bb + (k + 1) * tick, nd): rng.uniform(20, 60) for k in range(levels)},
                 "pending_gap": None}
    dt = 0.1
    tau = 15.0
    steps = int(minutes * 60 / dt)
    next_audit = start_ms + int(audit_interval_s * 1000)

    def snapshot(s: str, t: int) -> dict:
        b = sorted(st[s]["bids"].items(), key=lambda kv: -kv[0])
        a = sorted(st[s]["asks"].items())
        return {"lastUpdateId": st[s]["u"], "E": t + 1, "T": t,
                "bids": [[_fmt(p, nd), _fmt(q, 3)] for p, q in b], "asks": [[_fmt(p, nd), _fmt(q, 3)] for p, q in a]}

    for k in range(steps):
        t = start_ms + int(k * dt * 1000)
        for s in symbols:
            S = st[s]
            sl = s.lower()
            lat = rng.randint(3, 30)
            S["z"] += -S["z"] * dt / tau + math.sqrt(2 / tau) * math.sqrt(dt) * rng.gauss(0, 1)
            z = S["z"]
            bids, asks = S["bids"], S["asks"]
            changed: dict[tuple[int, float], float] = {}
            trades = []

            def setq(side: int, p: float, q: float) -> None:
                book = bids if side > 0 else asks
                p = round(p, nd)
                if q < 0.0005:                      # rounds to "0.000": a removal on the wire
                    book.pop(p, None)
                    q = 0.0
                else:
                    book[p] = q
                changed[(side, p)] = q

            # --- price move: market orders sweep the touch (aggressor mix uninformative)
            p_up = min(max(0.5 + edge * math.tanh(z), 0.02), 0.98)
            r = rng.random()
            move = 1 if r < move_prob * p_up else (-1 if r < move_prob else 0)
            best_bid, best_ask = max(bids), min(asks)
            if move == 1:
                q = asks[best_ask]
                trades.append((best_ask, q, True))
                setq(-1, best_ask, 0)
                setq(+1, best_ask, rng.uniform(20, 60))             # new bid queue at the old ask
                setq(-1, round(max(asks) + tick, nd), rng.uniform(20, 60))
                setq(+1, round(min(bids), nd), 0) if len(bids) > levels else None
            elif move == -1:
                q = bids[best_bid]
                trades.append((best_bid, q, False))
                setq(+1, best_bid, 0)
                setq(-1, best_bid, rng.uniform(20, 60))
                setq(+1, round(min(bids) - tick, nd), rng.uniform(20, 60))
                setq(-1, round(max(asks), nd), 0) if len(asks) > levels else None
            # --- small uninformative trades at the touch
            for _ in range(rng.randint(0, 3)):
                buy = rng.random() < 0.5
                book = asks if buy else bids
                if not book:
                    continue
                p = min(asks) if buy else max(bids)
                q = min(rng.expovariate(1 / 2.0), book[p] * 0.5)
                if q < 0.001:
                    continue
                trades.append((p, q, buy))
                setq(-1 if buy else +1, p, book[p] - q)
            # --- uninformative add/cancel noise everywhere (incl. L1)
            for side, book in ((+1, bids), (-1, asks)):
                lv = sorted(book, reverse=side > 0)[:20]
                for _ in range(4):
                    p = lv[min(int(rng.expovariate(1 / 4.0)), len(lv) - 1)]
                    if p in book:
                        setq(side, p, min(max(book[p] * rng.uniform(0.6, 1.5), 0.5), 400.0))
            # --- PLANTED L2 signal: deep-level pulling on the side the price will move toward,
            #     queue replenishment on the other side (levels 2-10 only; L1 untouched)
            if edge > 0:
                pull_side, fill_side = (-1, +1) if z > 0 else (+1, -1)
                intensity = min(abs(z), 2.5)
                for side, book, f in ((pull_side, asks if pull_side < 0 else bids, 1 - 0.25 * intensity),
                                      (fill_side, bids if fill_side > 0 else asks, 1 + 0.25 * intensity)):
                    lv = sorted(book, reverse=side > 0)[1:10]
                    for p in rng.sample(lv, min(3, len(lv))):
                        if p in book:
                            setq(side, p, max(book[p] * max(f, 0.1) * rng.uniform(0.9, 1.1), 0.5))
            if max(bids) >= min(asks):          # keep the synthetic book sane
                for p in [p for p in bids if p >= min(asks)]:
                    setq(+1, p, 0)
            # --- emit what the recorder would have received
            prev = S["u"]
            S["u"] += max(len(changed), 1)
            ev = {"e": "depthUpdate", "E": t, "T": t - 1, "s": s, "U": prev + 1, "u": S["u"], "pu": prev,
                  "b": [[_fmt(p, nd), _fmt(q, 3)] for (sd, p), q in changed.items() if sd > 0],
                  "a": [[_fmt(p, nd), _fmt(q, 3)] for (sd, p), q in changed.items() if sd < 0]}
            w = writers[s]
            for j, (p, q, buy) in enumerate(trades):
                S["agg"] += 1
                w.write("detail", f"{sl}@aggTrade", {"e": "aggTrade", "E": t - 1, "s": s, "a": S["agg"],
                                                    "p": _fmt(p, nd), "q": _fmt(q, 3), "f": S["agg"], "l": S["agg"],
                                                    "T": t - 2, "m": not buy}, t - 1 + lat)
            lost = gap_prob > 0 and rng.random() < gap_prob
            if not lost:
                w.write("detail", f"{sl}@depth@100ms", ev, t + lat)
            else:
                S["pending_gap"] = t + lat + 100     # the next diff's pu will not match
            b20 = sorted(bids.items(), key=lambda kv: -kv[0])[:20]
            a20 = sorted(asks.items())[:20]
            w.write("detail", f"{sl}@depth20@100ms", {**ev, "b": [[_fmt(p, nd), _fmt(q, 3)] for p, q in b20],
                                                     "a": [[_fmt(p, nd), _fmt(q, 3)] for p, q in a20]}, t + lat)
            w.write("detail", f"{sl}@bookTicker", {"e": "bookTicker", "u": S["u"], "E": t, "T": t - 1, "s": s,
                                                   "b": _fmt(b20[0][0], nd), "B": _fmt(b20[0][1], 3),
                                                   "a": _fmt(a20[0][0], nd), "A": _fmt(a20[0][1], 3)}, t + lat + 1)
            if k == 3:      # recorder start: diffs buffered first, then the REST snapshot
                w.write("detail", f"__snapshot__@{s}", snapshot(s, t), t + lat + 400)
                w.write("detail", f"__resync__@{s}", {"why": "connect", "status": "ok"}, t + lat + 400)
            if S["pending_gap"] is not None and t + lat >= S["pending_gap"]:
                w.write("detail", f"__gap__@{s}", {"reason": "pu_mismatch", "last_u": prev}, t + lat)
                w.write("detail", f"__snapshot__@{s}", snapshot(s, t), t + lat + 400)
                w.write("detail", f"__resync__@{s}", {"why": "gap", "status": "buffering"}, t + lat + 400)
                S["pending_gap"] = None
        if t >= next_audit:
            for s in symbols:
                writers[s].write("detail", f"__snapshot_audit__@{s}", snapshot(s, t), t + 50)
            next_audit += int(audit_interval_s * 1000)
    for w in writers.values():
        w.close()
    return {"symbols": list(symbols), "start_ms": start_ms, "end_ms": start_ms + int(minutes * 60_000)}
