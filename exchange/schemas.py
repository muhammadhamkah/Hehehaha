"""Validation of Binance USDT-M WebSocket payloads + feed-health statistics.

Every message is checked against the fields/types the parsers rely on. Malformed or
unexpected messages are COUNTED and LOGGED (rate-limited), never silently dropped.
Sequence and timing health is tracked per stream:

  * aggTrade      aggregate-trade id gaps / duplicates, T <= E, price/qty > 0
  * depth         U <= u, pu continuity (``pu`` == previous ``u``), sorted levels,
                  uncrossed book, non-negative quantities
  * bookTicker    update-id monotonicity, bid <= ask
  * all streams   event time (E) monotonicity and receive latency (local - E)

Parsed values are returned so the handlers never index raw dicts themselves.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger("feed")

LOG_FIRST_N = 20
LOG_EVERY_N = 1000
LATENCY_WARN_MS = 5000
LATENCY_SAMPLES = 2000


class MalformedMessage(ValueError):
    pass


@dataclass(slots=True)
class ParsedTrade:
    symbol: str
    agg_id: int
    price: float
    qty: float
    trade_ts: int
    event_ts: int
    is_buyer_maker: bool


@dataclass(slots=True)
class ParsedDepth:
    symbol: str
    event_ts: int
    trans_ts: int
    first_id: int
    final_id: int
    prev_final_id: int | None
    bids: list[tuple[float, float]]
    asks: list[tuple[float, float]]
    raw: dict


@dataclass(slots=True)
class ParsedBookTicker:
    symbol: str
    update_id: int
    bid: float
    bid_qty: float
    ask: float
    ask_qty: float
    event_ts: int


@dataclass
class StreamHealth:
    messages: int = 0
    malformed: int = 0
    last_event_ts: int = 0
    non_monotonic_ts: int = 0
    seq_gaps: int = 0
    missing_ids: int = 0
    duplicates: int = 0
    crossed: int = 0
    latency_ms: list[int] = field(default_factory=list)
    high_latency: int = 0


def _f(d: dict, key: str) -> float:
    try:
        return float(d[key])
    except KeyError as exc:
        raise MalformedMessage(f"missing field {key!r}") from exc
    except (TypeError, ValueError) as exc:
        raise MalformedMessage(f"field {key!r} not numeric: {d.get(key)!r}") from exc


def _i(d: dict, key: str) -> int:
    try:
        v = d[key]
    except KeyError as exc:
        raise MalformedMessage(f"missing field {key!r}") from exc
    if isinstance(v, bool) or not isinstance(v, (int, str)):
        raise MalformedMessage(f"field {key!r} not an integer: {v!r}")
    try:
        return int(v)
    except ValueError as exc:
        raise MalformedMessage(f"field {key!r} not an integer: {v!r}") from exc


def _levels(raw: Any, name: str) -> list[tuple[float, float]]:
    if not isinstance(raw, list):
        raise MalformedMessage(f"{name} is not a list")
    out = []
    for lv in raw:
        if not isinstance(lv, (list, tuple)) or len(lv) < 2:
            raise MalformedMessage(f"bad {name} level {lv!r}")
        try:
            p, q = float(lv[0]), float(lv[1])
        except (TypeError, ValueError) as exc:
            raise MalformedMessage(f"non-numeric {name} level {lv!r}") from exc
        if p <= 0 or q < 0:
            raise MalformedMessage(f"invalid {name} level {lv!r}")
        out.append((p, q))
    return out


class FeedValidator:
    def __init__(self) -> None:
        self.health: dict[str, StreamHealth] = defaultdict(StreamHealth)
        self.last_agg_id: dict[str, int] = {}
        self.last_depth_u: dict[str, int] = {}
        self.last_bt_u: dict[str, int] = {}
        self.unexpected: dict[str, int] = defaultdict(int)
        self.malformed_by_kind: dict[str, int] = defaultdict(int)

    # ------------------------------------------------------------------ helpers
    def _report(self, kind: str, stream: str, reason: str, payload: Any) -> None:
        self.malformed_by_kind[kind] += 1
        n = self.malformed_by_kind[kind]
        if n <= LOG_FIRST_N or n % LOG_EVERY_N == 0:
            log.warning("malformed %s message #%d on %s: %s | payload=%.300s", kind, n, stream, reason, payload)

    def report_unexpected(self, stream: str, payload: Any) -> None:
        self.unexpected[stream] += 1
        n = self.unexpected[stream]
        if n <= LOG_FIRST_N or n % LOG_EVERY_N == 0:
            log.warning("unexpected message #%d on %s: %.300s", n, stream, payload)

    def _timing(self, h: StreamHealth, event_ts: int, local_ts: int) -> None:
        h.messages += 1
        if event_ts < h.last_event_ts:
            h.non_monotonic_ts += 1
        h.last_event_ts = max(h.last_event_ts, event_ts)
        lat = local_ts - event_ts
        if len(h.latency_ms) < LATENCY_SAMPLES:
            h.latency_ms.append(lat)
        else:
            h.latency_ms[h.messages % LATENCY_SAMPLES] = lat
        if abs(lat) > LATENCY_WARN_MS:
            h.high_latency += 1
            if h.high_latency <= LOG_FIRST_N:
                log.warning("high receive latency %d ms (clock skew or stale feed)", lat)

    @staticmethod
    def _check_symbol(d: dict, expected: str | None) -> str:
        s = d.get("s")
        if not isinstance(s, str) or not s:
            raise MalformedMessage("missing symbol 's'")
        if expected and s != expected:
            raise MalformedMessage(f"symbol mismatch: stream {expected} payload {s}")
        return s

    # ------------------------------------------------------------------ parsers
    def agg_trade(self, stream: str, d: Any, local_ts: int, symbol: str | None = None) -> ParsedTrade | None:
        h = self.health[f"aggTrade:{symbol or '*'}"]
        try:
            if not isinstance(d, dict):
                raise MalformedMessage("payload is not an object")
            if d.get("e") not in (None, "aggTrade"):
                raise MalformedMessage(f"unexpected event type {d.get('e')!r}")
            sym = self._check_symbol(d, symbol)
            t = ParsedTrade(sym, _i(d, "a"), _f(d, "p"), _f(d, "q"), _i(d, "T"),
                            _i(d, "E") if "E" in d else _i(d, "T"), d.get("m"))
            if not isinstance(t.is_buyer_maker, bool):
                raise MalformedMessage(f"'m' is not a boolean: {d.get('m')!r}")
            if t.price <= 0 or t.qty <= 0:
                raise MalformedMessage("non-positive price/qty")
        except MalformedMessage as exc:
            h.malformed += 1
            self._report("aggTrade", stream, str(exc), d)
            return None
        last = self.last_agg_id.get(sym)
        if last is not None:
            if t.agg_id <= last:
                h.duplicates += 1
            elif t.agg_id > last + 1:
                h.missing_ids += t.agg_id - last - 1
                h.seq_gaps += 1
        self.last_agg_id[sym] = max(t.agg_id, last or 0)
        self._timing(h, t.event_ts, local_ts)
        return t

    def depth(self, stream: str, d: Any, local_ts: int, symbol: str | None = None,
              partial: bool = True) -> ParsedDepth | None:
        h = self.health[f"depth:{symbol or '*'}"]
        try:
            if not isinstance(d, dict):
                raise MalformedMessage("payload is not an object")
            if d.get("e") not in (None, "depthUpdate"):
                raise MalformedMessage(f"unexpected event type {d.get('e')!r}")
            sym = self._check_symbol(d, symbol) if "s" in d or symbol is None else symbol
            p = ParsedDepth(
                sym, _i(d, "E"), _i(d, "T") if "T" in d else _i(d, "E"),
                _i(d, "U"), _i(d, "u"), _i(d, "pu") if "pu" in d else None,
                _levels(d.get("b"), "bids"), _levels(d.get("a"), "asks"), d,
            )
            if p.first_id > p.final_id:
                raise MalformedMessage(f"U > u ({p.first_id} > {p.final_id})")
            if partial:
                if not p.bids or not p.asks:
                    raise MalformedMessage("empty side in partial depth snapshot")
                if any(a[0] <= b[0] for a, b in zip(p.bids, p.bids[1:])):
                    raise MalformedMessage("bids not strictly descending")
                if any(b[0] <= a[0] for a, b in zip(p.asks, p.asks[1:])):
                    raise MalformedMessage("asks not strictly ascending")
                if p.bids[0][0] >= p.asks[0][0]:
                    h.crossed += 1
                    raise MalformedMessage("crossed partial book")
        except MalformedMessage as exc:
            h.malformed += 1
            self._report("depth", stream, str(exc), d)
            return None
        last_u = self.last_depth_u.get(sym)
        if last_u is not None and p.prev_final_id is not None and p.prev_final_id != last_u:
            h.seq_gaps += 1
        if last_u is not None and p.final_id < last_u:
            h.duplicates += 1
        self.last_depth_u[sym] = max(p.final_id, last_u or 0)
        self._timing(h, p.event_ts, local_ts)
        return p

    def book_ticker(self, stream: str, d: Any, local_ts: int, symbol: str | None = None) -> ParsedBookTicker | None:
        h = self.health[f"bookTicker:{symbol or '*'}"]
        try:
            if not isinstance(d, dict):
                raise MalformedMessage("payload is not an object")
            if d.get("e") not in (None, "bookTicker"):
                raise MalformedMessage(f"unexpected event type {d.get('e')!r}")
            sym = self._check_symbol(d, symbol)
            ev_ts = _i(d, "E") if "E" in d else (_i(d, "T") if "T" in d else local_ts)
            bt = ParsedBookTicker(sym, _i(d, "u") if "u" in d else 0, _f(d, "b"), _f(d, "B"),
                                  _f(d, "a"), _f(d, "A"), ev_ts)
            if bt.bid <= 0 or bt.ask <= 0 or bt.bid_qty < 0 or bt.ask_qty < 0:
                raise MalformedMessage("non-positive price or negative qty")
            if bt.bid > bt.ask:
                h.crossed += 1
                raise MalformedMessage("bid > ask")
        except MalformedMessage as exc:
            h.malformed += 1
            self._report("bookTicker", stream, str(exc), d)
            return None
        last = self.last_bt_u.get(sym)
        if last is not None and bt.update_id and bt.update_id < last:
            h.duplicates += 1
        if bt.update_id:
            self.last_bt_u[sym] = max(bt.update_id, last or 0)
        self._timing(h, bt.event_ts, local_ts)
        return bt

    def ticker_arr(self, stream: str, d: Any, local_ts: int) -> list[dict]:
        h = self.health["ticker_arr"]
        if not isinstance(d, list):
            h.malformed += 1
            self._report("ticker_arr", stream, "payload is not a list", d)
            return []
        out = []
        max_e = 0
        for t in d:
            try:
                if not isinstance(t, dict):
                    raise MalformedMessage("element is not an object")
                if t.get("e") not in (None, "24hrTicker"):
                    raise MalformedMessage(f"unexpected event type {t.get('e')!r}")
                self._check_symbol(t, None)
                _f(t, "c")
                _f(t, "q")
                n = t.get("n", 0)
                if not isinstance(n, int) or isinstance(n, bool):
                    raise MalformedMessage(f"'n' not int: {n!r}")
                max_e = max(max_e, _i(t, "E")) if "E" in t else max_e
                out.append(t)
            except MalformedMessage as exc:
                h.malformed += 1
                self._report("ticker", stream, str(exc), t)
        self._timing(h, max_e or local_ts, local_ts)
        return out

    # ------------------------------------------------------------------ summary
    def summary(self) -> dict[str, Any]:
        def pct(xs: list[int], q: float) -> int | None:
            if not xs:
                return None
            s = sorted(xs)
            return s[min(int(q * len(s)), len(s) - 1)]

        agg: dict[str, dict[str, Any]] = {}
        for key, h in self.health.items():
            kind = key.split(":")[0]
            a = agg.setdefault(kind, {"streams": 0, "messages": 0, "malformed": 0, "non_monotonic_ts": 0,
                                      "seq_gaps": 0, "missing_ids": 0, "duplicates": 0, "crossed": 0,
                                      "high_latency": 0, "_lat": []})
            a["streams"] += 1
            for k in ("messages", "malformed", "non_monotonic_ts", "seq_gaps", "missing_ids",
                      "duplicates", "crossed", "high_latency"):
                a[k] += getattr(h, k)
            a["_lat"].extend(h.latency_ms)
        for a in agg.values():
            lat = a.pop("_lat")
            a["latency_ms_p50"] = pct(lat, 0.5)
            a["latency_ms_p99"] = pct(lat, 0.99)
        return {"by_kind": agg, "unexpected": dict(self.unexpected),
                "malformed_by_kind": dict(self.malformed_by_kind)}
