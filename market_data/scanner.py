"""Stage-1 market-wide scanner.

Runs on cheap all-market streams only (``!ticker@arr`` and ``!bookTicker``) and
ranks every eligible USDT-M perpetual by a weighted sum of cross-sectional
percentile ranks of:

  * 24h quote volume (liquidity)
  * bid/ask spread (lower is better)
  * short-term volatility (price-change dispersion over ``volatility_window_s``)
  * recent trade activity (trade-count rate)
  * volume acceleration (recent quote-volume rate vs. older rate)
  * |top-of-book imbalance|

The top ``top_n`` symbols are shortlisted for the expensive microstructure engine,
with hysteresis so incumbents are not churned on small rank changes. Symbols can be
"pinned" (open position, pending signal labels) so they stay subscribed.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

from config import ScannerConfig
from utils.mathx import imbalance, percentile_ranks


@dataclass
class SymbolStats:
    symbol: str
    last_price: float = 0.0
    quote_volume_24h: float = 0.0
    trade_count_24h: int = 0
    bid: float = 0.0
    ask: float = 0.0
    bid_qty: float = 0.0
    ask_qty: float = 0.0
    first_seen_ms: int = 0
    last_update_ms: int = 0
    # (ts_ms, last_price, quote_volume_24h, trade_count_24h)
    samples: deque = field(default_factory=lambda: deque(maxlen=400))

    @property
    def spread_bps(self) -> float:
        if self.bid <= 0 or self.ask <= 0:
            return float("inf")
        return (self.ask - self.bid) / (0.5 * (self.ask + self.bid)) * 1e4

    @property
    def tob_imbalance(self) -> float:
        return imbalance(self.bid * self.bid_qty, self.ask * self.ask_qty)


@dataclass
class RankedSymbol:
    symbol: str
    score: float
    quote_volume_24h: float
    spread_bps: float
    volatility_bps: float
    activity: float
    volume_accel: float
    tob_imbalance: float


class MarketScanner:
    def __init__(self, cfg: ScannerConfig) -> None:
        self.cfg = cfg
        self.stats: dict[str, SymbolStats] = {}
        self.eligible: set[str] = set()     # from exchangeInfo (TRADING PERPETUAL USDT)
        self.selected: list[str] = []
        self.pinned: dict[str, set[str]] = {}   # symbol -> reasons
        self.last_ranking: list[RankedSymbol] = []

    # ------------------------------------------------------------------ inputs
    def set_universe(self, symbols: set[str]) -> None:
        excl = set(self.cfg.exclude_symbols)
        self.eligible = {s for s in symbols if s not in excl}
        for s in list(self.stats):
            if s not in self.eligible:
                del self.stats[s]

    def _get(self, symbol: str, now_ms: int) -> SymbolStats | None:
        if self.eligible and symbol not in self.eligible:
            return None
        st = self.stats.get(symbol)
        if st is None:
            st = SymbolStats(symbol=symbol, first_seen_ms=now_ms)
            self.stats[symbol] = st
        return st

    def on_ticker(self, t: dict, now_ms: int) -> None:
        """24hr ticker element from ``!ticker@arr`` (keys: s, c, q, n, E)."""
        st = self._get(t["s"], now_ms)
        if st is None:
            return
        st.last_price = float(t["c"])
        st.quote_volume_24h = float(t["q"])
        st.trade_count_24h = int(t.get("n", 0))
        st.last_update_ms = now_ms
        st.samples.append((now_ms, st.last_price, st.quote_volume_24h, st.trade_count_24h))

    def on_book_ticker(self, b: dict, now_ms: int) -> None:
        """Element from ``!bookTicker`` (keys: s, b, B, a, A)."""
        st = self._get(b["s"], now_ms)
        if st is None:
            return
        st.bid = float(b["b"])
        st.bid_qty = float(b["B"])
        st.ask = float(b["a"])
        st.ask_qty = float(b["A"])
        st.last_update_ms = max(st.last_update_ms, now_ms)

    # ------------------------------------------------------------------ metrics
    def _metrics(self, st: SymbolStats, now_ms: int) -> tuple[float, float, float] | None:
        win_v = int(self.cfg.volatility_window_s * 1000)
        win_a = int(self.cfg.activity_window_s * 1000)
        samples = [s for s in st.samples if now_ms - s[0] <= max(win_v, 2 * win_a)]
        if len(samples) < 3:
            return None
        vol_samples = [s for s in samples if now_ms - s[0] <= win_v]
        rets = []
        for (t0, p0, _, _), (t1, p1, _, _) in zip(vol_samples, vol_samples[1:]):
            if p0 > 0 and p1 > 0:
                rets.append(math.log(p1 / p0))
        volatility_bps = math.sqrt(sum(r * r for r in rets)) * 1e4 if rets else 0.0

        recent = [s for s in samples if now_ms - s[0] <= win_a]
        older = [s for s in samples if win_a < now_ms - s[0] <= 2 * win_a]

        def rate(group: list, idx: int) -> float:
            if len(group) < 2:
                return 0.0
            dt = (group[-1][0] - group[0][0]) / 1000.0
            dv = group[-1][idx] - group[0][idx]
            # 24h rolling counters can decrease as old volume rolls off; clamp at 0.
            return max(dv, 0.0) / dt if dt > 0 else 0.0

        activity = rate(recent, 3)                    # trades / s
        vol_recent = rate(recent, 2)                  # quote volume / s
        vol_older = rate(older, 2)
        if vol_older > 0:
            accel = vol_recent / vol_older
        else:
            accel = 1.0 if vol_recent == 0 else 2.0
        return volatility_bps, activity, accel

    # ------------------------------------------------------------------ ranking
    def rank(self, now_ms: int) -> list[RankedSymbol]:
        rows: list[tuple[SymbolStats, float, float, float]] = []
        min_age = int(self.cfg.min_age_for_ranking_s * 1000)
        for st in self.stats.values():
            if st.quote_volume_24h < self.cfg.min_quote_volume_24h:
                continue
            if st.spread_bps > self.cfg.max_spread_bps:
                continue
            if now_ms - st.first_seen_ms < min_age:
                continue
            m = self._metrics(st, now_ms)
            if m is None:
                continue
            rows.append((st, *m))
        if not rows:
            self.last_ranking = []
            return []

        c = self.cfg
        r_vol = percentile_ranks([math.log10(max(r[0].quote_volume_24h, 1.0)) for r in rows])
        r_spread = percentile_ranks([-r[0].spread_bps for r in rows])
        r_volat = percentile_ranks([r[1] for r in rows])
        r_act = percentile_ranks([r[2] for r in rows])
        r_acc = percentile_ranks([r[3] for r in rows])
        r_imb = percentile_ranks([abs(r[0].tob_imbalance) for r in rows])
        wsum = c.w_volume + c.w_spread + c.w_volatility + c.w_activity + c.w_volume_accel + c.w_imbalance

        ranked = []
        for i, (st, volat, act, acc) in enumerate(rows):
            score = (
                c.w_volume * r_vol[i]
                + c.w_spread * r_spread[i]
                + c.w_volatility * r_volat[i]
                + c.w_activity * r_act[i]
                + c.w_volume_accel * r_acc[i]
                + c.w_imbalance * r_imb[i]
            ) / wsum
            ranked.append(
                RankedSymbol(
                    symbol=st.symbol,
                    score=score,
                    quote_volume_24h=st.quote_volume_24h,
                    spread_bps=st.spread_bps,
                    volatility_bps=volat,
                    activity=act,
                    volume_accel=acc,
                    tob_imbalance=st.tob_imbalance,
                )
            )
        ranked.sort(key=lambda r: r.score, reverse=True)
        self.last_ranking = ranked
        return ranked

    def select(self, now_ms: int) -> list[str]:
        """Update and return the shortlist (top_n with hysteresis) plus pinned symbols."""
        ranked = self.rank(now_ms)
        if self.cfg.static_symbols:
            self.selected = [s for s in self.cfg.static_symbols if not self.eligible or s in self.eligible]
            return self.active_symbols()
        order = [r.symbol for r in ranked]
        pos = {s: i for i, s in enumerate(order)}
        n = self.cfg.top_n
        keep_limit = int(math.ceil(n * self.cfg.hysteresis))
        incumbents = [s for s in self.selected if pos.get(s, 10**9) < keep_limit]
        chosen = list(incumbents)
        for s in order:
            if len(chosen) >= n:
                break
            if s not in chosen:
                chosen.append(s)
        chosen.sort(key=lambda s: pos.get(s, 10**9))
        chosen = chosen[:n]
        self.selected = chosen
        return self.active_symbols()

    def active_symbols(self) -> list[str]:
        out = list(self.selected)
        for s in sorted(self.pinned):
            if s not in out:
                out.append(s)
        return out

    def pin(self, symbol: str, reason: str) -> None:
        self.pinned.setdefault(symbol, set()).add(reason)

    def unpin(self, symbol: str, reason: str) -> None:
        reasons = self.pinned.get(symbol)
        if reasons is None:
            return
        reasons.discard(reason)
        if not reasons:
            del self.pinned[symbol]
