"""V3 L2 / queue-dynamics / flow feature engine.

One ``V3Engine`` per symbol is fed every raw recorded message (diff depth, REST snapshots,
aggTrades, bookTicker) in replay-time order -- by the dataset builder AND by the bot's V3
signal engine, so training and serving compute identical features.

Feature groups (column prefixes):
  v3d_  L2 depth / queue / cancellation / microprice / book shape   (needs the full book)
  v3f_  aggressive flow incl. sweeps and large-trade imbalance        (trades only)
  v3t_  temporal: changes over 100 ms-10 s, persistence, acceleration, slope, sign flips
  v3x_  explicit directional events: queue depletion, liquidity pulling, absorption
        (combine depth and flow)

Attribution of book changes (per diff, per price level):
  decrease = executed (matched by aggTrades at that price with trade time <= the diff's
             transaction time) + cancelled (the rest)        -> cancellation estimate
  increase = added / replenished
Best-queue depletion = executed + cancelled at the best price; replenishment = additions at
the best price. A level that empties because price traded through counts as depletion.
All notionals are in quote currency. Windows are measured on the replay clock.
"""
from __future__ import annotations

import math
from bisect import bisect_right
from collections import deque
from dataclasses import dataclass

from v3.book import ASK, BID, GAP, L2Book

FEATURE_VERSION = "v3-features-1"     # bump whenever any feature definition changes
NAN = float("nan")
DEPTH_LEVELS = (1, 3, 5, 10, 20)
WINDOWS_MS = (1000, 3000, 10000)
DELTAS_MS = (100, 250, 500, 1000, 2000, 5000, 10000)
WARMUP_MS = 10_000          # book must have been valid this long before features are emitted
STALE_MS = 2_000            # no diff for this long -> no features
TEMPORAL_KEYS = ("mid", "micro1", "micro5", "imb1", "imb5", "wimb20", "ofi", "mlofi", "sflow", "cancel_asym")
CUMULATIVE = {"ofi", "mlofi", "sflow", "cancel_asym"}


def _slog(x: float) -> float:
    return math.copysign(math.log1p(abs(x)), x)


def _imb(a: float, b: float) -> float:
    return (a - b) / (a + b) if a + b > 0 else 0.0


@dataclass(slots=True)
class DiffRec:
    ts: int
    dep_b: float = 0.0       # best-bid queue depletion (exec + cancel), notional
    dep_a: float = 0.0
    rep_b: float = 0.0       # best-queue replenishment
    rep_a: float = 0.0
    can_b5: float = 0.0      # cancellations within the top 5 / 10 levels
    can_a5: float = 0.0
    can_b10: float = 0.0
    can_a10: float = 0.0
    add_b10: float = 0.0
    add_a10: float = 0.0
    exe_b: float = 0.0
    exe_a: float = 0.0
    rapid_b: int = 0         # top-10 levels losing >= 50% to cancellation in one update
    rapid_a: int = 0
    bid_up: int = 0          # best-price moves
    bid_dn: int = 0
    ask_up: int = 0
    ask_dn: int = 0


@dataclass(slots=True)
class TradeRec:
    ts: int
    T: int
    price: float
    notional: float
    buy: bool


class V3Engine:
    def __init__(self, symbol: str, tick: float | None = None) -> None:
        self.symbol = symbol
        self.book = L2Book(symbol)
        self.tick = tick
        self.diffs: deque[DiffRec] = deque()
        self.trades: deque[TradeRec] = deque()
        self.unattributed: list[TradeRec] = []
        self.hist_ts: deque[int] = deque()
        self.hist: deque[tuple] = deque()
        self.cum = {"ofi": 0.0, "mlofi": 0.0, "sflow": 0.0, "cancel_asym": 0.0}
        self.prev_l1: tuple | None = None
        self.prev_top5: tuple | None = None
        self.best_since = {BID: 0, ASK: 0}
        self.best_px = {BID: 0.0, ASK: 0.0}
        self.lifetimes = {BID: deque(maxlen=50), ASK: deque(maxlen=50)}
        self.trade_sizes: deque[float] = deque(maxlen=2000)
        self.last_diff_ts = 0
        self.n_gaps = 0

    # ================================================================== events
    def on_message(self, kind: str, data, ts: int) -> None:
        """kind = stream suffix ('depth@100ms', 'aggTrade', ...) or a '__marker__' name."""
        if kind == "depth@100ms":
            self._on_diff(data, ts)
        elif kind == "aggTrade":
            self._on_trade(data, ts)
        elif kind in ("__snapshot__", "__snapshot_audit__"):
            if not self.book.valid:      # audit snapshots only (re)initialise an invalid book
                status, _ = self.book.on_snapshot(data, ts)
                self._reset_after_sync(ts)

    def _reset_after_sync(self, ts: int) -> None:
        self.prev_l1 = None
        self.prev_top5 = None
        self.unattributed.clear()

    def _on_trade(self, d: dict, ts: int) -> None:
        try:
            p, q = float(d["p"]), float(d["q"])
            T = int(d.get("T", ts))
            buy = not bool(d["m"])
        except (KeyError, TypeError, ValueError):
            return
        tr = TradeRec(ts, T, p, p * q, buy)
        self.trades.append(tr)
        self.unattributed.append(tr)
        self.trade_sizes.append(tr.notional)
        self.cum["sflow"] += tr.notional if buy else -tr.notional
        cut = ts - 30_000
        while self.trades and self.trades[0].ts < cut:
            self.trades.popleft()

    def _on_diff(self, d: dict, ts: int) -> None:
        book = self.book
        was_valid = book.valid
        pre_b, pre_a = book.top(10) if was_valid else ([], [])
        status, changes = book.on_diff(d, ts)
        if status == GAP:
            self.n_gaps += 1
            return
        if not book.valid:
            return
        if not was_valid:
            self._reset_after_sync(ts)
            pre_b, pre_a = book.top(10)
        self.last_diff_ts = ts
        T = int(d.get("T", d.get("E", ts)))
        execd = {BID: {}, ASK: {}}
        keep = []
        for tr in self.unattributed:
            if tr.T <= T:
                side = ASK if tr.buy else BID        # buyer-aggressor consumes asks
                execd[side][tr.price] = execd[side].get(tr.price, 0.0) + tr.notional / tr.price
            else:
                keep.append(tr)
        self.unattributed = keep[-500:]
        rec = DiffRec(ts)
        rank = {BID: {p: i for i, (p, _) in enumerate(pre_b)}, ASK: {p: i for i, (p, _) in enumerate(pre_a)}}
        best = {BID: pre_b[0][0] if pre_b else 0.0, ASK: pre_a[0][0] if pre_a else 0.0}
        lvl_med = sorted(p * q for p, q in pre_b + pre_a)[len(pre_b + pre_a) // 2] if pre_b and pre_a else 0.0
        for ch in changes:
            s = ch.side
            r = rank[s].get(ch.price)
            px = ch.price
            if ch.new < ch.old:
                dec = ch.old - ch.new
                ex = min(dec, execd[s].pop(px, 0.0))
                can = dec - ex
                if s == BID:
                    rec.exe_b += ex * px
                else:
                    rec.exe_a += ex * px
                if px == best[s]:
                    if s == BID:
                        rec.dep_b += dec * px
                    else:
                        rec.dep_a += dec * px
                if r is not None:
                    if r < 5:
                        if s == BID:
                            rec.can_b5 += can * px
                        else:
                            rec.can_a5 += can * px
                    if s == BID:
                        rec.can_b10 += can * px
                    else:
                        rec.can_a10 += can * px
                    if can >= 0.5 * ch.old and ch.old * px >= lvl_med:
                        if s == BID:
                            rec.rapid_b += 1
                        else:
                            rec.rapid_a += 1
            else:
                inc = ch.new - ch.old
                if px == best[s]:
                    if s == BID:
                        rec.rep_b += inc * px
                    else:
                        rec.rep_a += inc * px
                if r is not None or (s == BID and px > best[s]) or (s == ASK and px < best[s]):
                    if s == BID:
                        rec.add_b10 += inc * px
                    else:
                        rec.add_a10 += inc * px
        bb, bq, ba, aq = book.best()
        for s, px in ((BID, bb), (ASK, ba)):
            old = self.best_px[s]
            if px != old:
                if old:
                    if self.best_since[s]:
                        self.lifetimes[s].append(ts - self.best_since[s])
                    if s == BID:
                        rec.bid_up += px > old
                        rec.bid_dn += px < old
                    else:
                        rec.ask_up += px > old
                        rec.ask_dn += px < old
                self.best_px[s] = px
                self.best_since[s] = ts
        self.diffs.append(rec)
        cut = ts - 30_000
        while self.diffs and self.diffs[0].ts < cut:
            self.diffs.popleft()
        self.cum["cancel_asym"] += rec.can_a10 - rec.can_b10
        # L1 order-flow imbalance (Cont-Kukanov-Stoikov) and a 5-level version
        b5, a5 = book.top(5)
        if self.prev_l1 is not None:
            pb, pbq, pa, paq = self.prev_l1
            e = (bq * bb if bb >= pb else 0.0) - (pbq * pb if bb <= pb else 0.0) \
                - (aq * ba if ba <= pa else 0.0) + (paq * pa if ba >= pa else 0.0)
            self.cum["ofi"] += e
            ml = 0.0
            for i in range(min(len(b5), len(a5), len(self.prev_top5[0]), len(self.prev_top5[1]))):
                (cbp, cbq), (cap, caq) = b5[i], a5[i]
                (obp, obq), (oap, oaq) = self.prev_top5[0][i], self.prev_top5[1][i]
                ml += (cbq * cbp if cbp >= obp else 0.0) - (obq * obp if cbp <= obp else 0.0) \
                    - (caq * cap if cap <= oap else 0.0) + (oaq * oap if cap >= oap else 0.0)
            self.cum["mlofi"] += ml
        self.prev_l1 = (bb, bq, ba, aq)
        self.prev_top5 = (b5, a5)
        self._record_state(ts)

    # ================================================================== state history
    def _book_state(self) -> dict[str, float]:
        b, a = self.book.top(20)
        bb, bq = b[0]
        ba, aq = a[0]
        mid = 0.5 * (bb + ba)
        micro1 = (bb * aq + ba * bq) / (aq + bq) if aq + bq > 0 else mid

        def ml(k):
            qb = sum(q for _, q in b[:k])
            qa = sum(q for _, q in a[:k])
            vb = sum(p * q for p, q in b[:k]) / qb if qb else bb
            va = sum(p * q for p, q in a[:k]) / qa if qa else ba
            return (vb * qa + va * qb) / (qa + qb) if qa + qb else mid

        nb = [p * q for p, q in b]
        na = [p * q for p, q in a]
        wb = sum(x / (1 + i) for i, x in enumerate(nb))
        wa = sum(x / (1 + i) for i, x in enumerate(na))
        return {"mid": mid, "micro1": (micro1 - mid) / mid * 1e4, "micro5": (ml(5) - mid) / mid * 1e4,
                "micro20": (ml(20) - mid) / mid * 1e4, "imb1": _imb(nb[0], na[0]), "imb5": _imb(sum(nb[:5]), sum(na[:5])),
                "wimb20": _imb(wb, wa)}

    def _record_state(self, ts: int) -> None:
        s = self._book_state()
        row = tuple(s[k] if k in s else self.cum[k] for k in TEMPORAL_KEYS)
        self.hist_ts.append(ts)
        self.hist.append(row)
        cut = ts - 12_000
        while self.hist_ts and self.hist_ts[0] < cut:
            self.hist_ts.popleft()
            self.hist.popleft()

    def _state_at(self, ts: int) -> tuple | None:
        i = bisect_right(self.hist_ts, ts) - 1
        return self.hist[i] if i >= 0 else None

    # ================================================================== features
    def ready(self, now: int) -> bool:
        b = self.book
        return (b.valid and b.valid_since_ms is not None and now - b.valid_since_ms >= WARMUP_MS
                and now - self.last_diff_ts <= STALE_MS and len(self.hist) > 20)

    def features(self, now: int) -> dict[str, float] | None:
        if not self.ready(now):
            return None
        f: dict[str, float] = {}
        b, a = self.book.top(50)
        if len(b) < 20 or len(a) < 20:
            return None
        bb, ba = b[0][0], a[0][0]
        mid = 0.5 * (bb + ba)
        spread_bps = (ba - bb) / mid * 1e4
        nb = [p * q for p, q in b]
        na = [p * q for p, q in a]
        cb, ca = [], []
        for side_n, cum in ((nb, cb), (na, ca)):
            c = 0.0
            for x in side_n:
                c += x
                cum.append(c)
        # ---------------- multi-level depth
        for k in DEPTH_LEVELS:
            B, A = cb[k - 1], ca[k - 1]
            f[f"v3d_lbid_{k}"] = math.log1p(B)
            f[f"v3d_lask_{k}"] = math.log1p(A)
            f[f"v3d_imb_{k}"] = _imb(B, A)
            f[f"v3d_ratio_{k}"] = math.log((B + 1) / (A + 1))
            wb = sum(x / (1 + i) for i, x in enumerate(nb[:k]))
            wa = sum(x / (1 + i) for i, x in enumerate(na[:k]))
            f[f"v3d_wimb_{k}"] = _imb(wb, wa)
        for side, lv, cum in (("bid", b, cb), ("ask", a, ca)):
            for k in (5, 20):
                xs = [abs(p - mid) / mid * 1e4 for p, _ in lv[:k]]
                ys = [math.log1p(c) for c in cum[:k]]
                mx, my = sum(xs) / k, sum(ys) / k
                den = sum((x - mx) ** 2 for x in xs)
                f[f"v3d_slope_{side}_{k}"] = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den if den > 0 else 0.0
            f[f"v3d_convex_{side}"] = math.log((cum[19] - cum[4] + 1) / (cum[4] + 1))
            q20 = [q for _, q in lv[:20]]
            tot = sum(q20)
            f[f"v3d_hhi_{side}"] = sum((q / tot) ** 2 for q in q20) if tot else 0.0
            ratios = [math.log((q20[i] + 1e-12) / (q20[i + 1] + 1e-12)) for i in range(10)]
            f[f"v3d_cliff_{side}"] = max(ratios)
            f[f"v3d_cliff_lvl_{side}"] = float(ratios.index(max(ratios)))
            tick = self.tick or self._infer_tick()
            span = abs(lv[19][0] - lv[0][0])
            f[f"v3d_empty_{side}"] = max(span / tick - 19, 0.0) if tick else 0.0
            qs = sorted(q for _, q in lv)
            med = qs[len(qs) // 2]
            big = next(((p, q) for p, q in lv if q >= 5 * med), None)
            f[f"v3d_bigdist_{side}"] = abs(big[0] - mid) / mid * 1e4 if big else 50.0
            f[f"v3d_bigsize_{side}"] = math.log1p(big[0] * big[1]) if big else 0.0
        f["v3d_convex_asym"] = f["v3d_convex_bid"] - f["v3d_convex_ask"]
        f["v3d_hhi_asym"] = f["v3d_hhi_bid"] - f["v3d_hhi_ask"]
        f["v3d_cliff_asym"] = f["v3d_cliff_bid"] - f["v3d_cliff_ask"]
        f["v3d_bigdist_asym"] = f["v3d_bigdist_ask"] - f["v3d_bigdist_bid"]
        f["v3d_empty_asym"] = f["v3d_empty_ask"] - f["v3d_empty_bid"]
        f["v3d_spread_bps"] = spread_bps
        # ---------------- microprice
        st = self._book_state()
        for k in ("micro1", "micro5", "micro20"):
            f[f"v3d_{k}_bps"] = st[k]
            f[f"v3d_{k}_rel"] = st[k] / spread_bps if spread_bps > 0 else 0.0
        f["v3d_micro_div"] = st["micro5"] - st["micro1"]
        # ---------------- queue dynamics / cancellations (windows)
        qb, qa = nb[0], na[0]
        b10, a10 = cb[9], ca[9]
        for w in WINDOWS_MS:
            tag = f"{w // 1000}s"
            s = self._sum_diffs(now - w, now)
            sec = w / 1000
            f[f"v3d_dep_b_{tag}"] = s.dep_b / (qb + 1) / sec
            f[f"v3d_dep_a_{tag}"] = s.dep_a / (qa + 1) / sec
            f[f"v3d_rep_b_{tag}"] = s.rep_b / (qb + 1) / sec
            f[f"v3d_rep_a_{tag}"] = s.rep_a / (qa + 1) / sec
            f[f"v3d_repvel_b_{tag}"] = math.log1p(s.rep_b / sec)
            f[f"v3d_repvel_a_{tag}"] = math.log1p(s.rep_a / sec)
            f[f"v3d_repratio_b_{tag}"] = min(s.rep_b / (s.dep_b + 1e-9), 3.0) if s.dep_b > 0 else 1.0
            f[f"v3d_repratio_a_{tag}"] = min(s.rep_a / (s.dep_a + 1e-9), 3.0) if s.dep_a > 0 else 1.0
            f[f"v3d_dep_asym_{tag}"] = _imb(s.dep_a, s.dep_b)
            f[f"v3d_can_b_{tag}"] = s.can_b10 / (b10 + 1) / sec
            f[f"v3d_can_a_{tag}"] = s.can_a10 / (a10 + 1) / sec
            f[f"v3d_can5_asym_{tag}"] = _imb(s.can_a5, s.can_b5)
            f[f"v3d_can_asym_{tag}"] = _imb(s.can_a10, s.can_b10)
            f[f"v3d_pull_b_{tag}"] = (s.can_b10 - s.add_b10) / (b10 + 1)
            f[f"v3d_pull_a_{tag}"] = (s.can_a10 - s.add_a10) / (a10 + 1)
            f[f"v3d_pull_asym_{tag}"] = f[f"v3d_pull_a_{tag}"] - f[f"v3d_pull_b_{tag}"]
            f[f"v3d_rapid_b_{tag}"] = float(s.rapid_b)
            f[f"v3d_rapid_a_{tag}"] = float(s.rapid_a)
            f[f"v3d_bestmove_{tag}"] = float(s.bid_up + s.ask_up - s.bid_dn - s.ask_dn)
        f["v3d_qpersist_b"] = min((now - self.best_since[BID]) / 1000, 30.0)
        f["v3d_qpersist_a"] = min((now - self.best_since[ASK]) / 1000, 30.0)
        f["v3d_qlife_b"] = (sum(self.lifetimes[BID]) / len(self.lifetimes[BID]) / 1000) if self.lifetimes[BID] else 30.0
        f["v3d_qlife_a"] = (sum(self.lifetimes[ASK]) / len(self.lifetimes[ASK]) / 1000) if self.lifetimes[ASK] else 30.0
        f["v3d_qpersist_asym"] = f["v3d_qpersist_b"] - f["v3d_qpersist_a"]
        # ---------------- aggressive flow / sweeps
        large = sorted(self.trade_sizes)[int(len(self.trade_sizes) * 0.95)] if len(self.trade_sizes) > 50 else float("inf")
        for w in WINDOWS_MS:
            tag = f"{w // 1000}s"
            buy = sell = nb_ = ns_ = lb = ls = 0.0
            groups: dict[tuple[int, bool], list[TradeRec]] = {}
            for tr in reversed(self.trades):
                if tr.ts < now - w:
                    break
                if tr.ts > now:
                    continue
                if tr.buy:
                    buy += tr.notional
                    nb_ += 1
                    lb += tr.notional if tr.notional >= large else 0.0
                else:
                    sell += tr.notional
                    ns_ += 1
                    ls += tr.notional if tr.notional >= large else 0.0
                groups.setdefault((tr.T, tr.buy), []).append(tr)
            f[f"v3f_buy_{tag}"] = math.log1p(buy)
            f[f"v3f_sell_{tag}"] = math.log1p(sell)
            f[f"v3f_imb_{tag}"] = _imb(buy, sell)
            f[f"v3f_cntimb_{tag}"] = _imb(nb_, ns_)
            f[f"v3f_largeimb_{tag}"] = _imb(lb, ls)
            sw_b = sw_s = 0.0
            sw_n_b = sw_n_s = 0
            sw_depth = 0.0
            for (_, is_buy), trs in groups.items():
                pxs = {t.price for t in trs}
                if len(pxs) >= 2:
                    n = sum(t.notional for t in trs)
                    depth = (max(pxs) - min(pxs)) / mid * 1e4
                    sw_depth = max(sw_depth, depth)
                    if is_buy:
                        sw_b += n
                        sw_n_b += 1
                    else:
                        sw_s += n
                        sw_n_s += 1
            f[f"v3f_sweepimb_{tag}"] = _imb(sw_b, sw_s)
            f[f"v3f_sweepcnt_{tag}"] = float(sw_n_b - sw_n_s)
            f[f"v3f_sweepdepth_{tag}"] = sw_depth
        # ---------------- temporal
        cur = self.hist[-1]
        cur_d = dict(zip(TEMPORAL_KEYS, cur))
        scale = {"ofi": qb + qa + 1, "mlofi": cb[4] + ca[4] + 1, "sflow": 1.0, "cancel_asym": b10 + a10 + 1}
        for dms in DELTAS_MS:
            past = self._state_at(now - dms)
            for k, v in cur_d.items():
                name = f"v3t_{k}_d{dms}"
                if past is None:
                    f[name] = NAN
                    continue
                pv = past[TEMPORAL_KEYS.index(k)]
                if k == "mid":
                    f[name] = (v - pv) / pv * 1e4 if pv else NAN
                elif k in CUMULATIVE:
                    f[name] = _slog((v - pv) / scale[k]) if k != "sflow" else _slog(v - pv)
                else:
                    f[name] = v - pv
        for k in ("imb5", "micro5", "wimb20"):
            i = TEMPORAL_KEYS.index(k)
            xs = [(t, r[i]) for t, r in zip(self.hist_ts, self.hist) if t >= now - 5000]
            sgn = math.copysign(1.0, cur[i]) if cur[i] else 0.0
            f[f"v3t_{k}_persist5s"] = sum(1 for _, x in xs if x * sgn > 0) / len(xs) if xs and sgn else 0.0
            if len(xs) > 3:
                mt = sum(t for t, _ in xs) / len(xs)
                mv = sum(x for _, x in xs) / len(xs)
                den = sum((t - mt) ** 2 for t, _ in xs)
                f[f"v3t_{k}_slope5s"] = sum((t - mt) * (x - mv) for t, x in xs) / den * 1000 if den else 0.0
            else:
                f[f"v3t_{k}_slope5s"] = 0.0
            ys = [r[i] for t, r in zip(self.hist_ts, self.hist) if t >= now - 10_000]
            f[f"v3t_{k}_flips10s"] = float(sum(1 for x, y in zip(ys, ys[1:]) if x * y < 0))
        for k in ("ofi", "sflow", "mid"):
            p1, p2 = self._state_at(now - 1000), self._state_at(now - 2000)
            i = TEMPORAL_KEYS.index(k)
            if p1 is None or p2 is None:
                f[f"v3t_{k}_accel"] = NAN
                continue
            d1, d0 = cur[i] - p1[i], p1[i] - p2[i]
            if k == "mid":
                f[f"v3t_{k}_accel"] = (d1 - d0) / cur[i] * 1e4
            else:
                f[f"v3t_{k}_accel"] = _slog((d1 - d0) / (scale[k] if k != "sflow" else 1.0))
        # ---------------- explicit directional events
        s3 = self._sum_diffs(now - 3000, now)
        s1 = self._sum_diffs(now - 1000, now)
        buckets = [0.0] * 6
        for tr in reversed(self.trades):
            if tr.ts < now - 3000:
                break
            if tr.ts <= now:
                k = min(int((now - tr.ts) // 500), 5)
                buckets[k] += tr.notional if tr.buy else -tr.notional
        buy_persist = sum(1 for x in buckets if x > 0) / 6
        sell_persist = sum(1 for x in buckets if x < 0) / 6
        rep_ratio_a = min(s3.rep_a / s3.dep_a, 2.0) if s3.dep_a > 0 else 1.0
        rep_ratio_b = min(s3.rep_b / s3.dep_b, 2.0) if s3.dep_b > 0 else 1.0
        turn_a = s3.dep_a / (qa + 1)
        turn_b = s3.dep_b / (qb + 1)
        up_dep = turn_a * max(1 - rep_ratio_a, 0.0) * buy_persist
        dn_dep = turn_b * max(1 - rep_ratio_b, 0.0) * sell_persist
        f["v3x_depletion_up"] = up_dep
        f["v3x_depletion_dn"] = dn_dep
        f["v3x_depletion_net"] = up_dep - dn_dep
        buy1 = sum(t.notional for t in self.trades if now - 1000 <= t.ts <= now and t.buy)
        sell1 = sum(t.notional for t in self.trades if now - 1000 <= t.ts <= now and not t.buy)
        tot1 = buy1 + sell1
        pull_dn = s1.can_b10 / (b10 + 1) * (sell1 / tot1 if tot1 else 0.0)
        pull_up = s1.can_a10 / (a10 + 1) * (buy1 / tot1 if tot1 else 0.0)
        f["v3x_pull_up"] = pull_up
        f["v3x_pull_dn"] = pull_dn
        f["v3x_pull_net"] = pull_up - pull_dn
        sell3 = sum(t.notional for t in self.trades if now - 3000 <= t.ts <= now and not t.buy)
        buy3 = sum(t.notional for t in self.trades if now - 3000 <= t.ts <= now and t.buy)
        p3 = self._state_at(now - 3000)
        mid3 = p3[0] if p3 else mid
        held_bid = 1.0 if bb >= mid3 * (1 - spread_bps / 2e4) - 1e-12 else 0.0
        held_ask = 1.0 if ba <= mid3 * (1 + spread_bps / 2e4) + 1e-12 else 0.0
        absorb_bid = sell3 / (cb[4] + 1) * held_bid * min(rep_ratio_b, 2.0)
        absorb_ask = buy3 / (ca[4] + 1) * held_ask * min(rep_ratio_a, 2.0)
        f["v3x_absorb_bid"] = absorb_bid          # bullish: selling absorbed by a holding bid
        f["v3x_absorb_ask"] = absorb_ask          # bearish
        f["v3x_absorb_net"] = absorb_bid - absorb_ask
        f["v3x_pressure_net"] = (up_dep - dn_dep) + (pull_up - pull_dn) + (absorb_bid - absorb_ask)
        return f

    def _sum_diffs(self, t0: int, t1: int) -> DiffRec:
        s = DiffRec(t1)
        for r in reversed(self.diffs):
            if r.ts < t0:
                break
            if r.ts > t1:
                continue
            s.dep_b += r.dep_b
            s.dep_a += r.dep_a
            s.rep_b += r.rep_b
            s.rep_a += r.rep_a
            s.can_b5 += r.can_b5
            s.can_a5 += r.can_a5
            s.can_b10 += r.can_b10
            s.can_a10 += r.can_a10
            s.add_b10 += r.add_b10
            s.add_a10 += r.add_a10
            s.exe_b += r.exe_b
            s.exe_a += r.exe_a
            s.rapid_b += r.rapid_b
            s.rapid_a += r.rapid_a
            s.bid_up += r.bid_up
            s.bid_dn += r.bid_dn
            s.ask_up += r.ask_up
            s.ask_dn += r.ask_dn
        return s

    def _infer_tick(self) -> float:
        b, a = self.book.top(20)
        diffs = [round(x[0] - y[0], 10) for x, y in zip(b, b[1:])] + [round(y[0] - x[0], 10) for x, y in zip(a, a[1:])]
        diffs = [d for d in diffs if d > 0]
        if diffs:
            self.tick = min(diffs)
        return self.tick or 0.0

    def timeline(self, now: int) -> dict[str, float] | None:
        """Compact per-evaluation record for the entry-timing study."""
        if not self.ready(now) or not self.hist:
            return None
        r = dict(zip(TEMPORAL_KEYS, self.hist[-1]))
        s1 = self._sum_diffs(now - 1000, now)
        buy1 = sum(t.notional for t in self.trades if now - 1000 <= t.ts <= now and t.buy)
        sell1 = sum(t.notional for t in self.trades if now - 1000 <= t.ts <= now and not t.buy)
        return {"mid": r["mid"], "micro5": r["micro5"], "imb5": r["imb5"], "wimb20": r["wimb20"],
                "ofi": r["ofi"], "sflow": r["sflow"], "can_asym1s": _imb(s1.can_a10, s1.can_b10),
                "dep_asym1s": _imb(s1.dep_a, s1.dep_b), "flow_imb1s": _imb(buy1, sell1)}


def feature_group(name: str) -> str:
    """Ablation group of a V3-dataset column."""
    if name.startswith("v3d_"):
        # touch-only quantities are L1 information even when computed from the L2 book:
        # level-1 depth, spread, L1 microprice and best-queue depletion/replenishment/lifetime
        if name.endswith("_1") or name.startswith(L1_V3D):
            return "l1"
        return "l2"
    if name.startswith("v3t_"):
        base = name[4:].split("_")[0]
        return "flow" if base == "sflow" else ("l1" if base in ("mid", "micro1", "imb1", "ofi") else "l2")
    if name.startswith("v3f_"):
        return "flow"
    if name.startswith("v3x_"):
        return "l2flow"
    return v2_group(name)


L1_V3D = ("v3d_spread", "v3d_micro1", "v3d_dep_", "v3d_rep", "v3d_qpersist", "v3d_qlife", "v3d_bestmove")
V2_FLOW_PREFIXES = ("buy_notional", "sell_notional", "flow_imb", "aggr_buy_ratio", "trades_per_s", "notional_per_s",
                    "avg_trade", "trade_ret", "vol_accel", "velocity_change", "large_trade", "flow_price_agree",
                    "fimb_", "ntr_rate_", "vol_rate_log_", "cnt_accel_", "vol_accel_", "tret_")


def v2_group(name: str) -> str:
    base = name
    for pre in ("rmean_", "rmax_", "rmin_", "rslope_", "rpers_", "rchg_"):
        if name.startswith(pre):
            base = name[len(pre):]
    return "flow" if base.startswith(V2_FLOW_PREFIXES) else "l1"
