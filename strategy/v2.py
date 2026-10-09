"""V2 strategy: calibrated P(TP before SL) models -> expected NET value -> entry.

V1 (rule-based score -> drift -> barrier model) is untouched and remains the default;
V2 is selected with ``strategy.predictor = "v2"`` and plugs into the same bot, replay,
execution simulator, risk manager and trade logging.

Entry gate (all must hold):
  data quality / spread / abnormal-volatility / liquidity / slippage / size / risk gates
  (shared with V1), then for every (side, target) the model covers:
    p   = calibrated P(TP first)
    c   = round-trip cost bps from the live book (taker entry + taker exit)
    EV  = N/1e4 * (p*(T - c) - (1-p)*(S + c))
  enter the (side, target) with the best EV if p >= threshold, EV >= min_net and the
  target alone nets >= min_net. Exits are barrier exits (exit.mode = "barrier").
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np

from config import BotConfig
from features.v2_features import FeatureHistory, v2_features
from market_data.orderbook import OrderBook
from market_data.tradeflow import TradeFlow
from strategy.barrier import BarrierOutcome
from strategy.costs import CostModel
from strategy.entry_filter import EntryDecision, EntryPlan
from strategy.predictor import Prediction
from strategy.signal_engine import SignalEngine, SignalResult


# ====================================================================== models
class _Calib:
    def __init__(self, d: dict) -> None:
        self.method, self.a, self.b = d["method"], d.get("a", 1.0), d.get("b", 0.0)
        self.xs, self.ys = d.get("xs"), d.get("ys")

    def __call__(self, p: float) -> float:
        p = min(max(p, 1e-6), 1 - 1e-6)
        if self.method == "platt":
            return float(1 / (1 + np.exp(-(self.a * np.log(p / (1 - p)) + self.b))))
        if self.method == "isotonic":
            return float(np.interp(p, self.xs, self.ys))
        return p


class V2Model:
    """Loads a model directory written by research.v2_models (spec.json + artifacts)."""

    def __init__(self, model_dir: str) -> None:
        with open(os.path.join(model_dir, "spec.json"), encoding="utf-8") as fh:
            self.spec = spec = json.load(fh)
        self.kind = spec["kind"]
        self.features: list[str] = spec["features"]
        self.stop_bps: int = spec["stop_bps"]
        self.threshold: float = spec["threshold"]
        self.keys: list[tuple[str, int]] = []
        self._fns = {}
        for key, entry in spec["models"].items():
            side, t = key.rsplit("_", 1)
            k = (side, int(t))
            self.keys.append(k)
            self._fns[k] = (self._load(model_dir, entry), _Calib(entry["calibrator"]))

    def _load(self, d: str, entry: dict):
        if self.kind == "lightgbm":
            import lightgbm as lgb

            b = lgb.Booster(model_file=os.path.join(d, entry["file"]))
            return lambda x: float(b.predict(x, num_threads=1)[0])
        if self.kind == "xgboost":
            import xgboost as xgb

            b = xgb.Booster()
            b.load_model(os.path.join(d, entry["file"]))
            b.set_param({"nthread": 1})
            rng = (0, int(entry.get("best_iteration", -1)) + 1) if "best_iteration" in entry else (0, 0)
            return lambda x: float(b.inplace_predict(x, iteration_range=rng)[0])
        p = entry["params"]
        med, mu, sd = np.array(p["med"]), np.array(p["mu"]), np.array(p["sd"])
        if self.kind == "logistic":
            w, c = np.array(p["coef"]), p["intercept"]
            return lambda x: float(1 / (1 + np.exp(-(np.clip((np.where(np.isnan(x[0]), med, x[0]) - mu) / sd,
                                                             -8, 8) @ w + c))))
        if self.kind == "mlp":
            W = [np.array(w) for w in p["W"]]
            B = [np.array(b) for b in p["b"]]

            def f(x):
                h = np.clip((np.where(np.isnan(x[0]), med, x[0]) - mu) / sd, -8, 8)
                for i, (w, b) in enumerate(zip(W, B)):
                    h = h @ w + b
                    if i < len(W) - 1:
                        h = np.maximum(h, 0)
                return float(1 / (1 + np.exp(-h[0])))
            return f
        raise ValueError(f"unknown model kind {self.kind}")

    def vector(self, f: dict[str, float]) -> np.ndarray:
        x = np.array([[f.get(c, np.nan) for c in self.features]], dtype=np.float32)
        x[~np.isfinite(x)] = np.nan
        return x

    def predict(self, f: dict[str, float]) -> dict[tuple[str, int], float]:
        x = self.vector(f)
        return {k: cal(fn(x)) for k, (fn, cal) in self._fns.items()}


class TwoStageModel:
    """Research model from research.v2_twostage: Stage A P(|move| >= T bps within 60 s), then --
    only where Stage A passes its threshold -- Stage B P(UP | large move). The predicted side
    gets the calibrated P(TP first) of the combined score; the other side gets nothing."""

    def __init__(self, model_dir: str) -> None:
        with open(os.path.join(model_dir, "spec.json"), encoding="utf-8") as fh:
            self.spec = spec = json.load(fh)
        self.kind = f"twostage_{spec['base']}"
        self.features: list[str] = spec["features"]
        self.stop_bps: int = spec["stop_bps"]
        self.threshold: float = spec["threshold"]
        self.b_thr: float = spec["b_thr"]
        self.targets = [int(t) for t in spec["targets"]]
        self.keys = [(s, t) for t in self.targets for s in ("long", "short")]
        loader = V2Model.__new__(V2Model)
        loader.kind = spec["base"]
        self._a = {t: loader._load(model_dir, spec["stage_a"][str(t)]) for t in self.targets}
        self._b = {t: (loader._load(model_dir, spec["stage_b"][str(t)]), _Calib(spec["stage_b"][str(t)]["calibrator"]))
                   for t in self.targets}
        self._c = {t: _Calib(spec["combo_calib"][str(t)]) for t in self.targets}
        self.a_thr = {t: float(spec["a_thr"][str(t)]) for t in self.targets}
        self.last_reject = ""

    vector = V2Model.vector

    def predict(self, f: dict[str, float]) -> dict[tuple[str, int], float]:
        x = self.vector(f)
        out: dict[tuple[str, int], float] = {}
        any_a = False
        for t in self.targets:
            pa = self._a[t](x)
            if pa < self.a_thr[t]:
                continue
            any_a = True
            fn, cal = self._b[t]
            pb = cal(fn(x))
            conf = max(pb, 1 - pb)
            if conf < self.b_thr:
                continue
            out[("long" if pb >= 0.5 else "short", t)] = self._c[t](pa * conf)
        self.last_reject = "" if out else ("stage_b_direction_weak" if any_a else "stage_a_no_large_move")
        return out


def load_v2_model(model_dir: str):
    with open(os.path.join(model_dir, "spec.json"), encoding="utf-8") as fh:
        kind = json.load(fh)["kind"]
    return TwoStageModel(model_dir) if kind == "twostage" else V2Model(model_dir)


# ====================================================================== entry gate
@dataclass
class V2Choice:
    side: str
    target: int
    p: float
    ev: float
    cost_bps: float


class V2EntryFilter:
    def __init__(self, cfg: BotConfig, costs: CostModel, model: "V2Model | TwoStageModel") -> None:
        self.cfg = cfg
        self.costs = costs
        self.model = model
        self.threshold = cfg.strategy.v2_threshold if cfg.strategy.v2_threshold is not None else model.threshold

    def evaluate(self, symbol: str, probs: dict[tuple[str, int], float], f: dict[str, float],
                 book: OrderBook) -> EntryDecision:
        e = self.cfg.entry
        md = self.cfg.market_data
        if not f:
            return EntryDecision(False, "no_features")
        if f["book_age_ms"] > md.stale_after_ms or f["flow_age_ms"] > md.stale_after_ms * 5:
            return EntryDecision(False, "stale_data")
        if f["spread_bps"] > e.max_spread_bps:
            return EntryDecision(False, "spread_too_wide")
        if abs(f.get("imb_l10", 0.0)) > e.max_book_imbalance_abs:
            return EntryDecision(False, "abnormal_book_one_sided")
        if f.get("rv_1s_bps", 0.0) > e.max_realized_vol_bps_1s:
            return EntryDecision(False, "abnormal_volatility")
        if f.get("liquidity_change", 0.0) < e.min_liquidity_change:
            return EntryDecision(False, "liquidity_withdrawal")
        if f.get("trades_per_s_10s", 0.0) < e.min_trades_per_s:
            return EntryDecision(False, "thin_tape")
        if min(f.get("bid_depth_10bps", 0.0), f.get("ask_depth_10bps", 0.0)) < e.min_depth_usdt_within_10bps:
            return EntryDecision(False, "insufficient_depth")

        if not probs:
            return EntryDecision(False, getattr(self.model, "last_reject", "") or "no_model_output")
        S = self.model.stop_bps
        notional = min(self.cfg.sizing.position_notional_usdt, self.cfg.risk.max_position_notional_usdt)
        best: V2Choice | None = None
        best_any: V2Choice | None = None
        cost_cache: dict[str, tuple] = {}
        for (side, T), p in probs.items():
            if side not in cost_cache:
                d = 1 if side == "long" else -1
                ce = self.costs.estimate(book, d, notional, entry_maker=False)
                cost_cache[side] = (ce, ce.total_bps)
            ce, c = cost_cache[side]
            if not np.isfinite(c):
                continue
            ev = notional / 1e4 * (p * (T - c) - (1 - p) * (S + c))
            ch = V2Choice(side, T, p, ev, c)
            if best_any is None or ev > best_any.ev:
                best_any = ch
            if p >= self.threshold and ev >= e.min_net_profit_usdt and notional / 1e4 * (T - c) >= e.min_net_profit_usdt:
                if best is None or ev > best.ev:
                    best = ch
        details = {}
        if best_any is not None:
            details = {"p_target": best_any.p, "target_bps": best_any.target, "stop_bps": S,
                       "expected_net": best_any.ev, "costs_bps": best_any.cost_bps}
        if not cost_cache or all(not np.isfinite(v[1]) for v in cost_cache.values()):
            return EntryDecision(False, "slippage_too_high", details=details)
        if best is None:
            if best_any is not None and best_any.p < self.threshold:
                return EntryDecision(False, "p_target_too_low", details=details)
            return EntryDecision(False, "expected_net_below_min", details=details)
        ce, c = cost_cache[best.side]
        if ce.exit_slippage_bps > e.max_expected_slippage_bps or ce.entry_slippage_bps > e.max_expected_slippage_bps:
            return EntryDecision(False, "slippage_too_high", details=details)
        loss_if_stop = notional * S / 1e4 + ce.total_usdt
        if loss_if_stop > self.cfg.sizing.max_risk_per_trade_usdt * 1.05:
            return EntryDecision(False, "risk_per_trade_exceeded", details=details)
        horizon = self.cfg.strategy.max_hold_s
        plan = EntryPlan(
            symbol=symbol, direction=1 if best.side == "long" else -1, notional=notional, entry_maker=False,
            target_bps=float(best.target), stop_bps=float(S), horizon_s=horizon,
            outcome=BarrierOutcome(best.p, 1 - best.p, 0.0, best.p * best.target - (1 - best.p) * S, horizon / 2),
            costs=ce, expected_gross_usdt=best.ev + ce.total_usdt, expected_net_usdt=best.ev,
            net_if_target_usdt=notional * best.target / 1e4 - ce.total_usdt, loss_if_stop_usdt=loss_if_stop,
            ref_mid=f["mid"], confidence=best.p, score=best.p if best.side == "long" else -best.p,
        )
        details.update(p_target=best.p, target_bps=best.target, expected_net=best.ev)
        return EntryDecision(True, "ok", plan=plan, details=details)


# ====================================================================== signal engine
class SignalEngineV2(SignalEngine):
    """Same pipeline/recording as V1's SignalEngine, with the V2 model and gate."""

    def __init__(self, cfg: BotConfig, costs: CostModel, recorder, risk, model_dir: str) -> None:
        self.model = load_v2_model(model_dir)
        super().__init__(cfg, None, None, recorder, risk)   # predictor/filter replaced below
        self.v2_filter = V2EntryFilter(cfg, costs, self.model)
        self.histories: dict[str, FeatureHistory] = {}

    def _v2(self, symbol: str, book: OrderBook, flow: TradeFlow, now_ms: int):
        base = self.features(book, flow, now_ms)
        if not base:
            return base, {}, {}
        hist = self.histories.setdefault(symbol, FeatureHistory())
        fv = v2_features(book, flow, base, hist, now_ms)
        hist.add(now_ms, base)
        return base, fv, self.model.predict(fv)

    def evaluate(self, symbol: str, book: OrderBook, flow: TradeFlow, now_ms: int,
                 trading_enabled: bool, record: bool = True) -> SignalResult:
        f, fv, probs = self._v2(symbol, book, flow, now_ms)
        if not f:
            return SignalResult(symbol, now_ms, f, None, EntryDecision(False, "no_features"), "reject")
        pl = max((p for (s, _), p in probs.items() if s == "long"), default=0.0)
        ps = max((p for (s, _), p in probs.items() if s == "short"), default=0.0)
        tot = pl + ps
        pred = Prediction(score=pl - ps, p_long=pl / tot if tot else 0.5, p_short=ps / tot if tot else 0.5,
                          drift_bps_per_s=0.0, sigma_1s_bps=f.get("rv_1s_bps", 0.0), flow_confirms=True,
                          components={"p_long_best": pl, "p_short_best": ps}, model=f"v2:{self.model.kind}")
        decision = self.v2_filter.evaluate(symbol, probs, f, book)
        action = "reject"
        if decision.ok and decision.plan is not None:
            plan = decision.plan
            ok, reason = self.risk.can_open(symbol, plan.notional, self.cfg.sizing.leverage, f["spread_bps"],
                                            plan.costs.exit_slippage_bps)
            if not ok:
                decision = EntryDecision(False, f"risk:{reason}", plan=plan, details=decision.details)
            else:
                action = "enter" if trading_enabled else "would_enter"
        if not decision.ok:
            self.rejections[decision.reason] = self.rejections.get(decision.reason, 0) + 1
        if record:
            key = action if action != "reject" else decision.reason
            self.decisions[key] = self.decisions.get(key, 0) + 1
        result = SignalResult(symbol, now_ms, f, pred, decision, action)
        if record and self.recorder is not None:
            sample = self.recorder.should_record(symbol, abs(pred.score), action != "reject", now_ms)
            if sample is not None:
                result.signal_id = self.recorder.record_signal(
                    symbol, now_ms, f, pred, action, "" if action != "reject" else decision.reason, sample,
                    dict(decision.details))
        return result

    def recheck_taker(self, symbol: str, book: OrderBook, flow: TradeFlow, now_ms: int):
        f = self.features(book, flow, now_ms)
        if not f:
            return None
        hist = self.histories.get(symbol) or FeatureHistory()
        probs = self.model.predict(v2_features(book, flow, f, hist, now_ms))
        d = self.v2_filter.evaluate(symbol, probs, f, book)
        return d.plan if d.ok else None
