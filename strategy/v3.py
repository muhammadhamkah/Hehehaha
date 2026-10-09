"""V3 serving (research-only): L2 features -> calibrated P(TP before SL) per barrier pair.

Selected with ``strategy.predictor = "v3"`` and ``v3.model_dir``; refused in live mode.
The bot forwards every raw detail message to ``SignalEngineV3.on_detail``; each symbol
keeps the same ``v3.state.SymbolState`` the dataset builder uses, so features are
computed identically in research and in replay.

Entry gate: shared data-quality gates, optional Stage-A gate (P(large move) >= threshold
chosen on the selection days), then for every (side, TP, SL) the model covers
    c  = depth-based round-trip cost from the live L2 book (same function as the dataset)
    EV = N/1e4 * (p*(TP - c) - (1-p)*(SL + c))
enter the best EV if p >= threshold, EV >= min_net and TP alone nets >= min_net.
Exits: barrier mode (TP / SL / time stop at the pair horizon).
"""
from __future__ import annotations

import json
import os

import numpy as np

from config import BotConfig
from market_data.orderbook import OrderBook
from market_data.tradeflow import TradeFlow
from strategy.barrier import BarrierOutcome
from strategy.costs import CostModel
from strategy.entry_filter import EntryDecision, EntryPlan
from strategy.predictor import Prediction
from strategy.signal_engine import SignalEngine, SignalResult
from strategy.v2 import _Calib
from v3.dataset import costs_from_book
from v3.state import SymbolState


class V3Model:
    def __init__(self, model_dir: str) -> None:
        import lightgbm as lgb

        with open(os.path.join(model_dir, "spec.json"), encoding="utf-8") as fh:
            self.spec = spec = json.load(fh)
        self.name = spec["name"]
        self.features: list[str] = spec["features"]
        self.threshold: float = spec["threshold"]
        self.horizon_s: float = spec["horizon_s"]
        self.models = {}
        for key, e in spec["models"].items():
            side, T, S = key.split("_")
            b = lgb.Booster(model_file=os.path.join(model_dir, e["file"]))
            self.models[(side, int(T), int(S))] = (b, _Calib(e["calibrator"]))
        self.stage_a = None
        if spec.get("stage_a"):
            sa = spec["stage_a"]
            self.stage_a = (lgb.Booster(model_file=os.path.join(model_dir, sa["file"])), sa["features"], sa["threshold"])

    @staticmethod
    def _vec(row: dict, cols: list[str]) -> np.ndarray:
        x = np.array([[row.get(c, np.nan) for c in cols]], dtype=np.float32)
        x[~np.isfinite(x)] = np.nan
        return x

    def stage_a_pass(self, row: dict) -> tuple[bool, float | None]:
        if self.stage_a is None:
            return True, None
        b, cols, thr = self.stage_a
        p = float(b.predict(self._vec(row, cols), num_threads=1)[0])
        return p >= thr, p

    def predict(self, row: dict) -> dict[tuple[str, int, int], float]:
        x = self._vec(row, self.features)
        return {k: cal(float(b.predict(x, num_threads=1)[0])) for k, (b, cal) in self.models.items()}


class SignalEngineV3(SignalEngine):
    def __init__(self, cfg: BotConfig, costs: CostModel, recorder, risk, model_dir: str) -> None:
        super().__init__(cfg, None, None, recorder, risk)
        self.model = V3Model(model_dir)
        self.costs_model = costs
        self.states: dict[str, SymbolState] = {}
        self.threshold = cfg.v3.threshold if cfg.v3.threshold is not None else self.model.threshold
        self.notional = min(cfg.sizing.position_notional_usdt, cfg.risk.max_position_notional_usdt)

    # ------------------------------------------------------------------ raw input
    def on_detail(self, stream: str, data, ts: int) -> None:
        head, _, kind = stream.partition("@")
        sym = kind if head.startswith("__") else head.upper()
        st = self.states.get(sym)
        if st is None:
            st = self.states[sym] = SymbolState(sym, cfg=self.cfg)
        st.on_message(stream, data, ts)

    # ------------------------------------------------------------------ decision
    def _decide(self, symbol: str, f: dict, book: OrderBook, now_ms: int) -> tuple[EntryDecision, dict]:
        e, md = self.cfg.entry, self.cfg.market_data
        if not f:
            return EntryDecision(False, "no_features"), {}
        if f["book_age_ms"] > md.stale_after_ms or f["flow_age_ms"] > md.stale_after_ms * 5:
            return EntryDecision(False, "stale_data"), {}
        if f["spread_bps"] > e.max_spread_bps:
            return EntryDecision(False, "spread_too_wide"), {}
        if f.get("rv_1s_bps", 0.0) > e.max_realized_vol_bps_1s:
            return EntryDecision(False, "abnormal_volatility"), {}
        st = self.states.get(symbol)
        if st is None:
            return EntryDecision(False, "v3_no_state"), {}
        st.advance(now_ms)
        row = st.last_row
        if row is None or now_ms - st.last_eval_ts > 2 * st.eval_ms:
            return EntryDecision(False, "v3_features_unavailable"), {}
        ok, pa = self.model.stage_a_pass(row)
        if not ok:
            return EntryDecision(False, "stage_a_no_large_move", details={"p_stage_a": pa}), {}
        probs = self.model.predict(row)
        N = self.notional
        cost = costs_from_book(st.engine.book, self.cfg.costs, N)
        best, best_any = None, None
        for (side, T, S), p in probs.items():
            c = cost[f"cost_{side}_{int(N)}"]
            if not np.isfinite(c):
                continue
            ev = N / 1e4 * (p * (T - c) - (1 - p) * (S + c))
            ch = (ev, side, T, S, p, c)
            if best_any is None or ev > best_any[0]:
                best_any = ch
            if p >= self.threshold and ev >= e.min_net_profit_usdt and N / 1e4 * (T - c) >= e.min_net_profit_usdt:
                if best is None or ev > best[0]:
                    best = ch
        details = {}
        if best_any is not None:
            details = {"p_target": best_any[4], "target_bps": best_any[2], "stop_bps": best_any[3],
                       "expected_net": best_any[0], "costs_bps": best_any[5]}
        if best is None:
            if best_any is None:
                return EntryDecision(False, "slippage_too_high", details=details), probs
            reason = "p_target_too_low" if best_any[4] < self.threshold else "expected_net_below_min"
            return EntryDecision(False, reason, details=details), probs
        ev, side, T, S, p, c = best
        d = 1 if side == "long" else -1
        ce = self.costs_model.estimate(book, d, N, entry_maker=False)
        loss_if_stop = N * S / 1e4 + N * c / 1e4
        if loss_if_stop > self.cfg.sizing.max_risk_per_trade_usdt * 1.05:
            return EntryDecision(False, "risk_per_trade_exceeded", details=details), probs
        h = self.model.horizon_s
        plan = EntryPlan(symbol=symbol, direction=d, notional=N, entry_maker=False, target_bps=float(T),
                         stop_bps=float(S), horizon_s=h,
                         outcome=BarrierOutcome(p, 1 - p, 0.0, p * T - (1 - p) * S, h / 2), costs=ce,
                         expected_gross_usdt=ev + N * c / 1e4, expected_net_usdt=ev,
                         net_if_target_usdt=N * (T - c) / 1e4, loss_if_stop_usdt=loss_if_stop, ref_mid=f["mid"],
                         confidence=p, score=p if d > 0 else -p)
        details.update(p_target=p, target_bps=T, stop_bps=S, expected_net=ev)
        return EntryDecision(True, "ok", plan=plan, details=details), probs

    def evaluate(self, symbol: str, book: OrderBook, flow: TradeFlow, now_ms: int, trading_enabled: bool,
                 record: bool = True) -> SignalResult:
        f = self.features(book, flow, now_ms)
        decision, probs = self._decide(symbol, f, book, now_ms)
        pl = max((p for (s, _, _), p in probs.items() if s == "long"), default=0.0)
        ps = max((p for (s, _, _), p in probs.items() if s == "short"), default=0.0)
        tot = pl + ps
        pred = Prediction(score=pl - ps, p_long=pl / tot if tot else 0.5, p_short=ps / tot if tot else 0.5,
                          drift_bps_per_s=0.0, sigma_1s_bps=(f or {}).get("rv_1s_bps", 0.0), flow_confirms=True,
                          components={"p_long_best": pl, "p_short_best": ps}, model=f"v3:{self.model.name}")
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
        if record and self.recorder is not None and f:
            sample = self.recorder.should_record(symbol, abs(pred.score), action != "reject", now_ms)
            if sample is not None:
                result.signal_id = self.recorder.record_signal(
                    symbol, now_ms, f, pred, action, "" if action != "reject" else decision.reason, sample,
                    dict(decision.details))
        return result

    def recheck_taker(self, symbol: str, book: OrderBook, flow: TradeFlow, now_ms: int):
        d, _ = self._decide(symbol, self.features(book, flow, now_ms), book, now_ms)
        return d.plan if d.ok else None
