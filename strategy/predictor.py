"""Directional predictors.

Every predictor maps a flat feature dict to a :class:`Prediction`. The rest of the
system (entry filter, execution, exits) only sees ``Prediction``, so the rule-based
engine can be swapped for a trained model without touching execution code.

A prediction carries a signed drift (bps/s) and a volatility (bps/sqrt(s)); the entry
filter turns these into target-before-stop probabilities and expected value with the
finite-horizon barrier model (:mod:`strategy.barrier`).
"""
from __future__ import annotations

import json
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from config import HORIZONS_S, StrategyConfig
from strategy.barrier import BarrierOutcome, barrier_outcome
from utils.mathx import clip, sigmoid

MIN_SIGMA_BPS = 0.3


@dataclass
class Prediction:
    score: float                      # signed, [-1, 1]
    p_long: float
    p_short: float
    drift_bps_per_s: float            # signed expected mid drift
    sigma_1s_bps: float
    flow_confirms: bool
    components: dict[str, float] = field(default_factory=dict)
    model: str = "rule"

    @property
    def direction(self) -> int:
        if self.p_long > self.p_short:
            return 1
        if self.p_short > self.p_long:
            return -1
        return 0

    @property
    def confidence(self) -> float:
        return max(self.p_long, self.p_short)

    def expected_move_bps(self, horizon_s: float) -> float:
        return self.drift_bps_per_s * horizon_s

    def expected_moves(self) -> dict[int, float]:
        return {h: self.expected_move_bps(h) for h in HORIZONS_S}

    def outcome(self, direction: int, target_bps: float, stop_bps: float, horizon_s: float) -> BarrierOutcome:
        """Target-before-stop model for a trade in ``direction``."""
        mu = self.drift_bps_per_s * direction
        return barrier_outcome(target_bps, stop_bps, mu, max(self.sigma_1s_bps, MIN_SIGMA_BPS), horizon_s)


class BasePredictor(ABC):
    name = "base"

    @abstractmethod
    def predict(self, f: dict[str, float]) -> Prediction: ...


class RuleBasedPredictor(BasePredictor):
    """Weighted microstructure score -> logistic probability -> drift.

    Components (each in [-1, 1], positive == bullish):
      book        multi-level decayed depth imbalance
      ofi         order-flow imbalance from best-quote changes (3s)
      flow_short  aggressive buy/sell imbalance (3s)
      flow_long   aggressive buy/sell imbalance (10s)
      micro       microprice tilt within the spread
      momentum    3s mid return scaled by volatility
      persistence fraction of recent snapshots with bid-heavy book
      depletion   ask-side minus bid-side depth depletion

    The heuristic mapping (logistic_k, drift_per_score) is a placeholder until it is
    calibrated against recorded, labelled data (see research/).
    """

    name = "rule"

    def __init__(self, cfg: StrategyConfig) -> None:
        self.cfg = cfg

    def components(self, f: dict[str, float]) -> dict[str, float]:
        sigma = max(f.get("rv_1s_bps", 0.0), MIN_SIGMA_BPS)
        return {
            "book": f.get("imb_weighted", 0.0),
            "ofi": f.get("ofi_3s", 0.0),
            "flow_short": f.get("flow_imb_3s", 0.0),
            "flow_long": f.get("flow_imb_10s", 0.0),
            "micro": f.get("micro_tilt", 0.0),
            "momentum": math.tanh(f.get("mom_3s_bps", 0.0) / (sigma * math.sqrt(3.0))),
            "persistence": f.get("persistence", 0.0),
            "depletion": f.get("depletion_asym", 0.0),
        }

    def predict(self, f: dict[str, float]) -> Prediction:
        c = self.cfg
        comp = self.components(f)
        weights = {
            "book": c.w_book_imbalance,
            "ofi": c.w_ofi,
            "flow_short": c.w_flow_short,
            "flow_long": c.w_flow_long,
            "micro": c.w_microprice,
            "momentum": c.w_momentum,
            "persistence": c.w_persistence,
            "depletion": c.w_depletion,
        }
        wsum = sum(weights.values()) or 1.0
        score = sum(weights[k] * comp[k] for k in weights) / wsum

        # Down-weight thin tape: few trades means the flow components are noise.
        tps = f.get("trades_per_s_10s", 0.0)
        score *= clip(tps / 2.0, 0.0, 1.0)
        score = clip(score, -1.0, 1.0)

        book_side = (comp["book"] + comp["ofi"] + comp["micro"]) / 3.0
        flow_side = (comp["flow_short"] + comp["flow_long"]) / 2.0
        price_agree = f.get("flow_price_agree_3s", 0.0)
        flow_confirms = (
            abs(book_side) > 0.05
            and abs(flow_side) > 0.05
            and (book_side > 0) == (flow_side > 0)
            and (flow_side > 0) == (score > 0)
            and price_agree >= 0
        )

        p_long = sigmoid(c.logistic_k * score)
        sigma = max(f.get("rv_1s_bps", 0.0), MIN_SIGMA_BPS)
        drift = score * c.drift_per_score * sigma
        return Prediction(
            score=score,
            p_long=p_long,
            p_short=1.0 - p_long,
            drift_bps_per_s=drift,
            sigma_1s_bps=sigma,
            flow_confirms=flow_confirms,
            components=comp,
            model=self.name,
        )


class LinearModelPredictor(BasePredictor):
    """Logistic model trained offline (research/train_model.py).

    JSON format::
        {"features": [...], "mean": [...], "std": [...], "coef": [...],
         "intercept": float, "drift_per_edge": float}

    ``drift_per_edge`` maps (2p - 1) to drift in units of sigma_1s per second; it is
    estimated during training from realised forward returns.
    """

    name = "linear"

    def __init__(self, path: str, fallback: RuleBasedPredictor) -> None:
        with open(path, encoding="utf-8") as fh:
            m = json.load(fh)
        self.features: list[str] = m["features"]
        self.mean: list[float] = m["mean"]
        self.std: list[float] = m["std"]
        self.coef: list[float] = m["coef"]
        self.intercept: float = m["intercept"]
        self.drift_per_edge: float = m.get("drift_per_edge", 0.3)
        self.fallback = fallback

    def predict(self, f: dict[str, float]) -> Prediction:
        z = self.intercept
        for name, mu, sd, w in zip(self.features, self.mean, self.std, self.coef):
            x = f.get(name, mu)
            z += w * ((x - mu) / sd if sd > 0 else 0.0)
        p_long = sigmoid(z)
        edge = 2.0 * p_long - 1.0
        sigma = max(f.get("rv_1s_bps", 0.0), MIN_SIGMA_BPS)
        rule = self.fallback.predict(f)   # reuse for flow confirmation diagnostics
        return Prediction(
            score=edge,
            p_long=p_long,
            p_short=1.0 - p_long,
            drift_bps_per_s=edge * self.drift_per_edge * sigma,
            sigma_1s_bps=sigma,
            flow_confirms=rule.flow_confirms and (rule.score > 0) == (edge > 0),
            components={"rule_score": rule.score, **rule.components},
            model=self.name,
        )


def build_predictor(cfg: StrategyConfig) -> BasePredictor:
    rule = RuleBasedPredictor(cfg)
    if cfg.predictor == "linear":
        return LinearModelPredictor(cfg.model_path, rule)
    return rule
