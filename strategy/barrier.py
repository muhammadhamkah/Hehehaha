"""Finite-horizon double-barrier outcome model.

Given a (predicted) drift ``mu`` and volatility ``sigma`` of the mid price, a profit
barrier ``+a`` and a stop barrier ``-b`` (all in bps, per-second units), and a maximum
holding time ``T``, compute:

  * P(target first), P(stop first), P(time-out)
  * expected exit value (bps) including the time-out mark
  * expected holding time

Uses forward propagation of probability mass on a trinomial lattice (vectorized with
numpy). Falls back to the closed-form infinite-horizon result when the horizon is
long relative to the barriers.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from utils.mathx import prob_hit_upper_first


@dataclass(frozen=True)
class BarrierOutcome:
    p_target: float
    p_stop: float
    p_timeout: float
    expected_value_bps: float
    expected_hold_s: float


def barrier_outcome(a: float, b: float, mu: float, sigma: float, horizon_s: float,
                    max_steps: int = 3000) -> BarrierOutcome:
    if a <= 0 or b <= 0 or horizon_s <= 0:
        raise ValueError("a, b and horizon must be positive")
    if sigma <= 1e-9:
        # Deterministic drift.
        if mu > 0 and a / mu <= horizon_s:
            return BarrierOutcome(1.0, 0.0, 0.0, a, a / mu)
        if mu < 0 and b / -mu <= horizon_s:
            return BarrierOutcome(0.0, 1.0, 0.0, -b, b / -mu)
        return BarrierOutcome(0.0, 0.0, 1.0, mu * horizon_s, horizon_s)

    width = a + b
    dx_target = sigma * math.sqrt(3.0 * horizon_s / 400.0)
    m0 = int(min(max(round(width / dx_target), 4), 120))
    # Choose the grid size so the start point (b from the stop) falls on a node;
    # otherwise rounding biases the probabilities.
    m = min(range(m0, int(m0 * 1.5) + 2), key=lambda k: abs(b * k / width - round(b * k / width)))
    dx = width / m
    dt = dx * dx / (3.0 * sigma * sigma)
    steps = int(math.ceil(horizon_s / dt))
    if steps > max_steps:
        # Horizon is long relative to the barriers: the infinite-horizon result is accurate.
        p = prob_hit_upper_first(a, b, mu, sigma)
        hold = _approx_hold(a, b, mu, sigma, p)
        return BarrierOutcome(p, 1.0 - p, 0.0, p * a - (1.0 - p) * b, min(hold, horizon_s))
    dt = horizon_s / steps
    drift_term = mu * dt / (2.0 * dx)
    var_term = sigma * sigma * dt / (2.0 * dx * dx)   # (pu + pd) / 2, <= 1/6
    pu = min(max(var_term + drift_term, 0.0), 1.0)
    pd = min(max(var_term - drift_term, 0.0), 1.0)
    if pu + pd > 1.0:
        s = pu + pd
        pu, pd = pu / s, pd / s
    pm = 1.0 - pu - pd

    # Interior nodes 1..m-1; node 0 = stop (-b), node m = target (+a).
    start = int(round(b / dx))
    start = min(max(start, 1), m - 1)
    mass = np.zeros(m + 1)
    mass[start] = 1.0
    p_up = p_down = 0.0
    hold_acc = 0.0
    for k in range(1, steps + 1):
        new = pm * mass
        new[1:] += pu * mass[:-1]
        new[:-1] += pd * mass[1:]
        hit_up = new[m]
        hit_dn = new[0]
        p_up += hit_up
        p_down += hit_dn
        hold_acc += (hit_up + hit_dn) * k * dt
        new[0] = 0.0
        new[m] = 0.0
        mass = new
    p_timeout = float(mass.sum())
    xs = -b + dx * np.arange(m + 1)
    timeout_value = float((mass * xs).sum())
    ev = p_up * a - p_down * b + timeout_value
    hold = hold_acc + p_timeout * horizon_s
    return BarrierOutcome(float(p_up), float(p_down), p_timeout, float(ev), float(hold))


def _approx_hold(a: float, b: float, mu: float, sigma: float, p_up: float) -> float:
    # Driftless expected exit time is a*b/sigma^2; with drift use the distance/drift bound.
    driftless = a * b / (sigma * sigma)
    if abs(mu) < 1e-12:
        return driftless
    dist = p_up * a + (1 - p_up) * b
    return min(driftless, dist / abs(mu))
