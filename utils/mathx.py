"""Small numeric helpers (pure Python, no numpy needed on the hot path)."""
from __future__ import annotations

import math
from typing import Iterable, Sequence


def clip(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def safe_div(a: float, b: float, default: float = 0.0) -> float:
    return a / b if b else default


def imbalance(a: float, b: float) -> float:
    """(a - b) / (a + b) in [-1, 1]; 0 when both are zero."""
    s = a + b
    return (a - b) / s if s > 0 else 0.0


def bps(x: float, ref: float) -> float:
    return (x / ref) * 1e4 if ref else 0.0


def mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def stdev(xs: Sequence[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    m = sum(xs) / n
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))


def median(xs: Iterable[float]) -> float:
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return 0.0
    mid = n // 2
    return s[mid] if n % 2 else 0.5 * (s[mid - 1] + s[mid])


def percentile_ranks(values: Sequence[float]) -> list[float]:
    """Cross-sectional percentile rank in [0, 1] (ties share the average rank)."""
    n = len(values)
    if n == 0:
        return []
    if n == 1:
        return [0.5]
    order = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg / (n - 1)
        i = j + 1
    return ranks


def prob_hit_upper_first(a: float, b: float, mu: float, sigma: float) -> float:
    """P(Brownian motion with drift hits +a before -b), starting at 0.

    X_t = mu*t + sigma*W_t. Uses the scale function s(x) = exp(-2 mu x / sigma^2):
        P = (1 - exp(2 mu b / s2)) / (exp(-2 mu a / s2) - exp(2 mu b / s2))
    Falls back to the driftless b / (a + b) when mu ~ 0.
    a, b must be positive (same units as mu*t and sigma*sqrt(t)).
    """
    if a <= 0:
        return 1.0
    if b <= 0:
        return 0.0
    if sigma <= 0:
        return 1.0 if mu > 0 else 0.0 if mu < 0 else b / (a + b)
    s2 = sigma * sigma
    k = 2.0 * mu / s2
    if abs(k * (a + b)) < 1e-9:
        return b / (a + b)
    # Work in a numerically stable form: divide numerator/denominator by exp(k*b).
    # P = (exp(-k b) - 1) / (exp(-k (a + b)) - 1)
    x = clip(-k * b, -700, 700)
    y = clip(-k * (a + b), -700, 700)
    num = math.expm1(x)
    den = math.expm1(y)
    if den == 0:
        return b / (a + b)
    return clip(num / den, 0.0, 1.0)
