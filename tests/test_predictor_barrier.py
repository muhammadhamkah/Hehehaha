import math

from config import StrategyConfig
from strategy.barrier import barrier_outcome
from strategy.predictor import RuleBasedPredictor
from utils.mathx import percentile_ranks, prob_hit_upper_first


def test_closed_form_barrier():
    assert math.isclose(prob_hit_upper_first(1, 1, 0, 1), 0.5)
    assert math.isclose(prob_hit_upper_first(2, 1, 0, 1), 1 / 3)
    assert prob_hit_upper_first(1, 1, 1, 1) > 0.8
    assert prob_hit_upper_first(1, 1, -1, 1) < 0.2


def test_lattice_driftless_is_fair():
    r = barrier_outcome(6, 8, 0.0, 2.0, 600)
    assert math.isclose(r.p_target, 8 / 14, abs_tol=0.01)
    assert abs(r.expected_value_bps) < 0.05
    assert math.isclose(r.expected_hold_s, 6 * 8 / 4, rel_tol=0.05)


def test_lattice_drift_and_timeout():
    up = barrier_outcome(6, 8, 0.3, 2.0, 60)
    dn = barrier_outcome(6, 8, -0.3, 2.0, 60)
    assert up.p_target > 0.7 > dn.p_target
    assert up.expected_value_bps > 0 > dn.expected_value_bps
    quiet = barrier_outcome(10, 10, 0.0, 0.2, 30)
    assert quiet.p_timeout > 0.9
    total = up.p_target + up.p_stop + up.p_timeout
    assert math.isclose(total, 1.0, abs_tol=1e-6)


def _features(sign: float) -> dict:
    s = sign
    return {
        "imb_weighted": 0.6 * s, "ofi_3s": 0.5 * s, "flow_imb_3s": 0.7 * s, "flow_imb_10s": 0.5 * s,
        "micro_tilt": 0.5 * s, "mom_3s_bps": 3 * s, "persistence": 0.8 * s, "depletion_asym": 0.3 * s,
        "rv_1s_bps": 2.0, "trades_per_s_10s": 5.0, "flow_price_agree_3s": 1.0,
    }


def test_rule_predictor_direction_and_confirmation():
    p = RuleBasedPredictor(StrategyConfig())
    long = p.predict(_features(1))
    short = p.predict(_features(-1))
    assert long.direction == 1 and long.p_long > 0.75 and long.flow_confirms
    assert short.direction == -1 and short.p_short > 0.75 and short.flow_confirms
    assert long.drift_bps_per_s > 0 > short.drift_bps_per_s
    assert set(long.expected_moves()) == {1, 3, 5, 10, 30, 60}


def test_flow_contradicting_book_is_not_confirmed():
    f = _features(1)
    f["flow_imb_3s"] = -0.8
    f["flow_imb_10s"] = -0.8
    assert not RuleBasedPredictor(StrategyConfig()).predict(f).flow_confirms


def test_thin_tape_dampens_score():
    f = _features(1)
    f["trades_per_s_10s"] = 0.2
    assert abs(RuleBasedPredictor(StrategyConfig()).predict(f).score) < 0.1


def test_percentile_ranks():
    assert percentile_ranks([3, 1, 2]) == [1.0, 0.0, 0.5]
    assert percentile_ranks([1, 1]) == [0.5, 0.5]


def test_closed_form_lattice_matches_iterative_reference():
    import random

    from strategy import barrier as B

    rng = random.Random(5)
    for _ in range(60):
        a, b = rng.uniform(3, 60), rng.uniform(3, 30)
        mu, sigma, T = rng.uniform(-1.5, 1.5), rng.uniform(0.5, 6), rng.choice([10, 30, 60])
        width = a + b
        m = rng.randint(4, 80)
        dx = width / m
        dt = dx * dx / (3 * sigma * sigma)
        steps = max(1, int(math.ceil(T / dt)))
        if steps > 2500:
            continue          # barrier_outcome uses the infinite-horizon formula here
        dt = T / steps
        pu = min(max(sigma * sigma * dt / (2 * dx * dx) + mu * dt / (2 * dx), 0), 1)
        pd = min(max(sigma * sigma * dt / (2 * dx * dx) - mu * dt / (2 * dx), 0), 1)
        start = min(max(int(round(b / dx)), 1), m - 1)
        assert pu + pd <= 1.0 + 1e-12
        args = (m, start, steps, dt, dx, a, b, pu, pd, 1 - pu - pd, T)
        fast, ref = B._lattice_closed_form(*args), B._lattice_iterative(*args)
        assert math.isclose(fast.p_target, ref.p_target, abs_tol=1e-9)
        assert math.isclose(fast.p_stop, ref.p_stop, abs_tol=1e-9)
        assert math.isclose(fast.p_timeout, ref.p_timeout, abs_tol=1e-9)
        assert math.isclose(fast.expected_value_bps, ref.expected_value_bps, abs_tol=1e-7)
        assert math.isclose(fast.expected_hold_s, min(ref.expected_hold_s, T), rel_tol=1e-7, abs_tol=1e-7)
