import numpy as np
import pytest

from asttel.monitor import Monitor
from asttel.params import THETA, THETA_SI, UPSILON_COMPLETE

SCHEMA = ["a", "b", "c"]


def basis(i, n=8):
    v = np.zeros(n)
    v[i] = 1.0
    return v


def random_script(rng, steps=40, dim=16):
    for _ in range(steps):
        target = rng.choice(SCHEMA + ["general"])
        if rng.random() < 0.1:
            yield None, None, target
        else:
            yield rng.normal(size=dim), {d: float(rng.random() * 0.6) for d in rng.choice(SCHEMA, 2)}, target


def test_frozen_configuration():
    assert (THETA.alpha, THETA.beta, THETA.tau_g, THETA.upsilon_star, THETA.n) == (0.5, 0.5, 0.01, 0.5, 4)
    assert (THETA.lam, THETA.d_min, THETA.eps_p, THETA.eps_g) == (5.0, 0.2, 1e-9, 1e-12)
    assert (THETA_SI, UPSILON_COMPLETE) == (0.20, 0.80)


def test_first_exchange_has_no_friction():
    _, si, _ = Monitor(SCHEMA).step(basis(0), {"a": 0.3}, "a")
    assert si == 0.0


def test_no_evidence_leaves_state_unchanged():
    m = Monitor(SCHEMA)
    m.step(basis(0), {"a": 0.3}, "a")
    before = dict(m.upsilon), dict(m.attempts)
    _, si, gains = m.step(None, None, "a")
    assert si == 0.0 and not any(gains.values())
    assert (m.upsilon, m.attempts) == before


def test_resolution_is_monotone_and_bounded():
    rng = np.random.default_rng(0)
    for _ in range(20):
        m, prev = Monitor(SCHEMA), dict.fromkeys(SCHEMA, 0.0)
        for embedding, scores, target in random_script(rng):
            pe, si, _ = m.step(embedding, scores, target)
            assert 0.0 <= si <= 1.0
            assert all(v >= 0.0 for v in pe.values())
            assert all(m.upsilon[d] >= prev[d] for d in SCHEMA)
            prev = dict(m.upsilon)


def test_recurrent_evidence_without_gain_registers_friction():
    m = Monitor(SCHEMA)
    m.step(basis(0), {"a": 0.2}, "a")
    _, si, _ = m.step(basis(0), {"a": 0.2}, "a")
    assert si == pytest.approx(1.0 - 0.2, abs=1e-9)
    assert si > THETA_SI


def test_novel_evidence_registers_no_friction():
    m = Monitor(SCHEMA)
    m.step(basis(0), {"a": 0.2}, "a")
    _, si, _ = m.step(basis(1), {"a": 0.2}, "a")
    assert si == pytest.approx(0.0, abs=1e-9)


def test_realised_gain_dampens_friction():
    m = Monitor(SCHEMA)
    m.step(basis(0), {"a": 0.2}, "a")
    _, si, _ = m.step(basis(0), {"a": 0.3}, "a")
    assert si == pytest.approx(max(THETA.d_min, 1 - THETA.lam * 0.1) * (1 - 0.3), abs=1e-9)


def test_regressive_evidence_amplifies_friction():
    m = Monitor(SCHEMA)
    m.step(basis(0), {"a": 0.5}, "a")
    _, si, _ = m.step(basis(0), {"a": 0.4}, "a")
    assert si == pytest.approx((1 + THETA.lam * 0.1) * (1 - 0.5), abs=1e-9)


def test_saturated_dimensions_register_no_friction():
    m = Monitor(SCHEMA)
    m.step(basis(0), {"a": 1.0}, "a")
    _, si, _ = m.step(basis(0), {"a": 1.0}, "a")
    assert si == pytest.approx(0.0, abs=1e-9)


def test_unattributed_evidence_uses_current_evidence_and_aggregate_gain():
    m = Monitor(SCHEMA)
    m.step(basis(0), {"a": 0.2}, "a")
    m.step(basis(1), {"b": 0.4}, "b")
    _, si, _ = m.step(basis(1), {"c": 0.0}, "general")
    assert si == pytest.approx(1.0 - 0.4, abs=1e-9)


def test_progress_prefers_unexplored_dimensions():
    m = Monitor(SCHEMA)
    for _ in range(3):
        m.step(basis(0), {"a": 0.0}, "a")
    pe = m.progress()
    assert pe["b"] > pe["a"] and pe["b"] == pytest.approx(pe["c"])


def test_dependencies_gate_progress():
    m = Monitor(SCHEMA, dependencies={"b": ["a"]})
    pe = m.progress()
    assert pe["b"] == pytest.approx(0.5 * pe["c"])
