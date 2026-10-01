# tests/test_lh_closed_form.py
# Tests for the exact closed-form Langmuir-Hinshelwood solution.
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from scipy.integrate import solve_ivp

from catlab.kinetics_engine import (_fit_nonlinear, _lh_model, _lh_t_half)


def _lh_reference(t, k, K, C0):
    """High-accuracy reference for dC/dt = -k*K*C/(1+K*C).

    Integrates the log-transformed state x = ln C (which makes the tail of the
    decay well-conditioned) with a tight implicit solver.  Compared against a
    80-digit mpmath Lambert-W evaluation this stays below ~3e-12 relative
    accuracy over the whole grid, far tighter than the 1e-8 test tolerance.
    """
    t = np.asarray(t, dtype=float)

    def dx(x, tt):
        return [-k * K / (1.0 + K * np.exp(x[0]))]

    sol = solve_ivp(
        lambda tt, x: dx(x, tt),
        [t[0], t[-1]], [np.log(C0)],
        t_eval=t, method="Radau", rtol=1e-12, atol=1e-16)
    return np.maximum(np.exp(sol.y[0]), 0.0)


def test_closed_form_matches_reference():
    C0 = 1.0
    t = np.linspace(0.0, 500.0, 501)
    for Kc in (1e-6, 0.1, 1.0, 10.0, 1e3):
        K = Kc / C0
        for k in (1e-5, 1e-2, 1.0):
            ref = _lh_reference(t, k, K, C0)
            got = _lh_model(t, k, K, C0)
            mask = ref > 1e-12 * C0
            rel = np.max(np.abs(got[mask] - ref[mask]) / ref[mask])
            assert rel < 1e-8, f"Kc={Kc}, k={k}: max rel err {rel}"


def test_large_argument_no_overflow():
    C0 = 1.0
    K = 1e3 / C0
    t = np.array([0.0, 1e-3, 1e-2, 0.1, 1.0])
    C = _lh_model(t, 1e-2, K, C0)
    assert np.all(np.isfinite(C))
    assert not np.any(np.isnan(C))
    assert np.all(C >= 0.0)


def test_kads_zero_and_k_zero_return_C0():
    C0 = 0.01559
    t = np.array([0.0, 5.0, 50.0, 500.0])
    assert np.allclose(_lh_model(t, 0.01, 0.0, C0), C0)
    assert np.allclose(_lh_model(t, 0.0, 10.0, C0), C0)
    assert np.allclose(_lh_model(t, 0.0, 0.0, C0), C0)


def test_first_order_limit_for_small_kc():
    C0 = 0.01559
    K = 1e-12 / C0
    k = 0.02
    t = np.array([0.0, 5.0, 50.0, 500.0])
    got = _lh_model(t, k, K, C0)
    ref = C0 * np.exp(-k * K * t)
    assert np.allclose(got, ref, rtol=1e-8, atol=1e-12)


def test_t_half_consistent_with_model():
    C0 = 0.01559
    for k in (1e-6, 1e-5):
        for K in (0.1, 1.0, 10.0):
            t_half = _lh_t_half(C0, k, K)
            C = _lh_model(np.array([t_half]), k, K, C0)[0]
            assert abs(C - C0 / 2.0) / (C0 / 2.0) < 1e-9, \
                f"k={k}, K={K}: C(t_half)={C}, expected {C0/2}"


def test_double_exponential_k1_always_fast():
    """k1 (fast) must exceed k2, and A must be rescaled accordingly."""
    C0 = 0.01559
    T = np.array([0, 5, 10, 20, 40, 60, 90, 120, 180, 240, 300.0])
    k_slow, k_fast, A = 0.005, 0.08, 0.3
    Ct = C0 * (A * np.exp(-k_slow * T) + (1 - A) * np.exp(-k_fast * T))
    r = _fit_nonlinear(T, Ct, C0)["Double-Exponential"]
    assert r["k"] > r["k2"]
    assert abs(r["params"][2] - (1 - A)) < 0.05
    assert "k2" in r and "k2_se" in r


def test_de_prepare_swaps_slow_first_pair():
    """The fit above converges with k1 > k2 already, so it never reaches the
    swap; exercise _de_prepare directly."""
    from catlab.kinetics_engine import _de_prepare
    p, se = _de_prepare(np.array([0.005, 0.08, 0.3]), np.array([1.0, 2.0, 3.0]))
    np.testing.assert_allclose(p, [0.08, 0.005, 0.7])
    np.testing.assert_allclose(se, [2.0, 1.0, 3.0])
    p, se = _de_prepare(np.array([0.08, 0.005, 0.7]), np.array([2.0, 1.0, 3.0]))
    np.testing.assert_allclose(p, [0.08, 0.005, 0.7])   # already ordered: unchanged
