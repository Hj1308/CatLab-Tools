# tests/test_anchor.py
# The (t=0, C0) point is fitted exactly by every model because C0 is locked,
# so it must not count as data in n, R^2, AICc or the auto-saturation floor.
import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from catlab.kinetics_engine import (
    _fit_nonlinear,
    _first_order,
    _anchor_mask,
    _auto_saturation_exclusions,
    MODEL_NAMES,
    MIN_FIT_POINTS,
)

C0 = 7.798e-3
T = np.array([10, 20, 30, 45, 60, 90, 120.0])


def _noisy(seed=1):
    rng = np.random.default_rng(seed)
    return _first_order(T, 0.02, C0) * (1 + 0.03 * rng.standard_normal(len(T)))


def test_anchor_does_not_change_statistics():
    Ct = _noisy()
    without = _fit_nonlinear(T, Ct, C0)
    with_t0 = _fit_nonlinear(np.r_[0.0, T], np.r_[C0, Ct], C0)
    for m in MODEL_NAMES:
        a, b = without[m], with_t0[m]
        if not a.get("converged", True):
            continue
        assert b["R2"] == a["R2"], m
        assert b["aicc"] == a["aicc"], m
        assert b["n_fit"] == len(T) and b["n_anchor"] == 1


def test_pred_stays_aligned_with_input():
    Ct = _noisy()
    res = _fit_nonlinear(np.r_[0.0, T], np.r_[C0, Ct], C0)
    pred = res["Pseudo-first"]["pred"]
    assert len(pred) == len(T) + 1
    assert pred[0] == C0


def test_t0_below_c0_is_informative():
    """A t=0 point after a dark-adsorption step (C < C0) is real data."""
    t = np.array([0.0, 10.0])
    Ct = np.array([0.95 * C0, 0.8 * C0])
    assert _anchor_mask(t, Ct, C0).tolist() == [False, False]
    assert _anchor_mask([0.0], [C0], C0).tolist() == [True]


def test_saturation_floor_ignores_anchor():
    n = MIN_FIT_POINTS
    t = np.arange(n + 1, dtype=float)  # t=0 anchor + n informative
    rem = np.r_[0.0, np.linspace(20, 90, n)]
    excl, _, rem_keep, clamped = _auto_saturation_exclusions(t, rem, 0.5)
    assert excl == [] and clamped
    assert len(rem_keep) == n + 1
