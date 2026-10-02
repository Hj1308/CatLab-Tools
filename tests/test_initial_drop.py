# tests/test_initial_drop.py
# Tests for the "Pseudo-first (initial drop)" model: C(t) = A·C0·exp(-k·t).
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from catlab.kinetics_engine import (
    MODEL_NAMES,
    _best_model,
    _fit_nonlinear,
    _first_order_drop,
    _post_first_drop,
)

C0 = 7.798e-3  # mol/L (~250 ppmS)
T = np.array([10, 20, 40, 60, 90, 120, 180, 240.0])
DROP_MODEL = "Pseudo-first (initial drop)"


def _two_stage_data(seed):
    rng = np.random.default_rng(seed)
    clean = 0.75 * C0 * np.exp(-0.009 * T)
    return clean * (1.0 + 0.02 * rng.standard_normal(len(T)))


def _pure_pfo_data(seed):
    rng = np.random.default_rng(seed)
    clean = C0 * np.exp(-0.009 * T)
    return clean * (1.0 + 0.02 * rng.standard_normal(len(T)))


def test_first_order_drop_shape():
    # At t=0 the model returns A·C0 (value just after the drop), not C0.
    assert np.isclose(_first_order_drop(np.array([0.0]), 0.01, 0.75, C0)[0], 0.75 * C0)
    # For t > 0 it decays as A·C0·exp(-k t).
    t = np.array([60.0])
    assert np.isclose(_first_order_drop(t, 0.01, 0.75, C0)[0], 0.75 * C0 * np.exp(-0.01 * 60.0))


def test_two_stage_data_selects_initial_drop():
    wins = 0
    for seed in range(50):
        res = _fit_nonlinear(T, _two_stage_data(seed), C0)
        if _best_model(res, MODEL_NAMES) == DROP_MODEL:
            wins += 1
    assert wins >= 45, f"initial-drop model selected only {wins}/50 times"


def test_pure_pfo_data_rarely_selects_initial_drop():
    wins = 0
    for seed in range(50):
        res = _fit_nonlinear(T, _pure_pfo_data(seed), C0)
        if _best_model(res, MODEL_NAMES) == DROP_MODEL:
            wins += 1
    assert wins <= 5, f"initial-drop model selected {wins}/50 times on pure PFO data"


def test_noise_free_pfo_flags_A_at_upper_bound():
    res = _fit_nonlinear(T, C0 * np.exp(-0.009 * T), C0)
    mr = res[DROP_MODEL]
    assert mr.get("at_bound")
    assert any("A at upper bound" in b for b in mr["at_bound"])


def test_fitted_A_recovers_drop_fraction():
    A_values = []
    drop_pcts = []
    for seed in range(50):
        res = _fit_nonlinear(T, _two_stage_data(seed), C0)
        mr = res[DROP_MODEL]
        A_values.append(mr["A"])
        drop_pcts.append(mr["initial_drop_pct"])
    A_values = np.array(A_values)
    drop_pcts = np.array(drop_pcts)
    # Fitted A is within 0.05 of 0.75 for most seeds.
    assert np.sum(np.abs(A_values - 0.75) < 0.05) >= 45
    # initial_drop_pct ≈ 100 * (1 - 0.75) = 25.
    assert abs(float(np.mean(drop_pcts)) - 25.0) < 1.0


def test_t_half_low_A_is_zero():
    # A <= 0.5: half already removed by the drop.
    assert _post_first_drop([0.01, 0.4], [None, None], [], C0)["t_half"] == 0.0


def test_t_half_high_A_is_ln2A_over_k():
    k, A = 0.01, 0.9
    expected = np.log(2.0 * A) / k
    r = _post_first_drop([k, A], [None, None], [], C0)
    assert np.isclose(r["t_half"], expected)
