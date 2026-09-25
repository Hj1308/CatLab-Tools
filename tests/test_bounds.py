# tests/test_bounds.py
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

from catlab.kinetics_engine import (_fit_nonlinear, _best_model, MODEL_NAMES,
                                    _zero_order)

C0 = 7.798e-3
T = np.array([0, 15, 30, 45, 60, 90, 120.0])


def _avrami_data(n):
    return C0 * np.exp(-(0.02 * T) ** n)


def test_power_law_flags_n_at_lower_bound_on_zero_order_data():
    """Zero-order data wants n = 0, below the Power-Law bound of 0.1.
    The fit must be flagged and its SEs reported as NaN, not as numbers."""
    r = _fit_nonlinear(T, _zero_order(T, 4e-5, C0), C0)["Power-Law"]
    assert r["at_bound"] == ["n at lower bound 0.1"]
    assert np.isnan(r["k_se"]) and np.isnan(r["n_pl_se"])


def test_interior_fit_is_not_flagged():
    r = _fit_nonlinear(T, _avrami_data(2.0), C0)["Avrami"]
    assert r["at_bound"] == []
    assert np.isfinite(r["k_se"])


def test_avrami_bound_allows_n_up_to_4():
    """Regression: the bound was n <= 3, below the JMAK range, so n = 3.6
    data was pinned at 3.0.  It must now fit as an interior optimum."""
    res = _fit_nonlinear(T, _avrami_data(3.6), C0)
    r = res["Avrami"]
    assert abs(r["params"][1] - 3.6) < 1e-3
    assert r["at_bound"] == []
    assert _best_model(res, MODEL_NAMES) == "Avrami"


def test_avrami_beyond_4_is_flagged_but_stays_eligible():
    """At-bound fits are flagged, not vetoed: excluding them made sigmoidal
    data select Zero-order."""
    res = _fit_nonlinear(T, _avrami_data(4.5), C0)
    assert res["Avrami"]["at_bound"] == ["n at upper bound 4"]
    assert np.isnan(res["Avrami"]["k_se"])
    assert _best_model(res, MODEL_NAMES) == "Avrami"
