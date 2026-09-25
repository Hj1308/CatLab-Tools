# tests/test_fit_convergence.py
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from scipy.optimize import curve_fit

from catlab.kinetics_engine import _fit_nonlinear, _lh_model

C0 = 7.798e-3                                    # mol/L (250 ppmS)
T = np.array([0, 15, 30, 45, 60, 90, 120.0])     # typical 7-point ODS grid


def _noisy_lh(seed):
    rng = np.random.default_rng(seed)
    Ct = _lh_model(T, 0.02 * C0, 400.0, C0) * (1 + 0.02 * rng.standard_normal(len(T)))
    Ct[0] = C0
    return Ct


def test_lh_fit_reaches_the_least_squares_optimum():
    """Regression: with SciPy's default gtol the bounded solver stopped early.
    On this dataset the L-H SSE was 156x the attainable minimum and K_ads came
    out as 253 L/mol instead of ~395.  The engine must now match a fit run with
    very tight tolerances."""
    Ct = _noisy_lh(29)
    r = _fit_nonlinear(T, Ct, C0)["L-H"]
    sse_engine = np.sum((Ct - r["pred"]) ** 2)

    f = lambda t_, a, b: _lh_model(t_, a, b, C0)
    p_ref, _ = curve_fit(f, T, Ct, p0=[0.01, 10.0],
                         bounds=([0, 0], [np.inf, np.inf]),
                         ftol=1e-14, xtol=1e-14, gtol=1e-14, maxfev=20000)
    sse_ref = np.sum((Ct - f(T, *p_ref)) ** 2)

    assert sse_engine <= 1.01 * sse_ref
    assert abs(r["K_ads"] - p_ref[1]) / p_ref[1] < 0.05
