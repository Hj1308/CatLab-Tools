# tests/test_golden_master.py
# Golden-master regression for _fit_nonlinear: bit-for-bit output stability.
import os
import sys
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from catlab.kinetics_engine import (
    _fit_nonlinear,
    _first_order,
    _second_order,
    _lh_model,
    _elovich,
    _avrami,
    _double_exponential,
    MODEL_NAMES,
    _best_model,
)

C0 = 7.798e-3
T = np.array([0, 15, 30, 45, 60, 90, 120.0])
T_ANCHOR = np.array([0, 15, 30, 45, 60, 90, 120, 180.0])

# (name, seed, time grid, generator) — mirrors the generator that produced
# golden_fit.json.  PFO-anchor includes a (t=0, C0) anchor point on an 8-point
# grid; its golden values were produced by the pre-refactor engine.
_GEN = [
    ("PFO", 0, T, lambda t: _first_order(t, 0.02, C0)),
    ("PSO", 1, T, lambda t: _second_order(t, 4.0, C0)),
    ("L-H", 2, T, lambda t: _lh_model(t, 0.0001, 300.0, C0)),
    ("Elovich", 3, T, lambda t: _elovich(t, 1e-4, 500.0, C0)),
    ("Avrami", 4, T, lambda t: _avrami(t, 0.003, 2.0, C0)),
    ("Double-Exp", 5, T, lambda t: _double_exponential(t, 0.005, 0.08, 0.3, C0)),
    ("PFO-anchor", 6, T_ANCHOR, lambda t: _first_order(t, 0.02, C0)),
]

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "golden_fit.json")


def _compute_results():
    results = {}
    for name, seed, tgrid, gen in _GEN:
        rng = np.random.default_rng(seed)
        clean = gen(tgrid)
        Ct = clean * (1.0 + 0.03 * rng.standard_normal(len(clean)))
        Ct[0] = C0
        results[name] = _fit_nonlinear(tgrid, Ct, C0)
    return results


# The golden JSON is produced on one machine; numpy/scipy/BLAS builds differ
# between machines.  Statistics (R2, AIC, AICc) agree exactly across Windows and
# Linux, but parameters and SEs of models that do not describe the data (e.g.
# Double-Exponential on PFO data, L-H on Elovich data) sit on a flat SSE valley,
# and the optimiser stops at a slightly different point (SEs differed by up to
# 2.7x between a Windows and a Linux build).  So by default:
#   - statistics, key sets, flags and the selected model are compared tightly,
#   - "pred" loosely (the fitted curve agrees to ~1e-5 relative),
#   - parameter values, SEs and the labels that print them are not compared.
# For a refactor check on the SAME machine that generated the JSON, run with
# GOLDEN_STRICT=1: every float is then compared at rtol=1e-12.
STRICT = os.environ.get("GOLDEN_STRICT") == "1"
STAT_KEYS = {"R2", "adj_r2", "aic", "aicc", "n_fit", "n_anchor"}
FRAGILE_KEYS = {
    "params",
    "k",
    "k_se",
    "k2",
    "k2_se",
    "K_ads",
    "K_se",
    "K_er",
    "K_er_se",
    "n_pl",
    "n_pl_se",
    "t_half",
    "r0",
    "r0_se",
    "label",
}


def _tol(key):
    if STRICT:
        return dict(rtol=1e-12, atol=0.0)
    if key == "pred":
        return dict(rtol=1e-4, atol=1e-10)
    return dict(rtol=1e-9, atol=0.0)


def _compare(golden, fresh, key=""):
    if not STRICT and key in FRAGILE_KEYS:
        return
    if isinstance(golden, dict):
        assert isinstance(fresh, dict)
        assert set(golden.keys()) == set(fresh.keys()), (
            f"key mismatch: {sorted(set(golden) ^ set(fresh))}"
        )
        for k in golden:
            _compare(golden[k], fresh[k], k)
    elif isinstance(golden, (list, tuple)):
        assert isinstance(fresh, (list, tuple, np.ndarray))
        assert len(golden) == len(fresh)
        for g, f in zip(golden, fresh):
            _compare(g, f, key)
    elif isinstance(golden, bool):
        assert fresh == golden
    elif isinstance(golden, str):
        assert fresh == golden, key
    elif golden is None:
        assert fresh is None
    elif isinstance(golden, (int, float)):
        np.testing.assert_allclose(fresh, golden, equal_nan=True, err_msg=key, **_tol(key))
    else:
        raise AssertionError(f"unhandled type {type(golden)}")


def test_golden_master():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        golden = json.load(f)
    fresh = _compute_results()

    assert golden["C0"] == C0
    np.testing.assert_allclose(golden["T"], T, rtol=1e-12)

    assert set(golden["datasets"]) == {n for n, _, _, _ in _GEN}
    for name, _, _, _ in _GEN:
        assert set(golden["datasets"][name]) == set(MODEL_NAMES)
        for m in MODEL_NAMES:
            _compare(golden["datasets"][name][m], fresh[name][m])
        # The model the app would select must not change.
        assert _best_model(golden["datasets"][name], MODEL_NAMES) == _best_model(
            fresh[name], MODEL_NAMES
        ), name
