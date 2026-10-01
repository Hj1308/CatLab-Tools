# tests/test_golden_master.py
# Golden-master regression for _fit_nonlinear: bit-for-bit output stability.
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

from catlab.kinetics_engine import (
    _fit_nonlinear, _first_order, _second_order, _lh_model,
    _elovich, _avrami, _double_exponential, MODEL_NAMES,
)

C0 = 7.798e-3
T = np.array([0, 15, 30, 45, 60, 90, 120.0])

# (seed, generator) — mirrors the generator that produced golden_fit.json.
_GEN = [
    ("PFO",        0, lambda: _first_order(T, 0.02, C0)),
    ("PSO",        1, lambda: _second_order(T, 4.0, C0)),
    ("L-H",        2, lambda: _lh_model(T, 0.0001, 300.0, C0)),
    ("Elovich",    3, lambda: _elovich(T, 1e-4, 500.0, C0)),
    ("Avrami",     4, lambda: _avrami(T, 0.003, 2.0, C0)),
    ("Double-Exp", 5, lambda: _double_exponential(T, 0.005, 0.08, 0.3, C0)),
]

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "golden_fit.json")


def _compute_results():
    results = {}
    for name, seed, gen in _GEN:
        rng = np.random.default_rng(seed)
        clean = gen()
        Ct = clean * (1.0 + 0.03 * rng.standard_normal(len(clean)))
        Ct[0] = C0
        results[name] = _fit_nonlinear(T, Ct, C0)
    return results


def _compare(golden, fresh):
    if isinstance(golden, dict):
        assert isinstance(fresh, dict)
        assert set(golden.keys()) == set(fresh.keys()), \
            f"key mismatch: {sorted(set(golden) ^ set(fresh))}"
        for k in golden:
            _compare(golden[k], fresh[k])
    elif isinstance(golden, (list, tuple)):
        assert isinstance(fresh, (list, tuple, np.ndarray))
        assert len(golden) == len(fresh)
        for g, f in zip(golden, fresh):
            _compare(g, f)
    elif isinstance(golden, bool):
        assert fresh == golden
    elif isinstance(golden, str):
        assert fresh == golden
    elif golden is None:
        assert fresh is None
    elif isinstance(golden, (int, float)):
        np.testing.assert_allclose(fresh, golden, rtol=1e-12, equal_nan=True)
    else:
        raise AssertionError(f"unhandled type {type(golden)}")


def test_golden_master():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        golden = json.load(f)
    fresh = _compute_results()

    assert golden["C0"] == C0
    np.testing.assert_allclose(golden["T"], T, rtol=1e-12)

    assert set(golden["datasets"]) == {n for n, _, _ in _GEN}
    for name, _, _ in _GEN:
        assert set(golden["datasets"][name]) == set(MODEL_NAMES)
        for m in MODEL_NAMES:
            _compare(golden["datasets"][name][m], fresh[name][m])
