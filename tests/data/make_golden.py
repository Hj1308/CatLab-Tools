# tests/data/make_golden.py
# Regenerates tests/data/golden_fit.json.
#
# NOTE: the "PFO-anchor" entry is a true pre-refactor reference.  It was
# produced with catlab/kinetics_engine.py as of commit 3fb0529 (the commit
# before the data-driven _fit_nonlinear refactor):
#
#     git show 3fb0529:catlab/kinetics_engine.py > /tmp/prerefactor_engine.py
#
# then calling that module's _fit_nonlinear.  Re-running this script against the
# *refactored* engine is still bit-for-bit identical (the refactor preserves
# every float), so the committed JSON is authoritative either way.
#
# The default test tolerates numpy/scipy/BLAS differences between machines;
# GOLDEN_STRICT=1 compares every float at rtol=1e-12 and only holds on the
# machine that produced the JSON (see tests/test_golden_master.py).
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np

from catlab.kinetics_engine import (
    _fit_nonlinear, _first_order, _second_order, _lh_model,
    _elovich, _avrami, _double_exponential, MODEL_NAMES,
)

C0 = 7.798e-3
T = np.array([0, 15, 30, 45, 60, 90, 120.0])
T_ANCHOR = np.array([0, 15, 30, 45, 60, 90, 120, 180.0])

_GEN = [
    ("PFO",        0, T,        lambda t: _first_order(t, 0.02, C0)),
    ("PSO",        1, T,        lambda t: _second_order(t, 4.0, C0)),
    ("L-H",        2, T,        lambda t: _lh_model(t, 0.0001, 300.0, C0)),
    ("Elovich",    3, T,        lambda t: _elovich(t, 1e-4, 500.0, C0)),
    ("Avrami",     4, T,        lambda t: _avrami(t, 0.003, 2.0, C0)),
    ("Double-Exp", 5, T,        lambda t: _double_exponential(t, 0.005, 0.08, 0.3, C0)),
    ("PFO-anchor", 6, T_ANCHOR, lambda t: _first_order(t, 0.02, C0)),
]


def _to_json(v):
    if isinstance(v, dict):
        return {k: _to_json(vv) for k, vv in v.items()}
    if isinstance(v, (list, tuple)):
        return [_to_json(vv) for vv in v]
    if isinstance(v, np.ndarray):
        return [_to_json(vv) for vv in v.tolist()]
    if isinstance(v, np.floating):
        return float(v)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.bool_):
        return bool(v)
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    raise TypeError(type(v))


def main():
    out = {"C0": float(C0), "T": [float(x) for x in T], "datasets": {}}
    for name, seed, tgrid, gen in _GEN:
        rng = np.random.default_rng(seed)
        clean = gen(tgrid)
        Ct = clean * (1.0 + 0.03 * rng.standard_normal(len(clean)))
        Ct[0] = C0
        res = _fit_nonlinear(tgrid, Ct, C0)
        out["datasets"][name] = {m: _to_json(res[m]) for m in MODEL_NAMES}

    out_path = os.path.join(os.path.dirname(__file__), "golden_fit.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, allow_nan=True)
    print("wrote", out_path)


if __name__ == "__main__":
    main()
