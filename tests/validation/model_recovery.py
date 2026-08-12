#!/usr/bin/env python
# tests/validation/model_recovery.py
# Synthetic model-recovery validation with reproducible seeds.
# Generates ground-truth curves from each recoverable model, runs
# _fit_nonlinear + _best_model, applies auto-saturation sweep,
# and writes a confusion-style table to model_recovery_results.md.
#
# Base seed ensures reproducibility.  Parameters match the README
# validation setup: C0 = 7.798e-3 mol/L (~250 ppmS), 7 time points,
# +/-3 % multiplicative removal noise.

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
from catlab.kinetics_engine import (
    _fit_nonlinear, _best_model, _auto_saturation_exclusions,
    MODEL_NAMES,
)

C0 = 7.798e-3  # mol/L
T = np.array([0, 15, 30, 45, 60, 90, 120.0])
BASE_SEED = 20260812
N_SEEDS = 15
NOISE = 0.03                 # multiplicative removal noise
CUTOFFS = [1.00, 0.95, 0.90, 0.85, 0.80]


# ---- ground-truth generators (removal %) -------------------------------
def true_rem_pso(t, C0, k2):
    Ct = C0 / (1.0 + k2 * C0 * np.maximum(t, 0.0))
    return 100.0 * (1.0 - Ct / C0)

def true_rem_pfo(t, C0, k):
    Ct = C0 * np.exp(-k * np.maximum(t, 0.0))
    return 100.0 * (1.0 - Ct / C0)

def true_rem_pow(t, C0, k, n):
    inside = C0 ** (1 - n) - k * (1 - n) * np.maximum(t, 0.0)
    Ct = np.maximum(inside, 1e-12) ** (1.0 / (1 - n))
    return 100.0 * (1.0 - Ct / C0)

def true_rem_lh(t, C0, kLH, K):
    from scipy.integrate import odeint as _oi
    t_ = np.asarray(t, dtype=float)
    def dC(C, _):
        Cv = max(C[0], 0.0)
        return [-kLH * K * Cv / (1.0 + K * Cv)]
    if t_[0] == 0:
        sol = _oi(dC, [C0], t_, rtol=1e-6, atol=1e-9)
        return 100.0 * (1.0 - np.maximum(sol.flatten(), 0.0) / C0)
    tf = np.concatenate(([0.0], t_))
    sol = _oi(dC, [C0], tf, rtol=1e-6, atol=1e-9)
    return 100.0 * (1.0 - np.maximum(sol.flatten()[1:], 0.0) / C0)

def true_rem_avrami(t, C0, k, n):
    Ct = C0 * np.exp(-k * np.maximum(t, 0.0) ** n)
    return 100.0 * (1.0 - Ct / C0)


# ---- 7 recoverable archetypes -------------------------------------------
# (excludes BEST_MODEL_EXCLUDE = {Eley-Rideal})
ARCHETYPES = [
    ("PSO-A", "Pseudo-second-order", {"k2": 8.0},       "pso",  true_rem_pso),
    ("PSO-B", "Pseudo-second-order", {"k2": 14.0},      "pso",  true_rem_pso),
    ("PFO-A", "Pseudo-first",        {"k": 0.08},       "pfo",  true_rem_pfo),
    ("PFO-B", "Pseudo-first",        {"k": 0.13},       "pfo",  true_rem_pfo),
    ("PL-A",  "Power-Law",           {"k": 0.45, "n": 1.5}, "mech", true_rem_pow),
    ("LH-A",  "L-H",                 {"kLH": 0.70, "K": 70.0}, "mech", true_rem_lh),
    ("AV-A",  "Avrami",              {"k": 0.06, "n": 1.6},  "mech", true_rem_avrami),
]


def fit_best(t_min, rem_pct, cutoff):
    """Apply auto-saturation, fit, return best-model name or None."""
    excl, tk, rk = _auto_saturation_exclusions(t_min, rem_pct, cutoff)
    Ct = C0 * (1.0 - np.asarray(rk, dtype=float) / 100.0)
    res = _fit_nonlinear(np.asarray(tk, dtype=float), Ct, C0)
    return _best_model(res, MODEL_NAMES)


def main():
    rows_md = []
    rows_md.append("# Model Recovery Validation")
    rows_md.append(
        f"C0 = {C0:.4e} mol/L, N = {N_SEEDS} seeds x {len(ARCHETYPES)} "
        f"archetypes, +/ {NOISE*100:.0f} % noise, seed = {BASE_SEED}")
    rows_md.append("")
    rows_md.append("## Per-cutoff recovery rates")
    rows_md.append("")
    rows_md.append("| cutoff | overall % | PSO/PFO rec % | mech rec % | false-PSO on mech data % |")
    rows_md.append("|:---:|:---:|:---:|:---:|:---:|")

    per_cat_rows = {c: [] for c in CUTOFFS}

    for c in CUTOFFS:
        r = np.zeros(7, dtype=float)
        for a in ARCHETYPES:
            name, truth, params, family, fn = a
            votes = {}
            for seed in range(N_SEEDS):
                rng = np.random.default_rng(BASE_SEED + seed)
                rem_clean = fn(T, C0, **params)
                rem = np.clip(rem_clean + rng.normal(0, NOISE * rem_clean),
                              0.0, 99.5)
                best = fit_best(T, rem, c)
                r[1] += 1
                if best == truth:
                    r[0] += 1
                if family in ("pso", "pfo"):
                    r[3] += 1
                    if best == truth:
                        r[2] += 1
                else:  # mechanistic
                    r[5] += 1   # total mechanistic runs
                    if best == truth:
                        r[4] += 1  # mech correct
                    if best == "Pseudo-second-order":
                        r[6] += 1  # false PSO on mechanistic data
                votes[best] = votes.get(best, 0) + 1
            top = max(votes.items(), key=lambda kv: kv[1])
            per_cat_rows[c].append(
                f"{top[0]} ({100 * top[1] / N_SEEDS:.0f} %)")

        overall = 100 * r[0] / r[1]
        pso_rec = 100 * r[2] / r[3] if r[3] else float("nan")
        mech_rec = 100 * r[4] / r[5] if r[5] else float("nan")
        false_pso = 100 * r[6] / r[5] if r[5] else 0.0
        rows_md.append(
            f"| {c:.2f} | {overall:.1f} | {pso_rec:.1f} | "
            f"{mech_rec:.1f} | {false_pso:.1f} |")

    rows_md.append("")
    rows_md.append("## Per-catalyst best model (most common across seeds)")
    rows_md.append("")
    hdr = "| catalyst | true model | " + " | ".join(f"{c:.2f}" for c in CUTOFFS) + " |"
    rows_md.append(hdr)
    rows_md.append("|" + "---|" * (2 + len(CUTOFFS)))
    for i, a in enumerate(ARCHETYPES):
        name, truth = a[0], a[1]
        cells = " | ".join(per_cat_rows[c][i] for c in CUTOFFS)
        rows_md.append(f"| {name} | {truth} | {cells} |")

    out_dir = os.path.dirname(__file__)
    out_path = os.path.join(out_dir, "model_recovery_results.md")
    with open(out_path, "w") as f:
        f.write("\n".join(rows_md) + "\n")
    print("\n".join(rows_md))
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
