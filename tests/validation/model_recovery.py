#!/usr/bin/env python
# tests/validation/model_recovery.py
#
# Reproducible, rules-driven model-recovery validation.
#
# DESIGN
#   1. PAIRED CUTOFF SWEEP.  The outer loop is (archetype, replicate); the
#      inner loop is the 5 saturation cutoffs.  All five cutoffs are applied
#      to the SAME noisy dataset, so cutoff comparisons are paired and the
#      cutoff effect is isolated from noise-draw variance.
#   2. DETERMINISM.  Every task derives its own RNG as a PURE FUNCTION of the
#      task only — never from a shared counter, a parent RNG, or dict/set
#      iteration:
#
#          arch_idx = ARCHETYPE_ORDER.index(name)      # fixed module tuple
#          ss = np.random.SeedSequence([BASE_SEED, arch_idx, replicate_index])
#          rng = np.random.default_rng(ss)
#
#      BLAS/OpenMP threads are pinned to 1 (see the import block below) as a
#      precaution against oversubscription when N workers each spawn their own
#      thread pool.  This is hygiene, not the fix for the determinism failure:
#      that had two causes, both corrected below — a wall-clock mean_fit_ms
#      stored inside the compared per_archetype dict, and top3 ties broken by
#      dict insertion order (i.e. task completion order) instead of by name.
#      Aggregation uses only counts/sums plus total-order sorts, and results.json
#      is serialised with sort_keys=True, so the output is byte-identical for
#      any worker count or completion order.  Asserted by
#      tests/test_model_selection.py::TestHarnessDeterminism (a @pytest.mark.slow
#      test, run with `python -m pytest tests -m slow`).
#   3. PARALLELISM.  Tasks are parallelised across (archetype, replicate) with
#      concurrent.futures.ProcessPoolExecutor.  Each task is ~350 ms (five
#      fits), well above process-spawn overhead.
#   4. CHECKPOINTING.  Each completed (archetype, replicate) task is appended
#      to tests/validation/.checkpoint.jsonl as one JSON line, flushed, the
#      moment it finishes.  On startup the checkpoint is read and finished
#      tasks are skipped, so an interrupted run resumes.  The report is
#      regenerated purely from the checkpoint; `--report-only` refits nothing.
#   5. SMOKE MODE.  `--replicates N` (default 200) scales the run; start with
#      `--replicates 20` (~1 min) to validate the pipeline and the conversion-
#      window assertions before the full run.  Smoke numbers are never
#      results: the replicate count is stamped into the report header.
#
# RULES (written before any numbers are generated — do not tune to result):
#
#   1. GENERATOR PARAMETERS ARE CHOSEN BY TARGET CONVERSION, NOT BY RECOVERY
#      RATE.  Every archetype must reach 70–95 % final conversion at t=120 min
#      (the range reported for ODS of DBT with H2O2 in the literature).
#      Asserted inline.  A curve reaching >99 % conversion by 45 min is not
#      representative.
#
#   2. NOISE MODEL: sigma = max(0.03 * Ct_clean, 0.005 * C0), applied in Ct
#      space.  Pure multiplicative-on-Ct noise makes late points almost
#      noiseless and inflates recovery; multiplicative-on-removal% makes late
#      points excessively noisy and deflates it.  The absolute floor avoids
#      both artefacts.  Ct[0] is reset to C0 after noising (C0 is locked in
#      the fit).
#
#   3. Ground-truth generators exclude BEST_MODEL_EXCLUDE models
#      (Eley-Rideal), whose recovery is 0 % by construction.
#
#   4. Report per-archetype recovery AND the top-3 wrong selections for each.
#      Define every aggregate: state bucket membership and whether the
#      denominator is replicates or archetypes.
#
#   5. Report the mean number of retained points alongside recovery rates for
#      the saturation-cutoff sweep.  Flag cells where the cutoff was clamped
#      by MIN_FIT_POINTS.
#
#   6. Fixed base seed, >=200 replicates per archetype.
#
# USAGE
#   # validate the pipeline first (smoke mode):
#   python tests/validation/model_recovery.py --replicates 20 --workers 4
#
#   # full run (Unix):
#   nohup python -u tests/validation/model_recovery.py --workers 4 \
#         > tests/validation/run.log 2>&1 &
#   tail -f tests/validation/run.log
#
#   # full run (Windows — -u so progress lines are not buffered):
#   python -u tests\validation\model_recovery.py --workers 4 ^
#         > tests\validation\run.log 2>&1
#
#   # determinism check — outputs must be identical (fast subset):
#   python tests\validation\model_recovery.py --replicates 3 \
#         --archetypes PSO-A,LH-A --workers 1 --fresh
#   python tests\validation\model_recovery.py --replicates 3 \
#         --archetypes PSO-A,LH-A --workers 4 --fresh
#
#   # or as the pytest slow test (excluded from the default suite by pytest.ini):
#   python -m pytest tests -m slow
#
#   # regenerate the report from an existing checkpoint without refitting:
#   python tests\validation\model_recovery.py --report-only

import os, sys

# BLAS/OpenMP threads are pinned to 1 as a precaution against oversubscription
# when N workers each spawn their own thread pool.  This is hygiene, not the
# fix for the determinism failure — that had two causes, both corrected in the
# aggregation code below: a wall-clock mean_fit_ms stored inside the compared
# per_archetype dict, and top3 ties broken by dict insertion order (i.e. task
# completion order) instead of by model name.  Pin every threading knob to 1
# with ASSIGNMENT (not setdefault) so a caller's environment cannot override
# it.  Must happen before numpy/scipy.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import json
import time
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from catlab.kinetics_engine import (
    _fit_nonlinear, _best_model, _auto_saturation_exclusions,
    _lh_model, _avrami, _power_law,
    MODEL_NAMES,
)

C0 = 7.798e-3          # mol/L (~250 ppmS)
T  = np.array([0, 15, 30, 45, 60, 90, 120.0])
BASE_SEED = 20260812
N_SEEDS   = 200        # default replicates per archetype (--replicates)
CUTOFFS   = [1.00, 0.95, 0.90, 0.85, 0.80]
CUTOFF_KEYS = [f"{c:.2f}" for c in CUTOFFS]

# ---- archetypes: params chosen so 70 % <= X_final <= 95 % at t=120 ----
# Named top-level generators so tasks are picklable by reference.
def _gen_pso(t, C0, k2):
    return C0 / (1.0 + k2 * C0 * np.maximum(t, 0.0))

def _gen_pfo(t, C0, k):
    return C0 * np.exp(-k * np.maximum(t, 0.0))

def _gen_pl(t, C0, k, n):
    return _power_law(t, k, n, C0)

def _gen_lh(t, C0, kLH, K_ads):
    return _lh_model(t, kLH, K_ads, C0)

def _gen_av(t, C0, k_av, n_av):
    return _avrami(t, k_av, n_av, C0)

ARCHETYPES = []

def _register(name, true_model, family, fn, kwargs):
    Ct = fn(T, C0, **kwargs)
    X_final = (C0 - Ct[-1]) / C0
    assert 0.70 <= X_final <= 0.95, \
        f"{name}: final conversion {X_final:.2%} out of [0.70, 0.95]"
    ARCHETYPES.append((name, true_model, family, fn, kwargs))

# Pseudo-second-order
_register("PSO-A", "Pseudo-second-order", "pso", _gen_pso, {"k2": 4.0})
_register("PSO-B", "Pseudo-second-order", "pso", _gen_pso, {"k2": 7.0})
# Pseudo-first
_register("PFO-A", "Pseudo-first", "pfo", _gen_pfo, {"k": 0.012})
_register("PFO-B", "Pseudo-first", "pfo", _gen_pfo, {"k": 0.020})
# Power-Law
_register("PL-A", "Power-Law", "mech", _gen_pl, {"k": 0.45, "n": 1.5})
_register("PL-B", "Power-Law", "mech", _gen_pl, {"k": 0.10, "n": 1.3})
# Langmuir-Hinshelwood
_register("LH-A", "L-H", "mech", _gen_lh, {"kLH": 0.0001, "K_ads": 300.0})
# Avrami
_register("AV-A", "Avrami", "mech", _gen_av, {"k_av": 0.003, "n_av": 1.4})

# Canonical fixed ordering.  Seeds are a pure function of the position of the
# archetype NAME in this tuple — never of dict/set iteration, a shared counter,
# or completion order — so results are independent of worker count and order.
ARCHETYPE_ORDER = tuple(a[0] for a in ARCHETYPES)
ARCH = {a[0]: a for a in ARCHETYPES}   # name -> (name, truth, family, fn, kwargs)

# Clean curve computed once per archetype, reused across replicates.
CLEAN_CURVES = {a[0]: a[3](T, C0, **a[4]) for a in ARCHETYPES}


# ---- noise model (documented above) -----------------------------------
def apply_noise(rng, Ct_clean):
    sigma = np.maximum(0.03 * Ct_clean, 0.005 * C0)
    Ct = Ct_clean + rng.normal(0.0, sigma)
    Ct[0] = C0               # C0 locked in fit
    return np.maximum(Ct, 1e-9)


# ---- task (runs in a worker process) ----------------------------------
def _task(args):
    name, replicate_idx = args
    arch_idx = ARCHETYPE_ORDER.index(name)
    # Deterministic per-task RNG: a pure function of (BASE_SEED, arch_idx,
    # replicate_idx) only, independent of worker count, scheduling and order.
    ss = np.random.SeedSequence([BASE_SEED, arch_idx, replicate_idx])
    rng = np.random.default_rng(ss)
    Ct = apply_noise(rng, CLEAN_CURVES[name])
    rem = 100.0 * (1.0 - Ct / C0)
    t0 = time.time()
    cutoffs = {}
    for ck, c in zip(CUTOFF_KEYS, CUTOFFS):
        excl, tk, rk, clamped = _auto_saturation_exclusions(T, rem, c)
        Ct_fit = C0 * (1.0 - np.asarray(rk, float) / 100.0)
        res = _fit_nonlinear(np.asarray(tk, float), Ct_fit, C0)
        best = _best_model(res, MODEL_NAMES)
        cutoffs[ck] = {"best": best, "n_pts": int(len(rk)),
                       "clamped": bool(clamped)}
    return {"archetype": name, "replicate": replicate_idx,
            "cutoffs": cutoffs, "fit_ms": 1000.0 * (time.time() - t0)}


# ---- checkpoint ---------------------------------------------------------
def _checkpoint_path(out_dir):
    return os.path.join(out_dir, ".checkpoint.jsonl")

def _append_checkpoint(path, rec):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, sort_keys=True) + "\n")
        f.flush()

def _maybe_progress(completed, total, t0):
    if completed % 50 == 0 or completed == total:
        elapsed = time.time() - t0
        eta = elapsed / completed * (total - completed) if completed else 0.0
        print(f"[{completed}/{total}] {100.0*completed/total:.1f}%  "
              f"elapsed {elapsed:.0f}s  ETA {eta:.0f}s", flush=True)


# ---- aggregation (from the checkpoint) ---------------------------------
def _aggregate(records, replicates, names):
    per_cut = {}
    for ck in CUTOFF_KEYS:
        ok = tot = 0
        pso_ok = pso_tot = 0
        mech_ok = mech_tot = 0
        false_pso = 0
        n_pts_sum = 0
        n_clamped = 0
        for rec in records:
            truth = ARCH[rec["archetype"]][1]
            family = ARCH[rec["archetype"]][2]
            cc = rec["cutoffs"][ck]
            tot += 1
            n_pts_sum += cc["n_pts"]
            n_clamped += int(cc["clamped"])
            if cc["best"] == truth:
                ok += 1
            if family in ("pso", "pfo"):
                pso_tot += 1
                if cc["best"] == truth:
                    pso_ok += 1
            else:
                mech_tot += 1
                if cc["best"] == truth:
                    mech_ok += 1
                if cc["best"] == "Pseudo-second-order":
                    false_pso += 1
        def _pct(a, b):
            return 100.0 * a / b if b else float("nan")
        per_cut[ck] = {
            "overall_pct": _pct(ok, tot),
            "pso_pfo_pct": _pct(pso_ok, pso_tot),
            "mech_pct": _pct(mech_ok, mech_tot),
            "false_pso_pct": _pct(false_pso, mech_tot),
            "mean_n_pts": (n_pts_sum / tot) if tot else float("nan"),
            "n_clamped": n_clamped,
        }

    per_arch = {}
    by_arch = {}
    for rec in records:
        by_arch.setdefault(rec["archetype"], []).append(rec)
    for name in names:
        recs = by_arch.get(name, [])
        votes = {}
        for rec in recs:
            best = rec["cutoffs"]["1.00"]["best"]
            votes[best] = votes.get(best, 0) + 1
        per_arch[name] = {
            "truth": ARCH[name][1],
            "n_completed": len(recs),
            "recovery_pct": (100.0 * votes.get(ARCH[name][1], 0) / len(recs)
                             if recs else float("nan")),
            # Total-order sort with an explicit tie-break on the model name.
            # Ties in win counts are common at low replicate counts, and
            # breaking them by dict insertion order (which under as_completed
            # follows task completion order) would make the output depend on
            # worker scheduling.
            "top3": sorted(votes.items(),
                           key=lambda kv: (-kv[1], str(kv[0])))[:3],
        }

    # Timing is wall-clock and therefore non-deterministic; it is kept apart
    # from the recovery numbers so the rest of the aggregate is byte-identical
    # for any worker count or completion order.
    mean_fit_ms = {}
    for name in names:
        recs = by_arch.get(name, [])
        mean_fit_ms[name] = (float(np.mean([r["fit_ms"] for r in recs]))
                             if recs else float("nan"))
    ms = [v for v in mean_fit_ms.values() if np.isfinite(v)]
    median_ms = float(np.median(ms)) if ms else float("nan")
    flagged = [a for a in mean_fit_ms
               if np.isfinite(mean_fit_ms[a]) and mean_fit_ms[a] > 3.0 * median_ms]

    return {"replicates": replicates, "n_tasks_completed": len(records),
            "per_cutoff": per_cut, "per_archetype": per_arch,
            "timing": {"median_fit_ms": median_ms,
                       "per_archetype_mean_fit_ms": mean_fit_ms,
                       "stiff_archetypes": flagged}}


# ---- report -------------------------------------------------------------
def _fmt_pct(x):
    return "n/a" if not np.isfinite(x) else f"{x:.1f}"

def _render(agg, elapsed_s, out_dir, names):
    rows = ["# Model Recovery Validation",
            f"C0 = {C0:.4e} mol/L, N = {agg['replicates']} seeds x "
            f"{len(names)} archetypes, seed = {BASE_SEED}",
            f"Paired cutoff sweep: all 5 cutoffs applied to the same noisy "
            f"dataset per (archetype, replicate). Completed tasks: "
            f"{agg['n_tasks_completed']}. Wall clock: {elapsed_s:.0f}s.",
            "",
            "## Per-cutoff sweep (saturation cutoff vs recovery)",
            "",
            "| cutoff | overall % | PSO/PFO rec % | mech rec % | "
            "false-PSO on mech data % | mean n_pts | clamped? |",
            "|:---:|:---:|:---:|:---:|:---:|:---:|:---:|"]
    for ck, c in zip(CUTOFF_KEYS, CUTOFFS):
        pc = agg["per_cutoff"][ck]
        clamp_mark = " yes" if pc["n_clamped"] > 0 else ""
        rows.append(
            f"| {c:.2f} | {_fmt_pct(pc['overall_pct'])} | "
            f"{_fmt_pct(pc['pso_pfo_pct'])} | {_fmt_pct(pc['mech_pct'])} | "
            f"{_fmt_pct(pc['false_pso_pct'])} | {pc['mean_n_pts']:.1f} "
            f"|{clamp_mark} |")
    rows.append("")
    rows.append("**Buckets:** PSO/PFO = Pseudo-first + Pseudo-second-order "
                "(simplified, flexible). "
                "mech = Power-Law + L-H + Avrami (mechanistic). "
                "Denominator = replicates (N_SEEDS x N_archetypes_in_bucket).")
    rows.append("")
    rows.append("**Cutoff sweep granularity:** with only 7 raw time points and "
                "MIN_FIT_POINTS=6, at most 1 point can ever be excluded before "
                "clamping — so cutoffs 0.95 through 0.80 are clamped to the same "
                "retained set and are not independent evidence of a cutoff effect "
                "on this T array. Only cutoff 1.00 (no exclusion) differs "
                "meaningfully from the rest here.")
    rows.append("")
    rows.append("## Per-archetype recovery (top-3 selections) | cutoff 1.00")
    rows.append("")
    for name in names:
        pa = agg["per_archetype"][name]
        top_str = " | ".join(f"{m} ({c}/{pa['n_completed']})"
                             for m, c in pa["top3"])
        rows.append(f"| {name} | {pa['truth']} | {pa['recovery_pct']:.1f} % "
                    f"| {top_str} |")
    rows.append("")
    rows.append("## Per-archetype mean fit time")
    rows.append("")
    rows.append("| archetype | mean fit ms |")
    rows.append("|:---:|:---:|")
    for name in names:
        rows.append(f"| {name} | "
                    f"{agg['timing']['per_archetype_mean_fit_ms'][name]:.0f} |")
    stiff = agg["timing"]["stiff_archetypes"]
    if stiff:
        rows.append("")
        rows.append(f"**Stiffness signal:** {', '.join(stiff)} mean fit time "
                    f"> 3x median ({agg['timing']['median_fit_ms']:.0f} ms) — "
                    f"recheck their conversion window; do NOT loosen the "
                    f"odeint tolerances in catlab/.")
    md = "\n".join(rows) + "\n"

    with open(os.path.join(out_dir, "model_recovery_results.md"), "w",
              encoding="utf-8") as f:
        f.write(md)
    with open(os.path.join(out_dir, "results.json"), "w",
              encoding="utf-8") as f:
        json.dump(agg, f, indent=2, sort_keys=True)
    return md


# ---- driver -------------------------------------------------------------
def run(replicates=N_SEEDS, workers=None, out_dir=None,
        fresh=False, report_only=False, archetypes=None):
    names = list(archetypes) if archetypes else list(ARCHETYPE_ORDER)
    for n in names:
        if n not in ARCH:
            raise ValueError(f"unknown archetype: {n}")

    out_dir = out_dir or os.path.dirname(os.path.abspath(__file__))
    os.makedirs(out_dir, exist_ok=True)
    chk = _checkpoint_path(out_dir)
    if fresh and os.path.exists(chk):
        os.remove(chk)

    records = []
    if os.path.exists(chk):
        with open(chk, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    done = {(r["archetype"], r["replicate"]) for r in records}

    tasks = [(name, ri) for name in names for ri in range(replicates)]
    pending = [t for t in tasks if t not in done]
    elapsed = 0.0
    if not report_only and pending:
        workers = workers if workers is not None \
            else max(1, (os.cpu_count() or 2) - 1)
        t0 = time.time()
        total = len(pending)
        print(f"{total} tasks pending, {len(records)} in checkpoint "
              f"({workers} workers)", flush=True)
        completed = 0
        if workers == 1:
            for t in pending:
                records.append(_task(t))
                _append_checkpoint(chk, records[-1])
                completed += 1
                _maybe_progress(completed, total, t0)
        else:
            with ProcessPoolExecutor(max_workers=workers) as ex:
                futs = {ex.submit(_task, t): t for t in pending}
                for fut in as_completed(futs):
                    records.append(fut.result())
                    _append_checkpoint(chk, records[-1])
                    completed += 1
                    _maybe_progress(completed, total, t0)
        elapsed = time.time() - t0
        print(f"fitting done in {elapsed:.0f}s", flush=True)

    agg = _aggregate(records, replicates, names)
    agg["timing"]["elapsed_s"] = elapsed
    md = _render(agg, elapsed, out_dir, names)
    return md, agg


def main():
    ap = argparse.ArgumentParser(description="Model-recovery validation harness")
    ap.add_argument("--replicates", type=int, default=N_SEEDS,
                    help="replicates per archetype (default %(default)s; "
                         "--replicates 20 = smoke mode)")
    ap.add_argument("--workers", type=int, default=None,
                    help="worker processes (default os.cpu_count()-1; use 1 "
                         "for the determinism check)")
    ap.add_argument("--archetypes", default=None,
                    help="comma-separated archetype names to run "
                         "(default: all, e.g. PSO-A,LH-A)")
    ap.add_argument("--out", default=None,
                    help="output directory (default: this script's directory)")
    ap.add_argument("--fresh", action="store_true",
                    help="ignore and delete the existing checkpoint")
    ap.add_argument("--report-only", action="store_true",
                    help="regenerate the report from the checkpoint, no fitting")
    args = ap.parse_args()

    arch_names = None
    if args.archetypes:
        arch_names = [x.strip() for x in args.archetypes.split(",") if x.strip()]

    out_dir = args.out or os.path.dirname(os.path.abspath(__file__))
    md, _ = run(args.replicates, args.workers, args.out,
                args.fresh, args.report_only, archetypes=arch_names)
    print(md)
    print(f"Saved to {os.path.join(out_dir, 'model_recovery_results.md')}")


if __name__ == "__main__":
    main()
