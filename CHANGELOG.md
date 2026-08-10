# Changelog

## v3.5.4 (2026-08-10)

### Changed
- **`catlab/ods_kinetics._fit_kinetics`** now delegates to the shared nonlinear
  least-squares engine (`catlab/kinetics_engine._fit_nonlinear`) instead of
  performing its own linearised `linregress` fits. Linearised regression on
  transformed variables (ln(C), 1/C) systematically biases parameter estimates
  when the data contain even moderate noise (Kostoglou & Karapantsios,
  *Colloids Interfaces* 2022). Across 200 synthetic replicates with ±3 % noise
  the nonlinear engine reduces mean rate-constant error from 5.7 % to 4.0 %
  (pseudo-first-order) and from 2.3 % to 1.9 % (pseudo-second-order). The
  diagnostic plots (zero/first/second-order linearised views) now draw the
  through-origin line y = k·t using the reported nonlinear rate constant,
  guaranteeing the plotted slope matches the reported number.

## v3.0 (2026-06-15)

### New Features
- `app_ods.py`: k ± SE from covariance matrix for all kinetic models
- `app_ods.py`: r₀ (initial reaction rate) — model-independent activity metric
- `app_ods.py`: Langmuir-Hinshelwood (L-H) as 5th kinetic model (scipy ODE)
- `app_ods.py`: Tab 5 — Parameter Effect module
- `app_ods.py`: Tab 6 — Oxidant Efficiency (η H₂O₂ %)
- `app_ods.py`: Tab 7 — Condition Comparison (Thermal vs UV vs ECODS)
- `README.md`: Full v3.0 documentation

## v2.0
- Fixed Elovich t½ (was hard-coded to NaN)
- Per-sheet error handling in kinetics tab
- TOF/TON module
- Reusability module

## v1.4
- Initial public release
