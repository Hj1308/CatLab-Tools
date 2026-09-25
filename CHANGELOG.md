# Changelog

## v3.6.0 (2026-09-25)

### Fixed
- **One model-selection rule.** `KineticsAnalyser.best_fit()` selected by
  max(R²) over four models while the app selected by AICc over nine, so the
  same data could yield two different model names. `best_fit()` now uses the
  app's AICc selection over the full portfolio. On a synthetic L-H curve the
  package previously returned Zero-order (ΔAICc = 27.5); both now return L-H.
  The linearised Lagergren fit remains available as a diagnostic but is no
  longer a selection candidate, because its R² is on a different dependent
  variable.
- **Tab 2 no longer ranks models by linear R².** The linearisations use
  different dependent variables (C, ln(C₀/C), 1/C), so their R² values are not
  comparable. The table is now labelled a diagnostic. Its exclusion list
  (Elovich, Double-Exponential) was also out of step with Tab 1, and an
  unsourced claim that PSO and L-H are "more physically meaningful" was removed.
- **Auto-saturation is no longer attributed to Simonin (2016).** The rule uses
  the last observed removal as its own reference, so any setting below 1.0
  drops exactly one point; Simonin's criterion needs an independently measured
  equilibrium capacity. The slider, messages and README now say so.
- Excluded time points are reported exactly (77.5 min was shown as 77).
- `catlab.__version__` now matches the release (it said 1.1.0).
- `_get_valid_models()` no longer raises `KeyError` on a partial results dict.
- The release workflow now extracts this file's section for the release notes
  (it produced an empty body).

### Added
- **Initial-rate TOF in Tab 4** (`TOF_0`, from the fitted r₀) alongside the
  whole-run average, which depends on when the run was stopped. On synthetic
  data with the same k, runs stopped at 50 % and 90 % conversion gave average
  TOFs of 3.38 and 1.83 h⁻¹; the initial-rate TOF was 4.68 h⁻¹ for both.
  Existing columns are renamed `TOF_avg` / `TOF_mass_avg`. Avrami and
  Double-Exponential have no defined initial rate and report N/A.
- `best_fit()` also returns `aicc`, `delta_aicc`, `weight`, `n_params` and
  `selection_criterion`; `run_ods_analysis()` gains three summary columns.
- README: Kostoglou & Karapantsios (2022) on linearised versus non-linear
  fitting.
- Tests: 83 → 104.

### Note
- No rate constant, R², AIC, AICc or Akaike weight changes for any model; what
  changes is which model is named and how results are labelled.

## v3.5.5 (2026-08-16)

### Fixed
- AICc now correctly counts sigma^2 as an estimated parameter (K = p+1,
  Burnham & Anderson 2002 Sec 2.2).
- N_PARAMS["Double-Exponential"] corrected from 4 to 3 (k1, k2, A fitted;
  C0 locked).
- Arrhenius confidence intervals now use t(0.975, df=n_T-2) instead of a
  hardcoded z=1.96; 2-point fits (df=0) report point estimate only with
  an explicit warning.
- MIN_FIT_POINTS raised to 6 and an off-by-one fixed in
  _auto_saturation_exclusions that could let the retained dataset fall to
  2 points, silently making AICc infinite for every model. Clamping is
  now surfaced in the UI and in fitting_summary.csv.

### Changed
- _best_model selection is now purely AICc-based (fewest parameters, then
  lowest AICc); removed the hard-coded Pseudo-second-order preference.
- Added akaike_weights() reporting delta_i and w_i over the candidate set.

### Added
- Validation harness rewrite (tests/validation/model_recovery.py):
  deterministic per-task RNG, parallel execution with checkpointing and
  resume, paired cutoff sweep across 8 kinetic archetypes.
- 6 new tests (AICc boundaries, saturation floor, harness determinism,
  timing-report correctness).

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

## v3.5.3
- Power-Law n>1 bug fix
- Eley-Rideal excluded from auto-selection
- Arrhenius composite-k warning; CSV auto-separator; Tab 4 Option B
  mass-normalized TOF for carbon catalysts

## v3.5.2
- Auto-saturation detection (8%/15% thresholds)
- Per-catalyst point exclusion in Tab 1
- Linearized plots based on best model

## v3.5.1
- Power-Law numerical stability
- Tab 8 savefig fix
- Model classification in assumptions

## v3.5.0
- ppmS volumetric default (no density)
- AICc model selection; Pseudo-second-order rename
- Residual diagnostics ddof fix

## v3.4
- Tab 8 Arrhenius multi-temperature
- Tab 9 residual diagnostics

## v3.3
- L-H t½ analytical fix
- Centralised data loader
- Shared file uploader

## v3.2
- ppmS/ppm dual C₀ display
- Solvent selector
- Oxidant efficiency tab

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
