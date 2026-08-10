# Audit Report — `upgrade/audit-and-fixes`

Read-only review of `app_ods.py`, `catlab/catalyst_analytics.py`, `catlab/ods_kinetics.py`,
`tests/test_catlab.py`, and `.github/workflows/ci.yml`. No code was modified.

---

## 1. Functions duplicated or diverging between `catlab/` and `app_ods.py`

1. **Unit converter — three divergent implementations of the same concept.**
   - `catlab/ods_kinetics.py:28` `_to_mol_L()` returns **mol/L**.
   - `app_ods.py:186` `_to_mol_L()` returns mol/L but adds ppmS volumetric/mass toggle,
     fuel-density handling and `n_sulfur`; the two no longer agree on ppmS (the app treats
     ppmS as mg(S)/L, ods_kinetics divides by MW_S only).
   - `catlab/catalyst_analytics.py:33` `convert_to_mmol_L()` returns **mmol/L** (×1000 scale
     difference) and ignores density entirely.
   - Severity: **critical**. Fix: keep one canonical `to_mol_L()` in `catlab/` and have
     `app_ods.py` import it (pass density/volumetric flags as optional args).

2. **Kinetic fitting engines — duplicated and semantically diverged.**
   - `app_ods.py:458` `_fit_nonlinear()` uses nonlinear `curve_fit` with C0 locked.
   - `catlab/ods_kinetics.py:72` `_fit_kinetics()` uses linearized `linregress`.
   - `catlab/catalyst_analytics.py:144-169` `KineticsAnalyser.fit_*` also use `linregress`.
   - The two `catlab` files disagree with each other (`_fit_kinetics` fits zero/first/second;
     `KineticsAnalyser` fits zero/first/second/pseudo-first), and both diverge from the app.
   - Severity: **critical**. Fix: single fitting module (ideally the nonlinear engine) reused
     by app and package; keep linearization only as diagnostics.

3. **Model functions duplicated.**
   `_zero_order`/`_first_order`/`_second_order` exist in `app_ods.py:238/241/244` and are
   re-derived inline in `ods_kinetics.py:78-93` and `catalyst_analytics.py:144-164`.
   Severity: **medium**. Fix: shared rate-law library in `catlab/kinetics_models.py`.
   *Correction (verified via synthetic second-order data with known k and C0):* the
   second-order fit in `catalyst_analytics.py` (`y = 1/C` with a free intercept) and
   `ods_kinetics.py` (`y = 1/C - 1/C0`) use different conventions but are mathematically
   equivalent — `linregress` estimates the intercept freely, so the slope k is unaffected
   by the missing `1/C0` term. The real divergence is units/rounding only
   (mmol/L·h vs mol/L·min), not a math error.

4. **`MW_S = 32.06` hardcoded in three places** — `catlab/catalyst_analytics.py:27`,
   `catlab/ods_kinetics.py:22`, `app_ods.py:137`. Severity: **low**. Fix: single constant.

5. **`COLORS`/`MARKERS` palette duplicated** — `app_ods.py:139-140` and
   `catlab/ods_kinetics.py:146-148` (identical values). Severity: **low**. Fix: move to
   shared module.

6. **Pseudo-first-order t½ `ln(2)/k` recomputed in parallel** — `app_ods.py:493` and
   `catlab/ods_kinetics.py:96`. Severity: **low**. Fix: single `t_half()` helper.

7. **C0 conversion duplicated** — `_C0_both()`/`_C0()` in `app_ods.py:227` vs
   `SampleInfo.c0_mmol_L` property in `catalyst_analytics.py:72`. Severity: **medium**.
   Fix: package-level converter consumed by both.

8. **Template generators duplicated/diverged** — `generate_template()` in
   `catlab/ods_kinetics.py:45` (empty sheets) vs `create_advanced_template()` in
   `app_ods.py:784` (Metadata + Raw_Data + Catalyst_Properties + Instructions). Severity:
   **low**. Fix: one template builder with options.

---

## 2. `except Exception` blocks returning sentinel values instead of raising/logging

1. **`app_ods.py:478-639` — all nine models in `_fit_nonlinear`** swallow `curve_fit`
   failures into `{"R2": -999, "aic": inf, "aicc": inf, "error": ...}` and only store the
   string. Downstream `_get_valid_models()` (line 645) then silently drops the model.
   Severity: **critical** — a converged-but-wrong fit is indistinguishable from a failed one,
   and the `-999`/`inf` sentinels leak into user-facing tables via `_fmt_*`.
   Fix: log `logging.error` with traceback and raise or return a typed failure object.

2. **`app_ods.py:297`, `:367`, `:380` — `_power_law_t_half`, `_elovich_t_half`,
   `_lh_t_half`** all `return float("nan")` on any exception. Severity: **medium**.
   Fix: validate inputs explicitly and log unexpected exceptions instead of masking them.

3. **`app_ods.py:439` `_load_kinetic_data`** — on read failure shows `st.error` and returns
   `None, None, None`; callers must remember to check `df is None`. Severity: **medium**.
   Fix: raise a typed exception and catch once at the tab boundary.

4. **Silent swallow sites** — `app_ods.py:858` (column-width loop), `:991` (exclusion-label
   parse), `:1111` (saturation advisory), `:1390` (linearization per-model), `:1643` (BET
   sheet read), `:2110/:2151/:2177` (Shapiro/probplot in Tab 9) all `except Exception: pass`
   with no log. Severity: **medium** — these mask real I/O and stats errors in production.
   Fix: replace bare `pass` with `logging.exception` at least.

---

## 3. Hardcoded magic constants without documented rationale

1. **`BEST_MODEL_EXCLUDE`** — `app_ods.py:158`. Rationale is in the comment block, but the
   exclusion set itself (e.g. Elovich/Eley-Rideal always excluded) is a hardcoded scientific
   judgment with no user override and no citation. Severity: **medium**. Fix: expose as a
   sidebar/settings option and cite the ODS-heterogeneous-catalysis rationale.

2. **`SAT_THRESH_1 = 8.0` / `SAT_THRESH_2 = 15.0`** — `app_ods.py:1023-1024`. Auto-deletes
   real data points based on two unexplained removal-increment thresholds. Severity:
   **critical** — silently drops measurements. Fix: move to settings with defaults and add a
   per-run confirm step.

3. **`DELTA = 2.5`** — `app_ods.py:672` AICc parsimony window; no citation or tunable.
   Severity: **medium**. Fix: name it `AICC_DELTA`, document (Burnham & Anderson rule of
   thumb), make configurable.

4. **`curve_fit` `p0` initial guesses** — `app_ods.py:467,484,502,520,537,559,581,603,624`
   (e.g. `p0=[1e-6]`, `p0=[0.01, 10.0]`, `p0=[0.1,0.01,0.6]`) are magic values that bias
   convergence and are not validated against data scale. Severity: **medium**. Fix: derive
   p0 from linearized estimates or data-driven defaults.

5. **Numeric clip guards `1e-15` / `1e-12`** — `app_ods.py:248,1207,1214,1219,1351,1354`;
   `ods_kinetics.py:84,90`. Unexplained epsilon values that silently distort fits near zero.
   Severity: **low**. Fix: define `EPS` constant with rationale.

6. **Solver tolerances `rtol=1e-6, atol=1e-9`** — `app_ods.py:257,260,307`. Hardcoded, no
   comment. Severity: **low**. Fix: constant with rationale.

7. **`0.01` R² closeness window** — `app_ods.py:680` in `_best_model` step 4a. Severity:
   **low**. Fix: named constant.

8. **Arrhenius Ea-band thresholds** (`<20 / 20-40 / 40-80 / >100 kJ/mol`) —
   `app_ods.py:2042-2046`. Uncited literature thresholds. Severity: **low**. Fix: cite source
   in the interpretation text.

---

## 4. Numerical risks

1. **Fixed `p0` in every `curve_fit` call** — `app_ods.py:467-624`. With data spanning
   different magnitude scales (e.g. µM vs mM C0, minutes vs hours), fixed guesses can drive
   the solver to local minima or outright failure, which then becomes a silent `R2=-999`.
   Severity: **critical**. Fix: auto-scale p0 (e.g. from linearized regression) and add
   multiple-restart fitting.

2. **Unchecked `t[0] == 0` assumption in ODE models** — `app_ods.py:256` (`_lh_model`)
   branches on exact float `t[0] == 0`; `_eley_rideal` (`app_ods.py:306`) *always* prepends
   `0.0` regardless. The two ODE models behave inconsistently for the same input, and the
   exact float comparison is fragile. Severity: **medium**. Fix: normalize time once at the
   fit boundary (guaranteed sorted, `t[0]=0`) before calling the ODE models.

3. **ODE state clipping mid-integration** — `_lh_model` `Cv = max(C[0], 0.0)` and
   `_eley_rideal` `Cv = max(float(C[0]), 1e-12)` (`app_ods.py:254,304`) inject non-smooth
   kinks into the derivative that `odeint` must step over; combined with loose fixed
   tolerances this can bias solutions. Severity: **medium**. Fix: use a smooth
   reparameterization (log-concentration) or a stiff solver with adaptive tolerances.

4. **`_power_law` n is clipped to `[0.1, 5.0]`** — `app_ods.py:271`, and the fractional-power
   branches (`inside**exp`) can still produce NaN if `inside` is tiny-but-negative; the
   v3.5.4 fix only handles the `n>1` branch. Severity: **medium**. Fix: clip `inside` to a
   floor before the power and vectorise both branches identically.

5. **Linearized fits in `catlab` can feed NaN into `log`** — `ods_kinetics.py:84`
   `np.log(C0 / np.clip(C, 1e-15, None))` and `catalyst_analytics.py:149`
   `np.log(self.c / self.c0)` blow up (inf/NaN, then `linregress` fails) if `c <= 0` or
   `c0 <= 0`. Severity: **medium**. Fix: guard `C0/Ct > 0` and drop/raise on non-positive.

---

## 5. Functions in `app_ods.py` with zero direct test coverage

`tests/test_catlab.py` imports only from `catlab/` (line 7-10); nothing imports `app_ods.py`
(it also can't be imported cleanly under pytest because of module-level `st.set_page_config`
and `matplotlib.use("Agg")`). **Every function in `app_ods.py` has zero direct test
coverage**, notably:

1. `_fit_nonlinear` (`app_ods.py:458`) — the core engine. Severity: **critical**.
2. `_best_model` / `_get_valid_models` (`app_ods.py:644/647`) — the AICc selection logic.
   Severity: **critical**.
3. `_to_mol_L` / `_C0_both` (`app_ods.py:186/227`) — unit conversion incl. ppmS modes.
   Severity: **high**.
4. `_power_law`, `_lh_model`, `_eley_rideal`, `_avrami`, `_double_exponential`
   (`app_ods.py:264/251/300/310/315`) — all model functions. Severity: **high**.
5. `_aic` / `_aicc` / `_adj_r2` / `_r2` (`app_ods.py:321-352`). Severity: **high**.
6. `_load_kinetic_data` (`app_ods.py:420`). Severity: **high**.
7. `_lh_t_half`, `_elovich_t_half`, `_power_law_t_half` (`app_ods.py:285/356/370`).
   Severity: **medium**.
8. All `_tab_*` UI functions and `main()` (`app_ods.py:908-2233`). Severity: **medium**
   (UI; extract pure logic for tests).

Fix: factor the pure math (models, AICc, conversion, loader) into `catlab/` and add a
`tests/test_ods_kinetics_app.py` that covers the nonlinear engine against synthetic data
with known parameters.

---

## 6. CI rules that never fail the build, and missing LICENSE

1. **`--exit-zero` lint step** — `.github/workflows/ci.yml:45`:
   `flake8 . --count --exit-zero --max-complexity=10 --max-line-length=120 --statistics`.
   This step always returns 0 regardless of complexity/line-length violations, so the "Lint
   with flake8" job can never fail on those rules. Severity: **critical** — the build is
   green while lint debt accumulates unseen. Fix: drop `--exit-zero` (keep only the
   `E9,F63,F7,F82` hard-fail pass on line 43, or turn the second pass into a real gate with
   an allow-list).

2. **Coverage has no minimum threshold** — `ci.yml:49` runs `pytest --cov` and uploads to
   Codecov with `fail_ci_if_error: false`; there is no `--cov-fail-under`, so dropping
   coverage never fails CI. Severity: **medium**. Fix: add `--cov-fail-under=80` and set
   `fail_ci_if_error: true`.

3. **Missing `LICENSE` file** — no `LICENSE`/`LICENSE.md` exists in the repo root (only
   `CITATION.cff`, `README.md`, `CHANGELOG.md`). The package is published with no license,
   which is legally ambiguous for reuse. Severity: **medium**. Fix: add a `LICENSE` file
   (MIT suggested, matching the repo intent) and reference it from `README.md`/`CITATION.cff`.

---

## 7. Inline "FIX / vX.Y.Z" comments that should be in CHANGELOG

`CHANGELOG.md` stops at **v3.0**; all v3.1→v3.5.x history is documented as inline comments
inside `app_ods.py` instead:

1. **Module docstring changelog block** — `app_ods.py:9-71`: FIX 1-5, NEW 7-10, FIX
   A/B/D/E/G/H/I/J/L, NEW N/O/P, v3.5.0, NEW AC/AD/AE, FIX X/Y/Z/AA/AB/R/S/T/U/V/W.
   Severity: **medium**. Fix: move to `CHANGELOG.md` (keep a one-line pointer in the
   docstring).
2. **`_power_law` docstring "v3.5.4: Correct branch…"** — `app_ods.py:267`. Severity:
   **low**. Fix: CHANGELOG entry; keep only the math explanation in code.
3. **Inline `# FIX J/E/W/B/I/D/R/S/T/U/A` markers** — `app_ods.py:81,95,160,190,340,372,419,
   499,707,782,866,2025,2094`. Severity: **low**. Fix: convert to dated CHANGELOG entries and
   delete the inline `FIX` tags.
4. **Version-string drift** — `app_ods.py:7` claims `v3.5.0`, `:100` and `:2201` claim
   `v3.5.3`, `:267` references `v3.5.4`, and `README.md:4/11/198/235` says `v3.5.3` —
   despite FIX V ("version string unified to v3.5.0"). Severity: **low**. Fix: single
   `__version__` constant consumed by docstring, page config, header, and README badge.
