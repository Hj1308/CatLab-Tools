# tests/test_app_ods.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
st = pytest.importorskip("streamlit")
st.set_page_config = lambda *args, **kwargs: None

import numpy as np
import pandas as pd
from unittest.mock import MagicMock
from scipy import stats as scipy_stats

import app_ods
from catlab.kinetics_engine import MIN_FIT_POINTS, _first_order


class TestNonConvergedSentinelHandling:
    """Regression: failed fits carry converged=False (not R2=-999), and every
    consumer must skip them via the converged flag -- never via numeric R2
    comparisons, because np.nan comparisons are always False."""

    @staticmethod
    def _failed(error="Optimal parameters not found"):
        return {"R2": np.nan, "aicc": np.nan, "converged": False, "error": error}

    def test_get_valid_models_skips_nonconverged(self):
        ok = {"R2": 0.99, "aicc": 5.0}
        res = {"Zero-order": self._failed(), "Pseudo-first": ok}
        assert app_ods._get_valid_models(res, ["Zero-order", "Pseudo-first"]) == {"Pseudo-first": ok}

    def test_best_model_returns_none_when_all_nonconverged(self):
        res = {"Zero-order": self._failed()}
        assert app_ods._best_model(res, ["Zero-order"]) is None

    def test_best_model_excludes_nonconverged_from_competition(self):
        ok_pso = {"R2": 0.95, "aicc": 5.0}
        ok_fo  = {"R2": 0.90, "aicc": 8.0}
        res = {"Zero-order": self._failed(),
               "Pseudo-first": ok_fo,
               "Pseudo-second-order": ok_pso}
        assert app_ods._best_model(
            res, ["Zero-order", "Pseudo-first", "Pseudo-second-order"]
        ) == "Pseudo-second-order"

    def test_residuals_skips_nonconverged_model_without_keyerror(self, monkeypatch):
        st = MagicMock()
        st.columns.return_value = (MagicMock(), MagicMock())
        st.expander.return_value = MagicMock()
        st.selectbox.side_effect = ["Cat-A Removal (%)", "Zero-order"]  # catalyst, then model
        monkeypatch.setattr(app_ods, "st", st)

        df = pd.DataFrame({
            "Time (min)": [0, 5, 10, 15, 20],
            "Cat-A Removal (%)": [0, 20, 40, 60, 80],
        })
        monkeypatch.setattr(app_ods, "_load_kinetic_data",
                            lambda uploaded: (df, "Time (min)", ["Cat-A Removal (%)"]))
        all_res = {m: self._failed() for m in app_ods.MODEL_NAMES}
        monkeypatch.setattr(app_ods, "_fit_nonlinear", lambda t, c, C0: all_res)

        app_ods._tab_residuals({"C0": 0.015}, MagicMock())  # must return, not KeyError

        assert st.error.called


class TestAutoSaturationDetection:
    """Regression: Simonin (2016) fractional-uptake cutoff. Case #1 — final
    removal 91% → cutoff 77.35% at 0.85; every point above the cutoff is
    dropped, not just the last one (the old increment-based rule dropped only the
    final point here). The retained set is never allowed to fall below
    MIN_FIT_POINTS, which keeps AICc finite for the whole 9-model portfolio; a
    cutoff that would strip below the floor reports `clamped=True` instead.
    Default max_fractional_uptake is 1.0 (disabled); tests pass lower values
    explicitly to exercise the cutoff."""

    T   = np.array([0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0])
    REM = np.array([0.0, 16.0, 30.0, 50.0, 68.0, 80.0, 91.0])

    @staticmethod
    def _best(t, rem, c0):
        Ct = c0 * (1.0 - np.asarray(rem, dtype=float) / 100.0)
        res = app_ods._fit_nonlinear(np.asarray(t, dtype=float), Ct, c0)
        best = app_ods._best_model(res, app_ods.MODEL_NAMES)
        return best, res[best]["R2"] if best else None

    def test_drops_all_points_above_cutoff_when_enough_headroom(self):
        """A curve long enough to satisfy the cutoff without hitting the floor:
        every point above the cutoff is dropped and clamped is False."""
        T   = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        REM = np.array([0.0, 8.0, 16.0, 25.0, 35.0, 50.0, 68.0, 80.0, 91.0])
        excl, t_keep, rem_keep, clamped = app_ods._auto_saturation_exclusions(
            T, REM, max_fractional_uptake=0.85)
        assert excl == [8, 7]
        assert rem_keep.tolist() == [0.0, 8.0, 16.0, 25.0, 35.0, 50.0, 68.0]
        assert not clamped

    def test_cutoff_scales_with_final_removal(self):
        # final = 80 → cutoff 68.0 at 0.85; only the point above it (80) drops.
        rem2 = np.array([0.0, 12.0, 25.0, 40.0, 55.0, 68.0, 80.0])
        excl, t_keep, rem_keep, clamped = app_ods._auto_saturation_exclusions(
            self.T, rem2, max_fractional_uptake=0.85)
        assert excl == [6]
        assert rem_keep.tolist() == [0.0, 12.0, 25.0, 40.0, 55.0, 68.0]
        assert not clamped

    def test_lower_cutoff_drops_more_points(self):
        excl, t_keep, rem_keep, clamped = app_ods._auto_saturation_exclusions(
            self.T, self.REM, max_fractional_uptake=0.5)
        # cutoff = 45.5 → 50, 68, 80, 91 are all ABOVE it.  MIN_FIT_POINTS stops
        # stripping at 6 points, so only the tail point (91) is dropped; 50, 68
        # and 80 are retained only because of the length floor, which is exactly
        # why clamped is True.
        assert excl == [6]
        assert clamped
        assert len(rem_keep) == MIN_FIT_POINTS
        assert rem_keep.tolist() == [0.0, 16.0, 30.0, 50.0, 68.0, 80.0]

    def test_max_frac_1_0_disables(self):
        excl, t_keep, rem_keep, clamped = app_ods._auto_saturation_exclusions(
            self.T, self.REM, max_fractional_uptake=1.0)
        assert excl == []
        assert t_keep.tolist() == self.T.tolist()
        assert not clamped

    def test_default_1_0_disables(self):
        excl, t_keep, rem_keep, clamped = app_ods._auto_saturation_exclusions(self.T, self.REM)
        assert excl == []
        assert t_keep.tolist() == self.T.tolist()
        assert not clamped

    def test_best_model_flips_without_vs_with_exclusion(self):
        c0 = 500.0 / 32.06 / 1000.0
        best_before, _ = self._best(self.T, self.REM, c0)
        excl, t_keep, rem_keep, _ = app_ods._auto_saturation_exclusions(
            self.T, self.REM, max_fractional_uptake=0.85)
        assert excl == [6]
        best_after, _ = self._best(t_keep, rem_keep, c0)

        assert best_before == "L-H"
        assert best_after != best_before

    def test_cutoff_0_9_pfo_retains_at_least_min_fit_points(self):
        """Regression: with k=0.08 / cutoff=0.90, the old loop dropped below the
        floor and made AICc inf for all models.  MIN_FIT_POINTS stops it, and the
        cutoff genuinely could not be fully applied, so clamped is True."""
        C0 = 7.798e-3
        T = np.array([0, 15, 30, 45, 60, 90, 120.0])
        k = 0.08
        Ct_clean = _first_order(T, k, C0)
        rem = 100.0 * (1.0 - Ct_clean / C0)
        excl, t_keep, rem_keep, clamped = app_ods._auto_saturation_exclusions(
            T, rem, max_fractional_uptake=0.90)
        assert len(rem_keep) >= MIN_FIT_POINTS, f"retained only {len(rem_keep)} points"
        assert clamped

    def test_clamped_true_when_floor_blocks_all_stripping(self):
        """Two separate facts about a curve already at exactly MIN_FIT_POINTS
        points whose tail exceeds the cutoff:
          (1) the returned DATA is unchanged — nothing can be dropped without
              going below the floor, so excluded is empty, and
          (2) the cutoff was NOT honoured — clamped is True even though
              excluded is empty.
        """
        n = MIN_FIT_POINTS
        rem = 20.0 * np.arange(n, dtype=float)  # [0, 20, ..., 20*(n-1)]
        T = np.arange(n, dtype=float)
        excl, t_keep, rem_keep, clamped = app_ods._auto_saturation_exclusions(
            T, rem, max_fractional_uptake=0.5)
        assert excl == []
        assert clamped
        assert rem_keep.tolist() == rem.tolist()
        assert len(rem_keep) == MIN_FIT_POINTS


class TestEdgeCaseModels:
    """Defensive coverage: Power-Law n>1 and L-H with t[0]!=0 already work;
    these tests lock in the behavior to prevent regressions."""

    C0 = 500.0 / 32.06 / 1000.0  # ~0.01559 mol/L
    T  = np.array([0, 10, 20, 40, 60, 90, 120, 180, 240, 300.0])

    def test_power_law_n_gt_1_fits_and_produces_valid_curve(self):
        """Power-Law with n=2.0 converges, yields high R², and reproduces
        the synthetic signal well.  Individual k/n parameters are *not*
        asserted — they are highly covariant for n>1 and multiple (k,n)
        pairs give the same curve."""
        n_true, k_true = 2.0, 0.001
        Ct_clean = app_ods._power_law(self.T, k_true, n_true, self.C0)
        res = app_ods._fit_nonlinear(self.T, Ct_clean, self.C0)
        pl = res["Power-Law"]
        assert pl["R2"] > 0.99
        assert 0.1 < pl["n_pl"] < 5.0
        assert pl.get("k") is not None
        assert pl.get("k") > 0
        # predicted curve within 5 % of true signal (relative to C0)
        pred = np.asarray(pl["pred"])
        assert np.max(np.abs(pred - Ct_clean)) / self.C0 < 0.05

    def test_lh_model_with_nonzero_t0_recovers_parameters(self):
        """L-H integration from t[0]=5 (not zero) — the ODE solver prepends
        t=0 with C(0)=C0 and correctly returns values at the original times."""
        k_true, K_true = 0.01, 15.0
        T_shift = np.array([5, 15, 30, 45, 60, 90, 120, 180.0])
        Ct_clean = app_ods._lh_model(T_shift, k_true, K_true, self.C0)
        # confirm signal decays from the shifted start
        assert Ct_clean[0] < self.C0
        # verify t=0 prepend hook: integrating from 0 gives same value at t=5
        T_full = np.array([0.0, 5, 15, 30, 45, 60, 90, 120, 180.0])
        Ct_full = app_ods._lh_model(T_full, k_true, K_true, self.C0)
        assert abs(Ct_full[0] - self.C0) < 1e-12
        assert abs(Ct_full[1] - Ct_clean[0]) < 1e-12
        # fit on shifted data recovers parameters (L-H is well-posed)
        res = app_ods._fit_nonlinear(T_shift, Ct_clean, self.C0)
        lh = res["L-H"]
        assert abs(lh["k"] - k_true) / k_true < 0.02
        assert abs(lh["K_ads"] - K_true) / K_true < 0.02


class TestArrheniusConfidenceInterval:
    """Arrhenius 95% CI uses the t-distribution (df = n_T - 2), not the
    normal-approximation z = 1.96, which understates the interval by ~6x at
    n_T = 3 (df = 1) and ~1.6x at n_T = 5 (df = 3)."""

    @staticmethod
    def _synthetic_k(temps_C, noise=0.0):
        rng = np.random.default_rng(0)
        A = 1e8
        Ea = 50000.0  # J/mol
        T = np.asarray(temps_C, dtype=float) + 273.15
        k = A * np.exp(-Ea / (app_ods.R_GAS * T))
        if noise > 0:
            k = k * np.exp(rng.normal(0.0, noise, size=k.shape))
        return T, k

    @staticmethod
    def _fit(T, k):
        inv_T = 1.0 / T
        ln_k = np.log(k)
        coeffs, cov = np.polyfit(inv_T, ln_k, 1, cov=True)
        return coeffs, cov

    def test_ci_uses_t_distribution_3_points(self):
        T, k = self._synthetic_k([25.0, 40.0, 55.0], noise=0.02)
        coeffs, cov = self._fit(T, k)
        Ea_ci, lnA_ci, df = app_ods._arrhenius_ci(cov, 3)

        old_Ea_ci = np.sqrt(cov[0, 0]) * app_ods.R_GAS / 1000.0 * 1.96
        old_lnA_ci = np.sqrt(cov[1, 1]) * 1.96
        expected_ratio = scipy_stats.t.ppf(0.975, 1) / 1.96

        assert df == 1
        assert np.isclose(Ea_ci / old_Ea_ci, expected_ratio, rtol=1e-6)
        assert np.isclose(lnA_ci / old_lnA_ci, expected_ratio, rtol=1e-6)
        assert Ea_ci > old_Ea_ci

    def test_ci_uses_t_distribution_5_points(self):
        T, k = self._synthetic_k([25.0, 35.0, 45.0, 55.0, 65.0], noise=0.02)
        coeffs, cov = self._fit(T, k)
        Ea_ci, lnA_ci, df = app_ods._arrhenius_ci(cov, 5)

        old_Ea_ci = np.sqrt(cov[0, 0]) * app_ods.R_GAS / 1000.0 * 1.96
        expected_ratio = scipy_stats.t.ppf(0.975, 3) / 1.96

        assert df == 3
        assert np.isclose(Ea_ci / old_Ea_ci, expected_ratio, rtol=1e-6)
        assert Ea_ci > old_Ea_ci

    def test_two_points_returns_nan_and_zero_df(self):
        # cov is ignored when df < 1; pass a placeholder.
        Ea_ci, lnA_ci, df = app_ods._arrhenius_ci(np.zeros((2, 2)), 2)
        assert df == 0
        assert np.isnan(Ea_ci)
        assert np.isnan(lnA_ci)

    def test_two_point_fit_warns_no_ci(self, monkeypatch):
        st = MagicMock()

        class _FakeFile:
            def __init__(self, name):
                self.name = name

            def seek(self, *a, **k):
                pass

        f1, f2 = _FakeFile("T25.xlsx"), _FakeFile("T40.xlsx")
        st.file_uploader.return_value = [f1, f2]
        st.number_input.side_effect = [25.0, 40.0]
        st.selectbox.return_value = "Pseudo-first"
        st.button.return_value = True
        st.progress.return_value = MagicMock()
        monkeypatch.setattr(app_ods, "st", st)

        df = pd.DataFrame({
            "Time (min)": [0, 5, 10, 20],
            "Cat-A Removal (%)": [0, 20, 40, 60],
        })
        monkeypatch.setattr(app_ods, "_load_kinetic_data",
                            lambda uploaded: (df, "Time (min)", ["Cat-A Removal (%)"]))

        k_iter = iter([0.010, 0.014])
        monkeypatch.setattr(app_ods, "_fit_nonlinear",
                            lambda t, Ct, C0: {"Pseudo-first": {"k": next(k_iter), "converged": True}})

        app_ods._tab_arrhenius({"C0": 0.015})

        messages = [str(c.args[0]) for c in st.warning.call_args_list]
        assert any("no valid 95% confidence interval" in m.lower() for m in messages)
