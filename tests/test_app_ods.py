# tests/test_app_ods.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
st = pytest.importorskip("streamlit")
st.set_page_config = lambda *args, **kwargs: None

import numpy as np
import pandas as pd
from unittest.mock import MagicMock

import app_ods


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
    removal 91% → cutoff 77.35% at 0.85; every point above the cutoff (80, 91) is
    dropped, not just the last one (the old increment-based rule dropped only the
    final point here). Default max_fractional_uptake is 1.0 (disabled); tests pass
    0.85 explicitly to exercise the cutoff."""

    T   = np.array([0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0])
    REM = np.array([0.0, 16.0, 30.0, 50.0, 68.0, 80.0, 91.0])

    @staticmethod
    def _best(t, rem, c0):
        Ct = c0 * (1.0 - np.asarray(rem, dtype=float) / 100.0)
        res = app_ods._fit_nonlinear(np.asarray(t, dtype=float), Ct, c0)
        best = app_ods._best_model(res, app_ods.MODEL_NAMES)
        return best, res[best]["R2"] if best else None

    def test_excludes_all_points_above_cutoff(self):
        excl, t_keep, rem_keep = app_ods._auto_saturation_exclusions(
            self.T, self.REM, max_fractional_uptake=0.85)
        assert excl == [6, 4]
        assert t_keep.tolist() == [0.0, 0.5, 1.0, 2.0, 3.0]
        assert rem_keep.tolist() == [0.0, 16.0, 30.0, 50.0, 68.0]

    def test_cutoff_scales_with_final_removal(self):
        # final = 80 → cutoff 68.0 at 0.85; only the point above it (80) drops.
        rem2 = np.array([0.0, 12.0, 25.0, 40.0, 55.0, 68.0, 80.0])
        excl, t_keep, rem_keep = app_ods._auto_saturation_exclusions(
            self.T, rem2, max_fractional_uptake=0.85)
        assert excl == [6]
        assert rem_keep.tolist() == [0.0, 12.0, 25.0, 40.0, 55.0, 68.0]

    def test_lower_cutoff_drops_more_points(self):
        excl, t_keep, rem_keep = app_ods._auto_saturation_exclusions(
            self.T, self.REM, max_fractional_uptake=0.5)
        # cutoff = 45.5 → 50, 68, 80, 91 all dropped
        assert excl == [6, 4, 3, 2]
        assert t_keep.tolist() == [0.0, 0.5, 1.0]
        assert rem_keep.tolist() == [0.0, 16.0, 30.0]

    def test_max_frac_1_0_disables(self):
        excl, t_keep, rem_keep = app_ods._auto_saturation_exclusions(
            self.T, self.REM, max_fractional_uptake=1.0)
        assert excl == []
        assert t_keep.tolist() == self.T.tolist()

    def test_default_1_0_disables(self):
        excl, t_keep, rem_keep = app_ods._auto_saturation_exclusions(self.T, self.REM)
        assert excl == []
        assert t_keep.tolist() == self.T.tolist()

    def test_best_model_flips_without_vs_with_exclusion(self):
        c0 = 500.0 / 32.06 / 1000.0
        best_before, _ = self._best(self.T, self.REM, c0)
        excl, t_keep, rem_keep = app_ods._auto_saturation_exclusions(
            self.T, self.REM, max_fractional_uptake=0.85)
        assert excl == [6, 4]
        best_after, _ = self._best(t_keep, rem_keep, c0)

        assert best_before == "L-H"
        assert best_after == "Pseudo-first"
        assert best_before != best_after


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
