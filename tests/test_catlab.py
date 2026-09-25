# tests/test_catlab.py
import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from catlab import (
    convert_to_mmol_L, SampleInfo, KineticsAnalyser,
    calc_conversion, calc_tof, calc_toc_removal,
)

class TestUnitConverter:
    def test_mol_L(self):    assert convert_to_mmol_L(1.0, "mol/L") == 1000.0
    def test_mmol_L(self):   assert convert_to_mmol_L(5.0, "mmol/L") == 5.0
    def test_ppmS(self):     assert abs(convert_to_mmol_L(500.0, "ppmS") - 500/32.06) < 1e-4
    def test_mg_L(self):     assert abs(convert_to_mmol_L(180.0, "mg/L", mw=180.0) - 1.0) < 1e-6
    def test_g_L(self):      assert abs(convert_to_mmol_L(1.0, "g/L", mw=100.0) - 10.0) < 1e-6
    def test_missing_mw(self):
        with pytest.raises(ValueError): convert_to_mmol_L(50.0, "mg/L")
    def test_unknown_unit(self):
        with pytest.raises(ValueError): convert_to_mmol_L(1.0, "xyz")

class TestSampleInfo:
    def setup_method(self):
        self.info = SampleInfo("Cat", "desulfurization", 0.05, 0.05, 500.0, "ppmS",
                               active_sites_mmol_g=0.32)
    def test_c0_mmol_L(self):      assert abs(self.info.c0_mmol_L - 500/32.06) < 1e-3
    def test_loading(self):        assert self.info.catalyst_loading_g_L == 1.0
    def test_n0(self):             assert abs(self.info.n0_mmol - self.info.c0_mmol_L*0.05) < 1e-6
    def test_c0_mmol_L_multi_sulfur(self):
        """c0_mmol_L corrects ppmS to compound basis for di-sulfur substrates."""
        info2 = SampleInfo("Cat", "desulfurization", 0.05, 0.05, 500.0, "ppmS",
                           n_sulfur=2)
        assert abs(info2.c0_S_mmol_L - 500.0/32.06) < 1e-3
        assert abs(info2.c0_mmol_L - 500.0/32.06/2) < 1e-3

class TestKinetics:
    def setup_method(self):
        info = SampleInfo("MoS2", "desulfurization", 0.05, 0.05, 500.0, "ppmS",
                          active_sites_mmol_g=0.32)
        t = np.array([0, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0])
        c = np.array([convert_to_mmol_L(v, "ppmS") for v in [500,420,350,250,160,100,45]])
        self.an = KineticsAnalyser(t, c, info)
    def test_conversion(self):      assert self.an.full_report()["Conversion X (%)"] > 80
    def test_best_r2(self):         assert self.an.best_fit()["R2"] > 0.90
    def test_tof_positive(self):    assert self.an.full_report()["TOF (h\u207b\xb9)"] > 0
    def test_profile_len(self):     assert len(self.an.conversion_profile()) == 7

    def test_fit_methods_return_real_values_for_good_data(self):
        """Regression for AUD-5: converged-guard must not silently zero-out
        successful fits.  PFO synthetic data — all three nonlinear models
        should return k>0 and R2>0.9."""
        C0 = 15.59
        t = np.array([0, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0])
        Ct = C0 * np.exp(-0.3 * t)
        info = SampleInfo("Cat", "desulfurization", 0.05, 0.05, C0, "mmol/L",
                          active_sites_mmol_g=0.32)
        ka = KineticsAnalyser(t, Ct, info)
        z = ka.fit_zero_order()
        f = ka.fit_first_order()
        s = ka.fit_second_order()
        assert z["R2"] > 0.8, f"zero-order R2={z['R2']} — should recover"
        assert z["k (mmol/L/h)"] > 0
        assert f["R2"] > 0.9, f"first-order R2={f['R2']} — should recover"
        assert f["k (h\u207b\u00b9)"] > 0
        assert s["R2"] > 0.8, f"second-order R2={s['R2']} — should recover"
        assert s["k (L/mmol/h)"] > 0

    def test_converged_false_triggers_zero_return(self, monkeypatch):
        """AUD-5 regression: explicit converged=False must cause k=0,R2=0."""
        info = SampleInfo("x", "desulfurization", 0.05, 0.05, 15.59, "mmol/L")
        ka = KineticsAnalyser(np.array([0.0, 1.0]), np.array([15.59, 10.0]), info)
        fake = {"Zero-order": {"converged": False},
                "Pseudo-first": {"converged": False},
                "Pseudo-second-order": {"converged": False}}
        monkeypatch.setattr("catlab.catalyst_analytics._fit_nonlinear",
                            lambda *a, **kw: fake)
        ka = KineticsAnalyser(np.array([0.0, 1.0]), np.array([15.59, 10.0]), info)
        assert ka.fit_zero_order()["k (mmol/L/h)"] == 0.0
        assert ka.fit_zero_order()["R2"] == 0.0
        assert ka.fit_first_order()["k (h\u207b\u00b9)"] == 0.0
        assert ka.fit_first_order()["R2"] == 0.0
        assert ka.fit_second_order()["k (L/mmol/h)"] == 0.0
        assert ka.fit_second_order()["R2"] == 0.0


class TestConvergedContract:
    """The engine sets converged=False on failure and omits the key on success.
    All callers must interpret a missing key as a successful fit."""

    def test_engine_get_valid_models_defaults_to_true(self):
        from catlab.kinetics_engine import _get_valid_models
        ok = {"Pseudo-first": {"R2": 0.99, "aicc": 5.0}}
        assert "Pseudo-first" in _get_valid_models(ok, ["Pseudo-first"])

    def test_engine_get_valid_models_rejects_false(self):
        from catlab.kinetics_engine import _get_valid_models
        bad = {"Pseudo-first": {"R2": 0.99, "aicc": 5.0, "converged": False}}
        assert "Pseudo-first" not in _get_valid_models(bad, ["Pseudo-first"])

class TestHelpers:
    def test_conversion(self):   assert calc_conversion(100.0, 10.0) == 90.0
    def test_tof(self):          assert abs(calc_tof(1.0, 0.05, 0.32, 6.0) - 1.0/(0.016*6)) < 0.01
    def test_toc(self):          assert abs(calc_toc_removal(85.0, 12.0) - 85.88) < 0.1
    def test_toc_nan(self):
        import math; assert math.isnan(calc_toc_removal(0.0, 5.0))


# ─────────────────────────────────────────────────────────────────
# Roadmap phase 1 — one model-selection criterion across the repo.
# Before this change, KineticsAnalyser.best_fit() selected by max(R2)
# over 4 models while app_ods.py selected by AICc over 9.  On the L-H
# fixture below the package API returned "Zero-order" while the app
# returned "L-H" with an Akaike weight of ~0.9997 (Zero-order sat at
# dAICc = 27.52).  These tests pin the two interfaces together.
# ─────────────────────────────────────────────────────────────────
from scipy.integrate import odeint

from catlab.kinetics_engine import (
    _best_model, _fit_nonlinear, akaike_weights, MODEL_NAMES,
)


def _lh_fixture():
    """Synthetic Langmuir-Hinshelwood curve, 2% multiplicative noise."""
    info = SampleInfo("X", "desulfurization", 0.05, 0.010, 250.0, "ppmS",
                      active_sites_mmol_g=0.3)
    c0 = info.c0_mmol_L
    t = np.array([0, 5, 10, 20, 30, 45, 60, 90, 120], float)
    sol = odeint(lambda c, tt: [-0.25 * 0.9 * max(c[0], 0)
                                / (1 + 0.9 * max(c[0], 0))], [c0], t).flatten()
    rng = np.random.default_rng(7)
    return info, c0, t, sol * (1 + rng.normal(0, 0.02, sol.size))


class TestUnifiedModelSelection:
    def test_package_api_agrees_with_app_selection(self):
        """best_fit() must return exactly what the app's _best_model returns."""
        info, c0, t, c = _lh_fixture()
        pkg = KineticsAnalyser(t, c, info).best_fit()["model"]
        app = _best_model(_fit_nonlinear(t, c, c0), MODEL_NAMES)
        assert pkg == app, f"package reported {pkg!r}, app reported {app!r}"

    def test_lh_curve_is_reported_as_lh(self):
        """The 4-model R2 rule could never reach L-H; the AICc rule must."""
        info, c0, t, c = _lh_fixture()
        assert KineticsAnalyser(t, c, info).best_fit()["model"] == "L-H"
        w = akaike_weights(_fit_nonlinear(t, c, c0), model_names=MODEL_NAMES)
        assert w["L-H"]["weight"] == pytest.approx(0.9997, abs=5e-3)
        assert w["Zero-order"]["delta_aicc"] > 20.0

    def test_best_fit_contract(self):
        info, c0, t, c = _lh_fixture()
        b = KineticsAnalyser(t, c, info).best_fit()
        for key in ("model", "R2", "aicc", "delta_aicc", "weight",
                    "n_params", "selection_criterion"):
            assert key in b, f"missing key {key!r}"
        assert b["selection_criterion"] == "AICc"
        assert b["delta_aicc"] == 0.0
        assert 0.0 < b["weight"] <= 1.0
        assert b["n_params"] >= 1

    def test_lagergren_is_not_a_selection_candidate(self):
        """The linearised log(qe-qt) fit is on a different dependent
        variable, so it is not a valid AICc competitor.  It must stay
        available as a diagnostic but never be selected."""
        info, c0, t, c = _lh_fixture()
        an = KineticsAnalyser(t, c, info)
        assert an.best_fit()["model"] in MODEL_NAMES
        assert an.best_fit()["model"] != "Pseudo-first-order"
        assert an.fit_pseudo_first_order()["model"] == "Pseudo-first-order"

    def test_all_fits_failed(self, monkeypatch):
        info, c0, t, c = _lh_fixture()
        dead = {m: {"converged": False} for m in MODEL_NAMES}
        monkeypatch.setattr("catlab.catalyst_analytics._fit_nonlinear",
                            lambda *a, **kw: dead)
        b = KineticsAnalyser(t, c, info).best_fit()
        assert b["model"] == "All fits failed"
        assert b["R2"] == 0.0
        assert b["selection_criterion"] == "AICc"

    def test_partial_results_dict_does_not_raise(self):
        """_get_valid_models used to index res[m] unguarded, raising
        KeyError on a results dict that omits a model."""
        info, c0, t, c = _lh_fixture()
        full = _fit_nonlinear(t, c, c0)
        partial = {k: full[k] for k in ("Pseudo-first", "Zero-order")}
        assert _best_model(partial, MODEL_NAMES) in ("Pseudo-first", "Zero-order")
        assert akaike_weights(partial, model_names=MODEL_NAMES) != {}


class TestLagergrenDiagnostic:
    """fit_pseudo_first_order: per-gram uptake and non-monotonic points."""

    T = np.array([0, 0.25, 0.5, 1, 1.5, 2, 3, 4.0])

    @staticmethod
    def _info(m_g, V_L):
        return SampleInfo("Cat", "desulfurization", m_g, V_L, 500.0, "ppmS")

    def _curve(self, info):
        c0 = info.c0_mmol_L
        return c0 - 0.91 * c0 * (1 - np.exp(-1.2 * self.T))

    def test_qe_is_per_gram(self):
        """Regression: qe was the concentration drop (mmol/L) labelled mmol/g.
        With V/m = 2 L/g it must be twice the V/m = 1 value."""
        a = self._info(0.05, 0.05)
        b = self._info(0.05, 0.10)
        c = self._curve(a)
        qe_1 = KineticsAnalyser(self.T, c, a).fit_pseudo_first_order()["qe (mmol/g)"]
        qe_2 = KineticsAnalyser(self.T, c, b).fit_pseudo_first_order()["qe (mmol/g)"]
        assert qe_2 == pytest.approx(2.0 * qe_1, rel=1e-4)

    def test_non_monotonic_point_is_dropped_not_clipped(self):
        """Regression: a point below the final concentration was clipped to
        1e-12, giving a ln = -27.6 outlier (R2 fell to ~0.2).  It must be
        dropped and reported instead."""
        info = self._info(0.05, 0.10)
        c = self._curve(info)
        clean = KineticsAnalyser(self.T, c, info).fit_pseudo_first_order()
        c[5] = c[-1] - 0.05
        r = KineticsAnalyser(self.T, c, info).fit_pseudo_first_order()
        assert r["dropped_t"] == [2.0]
        assert r["n_points"] == 6
        assert r["R2"] > 0.99
        assert r["qe (mmol/g)"] == pytest.approx(clean["qe (mmol/g)"], rel=0.01)

    def test_too_few_usable_points_gives_nan(self):
        info = self._info(0.05, 0.05)
        t = np.array([0, 1, 2, 3.0])
        c0 = info.c0_mmol_L
        c = np.array([c0, 0.5 * c0, 0.2 * c0, 0.4 * c0])   # rises at the end
        r = KineticsAnalyser(t, c, info).fit_pseudo_first_order()
        assert r["n_points"] < 3
        assert np.isnan(r["k1 (h⁻¹)"]) and np.isnan(r["qe (mmol/g)"])
