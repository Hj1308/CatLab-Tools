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
