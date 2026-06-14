# tests/test_catlab.py
import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from catlab import (
    convert_to_mmol_L, SampleInfo, KineticsAnalyser,
    calc_conversion, calc_tof, calc_toc_removal,
    BETAnalyser, TPlotAnalyser
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

class TestKinetics:
    def setup_method(self):
        info = SampleInfo("MoS2", "desulfurization", 0.05, 0.05, 500.0, "ppmS",
                          active_sites_mmol_g=0.32)
        t = np.array([0, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0])
        c = np.array([convert_to_mmol_L(v, "ppmS") for v in [500,420,350,250,160,100,45]])
        self.an = KineticsAnalyser(t, c, info)
    def test_conversion(self):      assert self.an.full_report()["Conversion X (%)"] > 80
    def test_best_r2(self):         assert self.an.best_fit()["R2"] > 0.90
    def test_tof_positive(self):    assert self.an.full_report()["TOF (h⁻¹)"] > 0
    def test_profile_len(self):     assert len(self.an.conversion_profile()) == 7

class TestHelpers:
    def test_conversion(self):   assert calc_conversion(100.0, 10.0) == 90.0
    def test_tof(self):          assert abs(calc_tof(1.0, 0.05, 0.32, 6.0) - 1.0/(0.016*6)) < 0.01
    def test_toc(self):          assert abs(calc_toc_removal(85.0, 12.0) - 85.88) < 0.1
    def test_toc_nan(self):
        import math; assert math.isnan(calc_toc_removal(0.0, 5.0))

class TestBET:
    def setup_method(self):
        self.p = np.array([0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.50,0.70,0.90,0.99])
        self.v = np.array([85, 102, 115, 126, 135, 143, 150, 172, 200, 280, 520])
    def test_sbet_positive(self): assert BETAnalyser(self.p, self.v).fit_bet()["S_BET_m2g"] > 0
    def test_r2(self):            assert BETAnalyser(self.p, self.v).fit_bet()["R2"] > 0.99

class TestTPlot:
    def setup_method(self):
        p = np.array([0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.50,0.70,0.90,0.99])
        v = np.array([85, 102, 115, 126, 135, 143, 150, 172, 200, 280, 520])
        self.tp = TPlotAnalyser(p, v, s_bet=442.0, total_pore_volume=0.52)
    def test_pore_sum_100(self):
        res  = self.tp.fit_tplot()
        dist = self.tp.pore_distribution(res["V_micro_cm3g"])
        assert abs(dist["Micropore_%"]+dist["Mesopore_%"]+dist["Macropore_%"] - 100.0) < 0.2
    def test_vtot(self):
        res  = self.tp.fit_tplot()
        dist = self.tp.pore_distribution(res["V_micro_cm3g"])
        assert abs(dist["V_total_cm3g"] - 0.52) < 0.01
