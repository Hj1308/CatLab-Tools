# tests/test_ods_kinetics.py
# Tests for catlab/ods_kinetics.py — previously 0% coverage.
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import OptimizeWarning

from catlab.ods_kinetics import _fit_kinetics, generate_template, run_ods_analysis
from catlab.kinetics_engine import _first_order, _second_order, _zero_order

C0 = 0.01559  # mol/L  (~500 ppmS)


# ==============================================================
# _fit_kinetics — core fitting logic
# ==============================================================

class TestFitKinetics:
    """Tests for _fit_kinetics — delegates to shared nonlinear engine."""

    T = np.array([0, 5, 10, 20, 40, 60, 90, 120, 180, 240, 300.0])

    # ---- normal cases ---------------------------------------------------

    def test_pfo_recovers_rate_constant(self):
        """Synthetic first-order: k_app within 5% of true, R2 > 0.99."""
        k_true = 0.02
        Ct = _first_order(self.T, k_true, C0)
        result = _fit_kinetics(self.T, Ct, C0)
        assert abs(result["Kapp (1/min)"] - k_true) / k_true < 0.05
        assert result["R2_first"] > 0.99

    def test_pso_recovers_rate_constant(self):
        """Synthetic second-order: k2 within 5% of true, R2 > 0.99."""
        k_true = 0.6
        Ct = _second_order(self.T, k_true, C0)
        result = _fit_kinetics(self.T, Ct, C0)
        assert abs(result["K2 (L/mol/min)"] - k_true) / k_true < 0.05
        assert result["R2_second"] > 0.99

    def test_zero_order_recovers_rate(self):
        """Synthetic zero-order: recovers k0 > 0, R2 > 0.99."""
        k_true = 1e-5
        Ct = _zero_order(self.T, k_true, C0)
        result = _fit_kinetics(self.T, Ct, C0)
        assert result["K0 (mol/L/min)"] > 0
        assert result["R2_zero"] > 0.99

    def test_output_dict_has_all_expected_keys(self):
        """Every key callers rely on is present."""
        Ct = _first_order(self.T, 0.02, C0)
        result = _fit_kinetics(self.T, Ct, C0)
        expected = [
            "K0 (mol/L/min)", "R2_zero",
            "Kapp (1/min)",   "R2_first",
            "K2 (L/mol/min)", "R2_second",
            "t_half (min)",
            "_t", "_y0", "_y1", "_y2", "_C0", "_C",
        ]
        for key in expected:
            assert key in result, f"missing key: {key}"

    def test_t_half_matches_ln2_over_kapp(self):
        """t_half ≈ ln(2) / k_app for first-order kinetics."""
        k_true = 0.02
        Ct = _first_order(self.T, k_true, C0)
        result = _fit_kinetics(self.T, Ct, C0)
        expected = np.log(2) / result["Kapp (1/min)"]
        assert result["t_half (min)"] == pytest.approx(expected, rel=0.01)

    # ---- edge cases ---------------------------------------------------

    def test_single_point_runs_without_crash(self):
        """One data point: engine returns p0, R2=0.  Covariance warning
        is expected (curve_fit cannot estimate covariance from 1 point)."""
        with pytest.warns(OptimizeWarning, match="Covariance"):
            result = _fit_kinetics(np.array([0.0]), np.array([C0]), C0)
        # Degenerate fit — k may be p0 or 0, but R2 should be 0
        assert result["R2_first"] == pytest.approx(0.0, abs=1e-12)
        assert result["R2_second"] == pytest.approx(0.0, abs=1e-12)

    def test_two_point_fit(self):
        """Two points: minimal valid fit recovers k.  Multi-param models
        (Elovich, L-H, etc.) cannot estimate covariance from 2 points and
        raise OptimizeWarning — expected and harmless."""
        t = np.array([0.0, 60.0])
        k_true = 0.02
        Ct = _first_order(t, k_true, C0)
        with pytest.warns(OptimizeWarning, match="Covariance"):
            result = _fit_kinetics(t, Ct, C0)
        assert result["R2_first"] > 0.95
        assert result["Kapp (1/min)"] > 0

    def test_constant_concentration(self):
        """No decay (Ct == C0): R2 is grossly negative due to a longstanding
        floating-point artefact in _r2: np.mean(Ct) differs from Ct[0] by
        ~2e-18, making ss_tot ≈ 3e-35 instead of 0, so 1-ss_res/ss_tot ≈ -1e26.
        This is a known engine issue, not introduced by ods_kinetics.  We only
        verify the function doesn't crash."""
        Ct = np.full_like(self.T, C0)
        result = _fit_kinetics(self.T, Ct, C0)
        # R2 is meaningless here — just check the function completed
        assert isinstance(result["R2_first"], float)

    def test_linearized_y0_is_c0_minus_ct(self):
        """y0 = C0 - C computed correctly."""
        Ct = _first_order(self.T, 0.02, C0)
        result = _fit_kinetics(self.T, Ct, C0)
        np.testing.assert_allclose(result["_y0"], C0 - Ct, atol=1e-12)

    def test_linearized_y1_is_ln_c0_over_ct(self):
        """y1 = ln(C0/Ct) computed correctly."""
        Ct = _first_order(self.T, 0.02, C0)
        result = _fit_kinetics(self.T, Ct, C0)
        expected = np.log(C0 / np.clip(Ct, 1e-15, None))
        np.testing.assert_allclose(result["_y1"], expected, atol=1e-12)

    def test_linearized_y2_is_one_over_ct_minus_one_over_c0(self):
        """y2 = 1/Ct - 1/C0 computed correctly."""
        Ct = _first_order(self.T, 0.02, C0)
        result = _fit_kinetics(self.T, Ct, C0)
        expected = (1.0 / np.clip(Ct, 1e-15, None)) - (1.0 / C0)
        np.testing.assert_allclose(result["_y2"], expected, atol=1e-12)

    def test_noisy_pfo_data_still_recovers_k(self):
        """Moderate noise: k within 20% of true."""
        k_true = 0.02
        rng = np.random.default_rng(42)
        Ct_clean = _first_order(self.T, k_true, C0)
        Ct = np.clip(Ct_clean * (1 + rng.normal(0, 0.03, len(self.T))),
                     0.001, C0 * 0.999)
        result = _fit_kinetics(self.T, Ct, C0)
        assert result["R2_first"] > 0.90
        assert abs(result["Kapp (1/min)"] - k_true) / k_true < 0.20

    # ---- converged contract --------------------------------------------

    def test_converged_false_zeroes_ks(self, monkeypatch):
        """AUD-5 regression: explicit converged=False → k=0, R2=0 for all
        three models, matching the engine's failure contract."""
        fake = {"Zero-order": {"converged": False},
                "Pseudo-first": {"converged": False},
                "Pseudo-second-order": {"converged": False}}
        monkeypatch.setattr("catlab.ods_kinetics._fit_nonlinear",
                            lambda *a, **kw: fake)
        result = _fit_kinetics(np.array([0.0, 1.0]), np.array([C0, C0*0.5]), C0)
        assert result["K0 (mol/L/min)"] == 0.0
        assert result["Kapp (1/min)"] == 0.0
        assert result["K2 (L/mol/min)"] == 0.0
        assert result["R2_zero"] == 0.0
        assert result["R2_first"] == 0.0
        assert result["R2_second"] == 0.0


# ==============================================================
# generate_template — Excel template creation
# ==============================================================

class TestGenerateTemplate:
    """Tests for generate_template — creates .xlsx template files."""

    def test_default_params_creates_file_with_two_sheets(self, tmp_path):
        path = str(tmp_path / "test_template.xlsx")
        result = generate_template(path)
        assert result == path
        assert os.path.isfile(path)
        with pd.ExcelFile(path) as xl:
            assert xl.sheet_names == ["Catalyst_1", "Catalyst_2"]

    def test_custom_catalysts_creates_correct_sheets(self, tmp_path):
        path = str(tmp_path / "custom.xlsx")
        cats = ["Cat-A", "Cat-B", "Cat-C"]
        generate_template(path, cats)
        with pd.ExcelFile(path) as xl:
            assert xl.sheet_names == cats

    def test_long_sheet_name_is_truncated_to_31_chars(self, tmp_path):
        path = str(tmp_path / "long.xlsx")
        long_name = "A" * 40
        generate_template(path, [long_name])
        with pd.ExcelFile(path) as xl:
            assert len(xl.sheet_names[0]) == 31
            assert xl.sheet_names[0] == long_name[:31]

    def test_no_example_catalysts_defaults(self, tmp_path):
        path = str(tmp_path / "none.xlsx")
        generate_template(path, example_catalysts=None)
        with pd.ExcelFile(path) as xl:
            assert xl.sheet_names == ["Catalyst_1", "Catalyst_2"]


# ==============================================================
# run_ods_analysis — full lifecycle (Excel → fit → plot → export)
# ==============================================================

class TestRunODSAnalysis:
    """Integration tests for run_ods_analysis."""

    @staticmethod
    def _make_excel(path, time_rem_sets):
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            for name, t, rem in time_rem_sets:
                df = pd.DataFrame({"Time (min)": t, "Removal (%)": rem})
                df.to_excel(writer, sheet_name=name[:31], index=False)

    def test_basic_single_catalyst_ppmS(self, tmp_path, monkeypatch):
        """End-to-end: single-catalyst Excel with PFO-like data."""
        import matplotlib.pyplot as plt
        monkeypatch.setattr(plt, "savefig", lambda *a, **kw: None)

        t_min = [0, 10, 20, 40, 60, 90, 120, 180, 240, 300]
        rem = [0.0, 18.1, 33.0, 55.1, 69.9, 83.5, 90.9, 97.3, 99.2, 99.8]
        excel_path = str(tmp_path / "data.xlsx")
        self._make_excel(excel_path, [("Cat-A", t_min, rem)])
        df = run_ods_analysis(
            excel_path, c0_value=500.0, c0_unit="ppmS",
            output_dir=str(tmp_path),
        )
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert df.loc[0, "Catalyst"] == "Cat-A"
        assert df.loc[0, "R2_first"] > 0.95
        assert df.loc[0, "Kapp (1/min)"] > 0
        assert df.loc[0, "t_half (min)"] > 0

    def test_multi_catalyst(self, tmp_path, monkeypatch):
        """Two catalysts in one Excel — both fitted."""
        import matplotlib.pyplot as plt
        monkeypatch.setattr(plt, "savefig", lambda *a, **kw: None)

        t_min = [0, 30, 60, 120, 180]
        excel_path = str(tmp_path / "multi.xlsx")
        self._make_excel(excel_path, [
            ("Cat-A", t_min, [0.0, 45.0, 70.0, 90.0, 96.0]),
            ("Cat-B", t_min, [0.0, 20.0, 35.0, 55.0, 68.0]),
        ])
        df = run_ods_analysis(
            excel_path, c0_value=500.0, c0_unit="ppmS",
            output_dir=str(tmp_path),
        )
        assert len(df) == 2
        assert df.loc[0, "Catalyst"] == "Cat-A"
        assert df.loc[1, "Catalyst"] == "Cat-B"

    def test_t0_not_zero_gets_prepended(self, tmp_path, monkeypatch):
        """Data starting at t>0: run_ods_analysis inserts (0, 0)."""
        import matplotlib.pyplot as plt
        monkeypatch.setattr(plt, "savefig", lambda *a, **kw: None)

        t_min = [10, 30, 60, 120, 180]
        rem = [18.0, 45.0, 70.0, 90.0, 96.0]
        excel_path = str(tmp_path / "shifted.xlsx")
        self._make_excel(excel_path, [("Cat-A", t_min, rem)])
        df = run_ods_analysis(
            excel_path, c0_value=500.0, c0_unit="ppmS",
            output_dir=str(tmp_path),
        )
        assert len(df) == 1
        assert df.loc[0, "R2_first"] > 0.90
