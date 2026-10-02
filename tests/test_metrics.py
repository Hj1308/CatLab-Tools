# tests/test_metrics.py
# Unit tests for catlab/metrics.py — pure scientific helpers.
import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from scipy import stats as scipy_stats

from catlab.metrics import (
    _C0_both,
    _initial_tof_site,
    _initial_tof_mass,
    _arrhenius_ci,
    c_to_user,
)

MW_S = 32.06


# ---- _C0_both ---------------------------------------------------------
def test_C0_both_ppmS_volumetric():
    # 250 ppmS = 250 mg(S)/L -> 250/32.06/1000 mol/L compound (n_sulfur=1)
    c0_compound, c0_s = _C0_both(250.0, "ppmS", None, None, n_sulfur=1)
    expected = 250.0 / MW_S / 1000.0
    assert np.isclose(c0_compound, expected)
    assert np.isclose(c0_s, expected)


def test_C0_both_ppmS_mass_basis_uses_density():
    # mass basis: mg/kg fuel, so mg/L = value * rho (g/mL == kg/L)
    c0_compound, c0_s = _C0_both(250.0, "ppmS", None, 0.8, n_sulfur=1, ppms_volumetric=False)
    expected = 250.0 * 0.8 / MW_S / 1000.0
    assert np.isclose(c0_compound, expected)
    assert np.isclose(c0_s, expected)


def test_C0_both_non_ppmS_splits_by_n_sulfur():
    # DBT has 1 sulfur atom; a compound with n_sulfur=2 gives C0_S = 2*C0_compound
    c0_compound, c0_s = _C0_both(0.01, "mol/L", None, None, n_sulfur=2)
    assert np.isclose(c0_compound, 0.01)
    assert np.isclose(c0_s, 0.02)


# ---- _initial_tof_site / _initial_tof_mass -----------------------------
def test_initial_tof_site_arithmetic():
    # r0 * V / n_sites = 1.949470e-4 * 0.010 / 2.5e-5 = 0.0779788 min^-1
    val = _initial_tof_site(1.949470e-4, 0.010, 2.5e-5)
    assert np.isclose(val, 0.0779788, rtol=1e-9)


def test_initial_tof_mass_arithmetic():
    # r0 * V * 1000 / m = 1.949470e-4 * 0.010 * 1000 / 0.05 = 0.0389894 mmol/g/min
    val = _initial_tof_mass(1.949470e-4, 0.010, 0.05)
    assert np.isclose(val, 0.0389894, rtol=1e-9)


def test_initial_tof_nan_for_none_or_nonpositive():
    assert np.isnan(_initial_tof_site(None, 0.010, 2.5e-5))
    assert np.isnan(_initial_tof_site(1e-4, 0.010, 0.0))
    assert np.isnan(_initial_tof_site(1e-4, 0.010, -1.0))
    assert np.isnan(_initial_tof_mass(None, 0.010, 0.05))
    assert np.isnan(_initial_tof_mass(1e-4, 0.010, 0.0))
    assert np.isnan(_initial_tof_mass(1e-4, 0.010, -1.0))


# ---- _arrhenius_ci -----------------------------------------------------
def test_arrhenius_ci_three_points():
    # cov is a placeholder; only the SEs and n_T matter for the ratio.
    cov = np.array([[0.5, 0.1], [0.1, 0.2]])
    Ea_ci, lnA_ci, df = _arrhenius_ci(cov, 3)
    assert df == 1
    t_crit = scipy_stats.t.ppf(0.975, 1)
    assert np.isclose(Ea_ci, np.sqrt(0.5) * 8.314 / 1000.0 * t_crit)
    assert np.isclose(lnA_ci, np.sqrt(0.2) * t_crit)


def test_arrhenius_ci_five_points():
    cov = np.array([[0.5, 0.1], [0.1, 0.2]])
    Ea_ci, lnA_ci, df = _arrhenius_ci(cov, 5)
    assert df == 3
    t_crit = scipy_stats.t.ppf(0.975, 3)
    assert np.isclose(Ea_ci, np.sqrt(0.5) * 8.314 / 1000.0 * t_crit)
    assert np.isclose(lnA_ci, np.sqrt(0.2) * t_crit)


def test_arrhenius_ci_two_points_nan():
    Ea_ci, lnA_ci, df = _arrhenius_ci(np.zeros((2, 2)), 2)
    assert df == 0
    assert np.isnan(Ea_ci)
    assert np.isnan(lnA_ci)


# ---- c_to_user ---------------------------------------------------------
def test_c_to_user_ppmS():
    # mol/L -> mg(S)/L = *32.06*1000
    assert np.isclose(c_to_user(0.0077978789769, "ppmS", None), 250.0, rtol=1e-9)


def test_c_to_user_ppm_and_mgL():
    assert np.isclose(c_to_user(0.001, "ppm", 184.26), 184.26)
    assert np.isclose(c_to_user(0.001, "mg/L", 184.26), 184.26)


def test_c_to_user_mmol_and_gL():
    assert np.isclose(c_to_user(0.001, "mmol/L", None), 1.0)
    assert np.isclose(c_to_user(0.001, "g/L", 184.26), 0.18426)


def test_c_to_user_molL_passthrough():
    assert c_to_user(0.001, "mol/L", None) == 0.001
