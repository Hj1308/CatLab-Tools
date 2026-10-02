# catlab/metrics.py
# Pure scientific helpers (no streamlit) shared by the app and CLI package.
import numpy as np
from scipy import stats as scipy_stats

from .kinetics_engine import _to_mol_L, MW_S

R_GAS = 8.314  # J/(mol·K)


def _C0_both(c0_val, c0_unit, mw_poll, rho_g_per_mL, n_sulfur=1, ppms_volumetric=True):
    if c0_unit == "ppmS":
        C0_S = _to_mol_L(c0_val, "ppmS", MW_S, rho_g_per_mL, ppms_volumetric)
        C0_compound = C0_S / n_sulfur
    else:
        C0_compound = _to_mol_L(c0_val, c0_unit, mw_poll, rho_g_per_mL, ppms_volumetric)
        C0_S = C0_compound * n_sulfur
    return C0_compound, C0_S


def _initial_tof_site(r0, V_L, n_sites_mol):
    """Initial-rate turnover frequency, min^-1.

    TOF_0 = r0 * V / n_sites, with r0 in mol/L/min, V in L and n_sites in mol.
    Returns nan when r0 is None (the selected model has no defined initial rate)
    or when n_sites_mol is not strictly positive.
    """
    if r0 is None or n_sites_mol <= 0:
        return float("nan")
    return r0 * V_L / n_sites_mol


def _initial_tof_mass(r0, V_L, m_g):
    """Initial-rate mass-normalised activity, mmol/g/min.

    TOF_mass,0 = r0 * V * 1000 / m, with r0 in mol/L/min, V in L and m in g.
    Returns nan when r0 is None or when m_g is not strictly positive.
    """
    if r0 is None or m_g <= 0:
        return float("nan")
    return r0 * V_L * 1000.0 / m_g


def _arrhenius_ci(cov, n_T):
    """95% confidence intervals for Eₐ and ln A from an Arrhenius fit.

    Uses the t-distribution critical value t(0.975, n_T - 2) rather than the
    normal-approximation z = 1.96, which understates the interval when only a
    few temperature points are available. For n_T == 2 (df == 0) no valid CI
    exists: both bounds are returned as NaN and df is returned as 0 so the
    caller can surface a warning (point estimate of Eₐ only).
    """
    df = n_T - 2
    if df < 1:
        return np.nan, np.nan, df
    t_crit = scipy_stats.t.ppf(0.975, df)
    Ea_ci = np.sqrt(cov[0, 0]) * R_GAS / 1000.0 * t_crit
    lnA_ci = np.sqrt(cov[1, 1]) * t_crit
    return Ea_ci, lnA_ci, df


def linear_intercept_check(x, y, expected, conf=0.95):
    """Test whether a linearisation passes through its theoretical intercept.

    Fits y = slope * x + intercept with scipy.stats.linregress and flags the
    intercept as "deviating" when |intercept - expected| exceeds the confidence
    half-width t(0.975, n-2) * intercept_stderr.  With fewer than three points
    there is no valid intercept uncertainty, so deviates is returned False.

    Returns {"intercept", "ci_half", "expected", "deviates", "r2"}; r2 is the
    R² of the linear fit, so callers can ignore an intercept mismatch on a line
    that does not fit the data anyway.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    if n < 3:
        return {
            "intercept": float("nan"),
            "ci_half": float("nan"),
            "expected": float(expected),
            "deviates": False,
            "r2": float("nan"),
        }
    reg = scipy_stats.linregress(x, y)
    df = n - 2
    t_crit = scipy_stats.t.ppf(1.0 - (1.0 - conf) / 2.0, df)
    ci_half = t_crit * reg.intercept_stderr
    if ci_half == 0 or not np.isfinite(ci_half):
        # Zero residual: the intercept is known exactly, so a machine-epsilon
        # offset must not count as a real deviation.
        deviates = not np.isclose(reg.intercept, expected, rtol=1e-12, atol=1e-15)
    else:
        deviates = abs(reg.intercept - expected) > ci_half
    return {
        "intercept": float(reg.intercept),
        "ci_half": float(ci_half),
        "expected": float(expected),
        "deviates": bool(deviates),
        "r2": float(reg.rvalue**2),
    }


def c_to_user(Ct_mol, unit, mw):
    """Convert a concentration from mol/L to the user's display unit."""
    if unit == "ppmS":
        return Ct_mol * MW_S * 1000  # mol/L → mg(S)/L = ppmS volumetric
    elif unit in ("ppm", "mg/L"):
        return Ct_mol * mw * 1000
    elif unit == "mmol/L":
        return Ct_mol * 1000
    elif unit == "g/L":
        return Ct_mol * mw
    else:
        return Ct_mol
