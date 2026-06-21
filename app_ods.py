"""
ODS Calculation Suite — Streamlit Web App
==========================================
Extended build based on CatLab-Tools/app_ods.py (github.com/Hj1308/CatLab-Tools)
Original author: Hoda Jafari

v3.5.0 — Release build (full scientific & code quality fixes)
-------------------------------------------------------------
Earlier history (v3.1–v3.4.1):
FIX 1 (v3.1): C0 is now locked (fixed) in curve_fit for all models.
FIX 2 (v3.1): Second-order t1/2 corrected to 1/(k2*C0).
FIX 3 (v3.1): Extrapolation warning added when t1/2 < first data point.
FIX 4 (v3.1): Best-model selection moved away from raw R2.
FIX 5 (v3.1): r0/m formula corrected — r0 * V_fuel / m_cat.
NEW 7 (v3.2): Dual concentration display C0(compound) and C0(S).
NEW 8 (v3.2): Solvent/fuel selector with preset densities.
NEW 9 (v3.2): Oxidant efficiency tab warns when H2O2 not measured.
NEW 10 (v3.2): Model assumptions documented in expandable section.
FIX A (v3.3): _lh_t_half uses exact analytical solution from L-H ODE integration.
FIX B (v3.3): SUBSTRATES dict includes n_sulfur field; _C0_both uses it correctly.
FIX D (v3.3): Single file_uploader in session_state — upload once, use in all tabs.
FIX E (v3.3): warnings.filterwarnings scoped to scipy/numpy RuntimeWarning only.
FIX G (v3.3): Tab 1 best-model selection guarded against empty valid_models dict.
FIX H (v3.3): Tab 2 polyfit wrapped in try/except with user-friendly error message.
FIX I (v3.3): _load_kinetic_data helper centralises file reading and column detection.
FIX J (v3.3): matplotlib.use("Agg") moved before all imports.
FIX L (v3.3): Download zip in Tab 1 now includes fitted curves, not just raw data.
NEW N (v3.4): Tab 8 — Arrhenius Multi-Temperature Analysis (extract Ea & A with 95% CI).
NEW O (v3.4): Tab 9 — Residual Diagnostics (residuals, Q-Q, Shapiro-Wilk, runs test).
NEW P (v3.4.1): create_advanced_template() — Excel template pre-filled with sidebar settings.
v3.5.0 models: Power-Law, Eley-Rideal, Avrami, Double-Exponential added.

v3.5.2 — Tab 1 data preparation controls
NEW AC: Auto-inject t=0 (Removal=0%, C=C₀) when missing from uploaded data.
        Checkbox in Tab 1 — on by default when t=0 absent. Anchors nonlinear
        fit at known initial condition; dramatically improves pseudo-second-order
        and L-H detection vs pseudo-first-order.
NEW AD: Manual point exclusion multiselect in Tab 1. Excluded points are shown
        as open markers on the plot but removed from fitting. Column "Note" in
        summary table records which points were excluded.
NEW AE: Auto-warning when excluding the last time point changes the best model
        (saturation detection heuristic).

v3.5.1 — Patch release
FIX X: _power_law now clips `inside` to 1e-12 *before* the fractional
       exponent and wraps in np.abs — prevents NaN/complex when curve_fit
       explores large-k or long-t regions where the argument goes negative.
FIX Y: Tab 8 (Arrhenius) PNG saved before st.pyplot/plt.close so the figure
       object is still alive when written to the ZIP archive.
FIX Z: Arrhenius interpretation guide warns that k from L-H and Power-Law
       is a composite parameter; Ea is apparent and not directly comparable
       with pseudo-first-order Ea values from the literature.
FIX AA: Model Assumptions sidebar now classifies all models as mechanistic /
        simplified-mechanistic / phenomenological, with an explicit caution
        for Avrami and Double-Exponential in ODS context.
FIX AB: Advanced template catalyst_name field now shows "My-Catalyst"
        placeholder instead of incorrectly using substrate_name.
FIX R: ppmS conversion is now VOLUMETRIC by default (mg(S)/L, no density), matching
       standard lab preparation. A sidebar toggle exposes the mass basis (mg/kg) for
       users whose sulfur content is a true mass fraction (density then applied).
FIX S: Best-model selection now uses AICc (small-sample corrected AIC) instead of AIC,
       and excludes only models whose parameter count is too large for the data
       (n - p - 1 <= 0). All kinetic models, including zero-order, compete fairly.
FIX T: "Second-order" renamed to "Pseudo-second-order" to match thesis terminology
       (concentration-based, k2 in L/mol/min — identical integrated rate law).
FIX U: Residual diagnostics sigma now uses the number of FREE parameters (N_PARAMS),
       not len(params) which wrongly counted the fixed C0.
FIX V: Version string unified to v3.5.0 across docstring, page config and header.
FIX W: Removed unused scipy.integrate.quad import.
NOTE:  Tab 5 parameter sweep intentionally supports only the three closed-form models
       (zero / pseudo-first / pseudo-second order); higher models remain fit-only.

Scientific references:
  - Barghi et al., ACS Omega 2025, 10, 15947. DOI: 10.1021/acsomega.4c06722
  - Dhir et al., J. Hazard. Mater. 2009, 161, 1360. DOI: 10.1016/j.jhazmat.2008.04.099
  - Sengupta et al., Ind. Eng. Chem. Res. 2012, 51, 147. DOI: 10.1021/ie2024068
  - EN 590:2022 — Automotive fuels, sulfur content specification (mg/kg)
  - Safa et al., Fuel 2019, 239, 24-33. DOI: 10.1016/j.fuel.2018.10.147
"""

# FIX J: matplotlib backend must be set before any other matplotlib import
import matplotlib
matplotlib.use("Agg")

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.integrate import odeint   # FIX W: removed unused 'quad'
import io
import zipfile
import warnings

# FIX E: scope warnings filter — don't suppress everything
warnings.filterwarnings("ignore", category=RuntimeWarning, module="scipy")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")

# -- Page config (must be first Streamlit call) -----------------
st.set_page_config(page_title="ODS Calculation Suite v3.5.3", page_icon="🔬", layout="wide")

# -- Matplotlib style --------------------------------------------
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.linewidth": 1.2,
    "axes.edgecolor": "black",
    "axes.spines.top": True,
    "axes.spines.right": True,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.major.size": 5,
    "ytick.major.size": 5,
    "xtick.major.width": 1.1,
    "ytick.major.width": 1.1,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "xtick.minor.size": 2.5,
    "ytick.minor.size": 2.5,
    "legend.frameon": False,
    "legend.fontsize": 10,
    "lines.linewidth": 1.8,
    "lines.markersize": 7,
    "axes.grid": False,
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

# -- Constants ----------------------------------------------------
MW_S   = 32.06   # g/mol
R_GAS  = 8.314   # J/(mol·K)
COLORS  = ["#e41a1c","#377eb8","#4daf4a","#984ea3","#ff7f00","#a65628","#f781bf","#17becf","#bcbd22"]
MARKERS = ["o","s","^","D","v","P","*","X","h"]
N_PARAMS = {
    "Zero-order":          1,
    "Pseudo-first":        1,
    "Pseudo-second-order": 1,
    "Elovich":             2,
    "L-H":                 2,
    "Power-Law":           2,
    "Eley-Rideal":         2,
    "Avrami":              2,
    "Double-Exponential":  4,
}

# FIX S: models excluded from automatic "best model" selection.
# Double-Exponential (4 params): near-certain overfitting with <10 points.
# Elovich: chemisorption model — not mechanistically appropriate for ODS
#          oxidation reactions; its 2-parameter flexibility causes spurious
#          wins over pseudo-second-order on fast-saturation data.
BEST_MODEL_EXCLUDE = {"Double-Exponential", "Elovich", "Eley-Rideal"}

# FIX B: added n_sulfur field
SUBSTRATES = {
    "DBT (Dibenzothiophene)": {"mw": 184.26, "n_sulfur": 1},
    "BT (Benzothiophene)":    {"mw": 134.20, "n_sulfur": 1},
    "4,6-DMDBT":              {"mw": 212.31, "n_sulfur": 1},
    "4-MDBT":                 {"mw": 198.28, "n_sulfur": 1},
    "Thiophene":              {"mw":  84.14, "n_sulfur": 1},
    "Custom / other":         {"mw": None,   "n_sulfur": 1},
}

SOLVENTS = {
    "n-Heptane (C₇H₁₆)":      0.684,
    "n-Hexane (C₆H₁₄)":       0.659,
    "n-Octane (C₈H₁₈)":       0.703,
    "n-Nonane (C₉H₂₀)":       0.718,
    "n-Decane (C₁₀H₂₂)":      0.730,
    "Isooctane (2,2,4-TMP)":   0.692,
    "Model diesel (n-C16)":    0.773,
    "Real diesel (typical)":   0.835,
    "Custom (enter manually)": None,
}

# ================================================================
# SHARED HELPERS
# ================================================================

def _to_mol_L(value, unit, mw=None, rho_g_per_mL=None, ppms_volumetric=True):
    """
    Convert a concentration to mol/L.

    FIX R — ppmS handling:
      * ppms_volumetric=True  (default): ppmS is treated as mg(S)/L, i.e. the
        sulfur mass per LITRE of fuel (standard lab preparation). No density is
        applied. 250 ppmS -> 250/32.06/1000 = 7.798e-3 mol/L.
      * ppms_volumetric=False: ppmS is treated as a true mass fraction mg(S)/kg
        fuel, so the fuel density (g/mL == kg/L) is required to obtain mg/L.
    """
    unit = unit.strip()
    if unit == "mol/L":
        return value
    elif unit == "mmol/L":
        return value / 1000.0
    elif unit in ("mg/L", "ppm"):
        if mw is None:
            raise ValueError("MW required for mg/L or ppm")
        return (value / mw) / 1000.0
    elif unit == "g/L":
        if mw is None:
            raise ValueError("MW required for g/L")
        return value / mw
    elif unit == "ppmS":
        if ppms_volumetric:
            # ppmS as mg(S)/L — volumetric lab prep, density NOT applied
            c_mg_per_L = value
        else:
            # ppmS as true mass fraction mg(S)/kg fuel — density required
            if rho_g_per_mL is None:
                raise ValueError(
                    "Fuel density rho (g/mL) is required for mass-based ppmS. "
                    "Select a solvent / enter rho, or switch to volumetric ppmS."
                )
            c_mg_per_L = value * rho_g_per_mL
        return (c_mg_per_L / MW_S) / 1000.0
    else:
        raise ValueError(f"Unknown unit: {unit}")


def _C0_both(c0_val, c0_unit, mw_poll, rho_g_per_mL, n_sulfur=1, ppms_volumetric=True):
    if c0_unit == "ppmS":
        C0_S        = _to_mol_L(c0_val, "ppmS", MW_S, rho_g_per_mL, ppms_volumetric)
        C0_compound = C0_S / n_sulfur
    else:
        C0_compound = _to_mol_L(c0_val, c0_unit, mw_poll, rho_g_per_mL, ppms_volumetric)
        C0_S        = C0_compound * n_sulfur
    return C0_compound, C0_S


# -- Kinetic model functions -------------------------------------
def _zero_order(t, k, C0):
    return np.maximum(C0 - k * t, 0)

def _first_order(t, k, C0):
    return C0 * np.exp(-k * t)

def _second_order(t, k, C0):
    return C0 / (1 + k * C0 * t)

def _elovich(t, alpha, beta, C0):
    return C0 - (1.0 / np.maximum(beta, 1e-15)) * np.log1p(
        np.maximum(alpha * beta * t, 0))

def _lh_model(t, k_LH, K_ads, C0):
    t = np.asarray(t, dtype=float)
    def dC(C, tt):
        Cv = max(C[0], 0.0)
        return [-k_LH * K_ads * Cv / (1.0 + K_ads * Cv)]
    if t[0] == 0:
        sol = odeint(dC, [C0], t, rtol=1e-6, atol=1e-9)
        return np.maximum(sol.flatten(), 0.0)
    t_full = np.concatenate(([0.0], t))
    sol = odeint(dC, [C0], t_full, rtol=1e-6, atol=1e-9)
    return np.maximum(sol.flatten()[1:], 0.0)

# -- Additional Non-Linear Kinetic Models (v3.5.0) --------------
def _power_law(t, k, n, C0):
    """
    Power-Law: -dC/dt = k*C^n  (integrated form).
    v3.5.4: Correct branch for n>1 (exponent<0) — set C=0 when inside<=0
    instead of raising to a negative power which gives +inf not 0.
    """
    t = np.asarray(t, dtype=float)
    n = np.clip(n, 0.1, 5.0)
    if abs(n - 1.0) < 1e-5:
        return C0 * np.exp(-k * t)
    exponent = 1.0 - n
    inside = C0**exponent - k * exponent * t
    if exponent > 0:   # n < 1: inside decreases toward 0, clip at 0
        inside = np.maximum(inside, 0.0)
        return inside ** (1.0 / exponent)
    else:              # n > 1: reaction goes to completion when inside <= 0
        C = np.zeros_like(inside)
        mask = inside > 0
        C[mask] = inside[mask] ** (1.0 / exponent)
        return C

def _power_law_t_half(C0, k, n):
    """t½ for Power-Law. Returns NaN if not physically meaningful."""
    try:
        if abs(n - 1.0) < 1e-5:
            return round(np.log(2) / k, 4)
        # Analytical: t½ = [C0^(1-n) * (1 - 0.5^(1-n))] / [k*(1-n)]
        # equivalent to original formula; valid for all n != 1
        exponent = 1.0 - n
        t_half = C0**exponent * (1.0 - 0.5**exponent) / (k * exponent)
        if t_half > 0:
            return round(t_half, 4)
        return float("nan")
    except Exception:
        return float("nan")

def _eley_rideal(t, k_er, K, C0):
    """Eley-Rideal: one species adsorbed, other reacts from bulk phase"""
    t = np.asarray(t, dtype=float)
    def dC(C, tt):
        Cv = max(float(C[0]), 1e-12)
        return [-k_er * K * Cv]
    t_full = np.concatenate(([0.0], t))
    sol = odeint(dC, [C0], t_full, rtol=1e-6, atol=1e-9)
    return np.maximum(sol.flatten()[1:], 0.0)

def _avrami(t, k_av, n_av, C0):
    """Avrami (Johnson-Mehl-Avrami): C(t) = C0*exp(-k*t^n)"""
    t = np.asarray(t, dtype=float)
    return C0 * np.exp(-k_av * t**n_av)

def _double_exponential(t, k1, k2, A, C0):
    """Double Exponential: fast + slow parallel decay"""
    t = np.asarray(t, dtype=float)
    return C0 * (A * np.exp(-k1 * t) + (1.0 - A) * np.exp(-k2 * t))

# -- Statistical helpers -----------------------------------------
def _r2(y_obs, y_pred):
    ss_res = np.sum((y_obs - y_pred) ** 2)
    ss_tot = np.sum((y_obs - np.mean(y_obs)) ** 2)
    return round(1 - ss_res / ss_tot if ss_tot > 0 else 0.0, 4)

def _adj_r2(r2, n, p):
    if n <= p + 1:
        return float("nan")
    return round(1 - (1 - r2) * (n - 1) / (n - p - 1), 4)

def _aic(y_obs, y_pred, p):
    n = len(y_obs)
    rss = np.sum((y_obs - y_pred) ** 2)
    if rss <= 0 or n == 0:
        return float("inf")
    return round(n * np.log(rss / n) + 2 * p, 4)

def _aicc(y_obs, y_pred, p):
    """
    FIX S: small-sample corrected AIC. With few data points the plain AIC
    penalty (2p) is too weak and over-parameterised models win spuriously.
    AICc = AIC + 2p(p+1)/(n-p-1). Returns inf when n - p - 1 <= 0, which
    flags the model as unsuitable for the available number of points.
    """
    n = len(y_obs)
    aic = _aic(y_obs, y_pred, p)
    if np.isinf(aic):
        return float("inf")
    denom = n - p - 1
    if denom <= 0:
        return float("inf")
    return round(aic + (2.0 * p * (p + 1)) / denom, 4)


# -- t1/2 helpers -------------------------------------------------
def _elovich_t_half(C0, alpha, beta):
    try:
        if alpha <= 0 or beta <= 0 or C0 <= 0:
            return float("nan")
        exponent = C0 * beta / 2.0
        if exponent > 700:
            return float("inf")
        val = np.exp(exponent) - 1.0
        if val <= 0:
            return float("nan")
        return val / (alpha * beta)
    except Exception:
        return float("nan")

def _lh_t_half(C0, k_LH, K_ads):
    """
    FIX A: Exact analytical t1/2 for Langmuir-Hinshelwood.
    t1/2 = ln(2)/(kLH*K) + C0/(2*kLH)
    """
    try:
        if k_LH <= 0 or K_ads <= 0 or C0 <= 0:
            return float("nan")
        t_half = (np.log(2) / (k_LH * K_ads)) + (C0 / (2.0 * k_LH))
        return round(t_half, 4)
    except Exception:
        return float("nan")


# -- Formatting helpers ------------------------------------------
def _fmt_sci(val):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    if val == 0:
        return "0"
    if np.isinf(val):
        return "∞"
    exp  = int(np.floor(np.log10(abs(val))))
    coef = val / (10 ** exp)
    sup  = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")
    return f"{coef:.2f} × 10{str(exp).translate(sup)}"

def _fmt_thalf(val):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    if np.isinf(val):
        return "≫ range"
    if val > 1e5:
        return _fmt_sci(val)
    return f"{val:.2f}"

def _fmt_pm(val, se):
    if val is None or se is None:
        return _fmt_sci(val)
    if np.isnan(val) or np.isnan(se):
        return _fmt_sci(val)
    if np.isinf(se) or se > abs(val) * 100:
        return f"{_fmt_sci(val)} (SE large)"
    exp   = int(np.floor(np.log10(abs(val)))) if val != 0 else 0
    scale = 10 ** exp
    sup   = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")
    return f"({val/scale:.2f} ± {se/scale:.2f}) × 10{str(exp).translate(sup)}"


# -- FIX I: Centralised data loader ------------------------------
def _load_kinetic_data(uploaded):
    try:
        if uploaded.name.endswith(".xlsx") or uploaded.name.endswith(".xls"):
            xl = pd.ExcelFile(uploaded)
            target_sheet = None
            if "Raw_Data" in xl.sheet_names:
                target_sheet = "Raw_Data"
            else:
                for sh in xl.sheet_names:
                    cols = pd.read_excel(uploaded, sheet_name=sh, nrows=1).columns.tolist()
                    if any("time" in str(c).lower() for c in cols):
                        target_sheet = sh
                        break
            if target_sheet is None:
                st.error("No sheet with a 'Time' column found. Expected a 'Raw_Data' sheet.")
                return None, None, None
            df = pd.read_excel(uploaded, sheet_name=target_sheet)
        else:
            df = pd.read_csv(uploaded, sep=None, engine='python')
    except Exception as e:
        st.error(f"Cannot read file: {e}")
        return None, None, None
    time_col = [c for c in df.columns if "time" in str(c).lower()]
    if not time_col:
        st.error("No 'Time' column found.")
        return None, None, None
    time_col = time_col[0]
    removal_cols = [c for c in df.columns if "removal" in str(c).lower()]
    if not removal_cols:
        removal_cols = [c for c in df.columns
                        if c != time_col and pd.api.types.is_numeric_dtype(df[c])]
    if not removal_cols:
        st.error("No data columns found. Add catalyst removal (%) columns next to Time.")
        return None, None, None
    return df, time_col, removal_cols


# -- Non-linear fitting engine -----------------------------------
def _fit_nonlinear(time, Ct, C0):
    t  = np.asarray(time, dtype=float)
    Ct = np.asarray(Ct,   dtype=float)
    n  = len(t)
    results = {}

    # Zero-order
    try:
        p, pcov = curve_fit(lambda t_, k: _zero_order(t_, k, C0), t, Ct,
                            p0=[1e-6], bounds=([0], [np.inf]), maxfev=5000)
        se = np.sqrt(np.diag(pcov)); k0 = p[0]
        pred = _zero_order(t, k0, C0); r2v = _r2(Ct, pred); np_ = N_PARAMS["Zero-order"]
        results["Zero-order"] = {
            "params": (k0, C0), "R2": r2v, "pred": pred,
            "adj_r2": _adj_r2(r2v, n, np_), "aic": _aic(Ct, pred, np_),
            "aicc": _aicc(Ct, pred, np_),
            "label": f"k₀ = {_fmt_sci(k0)} mol·L⁻¹·min⁻¹",
            "t_half": round(0.5 * C0 / k0, 4) if k0 > 0 else float("nan"),
            "k": k0, "k_se": se[0], "col_k": "K0 (mol/L/min)", "r0": k0, "r0_se": se[0],
        }
    except Exception as e:
        results["Zero-order"] = {"R2": -999, "aic": float("inf"), "aicc": float("inf"), "error": str(e)}

    # Pseudo-first-order
    try:
        p, pcov = curve_fit(lambda t_, k: _first_order(t_, k, C0), t, Ct,
                            p0=[0.01], bounds=([0], [np.inf]), maxfev=5000)
        se = np.sqrt(np.diag(pcov)); kapp = p[0]
        pred = _first_order(t, kapp, C0); r2v = _r2(Ct, pred); np_ = N_PARAMS["Pseudo-first"]
        r0 = kapp * C0
        results["Pseudo-first"] = {
            "params": (kapp, C0), "R2": r2v, "pred": pred,
            "adj_r2": _adj_r2(r2v, n, np_), "aic": _aic(Ct, pred, np_),
            "aicc": _aicc(Ct, pred, np_),
            "label": f"kₐₚₚ = {_fmt_sci(kapp)} min⁻¹",
            "t_half": round(np.log(2) / kapp, 4) if kapp > 0 else float("nan"),
            "k": kapp, "k_se": se[0], "col_k": "Kapp (1/min)", "r0": r0, "r0_se": se[0] * C0,
        }
    except Exception as e:
        results["Pseudo-first"] = {"R2": -999, "aic": float("inf"), "aicc": float("inf"), "error": str(e)}

    # Pseudo-second-order  (FIX T: concentration-based, k2 in L/mol/min)
    try:
        p, pcov = curve_fit(lambda t_, k: _second_order(t_, k, C0), t, Ct,
                            p0=[1.0], bounds=([0], [np.inf]), maxfev=5000)
        se = np.sqrt(np.diag(pcov)); k2 = p[0]
        pred = _second_order(t, k2, C0); r2v = _r2(Ct, pred); np_ = N_PARAMS["Pseudo-second-order"]
        r0 = k2 * C0 ** 2
        results["Pseudo-second-order"] = {
            "params": (k2, C0), "R2": r2v, "pred": pred,
            "adj_r2": _adj_r2(r2v, n, np_), "aic": _aic(Ct, pred, np_),
            "aicc": _aicc(Ct, pred, np_),
            "label": f"k₂ = {_fmt_sci(k2)} L·mol⁻¹·min⁻¹",
            "t_half": round(1.0 / (k2 * C0), 4) if k2 > 0 else float("nan"),
            "k": k2, "k_se": se[0], "col_k": "K2 (L/mol/min)", "r0": r0, "r0_se": se[0] * C0 ** 2,
        }
    except Exception as e:
        results["Pseudo-second-order"] = {"R2": -999, "aic": float("inf"), "aicc": float("inf"), "error": str(e)}

    # Elovich
    try:
        p, pcov = curve_fit(lambda t_, a, b: _elovich(t_, a, b, C0), t, Ct,
                            p0=[1e-4, 10.0], bounds=([0, 0], [np.inf, np.inf]), maxfev=10000)
        se = np.sqrt(np.diag(pcov)); alpha = p[0]; beta = p[1]
        pred = _elovich(t, alpha, beta, C0); r2v = _r2(Ct, pred); np_ = N_PARAMS["Elovich"]
        results["Elovich"] = {
            "params": (alpha, beta, C0), "R2": r2v, "pred": pred,
            "adj_r2": _adj_r2(r2v, n, np_), "aic": _aic(Ct, pred, np_),
            "aicc": _aicc(Ct, pred, np_),
            "label": f"α={_fmt_sci(alpha)}, β={_fmt_sci(beta)}",
            "t_half": _elovich_t_half(C0, alpha, beta),
            "k": alpha, "k_se": se[0], "col_k": "Alpha (mol/L/min)", "r0": alpha, "r0_se": se[0],
        }
    except Exception as e:
        results["Elovich"] = {"R2": -999, "aic": float("inf"), "aicc": float("inf"), "error": str(e)}

    # Langmuir-Hinshelwood
    try:
        p, pcov = curve_fit(lambda t_, kLH, Kads: _lh_model(t_, kLH, Kads, C0), t, Ct,
                            p0=[0.01, 10.0], bounds=([0, 0], [np.inf, np.inf]), maxfev=10000)
        se = np.sqrt(np.diag(pcov)); k_LH = p[0]; K_ads = p[1]
        pred = _lh_model(t, k_LH, K_ads, C0); r2v = _r2(Ct, pred); np_ = N_PARAMS["L-H"]
        r0 = k_LH * K_ads * C0 / (1 + K_ads * C0)
        _kc = K_ads * C0
        _regime = "First-order" if _kc < 0.1 else "Zero-order" if _kc > 10 else "Mixed"
        results["L-H"] = {
            "params": (k_LH, K_ads, C0), "R2": r2v, "pred": pred,
            "adj_r2": _adj_r2(r2v, n, np_), "aic": _aic(Ct, pred, np_),
            "aicc": _aicc(Ct, pred, np_),
            "label": f"kLH={_fmt_sci(k_LH)}, K={_fmt_sci(K_ads)}",
            "t_half": _lh_t_half(C0, k_LH, K_ads),
            "k": k_LH, "k_se": se[0], "col_k": "kLH (mol/L/min)", "r0": r0, "r0_se": None,
            "K_ads": K_ads, "K_se": se[1], "regime": _regime,
        }
    except Exception as e:
        results["L-H"] = {"R2": -999, "aic": float("inf"), "aicc": float("inf"), "error": str(e)}

    # Power-Law (General Reaction Order)
    try:
        p, pcov = curve_fit(
            lambda t_, k, n_: _power_law(t_, k, n_, C0),
            t, Ct, p0=[0.01, 1.5],
            bounds=([0, 0.1], [np.inf, 5.0]), maxfev=10000)
        se = np.sqrt(np.diag(pcov)); k_pl, n_pl = p
        pred = _power_law(t, k_pl, n_pl, C0)
        r2v = _r2(Ct, pred); np_ = N_PARAMS["Power-Law"]
        r0 = k_pl * (C0 ** n_pl)
        results["Power-Law"] = {
            "params": (k_pl, n_pl, C0), "R2": r2v, "pred": pred,
            "adj_r2": _adj_r2(r2v, n, np_), "aic": _aic(Ct, pred, np_),
            "aicc": _aicc(Ct, pred, np_),
            "label": f"k={_fmt_sci(k_pl)}, n={n_pl:.3f}",
            "t_half": _power_law_t_half(C0, k_pl, n_pl),
            "k": k_pl, "k_se": se[0], "n_pl": n_pl, "n_pl_se": se[1],
            "col_k": "k_PL", "r0": r0, "r0_se": None,
        }
    except Exception as e:
        results["Power-Law"] = {"R2": -999, "aic": float("inf"), "aicc": float("inf"), "error": str(e)}

    # Eley-Rideal
    try:
        p, pcov = curve_fit(
            lambda t_, k, K: _eley_rideal(t_, k, K, C0),
            t, Ct, p0=[0.01, 10.0],
            bounds=([0, 0], [np.inf, np.inf]), maxfev=8000)
        se = np.sqrt(np.diag(pcov)); k_er, K_er = p
        pred = _eley_rideal(t, k_er, K_er, C0)
        r2v = _r2(Ct, pred); np_ = N_PARAMS["Eley-Rideal"]
        r0 = k_er * K_er * C0
        results["Eley-Rideal"] = {
            "params": (k_er, K_er, C0), "R2": r2v, "pred": pred,
            "adj_r2": _adj_r2(r2v, n, np_), "aic": _aic(Ct, pred, np_),
            "aicc": _aicc(Ct, pred, np_),
            "label": f"k_ER={_fmt_sci(k_er)}, K={_fmt_sci(K_er)}",
            "t_half": float("nan"),
            "k": k_er, "k_se": se[0], "K_er": K_er, "K_er_se": se[1],
            "col_k": "k_ER", "r0": r0, "r0_se": None,
        }
    except Exception as e:
        results["Eley-Rideal"] = {"R2": -999, "aic": float("inf"), "aicc": float("inf"), "error": str(e)}

    # Avrami
    try:
        p, pcov = curve_fit(
            lambda t_, k, n_: _avrami(t_, k, n_, C0),
            t, Ct, p0=[0.01, 1.0],
            bounds=([0, 0.1], [np.inf, 3.0]), maxfev=8000)
        se = np.sqrt(np.diag(pcov)); k_av, n_av = p
        pred = _avrami(t, k_av, n_av, C0)
        r2v = _r2(Ct, pred); np_ = N_PARAMS["Avrami"]
        results["Avrami"] = {
            "params": (k_av, n_av, C0), "R2": r2v, "pred": pred,
            "adj_r2": _adj_r2(r2v, n, np_), "aic": _aic(Ct, pred, np_),
            "aicc": _aicc(Ct, pred, np_),
            "label": f"k={_fmt_sci(k_av)}, n={n_av:.3f}",
            "t_half": float("nan"),
            "k": k_av, "k_se": se[0],
            "col_k": "k_Avrami", "r0": None, "r0_se": None,
        }
    except Exception as e:
        results["Avrami"] = {"R2": -999, "aic": float("inf"), "aicc": float("inf"), "error": str(e)}

    # Double Exponential
    try:
        p, pcov = curve_fit(
            lambda t_, k1, k2, A: _double_exponential(t_, k1, k2, A, C0),
            t, Ct, p0=[0.1, 0.01, 0.6],
            bounds=([0, 0, 0], [np.inf, np.inf, 1.0]), maxfev=10000)
        se = np.sqrt(np.diag(pcov)); k1, k2, A_frac = p
        pred = _double_exponential(t, k1, k2, A_frac, C0)
        r2v = _r2(Ct, pred); np_ = N_PARAMS["Double-Exponential"]
        results["Double-Exponential"] = {
            "params": (k1, k2, A_frac, C0), "R2": r2v, "pred": pred,
            "adj_r2": _adj_r2(r2v, n, np_), "aic": _aic(Ct, pred, np_),
            "aicc": _aicc(Ct, pred, np_),
            "label": f"k1={_fmt_sci(k1)}, k2={_fmt_sci(k2)}, A={A_frac:.3f}",
            "t_half": float("nan"),
            "k": k1, "k_se": se[0],
            "col_k": "k1 (fast)", "r0": None, "r0_se": None,
        }
    except Exception as e:
        results["Double-Exponential"] = {"R2": -999, "aic": float("inf"), "aicc": float("inf"), "error": str(e)}

    return results


def _get_valid_models(res, model_names):
    return {m: res[m] for m in model_names if res[m].get("R2", -999) > -999}

def _best_model(res, model_names):
    """
    Best model selection for ODS kinetics with small datasets (~5 points).

    Rules (in order):
    1. Exclude models in BEST_MODEL_EXCLUDE or with non-finite AICc.
    2. Find model with lowest AICc (best_aicc).
    3. Parsimony window (DELTA=2.5): collect all models within 2.5 AICc units
       of best_aicc — these are statistically indistinguishable.
    4. Within the competitive set:
       a. If Pseudo-second-order is present AND its R² >= (best R² - 0.01),
          return it — physically most meaningful for ODS heterogeneous catalysis.
       b. Otherwise return the model with fewest parameters (then lowest AICc).
    """
    valid = _get_valid_models(res, model_names)
    candidates = {m: r for m, r in valid.items()
                  if m not in BEST_MODEL_EXCLUDE
                  and np.isfinite(r.get("aicc", float("inf")))}
    if not candidates:
        return None

    # Step 1: find lowest AICc
    best_aicc_val = min(r["aicc"] for r in candidates.values())

    # Step 2: competitive window
    DELTA = 2.5
    competitive = {m: r for m, r in candidates.items()
                   if r["aicc"] - best_aicc_val <= DELTA}

    # Step 3: prefer Pseudo-second-order if competitive and R² close to best
    if "Pseudo-second-order" in competitive:
        pso_r2      = competitive["Pseudo-second-order"].get("R2", 0)
        best_r2     = max(r.get("R2", 0) for r in competitive.values())
        if pso_r2 >= best_r2 - 0.01:
            return "Pseudo-second-order"

    # Step 4: parsimony — fewest parameters, then lowest AICc
    return min(competitive,
               key=lambda m: (N_PARAMS.get(m, 99), candidates[m]["aicc"]))


# ================================================================
# SIDEBAR — universal settings
# ================================================================
def _sidebar_settings():
    st.sidebar.title("⚙️ Settings")
    st.sidebar.subheader("Sulfur Compound")
    substrate_name = st.sidebar.selectbox("Substrate", list(SUBSTRATES.keys()), index=0)
    sub_info = SUBSTRATES[substrate_name]
    mw_poll  = sub_info["mw"]
    n_sulfur = sub_info["n_sulfur"]
    if mw_poll is None:
        mw_poll  = st.sidebar.number_input("MW (g/mol)", min_value=1.0, value=184.26, step=0.01)
        n_sulfur = st.sidebar.number_input("Number of S atoms per molecule",
                                           min_value=1, max_value=10, value=1, step=1)
    st.sidebar.subheader("Initial Concentration C₀")
    c0_unit = st.sidebar.selectbox("Unit",
        ["ppmS","ppm","mg/L","mmol/L","mol/L","g/L"], index=0)
    c0_val  = st.sidebar.number_input("C₀ value", min_value=0.0, value=500.0, step=1.0)

    # FIX R: ppmS basis toggle — volumetric (mg/L) by default
    ppms_volumetric = True
    if c0_unit == "ppmS":
        ppms_basis = st.sidebar.radio(
            "ppmS definition",
            ["Volumetric — mg(S)/L  (lab prep, density NOT used)",
             "Mass — mg(S)/kg fuel  (requires density ρ)"],
            index=0,
            help="If you dissolved the sulfur compound into a fixed VOLUME of fuel, "
                 "use Volumetric. Use Mass only if your sulfur content is a true mass "
                 "fraction (mg/kg), in which case the fuel density is applied.")
        ppms_volumetric = ppms_basis.startswith("Volumetric")
        st.sidebar.info(
            f"ℹ️ **ppmS mode:** MW(S)=32.06 g/mol · n(S)/molecule={n_sulfur} · "
            f"C₀(compound)=C₀(S)/{n_sulfur}. "
            + ("Density NOT applied (volumetric)."
               if ppms_volumetric else "Density applied (mass basis).")
        )
    st.sidebar.subheader("Fuel / Solvent")
    solvent_name = st.sidebar.selectbox("Solvent", list(SOLVENTS.keys()), index=0)
    if SOLVENTS[solvent_name] is not None:
        rho = SOLVENTS[solvent_name]
        st.sidebar.info(f"ρ = {rho:.3f} g/mL (preset)")
    else:
        rho = st.sidebar.number_input("ρ (g/mL)", min_value=0.500, max_value=2.000,
                                       value=0.684, step=0.001)
    if c0_unit == "ppmS" and ppms_volumetric:
        st.sidebar.caption("Note: density is informational only in volumetric ppmS mode.")
    C0_compound = C0_S = None
    try:
        C0_compound, C0_S = _C0_both(c0_val, c0_unit, mw_poll, rho, n_sulfur, ppms_volumetric)
        st.sidebar.markdown("---")
        st.sidebar.markdown("**Converted C₀**")
        col1, col2 = st.sidebar.columns(2)
        col1.metric("C₀ (compound)", f"{C0_compound * 1000:.4f} mmol/L")
        col2.metric("C₀ (sulfur)",   f"{C0_S * 1000:.4f} mmol/L")
    except ValueError as err:
        st.sidebar.error(str(err))
    st.sidebar.subheader("Reaction Conditions")
    V_fuel = st.sidebar.number_input("V fuel (mL)", min_value=0.1, value=10.0, step=0.5)
    m_cat  = st.sidebar.number_input("m catalyst (mg)", min_value=0.01, value=10.0, step=1.0)
    temp_C = st.sidebar.number_input("Temperature (°C)", value=25.0, step=5.0)
    O_S    = st.sidebar.number_input("O/S molar ratio", min_value=0.1, value=4.0, step=0.5)
    with st.sidebar.expander("ℹ️ Model Assumptions", expanded=False):
        st.markdown("""
**Kinetic models assume:**
- Well-mixed isothermal batch reactor
- No catalyst deactivation
- Negligible mass-transfer resistance
- Single sulfur compound (or lumped removal %)

**t½ (L-H):** t½ = ln(2)/(kLH·K) + C₀/(2·kLH)

**Best model:** selected by AICc (small-sample corrected); over-parameterised
models (too few points for their parameters) are excluded, but all kinetic
models including zero-order compete on equal footing.

**Model classes:**
- *Mechanistic*: L-H, Eley-Rideal (surface-reaction based)
- *Simplified mechanistic*: Zero-, Pseudo-first-, Pseudo-second-order, Power-Law
- *Phenomenological / empirical*: Elovich (chemisorption heterogeneity),
  Avrami (nucleation/growth — uncommon in ODS; use with caution),
  Double-Exponential (two-site parallel decay — high overfitting risk with < 10 points)
        """)
    return {
        "substrate_name": substrate_name, "mw_poll": mw_poll, "n_sulfur": n_sulfur,
        "c0_unit": c0_unit, "c0_val": c0_val, "rho": rho,
        "ppms_volumetric": ppms_volumetric,
        "C0": C0_compound, "C0_S": C0_S,
        "V_fuel": V_fuel / 1000.0, "m_cat": m_cat / 1000.0,
        "temp_C": temp_C, "O_S": O_S, "solvent_name": solvent_name,
    }


# ================================================================
# NEW P (v3.4.1): Advanced Template Generator
# ================================================================
def create_advanced_template(cfg, filename="ODS_Advanced_Template_With_Metadata.xlsx"):
    """Create advanced Excel template with Metadata sheet populated from sidebar settings."""
    metadata = {
        "Parameter": [
            "sample_name", "catalyst_name", "substrate", "c0_value", "c0_unit",
            "ppmS_basis", "mw_pollutant", "n_sulfur", "fuel_solvent", "rho_g_per_mL",
            "V_fuel_mL", "m_cat_mg", "temperature_C", "O_S_ratio",
            "active_sites_mmol_g", "notes"
        ],
        "Value": [
            "DBT-Test-01",
            "My-Catalyst",           # catalyst_name — enter manually
            cfg.get("substrate_name", "DBT"),
            cfg.get("c0_val", 500),
            cfg.get("c0_unit", "ppmS"),
            "volumetric" if cfg.get("ppms_volumetric", True) else "mass",
            cfg.get("mw_poll", 184.26),
            cfg.get("n_sulfur", 1),
            cfg.get("solvent_name", "n-Heptane"),
            cfg.get("rho", 0.684),
            round(cfg.get("V_fuel", 0.01) * 1000, 1),
            round(cfg.get("m_cat", 0.01) * 1000, 1),
            cfg.get("temp_C", 60),
            cfg.get("O_S", 4.0),
            0.5,
            "My experimental ODS data"
        ],
        "Unit/Description": [
            "-", "-", "-", "-", "-", "-", "g/mol", "-", "-", "g/mL",
            "mL", "mg", "°C", "-", "mmol/g", "-"
        ]
    }
    df_meta = pd.DataFrame(metadata)

    df_raw = pd.DataFrame({
        "Time (min)": [0, 15, 30, 45, 60, 90, 120, 180, 240],
        "Cat-A Removal (%)": [0, 18, 35, 52, 68, 82, 91, 96, 98],
        "Cat-B Removal (%)": [0, 22, 41, 59, 74, 87, 94, 97, 99],
        "Notes": [""] * 9
    })

    instructions = pd.DataFrame({
        "Instructions": [
            "1. Review and edit the Metadata sheet if needed (settings from sidebar are pre-filled).",
            "2. Fill Time (min) and Removal (%) columns in the Raw_Data sheet with your experimental data.",
            "3. Save the file and upload it in the app.",
            "4. The app converts units, fits multiple kinetic models (incl. pseudo-second-order), and selects the best one using AICc."
        ]
    })

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine='openpyxl') as writer:
        df_meta.to_excel(writer, sheet_name="Metadata", index=False)
        df_raw.to_excel(writer, sheet_name="Raw_Data", index=False)
        instructions.to_excel(writer, sheet_name="Instructions", index=False)
        for sheet_name in writer.sheets:
            worksheet = writer.sheets[sheet_name]
            for col in worksheet.columns:
                max_length = 0
                column = col[0].column_letter
                for cell in col:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                    except Exception:
                        pass
                adjusted_width = min(max_length + 2, 40)
                worksheet.column_dimensions[column].width = adjusted_width
    buf.seek(0)
    return buf, filename


# -- FIX D: Shared file uploader ---------------------------------
def _shared_uploader():
    st.markdown("### 📂 Data File")
    uploaded = st.file_uploader(
        "Upload kinetic data (.xlsx or .csv) — shared across all tabs",
        type=["xlsx", "csv"], key="shared_file")
    if uploaded is not None:
        st.success(f"✅ File loaded: **{uploaded.name}**")
    else:
        st.info("Upload a file above to enable all analysis tabs.")
    return uploaded


# Master list of models used across fitting tabs
MODEL_NAMES = [
    "Zero-order", "Pseudo-first", "Pseudo-second-order",
    "Elovich", "L-H",
    "Power-Law", "Eley-Rideal", "Avrami", "Double-Exponential"
]


def _fit_curve(model, params, t_fine, C0):
    """Return the fitted curve for a given model over t_fine."""
    if model == "Zero-order":            return _zero_order(t_fine, params[0], C0)
    elif model == "Pseudo-first":        return _first_order(t_fine, params[0], C0)
    elif model == "Pseudo-second-order": return _second_order(t_fine, params[0], C0)
    elif model == "Elovich":             return _elovich(t_fine, params[0], params[1], C0)
    elif model == "L-H":                 return _lh_model(t_fine, params[0], params[1], C0)
    elif model == "Power-Law":           return _power_law(t_fine, params[0], params[1], C0)
    elif model == "Eley-Rideal":         return _eley_rideal(t_fine, params[0], params[1], C0)
    elif model == "Avrami":              return _avrami(t_fine, params[0], params[1], C0)
    elif model == "Double-Exponential":  return _double_exponential(t_fine, params[0], params[1], params[2], C0)
    return _lh_model(t_fine, params[0], params[1], C0)


# ================================================================
# TAB 1 — Kinetic Fitting
# ================================================================

# ================================================================
# TAB 1 — Kinetic Fitting
# ================================================================
def _tab_kinetics(cfg, uploaded):
    st.header("📈 Tab 1 — Kinetic Fitting")

    with st.expander("📋 Template & upload instructions", expanded=False):
        st.markdown("""
**Required columns:**
- `Time (min)` — reaction time
- One or more catalyst columns with `Removal (%)` values (0–100)

**Tips:**
- Minimum 4-8 time points per catalyst
- Include t=0 (Removal=0) if possible
        """)
        col1, col2 = st.columns(2)
        with col1:
            tmpl = pd.DataFrame({
                "Time (min)": [0, 10, 20, 30, 60, 90, 120],
                "Cat-A Removal (%)": [0, 15, 28, 40, 65, 80, 91],
                "Cat-B Removal (%)": [0, 22, 41, 55, 78, 89, 95],
            })
            buf = io.BytesIO()
            tmpl.to_excel(buf, index=False)
            st.download_button("⬇️ Download simple template", buf.getvalue(),
                               "ods_simple_template.xlsx",
                               "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        with col2:
            if st.button("📋 Generate Advanced Template (with Metadata)", type="primary"):
                buf, fname = create_advanced_template(cfg)
                st.download_button(
                    "⬇️ Download Advanced Template",
                    buf.getvalue(), fname,
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="adv_template")

    if uploaded is None:
        st.info("Upload a file above to begin fitting."); return
    df, time_col, removal_cols = _load_kinetic_data(uploaded)
    if df is None: return
    t_raw   = df[time_col].dropna().values.astype(float)
    C0      = cfg["C0"]
    c0_val  = cfg["c0_val"]
    c0_unit = cfg["c0_unit"]
    if C0 is None: st.error("C₀ conversion failed."); return

    # ── Data Preparation ─────────────────────────────────────────
    st.markdown("---")
    st.subheader("⚙️ Data Preparation")

    has_t0 = np.any(t_raw == 0)
    add_t0 = st.checkbox(
        "Auto-add t=0 point (Removal=0%, C=C₀)",
        value=(not has_t0),
        help="Recommended when t=0 is missing. Anchors the fit at C₀ — "
             "improves pseudo-second-order and L-H detection.")
    if add_t0 and not has_t0:
        st.info("✅ t=0 will be added automatically to all catalysts before fitting.")
    elif add_t0 and has_t0:
        st.info("ℹ️ t=0 already present — no duplication.")

    # Per-catalyst point exclusion
    # Apply reset BEFORE multiselects are rendered
    if st.session_state.get("reset_excl", False):
        st.session_state["reset_excl"] = False
        for col in removal_cols:
            st.session_state[f"excl_{col}"] = []

    st.markdown("**Point exclusion per catalyst** — select outlier / saturation points:")
    t_labels  = [f"t = {int(ti)} min" for ti in t_raw]
    n_cols_ui = min(len(removal_cols), 3)
    cols_ui   = st.columns(n_cols_ui)
    excl_per_cat = {}
    for ci, col in enumerate(removal_cols):
        cat_label = col.replace(" Removal (%)", "").strip()
        with cols_ui[ci % n_cols_ui]:
            excl = st.multiselect(
                f"**{cat_label}**",
                options=t_labels, default=[],
                key=f"excl_{col}",
                help=f"Excluded points shown as open markers on the plot.")
            excl_times = set()
            for lbl in excl:
                try:
                    excl_times.add(float(lbl.replace("t = ","").replace(" min","")))
                except Exception:
                    pass
            excl_per_cat[col] = excl_times

    st.markdown("---")

    # Run / Reset buttons
    run_col, reset_col, _ = st.columns([1, 1, 2])
    with run_col:
        run_analysis = st.button("▶ Run / Update Analysis", type="primary",
                                  help="Click after changing point exclusion or t=0 settings.")
    with reset_col:
        if st.button("🔄 Reset exclusions",
                     help="Clear all excluded points for all catalysts."):
            st.session_state["reset_excl"] = True
            if "tab1_ran" in st.session_state:
                del st.session_state["tab1_ran"]
            st.rerun()

    if run_analysis:
        st.session_state["tab1_ran"] = True
    if not st.session_state.get("tab1_ran", False):
        st.info("Press **▶ Run / Update Analysis** to start fitting.")
        return

    # ── Fitting ───────────────────────────────────────────────────
    model_names    = MODEL_NAMES
    all_results    = {}
    t_fit_per_cat  = {}
    Ct_fit_per_cat = {}

    # Auto-saturation thresholds
    SAT_THRESH_1 = 8.0   # if last interval < 8% → remove last point
    SAT_THRESH_2 = 15.0  # if penultimate interval also < 15% → remove that too

    for col in removal_cols:
        removal_raw = df[col].dropna().values[:len(t_raw)].astype(float)
        excl_times  = excl_per_cat[col]
        keep_mask   = np.array([ti not in excl_times for ti in t_raw])
        t_keep      = t_raw[keep_mask]
        rem_keep    = removal_raw[keep_mask]

        # ── Auto-saturation detection (only when no manual exclusion) ──
        auto_excl = []
        if not excl_times and len(rem_keep) >= 3:
            # Layer 1: last interval
            if (rem_keep[-1] - rem_keep[-2]) < SAT_THRESH_1:
                auto_excl.append(int(t_keep[-1]))
                t_keep  = t_keep[:-1]
                rem_keep = rem_keep[:-1]
            # Layer 2: penultimate interval
            if len(rem_keep) >= 3 and (rem_keep[-1] - rem_keep[-2]) < SAT_THRESH_2:
                auto_excl.append(int(t_keep[-1]))
                t_keep  = t_keep[:-1]
                rem_keep = rem_keep[:-1]

        if auto_excl:
            cat_label = col.replace(" Removal (%)","").strip()
            st.info(
                f"ℹ️ **{cat_label}**: auto-excluded saturation point(s) "
                f"t = {auto_excl} min (removal increment < {SAT_THRESH_1}%/"
                f"{SAT_THRESH_2}%). Use manual exclusion above to override.")

        Ct_keep = C0 * (1 - rem_keep / 100.0)

        if add_t0 and not has_t0:
            t_fit  = np.concatenate(([0.0], t_keep))
            Ct_fit = np.concatenate(([C0],  Ct_keep))
        else:
            t_fit  = t_keep
            Ct_fit = Ct_keep

        if len(t_fit) < 2:
            st.error(f"⚠️ **{col.replace(' Removal (%)','').strip()}**: only {len(t_fit)} point(s) remain after exclusion — minimum 2 needed. Please unselect some points above.")
            continue

        t_fit_per_cat[col]  = t_fit
        Ct_fit_per_cat[col] = Ct_fit
        # Show diagnostic before fitting
        n_excl = len(excl_times)
        n_pts  = len(t_fit)
        if n_excl > 0:
            st.caption(f"  {col.replace(' Removal (%)','').strip()}: "
                       f"{n_pts} points used ({n_excl} excluded)")
        all_results[col]    = _fit_nonlinear(t_fit, Ct_fit, C0)

    if not all_results:
        return

    # ── Saturation info per catalyst ──────────────────────────────
    try:
        for col in removal_cols:
            if col not in all_results: continue
            t_fit_s    = t_fit_per_cat[col]
            excl_times = excl_per_cat[col]
            if len(t_fit_s) >= 4 and not excl_times:
                removal_raw = df[col].dropna().values[:len(t_raw)].astype(float)
                t_nl_raw = t_fit_s[t_fit_s > 0][:-1]  # all non-zero points except last
                if len(t_nl_raw) < 2:
                    continue
                keep_nl = np.array([ti in set(t_nl_raw.tolist()) for ti in t_raw])
                Ct_nl   = C0 * (1 - removal_raw[keep_nl] / 100.0)
                if add_t0 and not has_t0:
                    t_nl2  = np.concatenate(([0.0], t_nl_raw))
                    Ct_nl2 = np.concatenate(([C0],  Ct_nl))
                else:
                    t_nl2  = t_nl_raw
                    Ct_nl2 = Ct_nl
                if len(t_nl2) < 3:
                    continue
                res_nl   = _fit_nonlinear(t_nl2, Ct_nl2, C0)
                best_all = _best_model(all_results[col], model_names)
                best_nl  = _best_model(res_nl, model_names)
                if best_all != best_nl:
                    cat_label = col.replace(" Removal (%)","").strip()
                    st.info(
                        f"ℹ️ **{cat_label}**: best model changes "
                        f"**{best_all} → {best_nl}** when "
                        f"t = {int(t_fit_s[-1])} min is excluded. "
                        f"Possible saturation — consider excluding it above.")
    except Exception:
        pass  # saturation detection is advisory only — never block main results

    # ── Helper: convert mol/L → user display unit ────────────────
    def _C_to_user(Ct_mol):
        if c0_unit == "ppmS":
            return Ct_mol * 32.06 * 1000        # mol/L → mg(S)/L = ppmS volumetric
        elif c0_unit in ("ppm", "mg/L"):
            mw = cfg.get("mw_poll") or 184.26
            return Ct_mol * mw * 1000
        elif c0_unit == "mmol/L":
            return Ct_mol * 1000
        elif c0_unit == "g/L":
            mw = cfg.get("mw_poll") or 184.26
            return Ct_mol * mw
        else:
            return Ct_mol

    u_label = c0_unit
    C0_user = _C_to_user(C0)

    # ── Raw Data Table ────────────────────────────────────────────
    with st.expander("📋 Raw data table", expanded=False):
        initial_data = []
        for col in removal_cols:
            removal_raw2 = df[col].dropna().values[:len(t_raw)].astype(float)
            cat_label2   = col.replace(" Removal (%)","").strip()
            for ti, rem in zip(t_raw, removal_raw2):
                C_t = C0 * (1 - rem / 100.0)
                row = {
                    "Catalyst":    cat_label2,
                    "Time (min)":  int(ti),
                    "Removal (%)": round(rem, 2),
                    "C (mmol/L)":  round(C_t * 1000, 4),
                }
                if c0_unit == "ppmS":
                    row[f"C ({u_label})"] = round(_C_to_user(C_t), 2)
                initial_data.append(row)
        df_initial = pd.DataFrame(initial_data)
        st.dataframe(df_initial, use_container_width=True, hide_index=True)
        csv_raw = df_initial.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download raw data table (CSV)",
                           csv_raw, "ods_raw_data.csv", "text/csv")

    t_fine     = np.linspace(0, t_raw.max(), 300)
    title_note = " [t=0 added]" if (add_t0 and not has_t0) else ""

    # ── Figure 1: C vs t (mmol/L) ────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    for ci, col in enumerate(removal_cols):
        if col not in all_results: continue
        removal_raw = df[col].dropna().values[:len(t_raw)].astype(float)
        excl_times  = excl_per_cat[col]
        color  = COLORS[ci % len(COLORS)]
        marker = MARKERS[ci % len(MARKERS)]
        cat_label = col.replace(" Removal (%)","").strip()

        for ti, ri in zip(t_raw, removal_raw):
            Ct_i = C0 * (1 - ri / 100.0)
            if ti in excl_times:
                ax.plot(ti, Ct_i * 1000, marker, color=color,
                        markersize=9, markerfacecolor="white",
                        markeredgewidth=1.5, zorder=5)
            else:
                ax.plot(ti, Ct_i * 1000, marker, color=color,
                        markersize=7, zorder=5)
        if add_t0 and not has_t0:
            ax.plot(0, C0 * 1000, marker, color=color, markersize=6,
                    markerfacecolor="none", markeredgewidth=1.2, zorder=4)
        ax.plot([], [], marker, color=color, label=cat_label)

        best = _best_model(all_results[col], model_names)
        if best is None: continue
        Ct_line = _fit_curve(best, all_results[col][best]["params"], t_fine, C0)
        ax.plot(t_fine, Ct_line * 1000, "-", color=color,
                label=f"{cat_label}: {best}")

    ax.set_xlabel("Time (min)"); ax.set_ylabel("C (mmol·L⁻¹)")
    ax.set_title(f"Concentration vs Time — Best Model (AICc){title_note}")
    ax.legend(fontsize=9); fig.tight_layout()
    st.pyplot(fig); plt.close(fig)

    # ── Figure 2: Linearized plot per catalyst ────────────────────
    st.markdown("#### Linearized plots (based on best model)")
    for ci, col in enumerate(removal_cols):
        if col not in all_results: continue
        best      = _best_model(all_results[col], model_names)
        if best is None: continue
        t_fit     = t_fit_per_cat[col]
        Ct_fit    = Ct_fit_per_cat[col]
        color     = COLORS[ci % len(COLORS)]
        marker    = MARKERS[ci % len(MARKERS)]
        cat_label = col.replace(" Removal (%)","").strip()
        C_user    = _C_to_user(Ct_fit)

        if best == "Pseudo-second-order":
            y_vals    = 1.0 / np.maximum(C_user, 1e-15)
            y_lbl     = f"1/C  ({u_label})⁻¹"
            title_lin = f"{cat_label} — Pseudo-second-order  |  1/C vs t"
        elif best in ("Pseudo-first", "Zero-order", "Power-Law",
                      "Eley-Rideal", "Avrami", "L-H",
                      "Elovich", "Double-Exponential"):
            # ln(C₀/C) vs t — valid for first-order regime; informative for others
            ratio  = np.maximum(C0_user / np.maximum(C_user, 1e-15), 1e-15)
            y_vals = np.log(ratio)
            y_lbl  = "ln(C₀/C)"
            title_lin = f"{cat_label} — {best}  |  ln(C₀/C) vs t"
        else:
            ratio  = np.maximum(C0_user / np.maximum(C_user, 1e-15), 1e-15)
            y_vals = np.log(ratio)
            y_lbl  = "ln(C₀/C)"
            title_lin = f"{cat_label} — {best}  |  ln(C₀/C) vs t"

        fig_lin, ax_lin = plt.subplots(figsize=(7, 4))
        ax_lin.scatter(t_fit, y_vals, color=color, marker=marker, s=60, zorder=5)
        if len(t_fit) >= 2:
            coeffs  = np.polyfit(t_fit, y_vals, 1)
            x_line  = np.linspace(t_fit.min(), t_fit.max(), 100)
            ax_lin.plot(x_line, np.polyval(coeffs, x_line), "--",
                        color=color, lw=1.5, alpha=0.9)
            r2_lin = _r2(y_vals, np.polyval(coeffs, t_fit))
            ax_lin.set_title(
                f"{title_lin}\nslope = {coeffs[0]:.4e}  |  R² = {r2_lin:.4f}",
                fontweight="bold")
        ax_lin.set_xlabel("Time (min)")
        ax_lin.set_ylabel(y_lbl)
        fig_lin.tight_layout()
        st.pyplot(fig_lin); plt.close(fig_lin)

    # ── Summary table ─────────────────────────────────────────────
    st.subheader("📊 Fitting Summary (Best Model by AICc)")
    rows = []
    for col, res in all_results.items():
        best = _best_model(res, model_names)
        cat_label = col.replace(" Removal (%)","").strip()
        if best is None:
            rows.append({"Catalyst": cat_label, "Best Model": "All fits failed"})
            continue
        br     = res[best]
        r0     = br.get("r0", float("nan"))
        V      = cfg["V_fuel"]; m = cfg["m_cat"]
        r0_m   = r0 * V / m if (r0 and m > 0) else float("nan")
        t_fit  = t_fit_per_cat[col]
        t_half = br.get("t_half", float("nan"))
        excl   = excl_per_cat[col]
        notes  = []
        first_pos = t_fit[t_fit > 0]
        if len(first_pos) > 0 and not np.isnan(t_half) and t_half < first_pos[0]:
            notes.append("⚠️ t½ < first data point")
        if excl:
            notes.append(f"excl: {', '.join([str(int(x)) for x in sorted(excl)])} min")
        if add_t0 and not has_t0:
            notes.append("t=0 added")
        rows.append({
            "Catalyst":           cat_label,
            "Best Model":         best,
            "k":                  _fmt_pm(br.get("k"), br.get("k_se")),
            "R²":                 br.get("R2","N/A"),
            "Adj-R²":             br.get("adj_r2","N/A"),
            "AICc":               br.get("aicc","N/A"),
            "AIC":                br.get("aic","N/A"),
            "t½ (min)":           _fmt_thalf(t_half),
            "r₀ (mol/L/min)":     _fmt_sci(r0),
            "r₀/m (mol/g/min)":   _fmt_sci(r0_m),
            "L-H Regime":         br.get("regime","–") if best == "L-H" else "–",
            "Note":               " | ".join(notes),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    with st.expander("🔍 All models for each catalyst", expanded=False):
        for col, res in all_results.items():
            cat_label = col.replace(" Removal (%)","").strip()
            st.markdown(f"**{cat_label}**")
            subrows = []
            for m in model_names:
                mr = res[m]
                if mr.get("R2", -999) > -999:
                    subrows.append({
                        "Model":    m,
                        "k (±SE)":  _fmt_pm(mr.get("k"), mr.get("k_se")),
                        "R²":       mr.get("R2","N/A"),
                        "Adj-R²":   mr.get("adj_r2","N/A"),
                        "AICc":     mr.get("aicc","N/A"),
                        "AIC":      mr.get("aic","N/A"),
                        "t½ (min)": _fmt_thalf(mr.get("t_half", float("nan"))),
                    })
                else:
                    subrows.append({"Model": m, "k (±SE)": "fit failed",
                                    "R²":"–","Adj-R²":"–","AICc":"–",
                                    "AIC":"–","t½ (min)":"–"})
            st.dataframe(pd.DataFrame(subrows), use_container_width=True)

    # ── Download ZIP ──────────────────────────────────────────────
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w") as zf:
        csv_buf = io.StringIO()
        pd.DataFrame(rows).to_csv(csv_buf, index=False)
        zf.writestr("fitting_summary.csv", csv_buf.getvalue())
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        for ci, (col, res) in enumerate(all_results.items()):
            removal_raw = df[col].dropna().values[:len(t_raw)].astype(float)
            color = COLORS[ci % len(COLORS)]
            cat_label = col.replace(" Removal (%)","").strip()
            Ct_all = C0 * (1 - removal_raw / 100.0)
            ax2.plot(t_raw, Ct_all * 1000, MARKERS[ci % len(MARKERS)],
                     color=color, label=cat_label)
            best = _best_model(res, model_names)
            if best is None: continue
            Ct_line = _fit_curve(best, res[best]["params"], t_fine, C0)
            ax2.plot(t_fine, Ct_line * 1000, "-", color=color,
                     label=f"{cat_label}: {best}")
        ax2.set_xlabel("Time (min)"); ax2.set_ylabel("C (mmol·L⁻¹)")
        ax2.set_title(f"Concentration vs Time (AICc){title_note}")
        ax2.legend(fontsize=9); fig2.tight_layout()
        png_buf = io.BytesIO()
        fig2.savefig(png_buf, dpi=300, bbox_inches="tight")
        zf.writestr("kinetics_plot.png", png_buf.getvalue())
        plt.close(fig2)
    st.download_button("⬇️ Download results (.zip)", zip_buf.getvalue(),
                       "kinetics_results.zip", "application/zip")

# ================================================================
# TAB 2 — Linearization
# ================================================================
def _tab_linearization(cfg, uploaded):
    st.header("📉 Tab 2 — Linearization Plots")
    if uploaded is None: st.info("Upload a file above."); return
    df, time_col, removal_cols = _load_kinetic_data(uploaded)
    if df is None: return
    t  = df[time_col].dropna().values.astype(float)
    C0 = cfg["C0"]
    if C0 is None: st.error("C₀ conversion failed."); return

    # Four classical linearized forms
    lin_models = [
        ("Zero-order  |  C vs t",
         "Time (min)", "C (mol·L⁻¹)",
         lambda t_, Ct: (t_, Ct)),
        ("Pseudo-first  |  ln(C₀/C) vs t",
         "Time (min)", "ln(C₀/C)",
         lambda t_, Ct: (t_[Ct > 0], np.log(C0 / np.maximum(Ct[Ct > 0], 1e-15)))),
        ("Pseudo-second-order  |  1/C vs t",
         "Time (min)", "1/C  (L·mol⁻¹)",
         lambda t_, Ct: (t_[Ct > 0], 1.0 / Ct[Ct > 0])),
        ("Elovich  |  C vs ln t",
         "ln t", "C (mol·L⁻¹)",
         lambda t_, Ct: (np.log(t_[t_ > 0]), Ct[t_ > 0])),
    ]

    # summary_rows collects one row per (model, catalyst)
    summary_rows = []

    for model_title, x_lbl, y_lbl, transform in lin_models:
        st.markdown(f"### {model_title}")
        fig, ax = plt.subplots(figsize=(8, 4))
        for ci, col in enumerate(removal_cols):
            removal   = df[col].dropna().values[:len(t)].astype(float)
            Ct        = C0 * (1 - removal / 100.0)
            color     = COLORS[ci % len(COLORS)]
            marker    = MARKERS[ci % len(MARKERS)]
            cat_label = col.replace(" Removal (%)","").strip()
            try:
                x_vals, y_vals = transform(t, Ct)
                if len(x_vals) < 2:
                    continue
                ax.scatter(x_vals, y_vals, color=color, marker=marker,
                           s=60, zorder=5, label=cat_label)
                coeffs = np.polyfit(x_vals, y_vals, 1)
                x_fit  = np.linspace(x_vals.min(), x_vals.max(), 100)
                ax.plot(x_fit, np.polyval(coeffs, x_fit), "--",
                        color=color, lw=1.5, alpha=0.8)
                r2_lin = _r2(y_vals, np.polyval(coeffs, x_vals))
                summary_rows.append({
                    "Model":     model_title.split("|")[0].strip(),
                    "Catalyst":  cat_label,
                    "slope":     round(coeffs[0], 6),
                    "intercept": round(coeffs[1], 6),
                    "R²":        round(r2_lin, 4),
                })
            except Exception:
                continue
        ax.set_xlabel(x_lbl); ax.set_ylabel(y_lbl)
        ax.set_title(model_title, fontweight="bold")
        ax.legend(fontsize=9)
        fig.tight_layout()
        st.pyplot(fig); plt.close(fig)

    if not summary_rows:
        return

    # ── Build pivot: best model per catalyst (highest linear R²) ──
    df_sum = pd.DataFrame(summary_rows)

    # Exclude same models as Tab 1 from "best" selection
    df_sum_eligible = df_sum[~df_sum["Model"].isin(
        {m.split("  |")[0].strip() for m in [
            "Elovich", "Double-Exponential"]}
    )]
    if df_sum_eligible.empty:
        df_sum_eligible = df_sum  # fallback if all excluded

    # For each catalyst find the model with max R² (from eligible models only)
    best_linear = (
        df_sum_eligible.loc[df_sum_eligible.groupby("Catalyst")["R²"].idxmax()]
        .set_index("Catalyst")[["Model", "R²"]]
        .rename(columns={"Model": "Best model (linear R²)",
                         "R²":    "Best R²"})
    )

    # Pivot so each row = one catalyst, columns = models
    df_pivot = df_sum.pivot_table(
        index="Catalyst", columns="Model", values="R²"
    ).reset_index()
    df_pivot.columns.name = None

    # Merge best-model column
    df_pivot = df_pivot.merge(best_linear, on="Catalyst", how="left")

    # Reorder: Catalyst | Best model | Best R² | individual model R²s
    model_cols = [c for c in df_pivot.columns
                  if c not in ("Catalyst", "Best model (linear R²)", "Best R²")]
    df_pivot = df_pivot[["Catalyst", "Best model (linear R²)", "Best R²"] + model_cols]

    st.markdown("---")
    st.markdown("### 🏆 Best Model by Linear R²")
    st.dataframe(best_linear.reset_index(), use_container_width=True, hide_index=True)

    st.markdown("### 📋 All Models — R² Comparison")
    st.dataframe(df_pivot, use_container_width=True, hide_index=True)

    st.info(
        "**How to interpret:** Use **Tab 1 (AICc + parsimony)** as the primary "
        "model selection. Use **Tab 2 (linear R²)** as supporting visual evidence. "
        "With only 5–6 points, **Pseudo-second-order** and **L-H** are generally "
        "more physically meaningful than Elovich or Power-Law."
    )



# ================================================================
# TAB 3 — Removal Efficiency
# ================================================================
def _tab_removal(cfg, uploaded):
    st.header("♻️ Tab 3 — Removal Efficiency")
    if uploaded is None: st.info("Upload a file above."); return
    df, time_col, removal_cols = _load_kinetic_data(uploaded)
    if df is None: return
    t = df[time_col].dropna().values.astype(float)
    fig, ax = plt.subplots(figsize=(8, 5))
    for ci, col in enumerate(removal_cols):
        removal = df[col].dropna().values[:len(t)].astype(float)
        ax.plot(t, removal, MARKERS[ci % len(MARKERS)] + "-",
                color=COLORS[ci % len(COLORS)], label=col)
    ax.set_xlabel("Time (min)"); ax.set_ylabel("Desulfurization efficiency (%)")
    ax.set_title("Desulfurization Efficiency"); ax.set_ylim(0, 105); ax.legend()
    fig.tight_layout(); st.pyplot(fig); plt.close(fig)
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    final_removals = []; labels = []
    for col in removal_cols:
        removal = df[col].dropna().values[:len(t)].astype(float)
        final_removals.append(removal[-1])
        labels.append(col.replace(" Removal (%)", "").strip())
    bars = ax2.bar(labels, final_removals, color=COLORS[:len(labels)],
                   edgecolor="black", linewidth=0.8)
    ax2.bar_label(bars, fmt="%.1f%%", padding=2, fontsize=10)
    ax2.set_ylabel("Desulfurization efficiency (%)"); ax2.set_ylim(0, 115)
    ax2.set_title(f"Desulfurization efficiency at t = {t[-1]:.0f} min")
    fig2.tight_layout(); st.pyplot(fig2); plt.close(fig2)


# ================================================================
# TAB 4 — TON / TOF
# ================================================================
_SITE_DENSITY_PRESETS = {
    "Metal oxides (MoO₃, V₂O₅, WO₃, TiO₂)":         {"rho": 2.0,  "range": "1–5",   "method": "NH₃-TPD or H₂-TPR"},
    "Zeolites (ZSM-5, USY, Beta, SAPO)":               {"rho": 5.0,  "range": "2–10",  "method": "Pyridine-FTIR or NH₃-TPD"},
    "Polyoxometalates (POM, HPW, PMo)":                {"rho": 1.0,  "range": "0.5–3", "method": "Formula-based or ³¹P NMR"},
    "Graphene / rGO / GO":                             {"rho": 1.5,  "range": "0.5–4", "method": "Boehm titration or XPS O/C"},
    "g-C₃N₄ / Carbon nitride":                        {"rho": 3.0,  "range": "1–6",   "method": "NH₃-TPD or XPS N 1s"},
    "N-doped carbon / N-graphene":                     {"rho": 2.5,  "range": "1–5",   "method": "XPS N content × BET"},
    "MOF-derived porous carbon":                       {"rho": 2.0,  "range": "0.5–4", "method": "Boehm titration or CO₂-TPD"},
    "Supported metal nanoparticles (Pd, Pt, Au)":      {"rho": 0.5,  "range": "0.1–2", "method": "CO chemisorption or TEM dispersion"},
    "Custom (enter manually)":                         {"rho": None, "range": "—",     "method": "User-defined"},
}

def _tab_ton_tof(cfg, uploaded):
    st.header("⚗️ Tab 4 — TON & TOF")
    st.markdown("""
**Definitions:**
- **TON** = n_substrate_converted / n_active_sites &nbsp;(dimensionless)
- **TOF** (min⁻¹) = TON / t_reaction
- **n_active_sites** can be entered directly (Option 1) or estimated from BET + material type (Option 2)
    """)
    if uploaded is None: st.info("Upload a file above."); return
    df, time_col, removal_cols = _load_kinetic_data(uploaded)
    if df is None: return
    t  = df[time_col].dropna().values.astype(float)
    C0 = cfg["C0"]; V = cfg["V_fuel"]; m = cfg["m_cat"]
    if C0 is None: st.error("C₀ conversion failed."); return

    method_choice = st.radio(
        "How to define active site density?",
        ["Option 1 — Direct input (from TPD / TPR / chemisorption / titration)",
         "Option 2 — Estimate from BET surface area + material type"],
        horizontal=True)

    if "Option 1" in method_choice:
        st.markdown("#### Direct input")
        col_a, col_b = st.columns(2)
        with col_a:
            site_density = st.number_input(
                "Active site density (mmol/g catalyst)",
                min_value=0.0001, value=0.5, step=0.05,
                help="From NH₃-TPD, H₂-TPR, CO chemisorption, Boehm titration, etc.")
        with col_b:
            char_method = st.selectbox(
                "Characterisation method used",
                ["NH₃-TPD", "H₂-TPR", "CO chemisorption",
                 "Pyridine-FTIR", "Boehm titration",
                 "XPS", "³¹P NMR (POM)", "Formula-based", "Other"])
        n_sites_mol = site_density * 1e-3 * m
        st.info(f"n_active_sites = **{n_sites_mol*1e6:.3f} µmol** "
                f"({site_density} mmol/g × {m} g catalyst) — method: {char_method}")
    else:
        st.markdown("#### BET-based estimation")
        col_a, col_b = st.columns(2)
        with col_a:
            s_bet = st.number_input("BET surface area (m²/g)",
                                    min_value=0.1, value=100.0, step=10.0)
            mat_type = st.selectbox("Material family", list(_SITE_DENSITY_PRESETS.keys()))
        preset = _SITE_DENSITY_PRESETS[mat_type]
        with col_b:
            if preset["rho"] is not None:
                rho_default = float(preset["rho"])
                st.markdown(f"""
**Typical ρ_site for {mat_type.split('(')[0].strip()}:**
- Range: **{preset["range"]} µmol/m²**
- Recommended method: *{preset["method"]}*
                """)
            else:
                rho_default = 1.0
                st.markdown("Enter your own ρ_site value below.")
            rho_site = st.number_input(
                "ρ_site — active site surface density (µmol/m²)",
                min_value=0.01, value=rho_default, step=0.1,
                help="Override the preset if you have measured this value.")
        n_sites_mol = s_bet * m * rho_site * 1e-6
        site_density_equiv = (n_sites_mol * 1e3) / m if m > 0 else float("nan")
        st.info(
            f"n_active_sites = S_BET × m × ρ_site = "
            f"{s_bet} × {m} × {rho_site} µmol/m² = "
            f"**{n_sites_mol*1e6:.3f} µmol** "
            f"(≡ {site_density_equiv:.4f} mmol/g)")
        with st.expander("ℹ️ How ρ_site is estimated per material family", expanded=False):
            st.markdown("""
| Material | ρ_site (µmol/m²) | Basis |
|----------|-----------------|-------|
| Metal oxides (MoO₃, V₂O₅, WO₃) | 1–5 | NH₃-TPD acid sites; terminal M=O counted as active |
| Zeolites (ZSM-5, USY, Beta) | 2–10 | Pyridine-FTIR Brønsted + Lewis; strong acid sites for ODS |
| POMs (HPW, PMo) | 0.5–3 | Keggin unit density on support; ³¹P NMR quantification |
| Graphene / rGO / GO | 0.5–4 | Boehm titration (COOH + C=O + OH groups); XPS O/C ratio |
| g-C₃N₄ / carbon nitride | 1–6 | Pyridinic + pyrrolic N from XPS N 1s; NH₃-TPD base sites |
| N-doped carbon / N-graphene | 1–5 | XPS N atomic % × BET; pyridinic N as primary active site |
| MOF-derived porous carbon | 0.5–4 | Boehm titration; CO₂-TPD for basic sites |
| Supported metals (Pd, Pt, Au) | 0.1–2 | CO chemisorption dispersion; TEM particle size → metal surface |

> **Note:** These are literature-based estimates. If you have measured site density directly (TPD, TPR, chemisorption), use **Option 1** for higher accuracy.
            """)

    st.markdown("---")
    st.markdown("### 📋 TON & TOF Results")
    if n_sites_mol <= 0:
        st.error("n_active_sites = 0. Check your inputs."); return
    rows = []
    for col in removal_cols:
        removal = df[col].dropna().values[:len(t)].astype(float)
        X_final = removal[-1] / 100.0
        n_conv  = C0 * V * X_final
        t_rxn   = t[-1]
        ton = n_conv / n_sites_mol
        tof = ton / t_rxn if t_rxn > 0 else float("nan")
        rows.append({
            "Catalyst":       col,
            "X_final (%)":    round(removal[-1], 1),
            "n_conv (µmol)":  round(n_conv * 1e6, 3),
            "n_sites (µmol)": round(n_sites_mol * 1e6, 3),
            "TON":            round(ton, 3),
            "TOF (min⁻¹)":    f"{tof:.5f}" if not np.isnan(tof) else "N/A",
            "TOF (h⁻¹)":      f"{tof*60:.3f}" if not np.isnan(tof) else "N/A",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True)
    if len(rows) > 1:
        fig, ax = plt.subplots(figsize=(7, 4))
        cats  = [r["Catalyst"] for r in rows]
        tofs  = [float(r["TOF (h⁻¹)"]) if r["TOF (h⁻¹)"] != "N/A" else 0 for r in rows]
        bars  = ax.bar(cats, tofs, color=COLORS[:len(cats)],
                       edgecolor="black", linewidth=0.8)
        ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=9)
        ax.set_ylabel("TOF (h⁻¹)"); ax.set_title("Turnover Frequency Comparison")
        fig.tight_layout(); st.pyplot(fig); plt.close(fig)


# ================================================================
# TAB 5 — Parameter Effect
# ================================================================
def _tab_parameter_effect(cfg, uploaded):
    st.header("🔬 Tab 5 — Parameter Effect")
    st.caption("Simulation tab — uses the three closed-form models "
               "(pseudo-first / zero / pseudo-second order).")
    C0 = cfg["C0"]
    if C0 is None: st.error("C₀ conversion failed."); return
    model_choice = st.selectbox("Kinetic model for simulation",
        ["Pseudo-first","Zero-order","Pseudo-second-order"])
    param  = st.selectbox("Parameter to sweep", [
        "Initial concentration C₀","Catalyst mass m",
        "Temperature (Arrhenius)","O/S molar ratio"])
    t_max  = st.number_input("t_max (min)", min_value=10, value=120, step=10)
    t_arr  = np.linspace(0, t_max, 300)
    fig, ax = plt.subplots(figsize=(8, 5))

    def _simulate(t_a, kapp, C0_v):
        if model_choice == "Pseudo-first": return _first_order(t_a, kapp, C0_v)
        elif model_choice == "Zero-order": return _zero_order(t_a, kapp, C0_v)
        else:                              return _second_order(t_a, kapp, C0_v)

    if param == "Initial concentration C₀":
        kapp_base = st.number_input("kapp", min_value=1e-6, value=0.05, format="%.5f")
        for f in [0.25, 0.5, 1.0, 2.0, 4.0]:
            C0_v = C0 * f; Ct = _simulate(t_arr, kapp_base, C0_v)
            ax.plot(t_arr, Ct * 1000, label=f"C₀ × {f}")
        ax.set_ylabel("C (mmol/L)")
    elif param == "Catalyst mass m":
        kapp_base = st.number_input("kapp at base m (min⁻¹)", min_value=1e-6, value=0.05, format="%.5f")
        for mf in [0.5, 1.0, 2.0, 3.0, 5.0]:
            kapp_v = kapp_base * mf; Ct = _simulate(t_arr, kapp_v, C0)
            rem = (1 - Ct / C0) * 100
            ax.plot(t_arr, rem, label=f"m × {mf}")
        ax.set_ylabel("Removal (%)")
    elif param == "Temperature (Arrhenius)":
        kapp_ref = st.number_input("k at ref T (min⁻¹)", min_value=1e-8, value=0.05, format="%.6f")
        Ea_kJ    = st.number_input("Eₐ (kJ/mol)", min_value=1.0, value=50.0, step=5.0)
        Ea = Ea_kJ * 1000.0; T_ref = cfg["temp_C"] + 273.15
        for T in [T_ref - 20, T_ref - 10, T_ref, T_ref + 10, T_ref + 20]:
            if T <= 200: continue
            k_T = kapp_ref * np.exp(-Ea / R_GAS * (1 / T - 1 / T_ref))
            rem = (1 - _simulate(t_arr, k_T, C0) / C0) * 100
            ax.plot(t_arr, rem, label=f"T = {T-273.15:.0f} °C")
        ax.set_ylabel("Removal (%)")
    else:
        kapp_base = st.number_input("k at O/S=1 (min⁻¹)", min_value=1e-6, value=0.05, format="%.5f")
        n_os = st.number_input("Reaction order in oxidant (n)", min_value=0.1, value=1.0, step=0.1)
        for os in [1, 2, 4, 6, 8]:
            k_os = kapp_base * (os ** n_os)
            rem  = (1 - _simulate(t_arr, k_os, C0) / C0) * 100
            ax.plot(t_arr, rem, label=f"O/S = {os}")
        ax.set_ylabel("Removal (%)")
    ax.set_xlabel("Time (min)"); ax.set_title(f"Effect of: {param} ({model_choice})")
    ax.legend(fontsize=9); fig.tight_layout(); st.pyplot(fig); plt.close(fig)


# ================================================================
# TAB 6 — Oxidant Efficiency
# ================================================================
def _tab_oxidant_efficiency(cfg, uploaded):
    st.header("🧪 Tab 6 — Oxidant Efficiency")
    measure_h2o2 = st.radio("H₂O₂ consumption measurement available?", [
        "No — use stoichiometric assumption (2 mol H₂O₂ per mol DBT)",
        "Yes — I will enter measured consumption"], index=0)
    if "No" in measure_h2o2:
        st.warning("⚠️ Stoichiometric assumption active — η will be ~100% and uninformative.")
    if uploaded is None: st.info("Upload a file above."); return
    df, time_col, removal_cols = _load_kinetic_data(uploaded)
    if df is None: return
    t = df[time_col].dropna().values.astype(float)
    C0 = cfg["C0"]; V = cfg["V_fuel"]; O_S = cfg["O_S"]
    if C0 is None: st.error("C₀ conversion failed."); return
    n_H2O2_initial = O_S * C0 * V
    rows = []
    for col in removal_cols:
        removal = df[col].dropna().values[:len(t)].astype(float)
        n_DBT_removed = C0 * V * (removal[-1] / 100.0)
        if "Yes" in measure_h2o2:
            safe_key = f"h2o2_{col.replace(' ','_').replace('(','').replace(')','')}"
            n_H2O2_consumed = st.number_input(
                f"n(H₂O₂)_consumed for {col} (mmol)",
                min_value=0.0, value=float(round(n_DBT_removed * 2 * 1000, 3)),
                step=0.001, key=safe_key) * 1e-3
        else:
            n_H2O2_consumed = 2 * n_DBT_removed
        eta = (n_DBT_removed / (n_H2O2_consumed / 2)) * 100 if n_H2O2_consumed > 0 else float("nan")
        rows.append({"Catalyst": col,
                     "n_DBT removed (µmol)": round(n_DBT_removed * 1e6, 2),
                     "n_H₂O₂ initial (µmol)": round(n_H2O2_initial * 1e6, 2),
                     "n_H₂O₂ consumed (µmol)": round(n_H2O2_consumed * 1e6, 2),
                     "η (%)": round(eta, 1) if not np.isnan(eta) else "N/A",
                     "H₂O₂ utilization (%)": round(n_H2O2_consumed / n_H2O2_initial * 100, 1)
                                              if n_H2O2_initial > 0 else "N/A"})
    st.dataframe(pd.DataFrame(rows), use_container_width=True)


# ================================================================
# TAB 7 — Condition Comparison
# ================================================================
def _tab_comparison(cfg):
    st.header("📊 Tab 7 — Condition Comparison")
    with st.expander("📋 Expected format", expanded=False):
        tmpl = pd.DataFrame({
            "Experiment": ["Run-1","Run-2","Run-3"],
            "T (°C)": [25, 40, 60], "O/S": [2, 4, 6],
            "kapp (1/min)": [0.012, 0.028, 0.055],
            "R2": [0.989, 0.994, 0.997],
            "t_half (min)": [57.8, 24.8, 12.6],
            "Removal_final (%)": [75, 88, 95],
        })
        buf = io.BytesIO(); tmpl.to_excel(buf, index=False)
        st.download_button("⬇️ Download comparison template", buf.getvalue(),
                           "comparison_template.xlsx",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    uploaded_cmp = st.file_uploader("Upload comparison table (.xlsx/.csv)",
                                    type=["xlsx","csv"], key="cmp_upload")
    if uploaded_cmp is None: st.info("Upload a summary table."); return
    try:
        df = pd.read_excel(uploaded_cmp) if uploaded_cmp.name.endswith(".xlsx") \
             else pd.read_csv(uploaded_cmp)
    except Exception as e:
        st.error(f"{e}"); return
    st.dataframe(df, use_container_width=True)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) >= 2:
        x_col = st.selectbox("X axis", numeric_cols, index=0)
        y_col = st.selectbox("Y axis", numeric_cols, index=min(1, len(numeric_cols)-1))
        fig, ax = plt.subplots(figsize=(7, 5))
        label_col = df.columns[0]
        for i, row in df.iterrows():
            ax.scatter(row[x_col], row[y_col], color=COLORS[i % len(COLORS)],
                       marker=MARKERS[i % len(MARKERS)], s=80, zorder=5)
            ax.annotate(str(row[label_col]), (row[x_col], row[y_col]),
                        textcoords="offset points", xytext=(5, 5), fontsize=9)
        ax.set_xlabel(x_col); ax.set_ylabel(y_col)
        ax.set_title(f"{y_col} vs {x_col}")
        fig.tight_layout(); st.pyplot(fig); plt.close(fig)


# ================================================================
# TAB 8 — Arrhenius Multi-Temperature Analysis
# ================================================================
def _tab_arrhenius(cfg):
    st.header("🌡️ Tab 8 — Arrhenius Analysis (Multi-Temperature)")
    st.markdown(r"""
**Purpose:** Upload kinetic datasets at **different temperatures**, fit each with the best kinetic
model (AICc), collect k(T), then fit the Arrhenius equation to extract **Eₐ** and **A**.

$$k(T) = A \cdot \exp\!\left(-\frac{E_a}{RT}\right)$$

Linearised: $\ln k = \ln A - \dfrac{E_a}{R} \cdot \dfrac{1}{T}$

| System | Eₐ (kJ/mol) | Source |
|--------|------------|--------|
| DBT / UiO-66-NO₂ | 38.5 | Barghi 2025 |
| DBT / MoO₃-Al₂O₃ | 52.3 | Dhir 2009 |
| BT / TS-1 | 44.1 | Sengupta 2012 |
    """)
    C0 = cfg["C0"]
    if C0 is None: st.error("C₀ conversion failed."); return
    st.subheader("Step 1 — Upload files at each temperature")
    uploaded_files = st.file_uploader("Upload one or more files (one per temperature)",
        type=["xlsx","csv"], accept_multiple_files=True, key="arrhenius_files")
    if not uploaded_files:
        st.info("Upload at least 2 files at different temperatures."); return
    st.subheader("Step 2 — Assign a temperature to each file")
    temps_C = []
    for uf in uploaded_files:
        T = st.number_input(f"Temperature for **{uf.name}** (°C)",
                            min_value=-20.0, max_value=200.0, value=25.0, step=5.0,
                            key=f"arr_T_{uf.name}")
        temps_C.append(T)
    model_names = MODEL_NAMES
    model_choice_arr = st.selectbox("Kinetic model for k extraction",
        ["Best (AICc)","Pseudo-first","Pseudo-second-order","L-H"], index=0, key="arr_model_choice")
    if model_choice_arr == "Best (AICc)":
        st.warning(
            "⚠️ **Warning:** 'Best (AICc)' may select **different models at different temperatures**, "
            "which violates the assumption of a constant reaction mechanism. "
            "Ea extracted from mixed-model k(T) values is physically meaningless. "
            "Select a **single fixed model** (e.g. Pseudo-second-order) for a valid Arrhenius plot."
        )
    if st.button("▶ Run Arrhenius Fitting", key="run_arrhenius"):
        results_per_T = {}; catalyst_names_all = None
        progress = st.progress(0)
        for idx, (uf, T_C) in enumerate(zip(uploaded_files, temps_C)):
            uf.seek(0)
            df, time_col, removal_cols = _load_kinetic_data(uf)
            if df is None: st.warning(f"Skipping {uf.name}."); continue
            t = df[time_col].dropna().values.astype(float); T_K = T_C + 273.15
            if catalyst_names_all is None: catalyst_names_all = removal_cols
            cat_k = {}
            for col in removal_cols:
                removal = df[col].dropna().values[:len(t)].astype(float)
                Ct  = C0 * (1 - removal / 100.0)
                res = _fit_nonlinear(t, Ct, C0)
                chosen = _best_model(res, model_names) if model_choice_arr == "Best (AICc)" \
                         else model_choice_arr
                cat_k[col] = res[chosen]["k"] if (chosen and res[chosen].get("R2",-999) > -999) else None
            results_per_T[T_K] = cat_k
            progress.progress((idx + 1) / len(uploaded_files))
        progress.empty()
        if not results_per_T or catalyst_names_all is None:
            st.error("No valid fits obtained."); return
        T_K_list = sorted(results_per_T.keys())
        n_cats = len(catalyst_names_all)
        fig, axes = plt.subplots(1, n_cats, figsize=(6 * n_cats, 5), squeeze=False)
        arrh_rows = []
        for ci, cat in enumerate(catalyst_names_all):
            ax = axes[0][ci]; color = COLORS[ci % len(COLORS)]
            k_vals = []; T_valid = []
            for T_K in T_K_list:
                k = results_per_T[T_K].get(cat)
                if k is not None and k > 0:
                    k_vals.append(k); T_valid.append(T_K)
            if len(k_vals) < 2:
                ax.set_title(f"{cat}\n(insufficient valid k)")
                arrh_rows.append({"Catalyst": cat, "Eₐ (kJ/mol)": "N/A", "n_T": len(k_vals)}); continue
            inv_T_v = np.array([1.0 / T for T in T_valid])
            ln_k    = np.log(np.array(k_vals))
            try:
                coeffs, cov = np.polyfit(inv_T_v, ln_k, 1, cov=True)
                slope = coeffs[0]; intercept = coeffs[1]
                Ea_kJ = -slope * R_GAS / 1000.0
                A_val = np.exp(intercept)
                Ea_ci = np.sqrt(cov[0,0]) * R_GAS / 1000.0 * 1.96
                lnA_ci = np.sqrt(cov[1,1]) * 1.96
                r2_arr = _r2(ln_k, np.polyval(coeffs, inv_T_v))
                ax.scatter(inv_T_v * 1000, ln_k, color=color,
                           marker=MARKERS[ci % len(MARKERS)], s=70, zorder=5, label=cat)
                x_fit = np.linspace(inv_T_v.min(), inv_T_v.max(), 100)
                ax.plot(x_fit * 1000, np.polyval(coeffs, x_fit), "--", color=color, linewidth=1.5)
                for x_, y_, T_ in zip(inv_T_v, ln_k, T_valid):
                    ax.annotate(f"{T_-273.15:.0f}°C", (x_*1000, y_),
                                textcoords="offset points", xytext=(4,4), fontsize=8)
                ax.set_xlabel("1000/T (K⁻¹)"); ax.set_ylabel("ln k")
                ax.set_title(f"{cat}\nEₐ = {Ea_kJ:.1f} ± {Ea_ci:.1f} kJ/mol\nR² = {r2_arr:.4f}")
                arrh_rows.append({"Catalyst": cat,
                                  "Eₐ (kJ/mol)": round(Ea_kJ, 2),
                                  "± 95% CI Eₐ (kJ/mol)": round(Ea_ci, 2),
                                  "ln A": round(intercept, 3),
                                  "± 95% CI ln A": round(lnA_ci, 3),
                                  "A": _fmt_sci(A_val),
                                  "R² (Arrhenius)": round(r2_arr, 4),
                                  "n_T": len(k_vals)})
            except Exception as e:
                ax.set_title(f"{cat}\nFit failed: {e}")
                arrh_rows.append({"Catalyst": cat, "Eₐ (kJ/mol)": f"Error: {e}", "n_T": len(k_vals)})
        fig.tight_layout()
        # FIX (v3.5.1): save PNG *before* st.pyplot/plt.close so the figure
        # object is still alive when we write it into the ZIP archive.
        png_buf = io.BytesIO()
        fig.savefig(png_buf, dpi=300, bbox_inches="tight")
        png_buf.seek(0)
        st.pyplot(fig); plt.close(fig)
        st.subheader("Arrhenius Parameters")
        st.dataframe(pd.DataFrame(arrh_rows), use_container_width=True)
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w") as zf:
            csv_buf = io.StringIO(); pd.DataFrame(arrh_rows).to_csv(csv_buf, index=False)
            zf.writestr("arrhenius_parameters.csv", csv_buf.getvalue())
            zf.writestr("arrhenius_plot.png", png_buf.getvalue())
        st.download_button("⬇️ Download Arrhenius results (.zip)", zip_buf.getvalue(),
                           "arrhenius_results.zip", "application/zip")
        with st.expander("📚 Interpretation guide", expanded=False):
            st.markdown("""
**Eₐ interpretation:**
- **< 20 kJ/mol** → likely mass-transfer limited
- **20–40 kJ/mol** → mixed regime
- **40–80 kJ/mol** → surface-reaction controlled (typical ODS)
- **> 100 kJ/mol** → possible deactivation artefact

**⚠️ Note on composite rate constants:**
When k is extracted from **L-H** or **Power-Law** models, it is a *composite*
parameter (e.g. k_LH × K_ads for L-H), not an elementary rate constant.
The Eₐ obtained therefore represents an *apparent* activation energy that
includes adsorption enthalpy contributions and cannot be compared directly
with Eₐ values from pseudo-first-order fits reported in the literature.
For cross-study comparisons, use **Pseudo-first-order** k values.

**References:**
- Barghi et al., *ACS Omega* **2025**, 10, 15947. DOI: 10.1021/acsomega.4c06722
- Dhir et al., *J. Hazard. Mater.* **2009**, 161, 1360. DOI: 10.1016/j.jhazmat.2008.04.099
- Sengupta et al., *Ind. Eng. Chem. Res.* **2012**, 51, 147. DOI: 10.1021/ie2024068
            """)


# ================================================================
# TAB 9 — Residual Diagnostics
# ================================================================
def _tab_residuals(cfg, uploaded):
    st.header("🔍 Tab 9 — Residual Diagnostics")
    st.markdown("""
**Purpose:** Check model quality via residual analysis.
Plots: Residuals vs Time, Residuals vs Fitted, Normal Q-Q, Standardised Residuals.
Statistical tests: Shapiro-Wilk normality, outlier detection (|z|>2), runs test for systematic misfit.

_Note: with only 5–9 points these tests have low statistical power and are indicative only._
    """)
    if uploaded is None: st.info("Upload a file above."); return
    df, time_col, removal_cols = _load_kinetic_data(uploaded)
    if df is None: return
    t  = df[time_col].dropna().values.astype(float)
    C0 = cfg["C0"]
    if C0 is None: st.error("C₀ conversion failed."); return
    model_names = MODEL_NAMES
    col1, col2 = st.columns(2)
    with col1: cat_choice   = st.selectbox("Select catalyst", removal_cols, key="resid_cat")
    with col2: model_choice = st.selectbox("Select model",    model_names,  key="resid_model")
    removal = df[cat_choice].dropna().values[:len(t)].astype(float)
    Ct_obs  = C0 * (1 - removal / 100.0)
    all_res = _fit_nonlinear(t, Ct_obs, C0)
    res = all_res[model_choice]
    if res.get("R2", -999) <= -999:
        st.error(f"Model '{model_choice}' failed. Error: {res.get('error','unknown')}"); return
    Ct_pred   = res["pred"]
    residuals = Ct_obs - Ct_pred
    n         = len(residuals)
    # FIX U: use number of FREE parameters (N_PARAMS), not len(params) which counts fixed C0
    p_free   = N_PARAMS.get(model_choice, 1)
    ddof_use = p_free if (n - p_free) > 0 else 0
    sigma    = np.std(residuals, ddof=ddof_use)
    std_resid = residuals / sigma if sigma > 0 else residuals
    from scipy import stats as scipy_stats
    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("R²",        f"{res['R2']:.4f}")
    col_b.metric("Adj-R²",    f"{res.get('adj_r2',float('nan')):.4f}"
                               if not np.isnan(res.get('adj_r2',float('nan'))) else "N/A")
    col_c.metric("AICc",      f"{res.get('aicc',float('nan')):.2f}"
                               if not np.isinf(res.get('aicc',float('inf'))) else "∞")
    col_d.metric("RMSE (mol/L)", f"{np.sqrt(np.mean(residuals**2)):.2e}")
    sw_stat = sw_p = float("nan")
    if n >= 3:
        try: sw_stat, sw_p = scipy_stats.shapiro(residuals)
        except Exception: pass
    if not np.isnan(sw_p):
        if sw_p > 0.05:
            st.success(f"✅ Shapiro-Wilk: W={sw_stat:.4f}, p={sw_p:.4f} — residuals normally distributed.")
        else:
            st.warning(f"⚠️ Shapiro-Wilk: W={sw_stat:.4f}, p={sw_p:.4f} — deviates from normality.")
    outlier_idx = np.where(np.abs(std_resid) > 2.0)[0]
    if len(outlier_idx) > 0:
        st.warning(f"⚠️ {len(outlier_idx)} outlier(s) at t = {[t[i] for i in outlier_idx]} min")
    else:
        st.success("✅ No outliers detected.")
    signs = np.sign(residuals)
    runs  = 1 + np.sum(signs[:-1] != signs[1:])
    n_pos = np.sum(signs > 0); n_neg = np.sum(signs < 0)
    expected_runs = 2 * n_pos * n_neg / n + 1 if n > 1 else 1
    if abs(runs - expected_runs) > 2 and n >= 8:
        st.warning(f"⚠️ Systematic misfit (runs={runs}, expected≈{expected_runs:.1f}) — consider a different model.")
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle(f"Residual Diagnostics — {cat_choice} | {model_choice} | R²={res['R2']:.4f}",
                 fontsize=13, fontweight="bold")
    ax1 = axes[0][0]
    ax1.scatter(t, residuals * 1000, color=COLORS[0], s=60, zorder=5)
    ax1.axhline(0, color="black", linewidth=1.0, linestyle="--")
    if len(outlier_idx) > 0:
        ax1.scatter(t[outlier_idx], residuals[outlier_idx]*1000,
                    color="red", s=100, zorder=6, label="Outlier (|z|>2)")
        ax1.legend(fontsize=9)
    ax1.set_xlabel("Time (min)"); ax1.set_ylabel("Residual (mmol/L)"); ax1.set_title("Residuals vs Time")
    ax2 = axes[0][1]
    ax2.scatter(Ct_pred * 1000, residuals * 1000, color=COLORS[1], s=60, zorder=5)
    ax2.axhline(0, color="black", linewidth=1.0, linestyle="--")
    ax2.set_xlabel("Fitted C (mmol/L)"); ax2.set_ylabel("Residual (mmol/L)"); ax2.set_title("Residuals vs Fitted")
    ax3 = axes[1][0]
    if n >= 3:
        try:
            (osm, osr), (slope_qq, intercept_qq, _) = scipy_stats.probplot(residuals, dist="norm", fit=True)
            ax3.scatter(osm, osr, color=COLORS[2], s=60, zorder=5)
            x_line = np.array([osm.min(), osm.max()])
            ax3.plot(x_line, slope_qq * x_line + intercept_qq, "--", color="black", linewidth=1.2)
            sw_label = f"W={sw_stat:.3f}, p={sw_p:.3f}" if not np.isnan(sw_p) else ""
            ax3.set_title(f"Normal Q-Q Plot\n{sw_label}")
        except Exception:
            ax3.set_title("Normal Q-Q (error)")
    ax3.set_xlabel("Theoretical Quantiles"); ax3.set_ylabel("Sample Quantiles")
    ax4 = axes[1][1]
    bar_colors = ["red" if abs(z) > 2 else COLORS[3] for z in std_resid]
    ax4.bar(t, std_resid, color=bar_colors, edgecolor="black", linewidth=0.6,
            width=(t[-1] - t[0]) / (len(t) * 1.5 + 1))
    ax4.axhline(2,  color="red", linewidth=1.0, linestyle="--", label="±2σ")
    ax4.axhline(-2, color="red", linewidth=1.0, linestyle="--")
    ax4.axhline(0,  color="black", linewidth=0.8)
    ax4.set_xlabel("Time (min)"); ax4.set_ylabel("Standardised Residual")
    ax4.set_title("Standardised Residuals"); ax4.legend(fontsize=9)
    fig.tight_layout(); st.pyplot(fig); plt.close(fig)
    with st.expander("📊 Compare all models", expanded=False):
        all_res_full = _fit_nonlinear(t, Ct_obs, C0)
        comp_rows = []
        for m in model_names:
            mr = all_res_full[m]
            if mr.get("R2",-999) <= -999:
                comp_rows.append({"Model": m, "R²": "fail", "AICc": "fail", "AIC": "fail",
                                  "RMSE (mmol/L)": "fail", "Shapiro-Wilk p": "fail"}); continue
            pred_m  = mr["pred"]; resid_m = Ct_obs - pred_m
            rmse_m  = np.sqrt(np.mean(resid_m**2))
            sw_p_m  = float("nan")
            if len(resid_m) >= 3:
                try: _, sw_p_m = scipy_stats.shapiro(resid_m)
                except Exception: pass
            signs_m = np.sign(resid_m); runs_m = 1 + np.sum(signs_m[:-1] != signs_m[1:])
            comp_rows.append({"Model": m, "R²": mr["R2"], "Adj-R²": mr.get("adj_r2",float("nan")),
                              "AICc": mr.get("aicc",float("nan")), "AIC": mr.get("aic",float("nan")),
                              "RMSE (mmol/L)": round(rmse_m * 1000, 5),
                              "Shapiro-Wilk p": round(sw_p_m,4) if not np.isnan(sw_p_m) else "N/A",
                              "Runs": runs_m})
        st.dataframe(pd.DataFrame(comp_rows), use_container_width=True)
    resid_df = pd.DataFrame({"Time (min)": t, "C_obs (mol/L)": Ct_obs,
                              "C_fitted (mol/L)": Ct_pred, "Residual (mol/L)": residuals,
                              "Standardised Residual": std_resid})
    csv_buf = io.StringIO(); resid_df.to_csv(csv_buf, index=False)
    st.download_button("⬇️ Download residuals (.csv)", csv_buf.getvalue(),
                       f"residuals_{cat_choice}_{model_choice}.csv", "text/csv")


# ================================================================
# MAIN APP
# ================================================================
def main():
    st.markdown("""
<div style='background:linear-gradient(90deg,#1a1a2e,#16213e);
            padding:18px 24px;border-radius:10px;margin-bottom:16px'>
  <h2 style='color:#e0e0e0;margin:0'>🔬 ODS Calculation Suite
    <span style='font-size:0.6em;color:#aaa'> v3.5.3 — CatLab-Tools</span></h2>
  <p style='color:#aaa;margin:4px 0 0'>
    Oxidative Desulfurization Kinetics &amp; Analysis |
    Author: Hoda Jafari |
    <a href='https://github.com/Hj1308/CatLab-Tools' style='color:#7eb8f7'>GitHub</a>
  </p>
</div>
    """, unsafe_allow_html=True)
    cfg      = _sidebar_settings()
    uploaded = _shared_uploader()
    tabs = st.tabs([
        "📈 Kinetic Fitting",
        "📉 Linearization",
        "♻️ Removal",
        "⚗️ TON/TOF",
        "🔬 Parameter Effect",
        "🧪 Oxidant Efficiency",
        "📊 Comparison",
        "🌡️ Arrhenius",
        "🔍 Residuals",
    ])
    with tabs[0]: _tab_kinetics(cfg, uploaded)
    with tabs[1]: _tab_linearization(cfg, uploaded)
    with tabs[2]: _tab_removal(cfg, uploaded)
    with tabs[3]: _tab_ton_tof(cfg, uploaded)
    with tabs[4]: _tab_parameter_effect(cfg, uploaded)
    with tabs[5]: _tab_oxidant_efficiency(cfg, uploaded)
    with tabs[6]: _tab_comparison(cfg)
    with tabs[7]: _tab_arrhenius(cfg)
    with tabs[8]: _tab_residuals(cfg, uploaded)

if __name__ == "__main__":
    main()
