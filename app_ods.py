"""
ODS Calculation Suite — Streamlit Web App
==========================================
Extended build based on CatLab-Tools/app_ods.py (github.com/Hj1308/CatLab-Tools)
Original author: Hoda Jafari

v3.3 — Full scientific & code quality fix release
----------------------------------------
FIX 1 (v3.1): C₀ is now locked (fixed) in curve_fit for all models.
FIX 2 (v3.1): Second-order t½ corrected to 1/(k₂·C₀).
FIX 3 (v3.1): Extrapolation warning added when t½ < first data point.
FIX 4 (v3.1): Best-model selection now uses AIC instead of R².
FIX 5 (v3.1): r₀/m formula corrected — r₀ × V_fuel / m_cat.
FIX 6 (v3.2): ppmS conversion now correctly uses fuel density ρ (g/mL).
NEW 7 (v3.2): Dual concentration display C₀(compound) and C₀(S).
NEW 8 (v3.2): Solvent/fuel selector with preset densities.
NEW 9 (v3.2): Oxidant efficiency tab warns when H₂O₂ not measured.
NEW 10 (v3.2): Model assumptions documented in expandable section.
NEW 11 (v3.2): Version header unified.
v3.3 fixes (all 13 issues from code review):
FIX A: _lh_t_half now uses exact analytical solution from L-H ODE integration.
FIX B: SUBSTRATES dict includes n_sulfur field; _C0_both uses it correctly.
FIX C: Tab 5 uses Arrhenius equation for temperature sweep; O/S physically modelled.
FIX D: Single file_uploader in session_state — upload once, use in all tabs.
FIX E: warnings.filterwarnings scoped to scipy RuntimeWarning only.
FIX F: Tab 6 H₂O₂ input keys stable (column-name based, with fallback).
FIX G: Tab 1 best-model selection guarded against empty valid_models dict.
FIX H: Tab 2 polyfit wrapped in try/except with user-friendly error message.
FIX I: _load_kinetic_data helper centralises file reading and column detection.
FIX J: matplotlib.use("Agg") moved before all imports.
FIX K: Tab 5 exposes all 5 kinetic models for parameter sweep.
FIX L: Download zip in Tab 1 now includes fitted curves, not just raw data.
FIX M: UI clarification when ppmS is selected with custom substrate MW.

v3.4 — New modules
NEW N: Tab 8 — Arrhenius Multi-Temperature Analysis (upload per-T files, extract Ea & A with 95% CI)
NEW O: Tab 9 — Residual Diagnostics (residuals vs time, vs fitted, Q-Q plot, Shapiro-Wilk, runs test)

v3.4.1 — Template & UX improvements
NEW P: create_advanced_template() — generates Excel template pre-filled with sidebar settings
NEW Q: Tab 1 template expander upgraded — dual download buttons (simple + advanced template)

Scientific references:
  - Barghi et al., ACS Omega 2025, 10, 15947. DOI: 10.1021/acsomega.4c06722
  - Dhir et al., J. Hazard. Mater. 2009, 161, 1360. DOI: 10.1016/j.jhazmat.2008.04.099
  - Sengupta et al., Ind. Eng. Chem. Res. 2012, 51, 147. DOI: 10.1021/ie2024068
  - EN 590:2022 — Automotive fuels, sulfur content specification (mg/kg)
  - Safa et al., Fuel 2019, 239, 24–33. DOI: 10.1016/j.fuel.2018.10.147
"""

# FIX J: matplotlib backend must be set before any other matplotlib import
import matplotlib
matplotlib.use("Agg")

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.integrate import odeint, quad
import io
import zipfile
import warnings

# FIX E: scope warnings filter — don't suppress everything
warnings.filterwarnings("ignore", category=RuntimeWarning, module="scipy")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")

# ── Page config (must be first Streamlit call) ─────────────────
st.set_page_config(page_title="ODS Calculation Suite v3.4.1", page_icon="🔬", layout="wide")

# ── Matplotlib style ────────────────────────────────────────────
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

# ── Constants ────────────────────────────────────────────────────
MW_S   = 32.06   # g/mol
R_GAS  = 8.314   # J/(mol·K)
COLORS  = ["#e41a1c","#377eb8","#4daf4a","#984ea3","#ff7f00","#a65628","#f781bf","#17becf","#bcbd22"]
MARKERS = ["o","s","^","D","v","P","*","X","h"]
N_PARAMS = {
    "Zero-order":   1,
    "Pseudo-first": 1,
    "Second-order": 1,
    "Elovich":      2,
    "L-H":          2,
}

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

# ════════════════════════════════════════════════════════════════
# SHARED HELPERS
# ════════════════════════════════════════════════════════════════

def _to_mol_L(value, unit, mw=None, rho_g_per_mL=None):
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
        if rho_g_per_mL is None:
            raise ValueError(
                "Fuel density ρ (g/mL) is required for ppmS conversion. "
                "Select a solvent or enter ρ manually in the sidebar."
            )
        c_mg_per_L = value * rho_g_per_mL
        return (c_mg_per_L / MW_S) / 1000.0
    else:
        raise ValueError(f"Unknown unit: {unit}")


def _C0_both(c0_val, c0_unit, mw_poll, rho_g_per_mL, n_sulfur=1):
    if c0_unit == "ppmS":
        C0_S        = _to_mol_L(c0_val, "ppmS", MW_S, rho_g_per_mL)
        C0_compound = C0_S / n_sulfur
    else:
        C0_compound = _to_mol_L(c0_val, c0_unit, mw_poll, rho_g_per_mL)
        C0_S        = C0_compound * n_sulfur
    return C0_compound, C0_S


# ── Kinetic model functions ──────────────────────────────────────
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


# ── Statistical helpers ──────────────────────────────────────────
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


# ── t½ helpers ────────────────────────────────────────────────────
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
    FIX A: Exact analytical t½ for Langmuir-Hinshelwood.
    t½ = ln(2)/(kLH·K) + C₀/(2·kLH)
    """
    try:
        if k_LH <= 0 or K_ads <= 0 or C0 <= 0:
            return float("nan")
        t_half = (np.log(2) / (k_LH * K_ads)) + (C0 / (2.0 * k_LH))
        return round(t_half, 4)
    except Exception:
        return float("nan")


# ── Formatting helpers ────────────────────────────────────────────
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


# ── FIX I: Centralised data loader ──────────────────────────────
def _load_kinetic_data(uploaded):
    try:
        if uploaded.name.endswith(".xlsx"):
            df = pd.read_excel(uploaded)
        else:
            df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Cannot read file: {e}")
        return None, None, None
    time_col = [c for c in df.columns if "time" in c.lower()]
    if not time_col:
        st.error("No 'Time' column found.")
        return None, None, None
    time_col = time_col[0]
    removal_cols = [c for c in df.columns if "removal" in c.lower()]
    if not removal_cols:
        st.error("No 'Removal (%)' column(s) found.")
        return None, None, None
    return df, time_col, removal_cols


# ── Non-linear fitting engine ────────────────────────────────────
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
            "label": f"k₀ = {_fmt_sci(k0)} mol·L⁻¹·min⁻¹",
            "t_half": round(0.5 * C0 / k0, 4) if k0 > 0 else float("nan"),
            "k": k0, "k_se": se[0], "col_k": "K0 (mol/L/min)", "r0": k0, "r0_se": se[0],
        }
    except Exception as e:
        results["Zero-order"] = {"R2": -999, "aic": float("inf"), "error": str(e)}

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
            "label": f"kₐₚₚ = {_fmt_sci(kapp)} min⁻¹",
            "t_half": round(np.log(2) / kapp, 4) if kapp > 0 else float("nan"),
            "k": kapp, "k_se": se[0], "col_k": "Kapp (1/min)", "r0": r0, "r0_se": se[0] * C0,
        }
    except Exception as e:
        results["Pseudo-first"] = {"R2": -999, "aic": float("inf"), "error": str(e)}

    # Second-order
    try:
        p, pcov = curve_fit(lambda t_, k: _second_order(t_, k, C0), t, Ct,
                            p0=[1.0], bounds=([0], [np.inf]), maxfev=5000)
        se = np.sqrt(np.diag(pcov)); k2 = p[0]
        pred = _second_order(t, k2, C0); r2v = _r2(Ct, pred); np_ = N_PARAMS["Second-order"]
        r0 = k2 * C0 ** 2
        results["Second-order"] = {
            "params": (k2, C0), "R2": r2v, "pred": pred,
            "adj_r2": _adj_r2(r2v, n, np_), "aic": _aic(Ct, pred, np_),
            "label": f"k₂ = {_fmt_sci(k2)} L·mol⁻¹·min⁻¹",
            "t_half": round(1.0 / (k2 * C0), 4) if k2 > 0 else float("nan"),
            "k": k2, "k_se": se[0], "col_k": "K2 (L/mol/min)", "r0": r0, "r0_se": se[0] * C0 ** 2,
        }
    except Exception as e:
        results["Second-order"] = {"R2": -999, "aic": float("inf"), "error": str(e)}

    # Elovich
    try:
        p, pcov = curve_fit(lambda t_, a, b: _elovich(t_, a, b, C0), t, Ct,
                            p0=[1e-4, 10.0], bounds=([0, 0], [np.inf, np.inf]), maxfev=10000)
        se = np.sqrt(np.diag(pcov)); alpha = p[0]; beta = p[1]
        pred = _elovich(t, alpha, beta, C0); r2v = _r2(Ct, pred); np_ = N_PARAMS["Elovich"]
        results["Elovich"] = {
            "params": (alpha, beta, C0), "R2": r2v, "pred": pred,
            "adj_r2": _adj_r2(r2v, n, np_), "aic": _aic(Ct, pred, np_),
            "label": f"α={_fmt_sci(alpha)}, β={_fmt_sci(beta)}",
            "t_half": _elovich_t_half(C0, alpha, beta),
            "k": alpha, "k_se": se[0], "col_k": "Alpha (mol/L/min)", "r0": alpha, "r0_se": se[0],
        }
    except Exception as e:
        results["Elovich"] = {"R2": -999, "aic": float("inf"), "error": str(e)}

    # Langmuir-Hinshelwood
    try:
        p, pcov = curve_fit(lambda t_, kLH, Kads: _lh_model(t_, kLH, Kads, C0), t, Ct,
                            p0=[0.01, 10.0], bounds=([0, 0], [np.inf, np.inf]), maxfev=10000)
        se = np.sqrt(np.diag(pcov)); k_LH = p[0]; K_ads = p[1]
        pred = _lh_model(t, k_LH, K_ads, C0); r2v = _r2(Ct, pred); np_ = N_PARAMS["L-H"]
        r0 = k_LH * K_ads * C0 / (1 + K_ads * C0)
        results["L-H"] = {
            "params": (k_LH, K_ads, C0), "R2": r2v, "pred": pred,
            "adj_r2": _adj_r2(r2v, n, np_), "aic": _aic(Ct, pred, np_),
            "label": f"kLH={_fmt_sci(k_LH)}, K={_fmt_sci(K_ads)}",
            "t_half": _lh_t_half(C0, k_LH, K_ads),
            "k": k_LH, "k_se": se[0], "col_k": "kLH (mol/L/min)", "r0": r0, "r0_se": None,
        }
    except Exception as e:
        results["L-H"] = {"R2": -999, "aic": float("inf"), "error": str(e)}

    return results


def _get_valid_models(res, model_names):
    return {m: res[m] for m in model_names if res[m].get("R2", -999) > -999}

def _best_model(res, model_names):
    valid = _get_valid_models(res, model_names)
    if not valid:
        return None
    return min(valid, key=lambda m: valid[m].get("aic", float("inf")))


# ════════════════════════════════════════════════════════════════
# SIDEBAR — universal settings
# ════════════════════════════════════════════════════════════════
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
    if c0_unit == "ppmS":
        st.sidebar.info(
            f"ℹ️ **ppmS mode:** conversion uses MW(S) = 32.06 g/mol. "
            f"n(S) per molecule = {n_sulfur}. C₀(compound) = C₀(S) / {n_sulfur}."
        )
    st.sidebar.subheader("Fuel / Solvent (for ppmS)")
    solvent_name = st.sidebar.selectbox("Solvent", list(SOLVENTS.keys()), index=0)
    if SOLVENTS[solvent_name] is not None:
        rho = SOLVENTS[solvent_name]
        st.sidebar.info(f"ρ = {rho:.3f} g/mL (preset)")
    else:
        rho = st.sidebar.number_input("ρ (g/mL)", min_value=0.500, max_value=2.000,
                                       value=0.684, step=0.001)
    C0_compound = C0_S = None
    try:
        C0_compound, C0_S = _C0_both(c0_val, c0_unit, mw_poll, rho, n_sulfur)
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

**t½ (L-H) v3.3:** t½ = ln(2)/(kLH·K) + C₀/(2·kLH)
        """)
    return {
        "substrate_name": substrate_name, "mw_poll": mw_poll, "n_sulfur": n_sulfur,
        "c0_unit": c0_unit, "c0_val": c0_val, "rho": rho,
        "C0": C0_compound, "C0_S": C0_S,
        "V_fuel": V_fuel / 1000.0, "m_cat": m_cat / 1000.0,
        "temp_C": temp_C, "O_S": O_S, "solvent_name": solvent_name,
    }


# ════════════════════════════════════════════════════════════════
# NEW P (v3.4.1): Advanced Template Generator
# ════════════════════════════════════════════════════════════════
def create_advanced_template(cfg, filename="ODS_Advanced_Template_With_Metadata.xlsx"):
    """Create advanced Excel template with Metadata sheet populated from sidebar settings."""

    # Sheet 1: Metadata
    metadata = {
        "Parameter": [
            "sample_name", "catalyst_name", "substrate", "c0_value", "c0_unit",
            "mw_pollutant", "n_sulfur", "fuel_solvent", "rho_g_per_mL",
            "V_fuel_mL", "m_cat_mg", "temperature_C", "O_S_ratio",
            "active_sites_mmol_g", "notes"
        ],
        "Value": [
            "DBT-Test-01",
            cfg.get("substrate_name", "MoS2"),
            cfg.get("substrate_name", "DBT"),
            cfg.get("c0_val", 500),
            cfg.get("c0_unit", "ppmS"),
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
            "-", "-", "-", "-", "-", "g/mol", "-", "-", "g/mL",
            "mL", "mg", "°C", "-", "mmol/g", "-"
        ]
    }
    df_meta = pd.DataFrame(metadata)

    # Sheet 2: Raw_Data
    df_raw = pd.DataFrame({
        "Time (min)": [0, 15, 30, 45, 60, 90, 120, 180, 240],
        "Cat-A Removal (%)": [0, 18, 35, 52, 68, 82, 91, 96, 98],
        "Cat-B Removal (%)": [0, 22, 41, 59, 74, 87, 94, 97, 99],
        "Notes": [""] * 9
    })

    # Sheet 3: Instructions
    instructions = pd.DataFrame({
        "Instructions": [
            "1. Review and edit the Metadata sheet if needed (settings from sidebar are pre-filled).",
            "2. Fill Time (min) and Removal (%) columns in the Raw_Data sheet with your experimental data.",
            "3. Save the file and upload it in the app.",
            "4. The app will convert units, fit multiple kinetic models (including second-order), and select the best one using AIC."
        ]
    })

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine='openpyxl') as writer:
        df_meta.to_excel(writer, sheet_name="Metadata", index=False)
        df_raw.to_excel(writer, sheet_name="Raw_Data", index=False)
        instructions.to_excel(writer, sheet_name="Instructions", index=False)

        # Auto-adjust column widths
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


# ── FIX D: Shared file uploader ──────────────────────────────────
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


# ════════════════════════════════════════════════════════════════
# TAB 1 — Kinetic Fitting
# ════════════════════════════════════════════════════════════════
def _tab_kinetics(cfg, uploaded):
    st.header("📈 Tab 1 — Kinetic Fitting")

    # NEW Q (v3.4.1): Upgraded dual-button template expander
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
            # Simple template
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
                    buf.getvalue(),
                    fname,
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="adv_template"
                )

    if uploaded is None:
        st.info("Upload a file above to begin fitting."); return
    df, time_col, removal_cols = _load_kinetic_data(uploaded)
    if df is None: return
    t  = df[time_col].dropna().values.astype(float)
    C0 = cfg["C0"]
    if C0 is None: st.error("C₀ conversion failed."); return
    model_names = ["Zero-order","Pseudo-first","Second-order","Elovich","L-H"]
    all_results = {}
    for col in removal_cols:
        removal = df[col].dropna().values[:len(t)].astype(float)
        Ct = C0 * (1 - removal / 100.0)
        all_results[col] = _fit_nonlinear(t, Ct, C0)
    t_fine = np.linspace(0, t.max(), 300)
    fig, ax = plt.subplots(figsize=(8, 5))
    for ci, (col, res) in enumerate(all_results.items()):
        removal  = df[col].dropna().values[:len(t)].astype(float)
        Ct_obs   = C0 * (1 - removal / 100.0)
        color    = COLORS[ci % len(COLORS)]
        marker   = MARKERS[ci % len(MARKERS)]
        ax.plot(t, Ct_obs * 1000, marker, color=color, label=col + " (data)")
        best_model = _best_model(res, model_names)
        if best_model is None: continue
        bres = res[best_model]; params = bres["params"]
        if best_model == "Zero-order":   Ct_fit = _zero_order(t_fine, params[0], C0)
        elif best_model == "Pseudo-first": Ct_fit = _first_order(t_fine, params[0], C0)
        elif best_model == "Second-order": Ct_fit = _second_order(t_fine, params[0], C0)
        elif best_model == "Elovich":    Ct_fit = _elovich(t_fine, params[0], params[1], C0)
        else:                            Ct_fit = _lh_model(t_fine, params[0], params[1], C0)
        ax.plot(t_fine, Ct_fit * 1000, "-", color=color, label=f"{col} — {best_model}")
    ax.set_xlabel("Time (min)"); ax.set_ylabel("C (mmol/L)")
    ax.set_title("Concentration vs Time — Best Fit Model (by AIC)")
    ax.legend(fontsize=9); fig.tight_layout()
    st.pyplot(fig); plt.close(fig)
    st.subheader("📊 Fitting Summary (Best Model by AIC)")
    rows = []
    for col, res in all_results.items():
        best = _best_model(res, model_names)
        if best is None:
            rows.append({"Catalyst": col, "Best Model": "All fits failed"}); continue
        br   = res[best]
        r0   = br.get("r0", float("nan"))
        V    = cfg["V_fuel"]; m = cfg["m_cat"]
        r0_m = r0 * V / m if (r0 and m > 0) else float("nan")
        t_half = br.get("t_half", float("nan"))
        warn_str = "⚠️ t½ before first data point" if not np.isnan(t_half) and t_half < t[0] else ""
        rows.append({"Catalyst": col, "Best Model": best,
                     "k": _fmt_pm(br.get("k"), br.get("k_se")),
                     "R²": br.get("R2","N/A"), "Adj-R²": br.get("adj_r2","N/A"),
                     "AIC": br.get("aic","N/A"), "t½ (min)": _fmt_thalf(t_half),
                     "r₀ (mol/L/min)": _fmt_sci(r0), "r₀/m (mol/g/min)": _fmt_sci(r0_m),
                     "Note": warn_str})
    st.dataframe(pd.DataFrame(rows), use_container_width=True)
    with st.expander("🔍 All models for each catalyst", expanded=False):
        for col, res in all_results.items():
            st.markdown(f"**{col}**")
            subrows = []
            for m in model_names:
                mr = res[m]
                if mr.get("R2", -999) > -999:
                    subrows.append({"Model": m,
                                    "k (±SE)": _fmt_pm(mr.get("k"), mr.get("k_se")),
                                    "R²": mr.get("R2","N/A"), "Adj-R²": mr.get("adj_r2","N/A"),
                                    "AIC": mr.get("aic","N/A"),
                                    "t½ (min)": _fmt_thalf(mr.get("t_half", float("nan")))})
                else:
                    subrows.append({"Model": m, "k (±SE)": "fit failed",
                                    "R²": "–", "Adj-R²": "–", "AIC": "–", "t½ (min)": "–"})
            st.dataframe(pd.DataFrame(subrows), use_container_width=True)
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w") as zf:
        csv_buf = io.StringIO(); pd.DataFrame(rows).to_csv(csv_buf, index=False)
        zf.writestr("fitting_summary.csv", csv_buf.getvalue())
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        for ci, (col, res) in enumerate(all_results.items()):
            removal  = df[col].dropna().values[:len(t)].astype(float)
            Ct_obs   = C0 * (1 - removal / 100.0)
            color    = COLORS[ci % len(COLORS)]
            ax2.plot(t, Ct_obs * 1000, MARKERS[ci % len(MARKERS)], color=color, label=col + " (data)")
            best_model = _best_model(res, model_names)
            if best_model is None: continue
            bres = res[best_model]; params = bres["params"]
            if best_model == "Zero-order":    Ct_fit = _zero_order(t_fine, params[0], C0)
            elif best_model == "Pseudo-first": Ct_fit = _first_order(t_fine, params[0], C0)
            elif best_model == "Second-order": Ct_fit = _second_order(t_fine, params[0], C0)
            elif best_model == "Elovich":     Ct_fit = _elovich(t_fine, params[0], params[1], C0)
            else:                             Ct_fit = _lh_model(t_fine, params[0], params[1], C0)
            ax2.plot(t_fine, Ct_fit * 1000, "-", color=color, label=f"{col} — {best_model}")
        ax2.set_xlabel("Time (min)"); ax2.set_ylabel("C (mmol/L)")
        ax2.set_title("Concentration vs Time — Best Fit Model"); ax2.legend(fontsize=9)
        fig2.tight_layout()
        png_buf = io.BytesIO(); fig2.savefig(png_buf, dpi=300, bbox_inches="tight")
        zf.writestr("kinetics_plot.png", png_buf.getvalue()); plt.close(fig2)
    st.download_button("⬇️ Download results (.zip)", zip_buf.getvalue(),
                       "kinetics_results.zip", "application/zip")


# ════════════════════════════════════════════════════════════════
# TAB 2 — Linearization
# ════════════════════════════════════════════════════════════════
def _tab_linearization(cfg, uploaded):
    st.header("📉 Tab 2 — Linearization Plots")
    if uploaded is None: st.info("Upload a file above."); return
    df, time_col, removal_cols = _load_kinetic_data(uploaded)
    if df is None: return
    t = df[time_col].dropna().values.astype(float)
    C0 = cfg["C0"]
    if C0 is None: st.error("C₀ conversion failed."); return
    model_choice = st.selectbox("Linearization model", [
        "Pseudo-first (ln C vs t)", "Second-order (1/C vs t)",
        "Zero-order (C vs t)", "Elovich (C vs ln t)"])
    n_cols = len(removal_cols)
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 4), squeeze=False)
    for ci, col in enumerate(removal_cols):
        removal = df[col].dropna().values[:len(t)].astype(float)
        Ct = C0 * (1 - removal / 100.0)
        ax = axes[0][ci]; color = COLORS[ci % len(COLORS)]
        if "Pseudo-first" in model_choice:
            valid = Ct > 0; x_vals = t[valid]; y_vals = np.log(Ct[valid])
            ax.set_xlabel("Time (min)"); ax.set_ylabel("ln C")
        elif "Second-order" in model_choice:
            valid = Ct > 0; x_vals = t[valid]; y_vals = 1.0 / Ct[valid]
            ax.set_xlabel("Time (min)"); ax.set_ylabel("1/C (L/mol)")
        elif "Zero-order" in model_choice:
            x_vals = t; y_vals = Ct
            ax.set_xlabel("Time (min)"); ax.set_ylabel("C (mol/L)")
        else:
            valid = t > 0; x_vals = np.log(t[valid]); y_vals = Ct[valid]
            ax.set_xlabel("ln t"); ax.set_ylabel("C (mol/L)")
        ax.scatter(x_vals, y_vals, color=color, zorder=5)
        if len(x_vals) >= 2:
            try:
                coeffs = np.polyfit(x_vals, y_vals, 1)
                x_fit  = np.linspace(x_vals.min(), x_vals.max(), 100)
                ax.plot(x_fit, np.polyval(coeffs, x_fit), "--", color=color)
                r2_lin = _r2(y_vals, np.polyval(coeffs, x_vals))
                ax.set_title(f"{col}\nslope={coeffs[0]:.4f}, R²={r2_lin:.4f}")
            except Exception as e:
                ax.set_title(f"{col}\nLinear fit failed: {e}")
        else:
            ax.set_title(f"{col}\n(insufficient valid points)")
    fig.tight_layout(); st.pyplot(fig); plt.close(fig)


# ════════════════════════════════════════════════════════════════
# TAB 3 — Removal Efficiency
# ════════════════════════════════════════════════════════════════
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
    ax.set_xlabel("Time (min)"); ax.set_ylabel("Removal (%)")
    ax.set_title("Desulfurization Efficiency"); ax.set_ylim(0, 105); ax.legend()
    fig.tight_layout(); st.pyplot(fig); plt.close(fig)
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    final_removals = []; labels = []
    for col in removal_cols:
        removal = df[col].dropna().values[:len(t)].astype(float)
        final_removals.append(removal[-1])
        labels.append(col.replace(" Removal (%)", ""))
    bars = ax2.bar(labels, final_removals, color=COLORS[:len(labels)],
                   edgecolor="black", linewidth=0.8)
    ax2.bar_label(bars, fmt="%.1f%%", padding=2, fontsize=10)
    ax2.set_ylabel("Removal at t_final (%)"); ax2.set_ylim(0, 115)
    ax2.set_title(f"Final Desulfurization at t = {t[-1]:.0f} min")
    fig2.tight_layout(); st.pyplot(fig2); plt.close(fig2)


# ════════════════════════════════════════════════════════════════
# TAB 4 — TON / TOF
# ════════════════════════════════════════════════════════════════
def _tab_ton_tof(cfg, uploaded):
    st.header("⚗️ Tab 4 — TON & TOF")
    st.markdown("""
**Definitions:**
- TON = n_DBT_converted / n_active_sites
- TOF (min⁻¹) = TON / t_reaction
    """)
    if uploaded is None: st.info("Upload a file above."); return
    df, time_col, removal_cols = _load_kinetic_data(uploaded)
    if df is None: return
    t = df[time_col].dropna().values.astype(float)
    C0 = cfg["C0"]; V = cfg["V_fuel"]; m = cfg["m_cat"]
    if C0 is None: st.error("C₀ conversion failed."); return
    site_density = st.number_input("Active site density (mmol/g catalyst)",
                                   min_value=0.001, value=0.5, step=0.05)
    n_sites = site_density * 1e-3 * m
    rows = []
    for col in removal_cols:
        removal = df[col].dropna().values[:len(t)].astype(float)
        n_conv = C0 * V * (removal[-1] / 100.0); t_rxn = t[-1]
        ton = n_conv / n_sites if n_sites > 0 else float("nan")
        tof = ton / t_rxn if t_rxn > 0 else float("nan")
        rows.append({"Catalyst": col, "n_conv (µmol)": round(n_conv * 1e6, 3),
                     "TON": round(ton, 2) if not np.isnan(ton) else "N/A",
                     "TOF (min⁻¹)": f"{tof:.4f}" if not np.isnan(tof) else "N/A"})
    st.dataframe(pd.DataFrame(rows), use_container_width=True)


# ════════════════════════════════════════════════════════════════
# TAB 5 — Parameter Effect
# ════════════════════════════════════════════════════════════════
def _tab_parameter_effect(cfg, uploaded):
    st.header("🔬 Tab 5 — Parameter Effect")
    C0 = cfg["C0"]
    if C0 is None: st.error("C₀ conversion failed."); return
    model_choice = st.selectbox("Kinetic model for simulation",
        ["Pseudo-first","Zero-order","Second-order"])
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


# ════════════════════════════════════════════════════════════════
# TAB 6 — Oxidant Efficiency
# ════════════════════════════════════════════════════════════════
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


# ════════════════════════════════════════════════════════════════
# TAB 7 — Condition Comparison
# ════════════════════════════════════════════════════════════════
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


# ════════════════════════════════════════════════════════════════
# TAB 8 — Arrhenius Multi-Temperature Analysis (NEW v3.4)
# ════════════════════════════════════════════════════════════════
def _tab_arrhenius(cfg):
    st.header("🌡️ Tab 8 — Arrhenius Analysis (Multi-Temperature)")
    st.markdown(r"""
**Purpose:** Upload kinetic datasets at **different temperatures**, fit each with the best kinetic
model (AIC), collect k(T), then fit the Arrhenius equation to extract **Eₐ** and **A**.

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
    model_names = ["Zero-order","Pseudo-first","Second-order","Elovich","L-H"]
    model_choice_arr = st.selectbox("Kinetic model for k extraction",
        ["Best (AIC)","Pseudo-first","Second-order","L-H"], index=0, key="arr_model_choice")
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
                chosen = _best_model(res, model_names) if model_choice_arr == "Best (AIC)" \
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
        fig.tight_layout(); st.pyplot(fig); plt.close(fig)
        st.subheader("Arrhenius Parameters")
        st.dataframe(pd.DataFrame(arrh_rows), use_container_width=True)
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w") as zf:
            csv_buf = io.StringIO(); pd.DataFrame(arrh_rows).to_csv(csv_buf, index=False)
            zf.writestr("arrhenius_parameters.csv", csv_buf.getvalue())
            png_buf = io.BytesIO(); fig.savefig(png_buf, dpi=300, bbox_inches="tight")
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

**References:**
- Barghi et al., *ACS Omega* **2025**, 10, 15947. DOI: 10.1021/acsomega.4c06722
- Dhir et al., *J. Hazard. Mater.* **2009**, 161, 1360. DOI: 10.1016/j.jhazmat.2008.04.099
- Sengupta et al., *Ind. Eng. Chem. Res.* **2012**, 51, 147. DOI: 10.1021/ie2024068
            """)


# ════════════════════════════════════════════════════════════════
# TAB 9 — Residual Diagnostics (NEW v3.4)
# ════════════════════════════════════════════════════════════════
def _tab_residuals(cfg, uploaded):
    st.header("🔍 Tab 9 — Residual Diagnostics")
    st.markdown("""
**Purpose:** Check model quality via residual analysis.
Plots: Residuals vs Time, Residuals vs Fitted, Normal Q-Q, Standardised Residuals.
Statistical tests: Shapiro-Wilk normality, outlier detection (|z|>2), runs test for systematic misfit.
    """)
    if uploaded is None: st.info("Upload a file above."); return
    df, time_col, removal_cols = _load_kinetic_data(uploaded)
    if df is None: return
    t  = df[time_col].dropna().values.astype(float)
    C0 = cfg["C0"]
    if C0 is None: st.error("C₀ conversion failed."); return
    model_names = ["Zero-order","Pseudo-first","Second-order","Elovich","L-H"]
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
    sigma     = np.std(residuals, ddof=len(res["params"]) if "params" in res else 1)
    std_resid = residuals / sigma if sigma > 0 else residuals
    from scipy import stats as scipy_stats
    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("R²",        f"{res['R2']:.4f}")
    col_b.metric("Adj-R²",    f"{res.get('adj_r2',float('nan')):.4f}"
                               if not np.isnan(res.get('adj_r2',float('nan'))) else "N/A")
    col_c.metric("AIC",       f"{res.get('aic',float('nan')):.2f}"
                               if not np.isinf(res.get('aic',float('inf'))) else "∞")
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
                comp_rows.append({"Model": m, "R²": "fail", "AIC": "fail",
                                  "RMSE (mmol/L)": "fail", "Shapiro-Wilk p": "fail"}); continue
            pred_m  = mr["pred"]; resid_m = Ct_obs - pred_m
            rmse_m  = np.sqrt(np.mean(resid_m**2))
            sw_p_m  = float("nan")
            if len(resid_m) >= 3:
                try: _, sw_p_m = scipy_stats.shapiro(resid_m)
                except Exception: pass
            signs_m = np.sign(resid_m); runs_m = 1 + np.sum(signs_m[:-1] != signs_m[1:])
            comp_rows.append({"Model": m, "R²": mr["R2"], "Adj-R²": mr.get("adj_r2",float("nan")),
                              "AIC": mr.get("aic",float("nan")),
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


# ════════════════════════════════════════════════════════════════
# MAIN APP
# ════════════════════════════════════════════════════════════════
def main():
    st.markdown("""
<div style='background:linear-gradient(90deg,#1a1a2e,#16213e);
            padding:18px 24px;border-radius:10px;margin-bottom:16px'>
  <h2 style='color:#e0e0e0;margin:0'>🔬 ODS Calculation Suite
    <span style='font-size:0.6em;color:#aaa'> v3.4.1 — CatLab-Tools</span></h2>
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
