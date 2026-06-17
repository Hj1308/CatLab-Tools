"""
ODS Calculation Suite — Streamlit Web App
==========================================
Extended build based on CatLab-Tools/app_ods.py (github.com/Hj1308/CatLab-Tools)
Original author: Hoda Jafari

v3.2 — Scientific fix release on top of v3.1
----------------------------------------
FIX 1 (v3.1): C₀ is now locked (fixed) in curve_fit for all models.
FIX 2 (v3.1): Second-order t½ corrected to 1/(k₂·C₀).
FIX 3 (v3.1): Extrapolation warning added when t½ < first data point.
FIX 4 (v3.1): Best-model selection now uses AIC instead of R².
FIX 5 (v3.1): r₀/m formula corrected — r₀ × V_fuel / m_cat.

NEW v3.2 — Concentration unit scientific overhaul
----------------------------------------
FIX 6: ppmS conversion now correctly uses fuel density ρ (g/mL).
        ppmS = mg S / kg fuel (mass-based, per EN590/ASTM D5453 standard).
        Correct formula: C₀(mol/L) = ppmS × ρ(g/mL) / MW_S / 1000
        Previous code treated ppmS as mg S/L (missing density), causing
        systematic error of factor ρ (~30% for n-heptane, ~15% for diesel).
NEW 7:  Dual concentration display: C₀(DBT, mol/L) and C₀(S, mol/L) shown
        simultaneously in sidebar — compound-based vs sulfur-based reporting.
NEW 8:  Solvent/fuel selector with preset densities for common ODS media
        (n-heptane, n-hexane, n-octane, n-nonane, n-decane, isooctane,
         model diesel, real diesel, custom).
NEW 9:  Oxidant efficiency tab warns when H₂O₂ consumption is not measured
        (stoichiometric assumption makes η uninformative).
NEW 10: Model assumptions explicitly documented in expandable section.
NEW 11: Version header unified to v3.2.

Scientific references:
  - Barghi et al., ACS Omega 2025, 10, 15947. DOI: 10.1021/acsomega.4c06722
  - Dhir et al., J. Hazard. Mater. 2009, 161, 1360. DOI: 10.1016/j.jhazmat.2008.04.099
  - Sengupta et al., Ind. Eng. Chem. Res. 2012, 51, 147. DOI: 10.1021/ie2024068
  - EN 590:2022 — Automotive fuels, sulfur content specification (mg/kg)
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.integrate import odeint
import io
import re
import zipfile
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="ODS Calculation Suite v3.2", page_icon="🔬", layout="wide")

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

MW_S = 32.06
COLORS  = ["#e41a1c","#377eb8","#4daf4a","#984ea3",
           "#ff7f00","#a65628","#f781bf","#17becf","#bcbd22"]
MARKERS = ["o","s","^","D","v","P","*","X","h"]

N_PARAMS = {
    "Zero-order":   1,
    "Pseudo-first": 1,
    "Second-order": 1,
    "Elovich":      2,
    "L-H":          2,
}

SUBSTRATES = {
    "DBT (Dibenzothiophene)":        184.26,
    "BT (Benzothiophene)":           134.20,
    "4,6-DMDBT":                     212.31,
    "4-MDBT":                        198.28,
    "Thiophene":                     84.14,
    "Custom / other":                None,
}

# NEW v3.2 — common ODS solvents/fuels with density at ~25°C (g/mL)
# Sources: NIST WebBook, literature ODS studies
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
#  SHARED HELPERS
# ════════════════════════════════════════════════════════════════

def _to_mol_L(value, unit, mw=None, rho_g_per_mL=None):
    """
    Convert concentration to mol/L.

    ppmS definition (v3.2 fix, per EN590 / ASTM D5453):
        ppmS = mg S / kg fuel  (mass-based)
        C [mol/L] = ppmS × ρ [g/mL] / MW_S [g/mol] / 1000
        NOTE: ρ is mandatory for ppmS. Without it the result is wrong
              by a factor of ρ (e.g. 46% error for n-heptane).

    ppm / mg/L definition:
        ppm ≡ mg compound / L solution  (volume-based, dilute organic solution)
        C [mol/L] = ppm / MW [g/mol] / 1000
        No density needed.
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
        # FIX 6 (v3.2): ppmS = mg S / kg fuel → needs fuel density ρ
        if rho_g_per_mL is None:
            raise ValueError(
                "Fuel density ρ (g/mL) is required for ppmS conversion. "
                "Select a solvent or enter ρ manually in the sidebar."
            )
        # mg S/kg × ρ [g/mL] = mg S/L  (since 1 g/mL = 1 kg/L)
        c_mg_per_L = value * rho_g_per_mL
        return (c_mg_per_L / MW_S) / 1000.0
    else:
        raise ValueError(f"Unknown unit: {unit}")


def _C0_both(c0_val, c0_unit, mw_poll, rho_g_per_mL):
    """
    Return (C0_compound_mol_L, C0_S_mol_L) for any input unit.
    C0_compound: molarity of the pollutant molecule (used in kinetic models)
    C0_S:        molarity of sulfur atoms (regulatory / ppmS reporting)
    For mono-S compounds: C0_S = C0_compound (same moles).
    """
    if c0_unit == "ppmS":
        # sulfur-based input → get C_S first, then convert to C_compound
        C0_S = _to_mol_L(c0_val, "ppmS", MW_S, rho_g_per_mL)
        # 1 mol DBT contains 1 mol S → same molarity
        C0_compound = C0_S  # valid for DBT, BT, 4,6-DMDBT (all mono-S)
    else:
        # compound-based input
        C0_compound = _to_mol_L(c0_val, c0_unit, mw_poll, rho_g_per_mL)
        C0_S = C0_compound  # mono-S assumption
    return C0_compound, C0_S


# ── Kinetic model functions ────────────────────────────────────
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
    """
    Langmuir-Hinshelwood: dC/dt = -kLH·K·C/(1+K·C), solved numerically.
    """
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
    try:
        if k_LH <= 0 or K_ads <= 0 or C0 <= 0:
            return float("nan")
        return (np.log(2) / (k_LH * K_ads)) + (C0 / 2.0) / k_LH
    except Exception:
        return float("nan")

def _fmt_sci(val):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    if val == 0:
        return "0"
    if np.isinf(val):
        return "∞"
    exp = int(np.floor(np.log10(abs(val))))
    coef = val / (10 ** exp)
    sup = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")
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
    exp = int(np.floor(np.log10(abs(val)))) if val != 0 else 0
    scale = 10 ** exp
    return f"({val/scale:.2f} ± {se/scale:.2f}) × 10{str(exp).translate(str.maketrans('0123456789-','⁰¹²³⁴⁵⁶⁷⁸⁹⁻'))}"


# ── Non-linear fitting engine ──────────────────────────────────
def _fit_nonlinear(time, Ct, C0):
    t  = np.asarray(time, dtype=float)
    Ct = np.asarray(Ct,   dtype=float)
    n  = len(t)
    results = {}

    # Zero-order
    try:
        p, pcov = curve_fit(
            lambda t_, k: _zero_order(t_, k, C0),
            t, Ct, p0=[1e-6], bounds=([0], [np.inf]), maxfev=5000
        )
        se = np.sqrt(np.diag(pcov))
        k0 = p[0]
        pred = _zero_order(t, k0, C0)
        r2v  = _r2(Ct, pred)
        np_  = N_PARAMS["Zero-order"]
        results["Zero-order"] = {
            "params": (k0, C0), "R2": r2v, "pred": pred,
            "adj_r2": _adj_r2(r2v, n, np_),
            "aic":    _aic(Ct, pred, np_),
            "label":  f"k₀ = {_fmt_sci(k0)} mol·L⁻¹·min⁻¹",
            "t_half": round(0.5 * C0 / k0, 2) if k0 > 0 else float("nan"),
            "k": k0, "k_se": se[0], "col_k": "K0 (mol/L/min)",
            "r0": k0, "r0_se": se[0],
        }
    except Exception as e:
        results["Zero-order"] = {"R2": -999, "aic": float("inf"), "error": str(e)}

    # Pseudo-first-order
    try:
        p, pcov = curve_fit(
            lambda t_, k: _first_order(t_, k, C0),
            t, Ct, p0=[0.01], bounds=([0], [np.inf]), maxfev=5000
        )
        se   = np.sqrt(np.diag(pcov))
        kapp = p[0]
        pred = _first_order(t, kapp, C0)
        r2v  = _r2(Ct, pred)
        np_  = N_PARAMS["Pseudo-first"]
        r0   = kapp * C0
        results["Pseudo-first"] = {
            "params": (kapp, C0), "R2": r2v, "pred": pred,
            "adj_r2": _adj_r2(r2v, n, np_),
            "aic":    _aic(Ct, pred, np_),
            "label":  f"kₐₚₚ = {_fmt_sci(kapp)} min⁻¹",
            "t_half": round(np.log(2) / kapp, 2) if kapp > 0 else float("nan"),
            "k": kapp, "k_se": se[0], "col_k": "Kapp (1/min)",
            "r0": r0, "r0_se": se[0] * C0,
        }
    except Exception as e:
        results["Pseudo-first"] = {"R2": -999, "aic": float("inf"), "error": str(e)}

    # Second-order
    try:
        p, pcov = curve_fit(
            lambda t_, k: _second_order(t_, k, C0),
            t, Ct, p0=[1.0], bounds=([0], [np.inf]), maxfev=5000
        )
        se = np.sqrt(np.diag(pcov))
        k2 = p[0]
        pred = _second_order(t, k2, C0)
        r2v  = _r2(Ct, pred)
        np_  = N_PARAMS["Second-order"]
        r0   = k2 * C0 ** 2
        results["Second-order"] = {
            "params": (k2, C0), "R2": r2v, "pred": pred,
            "adj_r2": _adj_r2(r2v, n, np_),
            "aic":    _aic(Ct, pred, np_),
            "label":  f"k₂ = {_fmt_sci(k2)} L·mol⁻¹·min⁻¹",
            "t_half": round(1.0 / (k2 * C0), 2) if k2 > 0 else float("nan"),
            "k": k2, "k_se": se[0], "col_k": "K2 (L/mol/min)",
            "r0": r0, "r0_se": se[0] * C0 ** 2,
        }
    except Exception as e:
        results["Second-order"] = {"R2": -999, "aic": float("inf"), "error": str(e)}

    # Elovich
    try:
        p, pcov = curve_fit(
            lambda t_, a, b: _elovich(t_, a, b, C0),
            t, Ct, p0=[1e-4, 10.0],
            bounds=([0, 0], [np.inf, np.inf]), maxfev=10000
        )
        se    = np.sqrt(np.diag(pcov))
        alpha = p[0]; beta = p[1]
        pred  = _elovich(t, alpha, beta, C0)
        r2v   = _r2(Ct, pred)
        np_   = N_PARAMS["Elovich"]
        r0    = alpha
        results["Elovich"] = {
            "params": (alpha, beta, C0), "R2": r2v, "pred": pred,
            "adj_r2": _adj_r2(r2v, n, np_),
            "aic":    _aic(Ct, pred, np_),
            "label":  f"α={_fmt_sci(alpha)}, β={_fmt_sci(beta)}",
            "t_half": _elovich_t_half(C0, alpha, beta),
            "k": alpha, "k_se": se[0], "col_k": "Alpha (mol/L/min)",
            "r0": r0, "r0_se": se[0],
        }
    except Exception as e:
        results["Elovich"] = {"R2": -999, "aic": float("inf"), "error": str(e)}

    # Langmuir-Hinshelwood
    try:
        p, pcov = curve_fit(
            lambda t_, kLH, Kads: _lh_model(t_, kLH, Kads, C0),
            t, Ct, p0=[0.01, 10.0],
            bounds=([0, 0], [np.inf, np.inf]), maxfev=10000
        )
        se    = np.sqrt(np.diag(pcov))
        k_LH  = p[0]; K_ads = p[1]
        pred  = _lh_model(t, k_LH, K_ads, C0)
        r2v   = _r2(Ct, pred)
        np_   = N_PARAMS["L-H"]
        r0    = k_LH * K_ads * C0 / (1 + K_ads * C0)
        results["L-H"] = {
            "params": (k_LH, K_ads, C0), "R2": r2v, "pred": pred,
            "adj_r2": _adj_r2(r2v, n, np_),
            "aic":    _aic(Ct, pred, np_),
            "label":  f"kLH={_fmt_sci(k_LH)}, K={_fmt_sci(K_ads)}",
            "t_half": _lh_t_half(C0, k_LH, K_ads),
            "k": k_LH, "k_se": se[0], "col_k": "kLH (mol/L/min)",
            "r0": r0, "r0_se": None,
        }
    except Exception as e:
        results["L-H"] = {"R2": -999, "aic": float("inf"), "error": str(e)}

    return results


# ════════════════════════════════════════════════════════════════
#  SIDEBAR — universal settings (v3.2: solvent + density + dual C₀)
# ════════════════════════════════════════════════════════════════

def _sidebar_settings():
    """
    Returns a dict with all global settings.
    New in v3.2: solvent selector, fuel density ρ, dual C₀ display.
    """
    st.sidebar.title("⚙️ Settings")

    # ── Substrate ──────────────────────────────────────────────
    st.sidebar.subheader("Sulfur Compound")
    substrate_name = st.sidebar.selectbox("Substrate", list(SUBSTRATES.keys()), index=0)
    mw_poll = SUBSTRATES[substrate_name]
    if mw_poll is None:
        mw_poll = st.sidebar.number_input("MW (g/mol)", min_value=1.0, value=184.26, step=0.01)

    # ── Initial concentration ──────────────────────────────────
    st.sidebar.subheader("Initial Concentration C₀")
    c0_unit = st.sidebar.selectbox(
        "Unit",
        ["ppmS", "ppm", "mg/L", "mmol/L", "mol/L", "g/L"],
        index=0,
        help=(
            "ppmS = mg S / kg fuel (mass-based, EN590 standard) — "
            "requires fuel density.\n"
            "ppm / mg/L = mg compound / L solution (volume-based)."
        ),
    )
    c0_val = st.sidebar.number_input("C₀ value", min_value=0.0, value=500.0, step=1.0)

    # ── NEW v3.2: Solvent / fuel density ─────────────────────
    st.sidebar.subheader("Fuel / Solvent (for ppmS)")
    solvent_name = st.sidebar.selectbox(
        "Solvent",
        list(SOLVENTS.keys()),
        index=0,
        help="Used ONLY for ppmS conversion. Ignored for other units.",
    )
    if SOLVENTS[solvent_name] is not None:
        rho = SOLVENTS[solvent_name]
        st.sidebar.info(f"ρ = {rho:.3f} g/mL (preset)")
    else:
        rho = st.sidebar.number_input(
            "ρ (g/mL)", min_value=0.500, max_value=2.000,
            value=0.684, step=0.001,
            help="Fuel density at ~25 °C in g/mL"
        )

    # ── Compute C₀ and show dual display ──────────────────────
    try:
        C0_compound, C0_S = _C0_both(c0_val, c0_unit, mw_poll, rho)

        # NEW v3.2: dual C₀ display
        st.sidebar.markdown("---")
        st.sidebar.markdown("**Converted C₀**")
        col1, col2 = st.sidebar.columns(2)
        col1.metric("C₀ (compound)", f"{C0_compound*1000:.4f} mmol/L")
        col2.metric("C₀ (sulfur)", f"{C0_S*1000:.4f} mmol/L")

        if c0_unit == "ppmS":
            st.sidebar.caption(
                f"Formula: {c0_val} ppmS × {rho} g/mL / {MW_S} g·mol⁻¹ / 1000 "
                f"= {C0_S*1000:.4f} mmol/L (S)"
            )

    except ValueError as err:
        st.sidebar.error(str(err))
        C0_compound = C0_S = None

    # ── Reaction conditions ────────────────────────────────────
    st.sidebar.subheader("Reaction Conditions")
    V_fuel  = st.sidebar.number_input("V fuel (mL)", min_value=0.1, value=10.0, step=0.5)
    m_cat   = st.sidebar.number_input("m catalyst (mg)", min_value=0.01, value=10.0, step=1.0)
    temp_C  = st.sidebar.number_input("Temperature (°C)", value=25.0, step=5.0)
    O_S     = st.sidebar.number_input(
        "O/S molar ratio", min_value=0.1, value=4.0, step=0.5,
        help="Moles of oxidant (H₂O₂) per mole of sulfur at t=0"
    )

    # ── Model assumptions expander (NEW v3.2) ─────────────────
    with st.sidebar.expander("ℹ️ Model Assumptions", expanded=False):
        st.markdown("""
**Kinetic models in this suite assume:**
- Well-mixed (CSTR) isothermal batch reactor
- Catalyst activity constant (no deactivation)
- Negligible internal & external mass-transfer resistance
- Single sulfur compound (or lumped removal %)
- Mono-sulfur stoichiometry (1 mol compound = 1 mol S)

**Fitted k values are therefore APPARENT constants** under the specific
hydrodynamic conditions used. Do not compare k across studies with
different stirring speeds or catalyst particle sizes without verifying
mass-transfer limitations (Mears & Weisz-Prater criteria).

**L-H model limitation:** uses bulk C(t) only; does not account for
surface coverage of intermediates (DBTO, DBTO₂). For a full
microkinetic treatment see: Barghi et al., ACS Omega 2025, 10, 15947.
        """)

    return {
        "substrate_name": substrate_name,
        "mw_poll":        mw_poll,
        "c0_unit":        c0_unit,
        "c0_val":         c0_val,
        "rho":            rho,
        "C0":             C0_compound,
        "C0_S":           C0_S,
        "V_fuel":         V_fuel / 1000.0,   # convert mL → L
        "m_cat":          m_cat / 1000.0,    # convert mg → g
        "temp_C":         temp_C,
        "O_S":            O_S,
        "solvent_name":   solvent_name,
    }


# ════════════════════════════════════════════════════════════════
#  TAB 1 — Kinetic Fitting
# ════════════════════════════════════════════════════════════════

def _tab_kinetics(cfg):
    st.header("📈 Tab 1 — Kinetic Fitting")

    with st.expander("📋 Template & upload instructions", expanded=False):
        st.markdown("""
**Required columns:**
- `Time (min)` — reaction time
- One or more catalyst columns with `Removal (%)` values (0–100)

**Tips:**
- Minimum 4 time points per catalyst for reliable fitting
- Include t=0 (Removal=0) if possible
- Use the download button below to get an example template
        """)
        tmpl = pd.DataFrame({
            "Time (min)": [0, 10, 20, 30, 60, 90, 120],
            "Cat-A Removal (%)": [0, 15, 28, 40, 65, 80, 91],
            "Cat-B Removal (%)": [0, 22, 41, 55, 78, 89, 95],
        })
        buf = io.BytesIO()
        tmpl.to_excel(buf, index=False)
        st.download_button("⬇️ Download template (.xlsx)", buf.getvalue(),
                           "ods_template.xlsx",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    uploaded = st.file_uploader("Upload kinetic data (.xlsx or .csv)", type=["xlsx","csv"])
    if uploaded is None:
        st.info("Upload a file to begin fitting.")
        return

    try:
        df = pd.read_excel(uploaded) if uploaded.name.endswith(".xlsx") else pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Cannot read file: {e}")
        return

    time_col = [c for c in df.columns if "time" in c.lower()]
    if not time_col:
        st.error("No 'Time' column found.")
        return
    time_col = time_col[0]

    removal_cols = [c for c in df.columns if "removal" in c.lower()]
    if not removal_cols:
        st.error("No 'Removal (%)' column(s) found.")
        return

    t = df[time_col].dropna().values.astype(float)
    C0 = cfg["C0"]
    if C0 is None:
        st.error("C₀ conversion failed. Check sidebar settings.")
        return

    # ── fit each catalyst ──────────────────────────────────────
    all_results = {}
    for col in removal_cols:
        removal = df[col].dropna().values[:len(t)].astype(float)
        Ct = C0 * (1 - removal / 100.0)
        all_results[col] = _fit_nonlinear(t, Ct, C0)

    # ── pick best model by AIC ─────────────────────────────────
    model_names = ["Zero-order","Pseudo-first","Second-order","Elovich","L-H"]

    # ── Plot C(t) ──────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    t_fine = np.linspace(0, t.max(), 300)

    for ci, (col, res) in enumerate(all_results.items()):
        removal = df[col].dropna().values[:len(t)].astype(float)
        Ct_obs  = C0 * (1 - removal / 100.0)
        color   = COLORS[ci % len(COLORS)]
        marker  = MARKERS[ci % len(MARKERS)]
        ax.plot(t, Ct_obs * 1000, marker, color=color, label=col + " (data)")

        best_model = min(
            {m: res[m] for m in model_names if res[m].get("R2", -999) > -999},
            key=lambda m: res[m].get("aic", float("inf"))
        )
        bres = res[best_model]
        if "pred" in bres:
            params = bres["params"]
            if best_model == "Zero-order":
                Ct_fit = _zero_order(t_fine, params[0], C0)
            elif best_model == "Pseudo-first":
                Ct_fit = _first_order(t_fine, params[0], C0)
            elif best_model == "Second-order":
                Ct_fit = _second_order(t_fine, params[0], C0)
            elif best_model == "Elovich":
                Ct_fit = _elovich(t_fine, params[0], params[1], C0)
            else:
                Ct_fit = _lh_model(t_fine, params[0], params[1], C0)
            ax.plot(t_fine, Ct_fit * 1000, "-", color=color,
                    label=f"{col} best: {best_model}")

    ax.set_xlabel("Time (min)")
    ax.set_ylabel("C (mmol/L)")
    ax.set_title("Concentration vs Time — Best Fit Model")
    ax.legend(fontsize=9)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # ── Summary table ──────────────────────────────────────────
    st.subheader("📊 Fitting Summary (Best Model by AIC)")
    rows = []
    for col, res in all_results.items():
        best = min(
            {m: res[m] for m in model_names if res[m].get("R2",-999) > -999},
            key=lambda m: res[m].get("aic", float("inf"))
        )
        br = res[best]
        r0   = br.get("r0", float("nan"))
        V    = cfg["V_fuel"]
        m    = cfg["m_cat"]
        r0_m = r0 * V / m if (r0 and m > 0) else float("nan")
        t_half = br.get("t_half", float("nan"))
        first_t = t[0]
        warn_str = ""
        if not np.isnan(t_half) and t_half < first_t:
            warn_str = "⚠️ t½ before first data point"
        rows.append({
            "Catalyst": col,
            "Best Model": best,
            "k": _fmt_pm(br.get("k"), br.get("k_se")),
            "R²": br.get("R2", "N/A"),
            "Adj-R²": br.get("adj_r2","N/A"),
            "AIC": br.get("aic","N/A"),
            "t½ (min)": _fmt_thalf(t_half),
            "r₀ (mol/L/min)": _fmt_sci(r0),
            "r₀/m (mol/g/min)": _fmt_sci(r0_m),
            "Note": warn_str,
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    # ── All models table ───────────────────────────────────────
    with st.expander("🔍 All models for each catalyst", expanded=False):
        for col, res in all_results.items():
            st.markdown(f"**{col}**")
            subrows = []
            for m in model_names:
                mr = res[m]
                if mr.get("R2",-999) > -999:
                    subrows.append({
                        "Model": m,
                        "k (±SE)": _fmt_pm(mr.get("k"), mr.get("k_se")),
                        "R²": mr.get("R2","N/A"),
                        "Adj-R²": mr.get("adj_r2","N/A"),
                        "AIC": mr.get("aic","N/A"),
                        "t½ (min)": _fmt_thalf(mr.get("t_half",float("nan"))),
                    })
                else:
                    subrows.append({"Model": m, "k (±SE)": "fit failed",
                                    "R²":"–","Adj-R²":"–","AIC":"–","t½ (min)":"–"})
            st.dataframe(pd.DataFrame(subrows), use_container_width=True)

    # ── Download zip ───────────────────────────────────────────
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w") as zf:
        # summary CSV
        summary_df = pd.DataFrame(rows)
        csv_buf = io.StringIO()
        summary_df.to_csv(csv_buf, index=False)
        zf.writestr("fitting_summary.csv", csv_buf.getvalue())
        # plot PNG
        fig2, ax2 = plt.subplots(figsize=(8,5))
        for ci,(col,res) in enumerate(all_results.items()):
            removal = df[col].dropna().values[:len(t)].astype(float)
            Ct_obs  = C0*(1-removal/100.0)
            ax2.plot(t, Ct_obs*1000, MARKERS[ci%len(MARKERS)],
                     color=COLORS[ci%len(COLORS)], label=col)
        ax2.set_xlabel("Time (min)"); ax2.set_ylabel("C (mmol/L)")
        ax2.legend()
        png_buf = io.BytesIO(); fig2.savefig(png_buf, dpi=300, bbox_inches="tight")
        zf.writestr("kinetics_plot.png", png_buf.getvalue())
        plt.close(fig2)
    st.download_button("⬇️ Download results (.zip)", zip_buf.getvalue(),
                       "kinetics_results.zip", "application/zip")


# ════════════════════════════════════════════════════════════════
#  TAB 2 — Linearization
# ════════════════════════════════════════════════════════════════

def _tab_linearization(cfg):
    st.header("📉 Tab 2 — Linearization Plots")

    uploaded = st.file_uploader("Upload kinetic data for linearization (.xlsx or .csv)",
                                type=["xlsx","csv"], key="lin_upload")
    if uploaded is None:
        st.info("Upload a file to generate linearization plots.")
        return

    try:
        df = pd.read_excel(uploaded) if uploaded.name.endswith(".xlsx") else pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Cannot read file: {e}"); return

    time_col = [c for c in df.columns if "time" in c.lower()]
    if not time_col:
        st.error("No Time column found."); return
    time_col = time_col[0]
    removal_cols = [c for c in df.columns if "removal" in c.lower()]
    if not removal_cols:
        st.error("No Removal (%) column found."); return

    t   = df[time_col].dropna().values.astype(float)
    C0  = cfg["C0"]
    if C0 is None:
        st.error("C₀ conversion failed."); return

    model_choice = st.selectbox("Linearization model",
        ["Pseudo-first (ln C vs t)", "Second-order (1/C vs t)",
         "Zero-order (C vs t)", "Elovich (C vs ln t)"])

    fig, axes = plt.subplots(1, len(removal_cols),
                             figsize=(5*len(removal_cols), 4), squeeze=False)

    for ci, col in enumerate(removal_cols):
        removal = df[col].dropna().values[:len(t)].astype(float)
        Ct      = C0 * (1 - removal/100.0)
        ax      = axes[0][ci]
        color   = COLORS[ci % len(COLORS)]

        if "Pseudo-first" in model_choice:
            valid = Ct > 0
            x_vals = t[valid]; y_vals = np.log(Ct[valid])
            ax.set_xlabel("Time (min)"); ax.set_ylabel("ln C")
        elif "Second-order" in model_choice:
            valid = Ct > 0
            x_vals = t[valid]; y_vals = 1.0/Ct[valid]
            ax.set_xlabel("Time (min)"); ax.set_ylabel("1/C (L/mol)")
        elif "Zero-order" in model_choice:
            x_vals = t; y_vals = Ct
            ax.set_xlabel("Time (min)"); ax.set_ylabel("C (mol/L)")
        else:  # Elovich
            valid = t > 0
            x_vals = np.log(t[valid]); y_vals = Ct[valid]
            ax.set_xlabel("ln t"); ax.set_ylabel("C (mol/L)")

        ax.scatter(x_vals, y_vals, color=color, zorder=5)
        if len(x_vals) >= 2:
            coeffs = np.polyfit(x_vals, y_vals, 1)
            x_fit  = np.linspace(x_vals.min(), x_vals.max(), 100)
            y_fit  = np.polyval(coeffs, x_fit)
            ax.plot(x_fit, y_fit, "--", color=color)
            r2_lin = _r2(y_vals, np.polyval(coeffs, x_vals))
            ax.set_title(f"{col}\nslope={coeffs[0]:.4f}, R²={r2_lin:.4f}")
        else:
            ax.set_title(col)

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# ════════════════════════════════════════════════════════════════
#  TAB 3 — Removal Efficiency
# ════════════════════════════════════════════════════════════════

def _tab_removal(cfg):
    st.header("♻️ Tab 3 — Removal Efficiency")

    uploaded = st.file_uploader("Upload removal data (.xlsx or .csv)",
                                type=["xlsx","csv"], key="rem_upload")
    if uploaded is None:
        st.info("Upload a file.")
        return

    try:
        df = pd.read_excel(uploaded) if uploaded.name.endswith(".xlsx") else pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Cannot read file: {e}"); return

    time_col = [c for c in df.columns if "time" in c.lower()]
    if not time_col:
        st.error("No Time column."); return
    time_col = time_col[0]
    removal_cols = [c for c in df.columns if "removal" in c.lower()]
    if not removal_cols:
        st.error("No Removal columns."); return

    t = df[time_col].dropna().values.astype(float)

    fig, ax = plt.subplots(figsize=(8,5))
    for ci, col in enumerate(removal_cols):
        removal = df[col].dropna().values[:len(t)].astype(float)
        ax.plot(t, removal, MARKERS[ci%len(MARKERS)]+"-",
                color=COLORS[ci%len(COLORS)], label=col)
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Removal (%)")
    ax.set_title("Desulfurization Efficiency")
    ax.set_ylim(0, 105)
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # bar chart at last time point
    fig2, ax2 = plt.subplots(figsize=(6,4))
    final_removals = []
    labels         = []
    for col in removal_cols:
        removal = df[col].dropna().values[:len(t)].astype(float)
        final_removals.append(removal[-1])
        labels.append(col.replace(" Removal (%)",""))
    bars = ax2.bar(labels, final_removals,
                   color=COLORS[:len(labels)], edgecolor="black", linewidth=0.8)
    ax2.bar_label(bars, fmt="%.1f%%", padding=2, fontsize=10)
    ax2.set_ylabel("Removal at t_final (%)")
    ax2.set_title(f"Final Desulfurization at t = {t[-1]:.0f} min")
    ax2.set_ylim(0, 115)
    fig2.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)


# ════════════════════════════════════════════════════════════════
#  TAB 4 — TON / TOF
# ════════════════════════════════════════════════════════════════

def _tab_ton_tof(cfg):
    st.header("⚗️ Tab 4 — TON & TOF")

    st.markdown("""
**Definitions used here:**
- TON (Turnover Number) = n_DBT_converted / n_active_sites
- TOF (Turnover Frequency, min⁻¹) = TON / t_reaction
- Active sites estimated from catalyst mass and assumed site density (user input)
    """)

    uploaded = st.file_uploader("Upload kinetic data (.xlsx/.csv)",
                                type=["xlsx","csv"], key="ton_upload")
    if uploaded is None:
        st.info("Upload a file."); return

    try:
        df = pd.read_excel(uploaded) if uploaded.name.endswith(".xlsx") else pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"{e}"); return

    time_col = [c for c in df.columns if "time" in c.lower()]
    if not time_col:
        st.error("No Time column."); return
    time_col = time_col[0]
    removal_cols = [c for c in df.columns if "removal" in c.lower()]
    if not removal_cols:
        st.error("No Removal columns."); return

    t   = df[time_col].dropna().values.astype(float)
    C0  = cfg["C0"]
    V   = cfg["V_fuel"]
    m   = cfg["m_cat"]
    if C0 is None:
        st.error("C₀ conversion failed."); return

    site_density = st.number_input(
        "Active site density (mmol/g catalyst)",
        min_value=0.001, value=0.5, step=0.05,
        help="From BET/CO-chemisorption/NH₃-TPD — see catalyst characterisation"
    )
    n_sites = site_density * 1e-3 * m   # mol active sites

    rows = []
    for col in removal_cols:
        removal  = df[col].dropna().values[:len(t)].astype(float)
        n_conv   = C0 * V * (removal[-1]/100.0)   # mol DBT converted at t_final
        t_rxn    = t[-1]
        ton      = n_conv / n_sites if n_sites > 0 else float("nan")
        tof      = ton / t_rxn if t_rxn > 0 else float("nan")
        rows.append({
            "Catalyst": col,
            "n_conv (µmol)": round(n_conv*1e6, 3),
            "TON": round(ton, 2) if not np.isnan(ton) else "N/A",
            "TOF (min⁻¹)": f"{tof:.4f}" if not np.isnan(tof) else "N/A",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True)


# ════════════════════════════════════════════════════════════════
#  TAB 5 — Parameter Effect
# ════════════════════════════════════════════════════════════════

def _tab_parameter_effect(cfg):
    st.header("🔬 Tab 5 — Parameter Effect")

    st.markdown("""
Simulate how changing reaction parameters affects the **pseudo-first-order**
removal curve (C vs t). Select a parameter to sweep.
    """)

    C0   = cfg["C0"]
    if C0 is None:
        st.error("C₀ conversion failed."); return

    param = st.selectbox("Parameter to sweep",
        ["Initial concentration C₀", "Catalyst mass m (g)",
         "Temperature (proxy: k multiplier)", "O/S ratio (proxy: k multiplier)"])

    kapp_base = st.number_input("Base kapp (min⁻¹)", min_value=1e-6, value=0.05,
                                 format="%.5f")
    t_max = st.number_input("t_max (min)", min_value=10, value=120, step=10)
    t_arr = np.linspace(0, t_max, 300)

    fig, ax = plt.subplots(figsize=(8, 5))

    if param == "Initial concentration C₀":
        factors = [0.25, 0.5, 1.0, 2.0, 4.0]
        for f in factors:
            C0_v = C0 * f
            Ct   = _first_order(t_arr, kapp_base, C0_v)
            ax.plot(t_arr, Ct*1000, label=f"C₀×{f} = {C0_v*1000:.3f} mmol/L")
        ax.set_ylabel("C (mmol/L)")
    elif param == "Catalyst mass m (g)":
        m_vals = [cfg["m_cat"]*f for f in [0.5, 1.0, 2.0, 3.0, 5.0]]
        for mv in m_vals:
            kapp_v = kapp_base * mv / cfg["m_cat"]
            Ct     = _first_order(t_arr, kapp_v, C0)
            rem    = (1 - Ct/C0)*100
            ax.plot(t_arr, rem, label=f"m = {mv*1000:.0f} mg")
        ax.set_ylabel("Removal (%)")
    else:
        multipliers = [0.5, 0.75, 1.0, 1.5, 2.0]
        lbl = "T multiplier" if "Temperature" in param else "O/S multiplier"
        for mf in multipliers:
            Ct  = _first_order(t_arr, kapp_base*mf, C0)
            rem = (1 - Ct/C0)*100
            ax.plot(t_arr, rem, label=f"{lbl} = {mf}")
        ax.set_ylabel("Removal (%)")

    ax.set_xlabel("Time (min)")
    ax.set_title(f"Effect of: {param}")
    ax.legend(fontsize=9)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# ════════════════════════════════════════════════════════════════
#  TAB 6 — Oxidant Efficiency  (v3.2: warning for unmeasured H₂O₂)
# ════════════════════════════════════════════════════════════════

def _tab_oxidant_efficiency(cfg):
    st.header("🧪 Tab 6 — Oxidant Efficiency")

    st.markdown("""
**Oxidant efficiency (η):**
$$\\eta = \\frac{n_{\\text{DBT,removed}}}{n_{\\text{H}_2\\text{O}_2,\\text{consumed}} / 2}$$

Stoichiometry: DBT + 2 H₂O₂ → DBTO₂ + 2 H₂O  (theoretical max η = 100%)
    """)

    # NEW v3.2: explicit warning about stoichiometric assumption
    measure_h2o2 = st.radio(
        "H₂O₂ consumption measurement available?",
        ["No — use stoichiometric assumption (2 mol H₂O₂ per mol DBT)",
         "Yes — I will enter measured consumption"],
        index=0
    )

    if "No" in measure_h2o2:
        st.warning(
            "⚠️ **Stoichiometric assumption active.**\n\n"
            "When H₂O₂ consumption is NOT measured, this tool assumes "
            "n(H₂O₂)_consumed = 2 × n(DBT)_removed, which forces η ≈ 100% "
            "and makes the efficiency metric **uninformative**.\n\n"
            "To obtain a meaningful η, measure actual H₂O₂ consumption by "
            "titration (e.g. cerimetry, iodometry) or spectrophotometry, "
            "as reported in: Safa et al., Fuel 2019, 239, 24–33."
        )

    uploaded = st.file_uploader("Upload removal data (.xlsx/.csv)",
                                type=["xlsx","csv"], key="ox_upload")
    if uploaded is None:
        st.info("Upload a file."); return

    try:
        df = pd.read_excel(uploaded) if uploaded.name.endswith(".xlsx") else pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"{e}"); return

    time_col = [c for c in df.columns if "time" in c.lower()]
    if not time_col:
        st.error("No Time column."); return
    time_col = time_col[0]
    removal_cols = [c for c in df.columns if "removal" in c.lower()]
    if not removal_cols:
        st.error("No Removal columns."); return

    t   = df[time_col].dropna().values.astype(float)
    C0  = cfg["C0"]
    V   = cfg["V_fuel"]
    O_S = cfg["O_S"]
    if C0 is None:
        st.error("C₀ conversion failed."); return

    n_H2O2_initial = O_S * C0 * V  # mol

    rows = []
    for col in removal_cols:
        removal = df[col].dropna().values[:len(t)].astype(float)
        n_DBT_removed = C0 * V * (removal[-1]/100.0)

        if "Yes" in measure_h2o2:
            n_H2O2_consumed = st.number_input(
                f"n(H₂O₂)_consumed for {col} (mmol)",
                min_value=0.0, value=float(round(n_DBT_removed*2*1000, 3)),
                step=0.001, key=f"h2o2_{col}"
            ) * 1e-3
        else:
            n_H2O2_consumed = 2 * n_DBT_removed  # stoichiometric assumption

        eta = (n_DBT_removed / (n_H2O2_consumed / 2)) * 100 if n_H2O2_consumed > 0 else float("nan")

        rows.append({
            "Catalyst": col,
            "n_DBT removed (µmol)": round(n_DBT_removed*1e6, 2),
            "n_H₂O₂ initial (µmol)": round(n_H2O2_initial*1e6, 2),
            "n_H₂O₂ consumed (µmol)": round(n_H2O2_consumed*1e6, 2),
            "η (%)": round(eta, 1) if not np.isnan(eta) else "N/A",
            "H₂O₂ utilization (%)": round(n_H2O2_consumed/n_H2O2_initial*100, 1)
                                     if n_H2O2_initial > 0 else "N/A",
        })

    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    with st.expander("📚 Scientific context", expanded=False):
        st.markdown("""
A high η (close to 100%) indicates that nearly all oxidant was used for
the target reaction (DBT → DBTO₂). Low η indicates:
- Parallel H₂O₂ decomposition (especially with metal catalysts at elevated T)
- Over-oxidation of solvent
- Competing side reactions

**Literature values:**  
η = 80–95% for MOF-based ODS at 25–60°C (Barghi 2025);  
η = 60–80% for V₂O₅/γ-Al₂O₃ at 60°C (Dhir 2009);  
η < 50% often indicates significant H₂O₂ wastage.

**Integration with kinetics (future work):**  
For a full mechanistic treatment coupling H₂O₂ consumption with  
DBT→DBTO→DBTO₂ stepwise oxidation, see:  
Barghi et al., ACS Omega 2025, DOI: 10.1021/acsomega.4c06722.
        """)


# ════════════════════════════════════════════════════════════════
#  TAB 7 — Condition Comparison
# ════════════════════════════════════════════════════════════════

def _tab_comparison(cfg):
    st.header("📊 Tab 7 — Condition Comparison")

    st.markdown("""
Compare kinetic results (k, R², t½, r₀) across multiple experimental
conditions (temperature, O/S ratio, catalyst loading, etc.).
Upload a summary table with one row per experiment.
    """)

    with st.expander("📋 Expected format", expanded=False):
        tmpl = pd.DataFrame({
            "Experiment": ["Run-1","Run-2","Run-3"],
            "T (°C)": [25, 40, 60],
            "O/S": [2, 4, 6],
            "kapp (1/min)": [0.012, 0.028, 0.055],
            "R2": [0.989, 0.994, 0.997],
            "t_half (min)": [57.8, 24.8, 12.6],
            "Removal_final (%)": [75, 88, 95],
        })
        buf = io.BytesIO()
        tmpl.to_excel(buf, index=False)
        st.download_button("⬇️ Download comparison template", buf.getvalue(),
                           "comparison_template.xlsx",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    uploaded = st.file_uploader("Upload comparison table (.xlsx/.csv)",
                                type=["xlsx","csv"], key="cmp_upload")
    if uploaded is None:
        st.info("Upload a summary table."); return

    try:
        df = pd.read_excel(uploaded) if uploaded.name.endswith(".xlsx") else pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"{e}"); return

    st.dataframe(df, use_container_width=True)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) >= 2:
        x_col = st.selectbox("X axis", numeric_cols, index=0)
        y_col = st.selectbox("Y axis", numeric_cols, index=min(1, len(numeric_cols)-1))

        fig, ax = plt.subplots(figsize=(7,5))
        label_col = df.columns[0]
        for i, row in df.iterrows():
            ax.scatter(row[x_col], row[y_col],
                       color=COLORS[i%len(COLORS)],
                       marker=MARKERS[i%len(MARKERS)], s=80, zorder=5)
            ax.annotate(str(row[label_col]),
                        (row[x_col], row[y_col]),
                        textcoords="offset points", xytext=(5,5), fontsize=9)
        ax.set_xlabel(x_col); ax.set_ylabel(y_col)
        ax.set_title(f"{y_col} vs {x_col}")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)


# ════════════════════════════════════════════════════════════════
#  MAIN APP
# ════════════════════════════════════════════════════════════════

def main():
    # ── App header ─────────────────────────────────────────────
    st.markdown("""
<div style="background:linear-gradient(90deg,#01696f,#0f3638);
            padding:1.2rem 1.8rem; border-radius:0.5rem; margin-bottom:1rem;">
    <h1 style="color:white;margin:0;font-size:1.8rem;">
        🔬 CatLab-Tools — ODS Calculation Suite
    </h1>
    <p style="color:#cedcd8;margin:0.3rem 0 0 0;font-size:0.95rem;">
        v3.2 | Oxidative Desulfurization Kinetics &amp; Analysis
    </p>
</div>
""", unsafe_allow_html=True)

    # ── Version changelog ──────────────────────────────────────
    with st.expander("📋 v3.2 Changelog & Scientific Notes", expanded=False):
        st.markdown("""
### What changed in v3.2 vs v3.1

| # | Issue | Fix |
|---|-------|-----|
| FIX 6 | **ppmS density bug** | `_to_mol_L` for ppmS now uses fuel density ρ. Without ρ, C₀ was wrong by factor ρ (~30–45% error for n-heptane) |
| NEW 7 | **Dual C₀ display** | Sidebar shows both C₀(compound, mol/L) and C₀(S, mol/L) simultaneously |
| NEW 8 | **Solvent presets** | Common ODS fuels/solvents with preset ρ (n-heptane, n-hexane, diesel, etc.) |
| NEW 9 | **H₂O₂ warning** | Tab 6 explicitly warns when stoichiometric assumption makes η uninformative |
| NEW 10 | **Assumptions box** | Sidebar lists all model assumptions for transparency |
| NEW 11 | **Version unified** | Header and all references now say v3.2 |

### Key scientific references
- Barghi et al., *ACS Omega* **2025**, 10, 15947 — complex reaction theory, UiO-66-NO₂
- Dhir et al., *J. Hazard. Mater.* **2009**, 161, 1360 — kinetic modeling
- Sengupta et al., *Ind. Eng. Chem. Res.* **2012**, 51, 147 — BT/TS-1 kinetics
- Safa et al., *Fuel* **2019**, 239, 24 — H₂O₂ consumption measurement
- EN 590:2022 — ppmS definition (mg S / kg fuel)
        """)

    # ── Sidebar ────────────────────────────────────────────────
    cfg = _sidebar_settings()

    # ── Tabs ───────────────────────────────────────────────────
    tabs = st.tabs([
        "📈 1 · Kinetic Fitting",
        "📉 2 · Linearization",
        "♻️ 3 · Removal Efficiency",
        "⚗️ 4 · TON & TOF",
        "🔬 5 · Parameter Effect",
        "🧪 6 · Oxidant Efficiency",
        "📊 7 · Comparison",
    ])

    with tabs[0]:
        _tab_kinetics(cfg)
    with tabs[1]:
        _tab_linearization(cfg)
    with tabs[2]:
        _tab_removal(cfg)
    with tabs[3]:
        _tab_ton_tof(cfg)
    with tabs[4]:
        _tab_parameter_effect(cfg)
    with tabs[5]:
        _tab_oxidant_efficiency(cfg)
    with tabs[6]:
        _tab_comparison(cfg)

    # ── Footer ─────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("""
<div style="text-align:center;color:#7a7974;font-size:0.82rem;padding:0.5rem 0;">
    CatLab-Tools v3.2 &nbsp;|&nbsp; 
    <a href="https://github.com/Hj1308/CatLab-Tools" target="_blank" 
       style="color:#01696f;">github.com/Hj1308/CatLab-Tools</a> &nbsp;|&nbsp;
    Original author: Hoda Jafari &nbsp;|&nbsp;
    Scientific references: ACS Omega 2025 · J. Hazard. Mater. 2009 · Fuel 2019
</div>
""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
