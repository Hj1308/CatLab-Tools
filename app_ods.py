"""
ODS Calculation Suite — Streamlit Web App
==========================================
Extended build based on CatLab-Tools/app_ods.py (github.com/Hj1308/CatLab-Tools)
Original author: Hoda Jafari

v3.0 — additions on top of v2.0
----------------------------------------
NEW  : k ± SE (standard error from curve_fit covariance matrix) for all models.
NEW  : r₀ (initial reaction rate) — model-independent activity metric:
         Pseudo-first:  r₀ = kapp × C₀  (mol·L⁻¹·min⁻¹)
         Second-order:  r₀ = k₂  × C₀²  (mol·L⁻¹·min⁻¹)
         Elovich:       r₀ = α           (mol·L⁻¹·min⁻¹)
         Zero-order:    r₀ = k₀          (mol·L⁻¹·min⁻¹)
NEW  : Langmuir-Hinshelwood (L-H) model as 5th kinetic model.
         rate = kLH·K·C / (1 + K·C)  — integrated numerically via ODE.
NEW  : Tab 5 — Parameter Effect: plots X%, k, or t½ vs a variable parameter
         (catalyst loading, H₂O₂ dose, DBT concentration, voltage, light intensity…)
         for multiple catalysts on one chart.
NEW  : Tab 6 — Oxidant Efficiency: η = n_DBT_removed / (n_H₂O₂_consumed × 2).
NEW  : Tab 7 — Condition Comparison: upload multiple kinetics_summary.xlsx files
         (one per condition: Thermal, UV, ECODS…) and compare t½ / k side-by-side
         with grouped bar charts.
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

st.set_page_config(page_title="ODS Calculation Suite", page_icon="🔬", layout="wide")

MW_S = 32.06
COLORS  = ["#e41a1c","#377eb8","#4daf4a","#984ea3",
           "#ff7f00","#a65628","#f781bf","#17becf","#bcbd22"]
MARKERS = ["o","s","^","D","v","P","*","X","h"]


# ════════════════════════════════════════════════════════════════
#  SHARED HELPERS
# ════════════════════════════════════════════════════════════════
def _to_mol_L(value, unit, mw=None):
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
        return (value / MW_S) / 1000.0
    else:
        raise ValueError(f"Unknown unit: {unit}")


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
    """Langmuir-Hinshelwood: dC/dt = -kLH·K·C/(1+K·C), solved numerically."""
    def dC(C, tt):
        Cv = max(C[0], 0.0)
        return [-k_LH * K_ads * Cv / (1.0 + K_ads * Cv)]
    sol = odeint(dC, [C0], t, rtol=1e-6, atol=1e-9)
    return np.maximum(sol.flatten(), 0.0)

def _r2(y_obs, y_pred):
    ss_res = np.sum((y_obs - y_pred) ** 2)
    ss_tot = np.sum((y_obs - np.mean(y_obs)) ** 2)
    return round(1 - ss_res / ss_tot if ss_tot > 0 else 0.0, 4)

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
    """Solve C(t½) = C0/2 analytically: t½ = [ln2/K + C0/2·(1-1/(K·C0))] ... use numerical."""
    try:
        if k_LH <= 0 or K_ads <= 0 or C0 <= 0:
            return float("nan")
        # analytical: t½ = (ln2 + K_ads*(C0 - C0/2)) / (kLH * K_ads) ... simplified:
        # integral from C0 to C0/2 of (1+KC)/kLH/K dC = ln2/kLH/K + C0/2/kLH
        t_half = (np.log(2) / (k_LH * K_ads)) + (C0 / 2.0) / k_LH
        return t_half
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
    """Format as 'val ± SE' in scientific notation."""
    if val is None or se is None:
        return _fmt_sci(val)
    if np.isnan(val) or np.isnan(se):
        return _fmt_sci(val)
    if np.isinf(se) or se > abs(val) * 100:
        return f"{_fmt_sci(val)} (SE large)"
    exp = int(np.floor(np.log10(abs(val)))) if val != 0 else 0
    scale = 10 ** exp
    return f"({val/scale:.2f} ± {se/scale:.2f}) × 10{str(exp).translate(str.maketrans('0123456789-','⁰¹²³⁴⁵⁶⁷⁸⁹⁻'))}"


# ── Non-linear fitting engine (v3.0 — captures pcov → k±SE, r₀) ──
def _fit_nonlinear(time, Ct, C0):
    t = np.asarray(time, dtype=float)
    Ct = np.asarray(Ct, dtype=float)
    results = {}

    # Zero-order
    try:
        p, pcov = curve_fit(_zero_order, t, Ct, p0=[1e-6, C0],
                            bounds=([0, C0*0.5], [np.inf, C0*1.5]), maxfev=5000)
        se = np.sqrt(np.diag(pcov))
        pred = _zero_order(t, *p)
        k0 = p[0]
        results["Zero-order"] = {
            "params": p, "R2": _r2(Ct, pred), "pred": pred,
            "label": f"k₀ = {_fmt_sci(k0)} mol·L⁻¹·min⁻¹",
            "t_half": round(0.5 * p[1] / k0, 2) if k0 > 0 else float("nan"),
            "k": k0, "k_se": se[0], "col_k": "K0 (mol/L/min)",
            "r0": k0, "r0_se": se[0],
        }
    except Exception as e:
        results["Zero-order"] = {"R2": -999, "error": str(e)}

    # Pseudo-first-order
    try:
        p, pcov = curve_fit(_first_order, t, Ct, p0=[0.01, C0],
                            bounds=([0, C0*0.5], [np.inf, C0*1.5]), maxfev=5000)
        se = np.sqrt(np.diag(pcov))
        pred = _first_order(t, *p)
        kapp = p[0]
        r0 = kapp * C0
        r0_se = se[0] * C0
        results["Pseudo-first"] = {
            "params": p, "R2": _r2(Ct, pred), "pred": pred,
            "label": f"kₐₚₚ = {_fmt_sci(kapp)} min⁻¹",
            "t_half": round(np.log(2) / kapp, 2) if kapp > 0 else float("nan"),
            "k": kapp, "k_se": se[0], "col_k": "Kapp (1/min)",
            "r0": r0, "r0_se": r0_se,
        }
    except Exception as e:
        results["Pseudo-first"] = {"R2": -999, "error": str(e)}

    # Second-order
    try:
        p, pcov = curve_fit(_second_order, t, Ct, p0=[1e-3, C0],
                            bounds=([0, C0*0.5], [np.inf, C0*1.5]), maxfev=5000)
        se = np.sqrt(np.diag(pcov))
        pred = _second_order(t, *p)
        k2 = p[0]
        r0 = k2 * C0 ** 2
        r0_se = se[0] * C0 ** 2
        results["Second-order"] = {
            "params": p, "R2": _r2(Ct, pred), "pred": pred,
            "label": f"k₂ = {_fmt_sci(k2)} L·mol⁻¹·min⁻¹",
            "t_half": round(1.0 / (k2 * p[1]), 2) if k2 > 0 else float("nan"),
            "k": k2, "k_se": se[0], "col_k": "K2 (L/mol/min)",
            "r0": r0, "r0_se": r0_se,
        }
    except Exception as e:
        results["Second-order"] = {"R2": -999, "error": str(e)}

    # Elovich
    try:
        if len(t) < 2:
            raise ValueError("Elovich needs at least 2 time points")
        p, pcov = curve_fit(
            lambda tt, a, b: _elovich(tt, a, b, C0),
            t, Ct,
            p0=[1e-4, 1e4],
            bounds=([1e-10, 1e-3], [1e3, 1e10]),
            maxfev=10000
        )
        se = np.sqrt(np.diag(pcov))
        alpha, beta = p
        pred_full = _elovich(t, alpha, beta, C0)
        results["Elovich"] = {
            "params": (alpha, beta, C0), "R2": _r2(Ct, pred_full), "pred": pred_full,
            "label": f"α={_fmt_sci(alpha)}, β={_fmt_sci(beta)}",
            "t_half": _elovich_t_half(C0, alpha, beta),
            "k": alpha, "k_se": se[0], "beta": beta, "col_k": "α (mol/L/min)",
            "r0": alpha, "r0_se": se[0],
        }
    except Exception as e:
        results["Elovich"] = {"R2": -999, "error": str(e)}

    # Langmuir-Hinshelwood (NEW in v3.0)
    try:
        if len(t) < 3:
            raise ValueError("L-H model needs at least 3 time points")
        p0_lh = [0.01, 10.0]
        p, pcov = curve_fit(
            lambda tt, kl, ka: _lh_model(tt, kl, ka, C0),
            t, Ct,
            p0=p0_lh,
            bounds=([1e-8, 1e-4], [1e3, 1e6]),
            maxfev=15000
        )
        se = np.sqrt(np.diag(pcov))
        k_LH, K_ads = p
        pred_lh = _lh_model(t, k_LH, K_ads, C0)
        r0_lh = k_LH * K_ads * C0 / (1.0 + K_ads * C0)
        results["L-H"] = {
            "params": (k_LH, K_ads, C0), "R2": _r2(Ct, pred_lh), "pred": pred_lh,
            "label": f"kLH={_fmt_sci(k_LH)}, K={_fmt_sci(K_ads)}",
            "t_half": _lh_t_half(C0, k_LH, K_ads),
            "k": k_LH, "k_se": se[0], "K_ads": K_ads, "K_ads_se": se[1],
            "col_k": "kLH (1/min)",
            "r0": r0_lh, "r0_se": float("nan"),
        }
    except Exception as e:
        results["L-H"] = {"R2": -999, "error": str(e)}

    candidates = [m for m in results if m != "Zero-order"]
    best_name = max(candidates, key=lambda m: results[m].get("R2", -999))
    for name in results:
        results[name]["is_best"] = (name == best_name)
    return results


def _plot_model(fits_all, sheets, model_name, ylabel, title):
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, sheet in enumerate(sheets):
        f = fits_all[sheet]
        res = f["fits"].get(model_name, {})
        if res.get("R2", -999) < -100:
            continue
        t = f["t"]; Ct = f["Ct"]
        c = COLORS[i % len(COLORS)]; m = MARKERS[i % len(MARKERS)]
        lbl = f"{sheet} R²={res['R2']:.4f}"
        if res.get("is_best"):
            lbl += " ★best"
        ax.scatter(t, Ct, color=c, marker=m, s=60, zorder=3,
                   edgecolors="white", linewidths=0.6)
        t_fit = np.linspace(0, t[-1], 300)
        if model_name == "Zero-order":
            pred_fit = _zero_order(t_fit, *res["params"])
        elif model_name == "Pseudo-first":
            pred_fit = _first_order(t_fit, *res["params"])
        elif model_name == "Second-order":
            pred_fit = _second_order(t_fit, *res["params"])
        elif model_name == "L-H":
            pred_fit = _lh_model(t_fit, *res["params"])
        else:
            pred_fit = _elovich(t_fit, *res["params"])
        ax.plot(t_fit, pred_fit, "--", color=c, lw=1.5, alpha=0.8, label=lbl)

        th = res.get("t_half", float("nan"))
        if res.get("is_best") and not np.isnan(th) and not np.isinf(th) and 0 < th <= t[-1] * 1.5:
            ax.axvline(th, color=c, ls=":", lw=1.2, alpha=0.7)
            ax.annotate("t½", xy=(th, ax.get_ylim()[1]*0.95), color=c,
                         fontsize=9, ha="center", fontweight="bold")

    ax.set_xlabel("Time (min)", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def _linreg_r2(x, y):
    A = np.vstack([x, np.ones_like(x)]).T
    slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
    yhat = slope * x + intercept
    return slope, intercept, _r2(y, yhat)


def _plot_linearization(fits_all, sheets, C0_mol, transform, ylabel, title):
    fig, ax = plt.subplots(figsize=(9, 5))
    lin_results = {}
    for i, sheet in enumerate(sheets):
        f = fits_all[sheet]
        t = f["t"]; Ct = f["Ct"]
        y = transform(Ct, C0_mol)
        slope, intercept, r2v = _linreg_r2(t, y)
        lin_results[sheet] = (slope, intercept, r2v)
        c = COLORS[i % len(COLORS)]; m = MARKERS[i % len(MARKERS)]
        ax.scatter(t, y, color=c, marker=m, s=60, zorder=3,
                   edgecolors="white", linewidths=0.6)
        t_fit = np.array([0, t[-1]])
        ax.plot(t_fit, slope*t_fit + intercept, "--", color=c, lw=1.5, alpha=0.8,
                label=f"{sheet} R²={r2v:.4f}")
    ax.set_xlabel("Time (min)", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig, lin_results


# ════════════════════════════════════════════════════════════════
#  TEMPLATE BUILDERS
# ════════════════════════════════════════════════════════════════
def _safe_sheet_name(name, used):
    safe = re.sub(r'[\\/?*\[\]:]', '-', str(name)).strip(" '")
    if not safe:
        safe = "Sheet"
    safe = safe[:31]
    base, i = safe, 2
    while safe in used:
        suffix = f"_{i}"
        safe = base[:31 - len(suffix)] + suffix
        i += 1
    used.add(safe)
    return safe


def make_kinetics_template(names):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        instr = pd.DataFrame({
            "Instructions": [
                "One sheet per catalyst (sheet name = catalyst name).",
                "Each sheet must have EXACTLY 2 columns: 'Time (min)' and 'Removal (%)'.",
                "Use at least 4-5 time points for a meaningful fit.",
                "Do not leave blank cells.",
                "Removal (%) = (C0 - Ct)/C0 * 100.",
            ]
        })
        instr.to_excel(w, sheet_name="Instructions", index=False)
        example = pd.DataFrame({
            "Time (min)": [0, 5, 10, 20, 30, 45, 60],
            "Removal (%)": [0, 18, 32, 50, 64, 78, 88],
        })
        example.to_excel(w, sheet_name="Example", index=False)
        used = {"Instructions", "Example"}
        mapping = {}
        for name in names:
            safe = _safe_sheet_name(name, used)
            mapping[name] = safe
            pd.DataFrame({"Time (min)": [None]*6, "Removal (%)": [None]*6}
                          ).to_excel(w, sheet_name=safe, index=False)
    buf.seek(0)
    return buf, mapping


def make_reusability_template(names, n_cycles=5):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        instr = pd.DataFrame({
            "Instructions": [
                "One sheet per catalyst.",
                "EXACTLY 2 columns: 'Cycle' and 'Removal (%)'.",
                "Retention (%) = X_cycle_n / X_cycle_1 * 100.",
            ]
        })
        instr.to_excel(w, sheet_name="Instructions", index=False)
        example = pd.DataFrame({
            "Cycle": list(range(1, n_cycles+1)),
            "Removal (%)": [92, 88, 83, 77, 70][:n_cycles],
        })
        example.to_excel(w, sheet_name="Example", index=False)
        used = {"Instructions", "Example"}
        mapping = {}
        for name in names:
            safe = _safe_sheet_name(name, used)
            mapping[name] = safe
            pd.DataFrame({"Cycle": list(range(1, n_cycles+1)),
                           "Removal (%)": [None]*n_cycles}
                          ).to_excel(w, sheet_name=safe, index=False)
    buf.seek(0)
    return buf, mapping


def make_tof_template(names):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        instr = pd.DataFrame({
            "Instructions": [
                "ONE sheet 'TOF_TON', one row per catalyst.",
                "TOF (h-1) = n_substrate_converted / (n_active_sites * time)",
                "TON = n_substrate_converted / n_active_sites",
            ]
        })
        instr.to_excel(w, sheet_name="Instructions", index=False)
        df = pd.DataFrame({
            "Catalyst": names,
            "Catalyst mass (g)": [None]*len(names),
            "Active sites (mmol/g)": [None]*len(names),
            "Fuel volume (L)": [None]*len(names),
            "Removal (%)": [None]*len(names),
            "Reaction time (h)": [None]*len(names),
        })
        df.to_excel(w, sheet_name="TOF_TON", index=False)
    buf.seek(0)
    return buf


def make_param_effect_template():
    """Template for Parameter Effect module (Tab 5)."""
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        instr = pd.DataFrame({
            "Instructions": [
                "Sheet name = parameter name (e.g. 'Catalyst Loading', 'H2O2 Dose').",
                "Column 1: parameter values (numeric). Header = parameter name + unit.",
                "Remaining columns: one per catalyst. Header = catalyst name.",
                "Cell values = X_final (%) OR kapp (1/min) OR t½ (min) — choose one metric.",
                "Example: Catalyst Loading (mg/L) | Cat-A | Cat-B | Cat-C",
                "You may have multiple sheets for different parameters.",
            ]
        })
        instr.to_excel(w, sheet_name="Instructions", index=False)
        example = pd.DataFrame({
            "Catalyst Loading (mg/L)": [5, 10, 15, 20, 25],
            "Cat-A": [45, 68, 82, 90, 92],
            "Cat-B": [38, 60, 75, 85, 88],
        })
        example.to_excel(w, sheet_name="Example_Loading", index=False)
    buf.seek(0)
    return buf


def make_ox_efficiency_template(names):
    """Template for Oxidant Efficiency module (Tab 6)."""
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        instr = pd.DataFrame({
            "Instructions": [
                "ONE sheet 'OxEff', one row per catalyst.",
                "n_DBT_removed (mmol): moles of DBT removed.",
                "n_H2O2_consumed (mmol): moles of H2O2 consumed (added minus residual).",
                "Stoichiometry: DBT + 2 H2O2 → DBTO2 + 2 H2O (1:2 ratio).",
                "η = n_DBT_removed / (n_H2O2_consumed / 2) × 100%",
                "η > 100% is impossible; if so, check your H2O2 measurement.",
            ]
        })
        instr.to_excel(w, sheet_name="Instructions", index=False)
        df = pd.DataFrame({
            "Catalyst": names,
            "n_DBT_removed (mmol)": [None]*len(names),
            "n_H2O2_consumed (mmol)": [None]*len(names),
        })
        df.to_excel(w, sheet_name="OxEff", index=False)
    buf.seek(0)
    return buf


# ════════════════════════════════════════════════════════════════
#  UI
# ════════════════════════════════════════════════════════════════
st.title("🔬 ODS Calculation Suite")
st.markdown(
    "**CatLab-Tools v3.0** — based on the work of Hoda Jafari · "
    "[GitHub](https://github.com/Hj1308/CatLab-Tools)"
)
st.markdown("---")

with st.sidebar:
    st.header("⚙️ Global Parameters")
    c0_val = st.number_input("C₀ (initial DBT concentration)", value=250.0, min_value=0.0001)
    c0_unit = st.selectbox("C₀ unit", ["ppmS", "ppm", "mg/L", "g/L", "mmol/L", "mol/L"])
    mw_poll = None
    if c0_unit in ("ppm", "mg/L", "g/L"):
        mw_poll = st.number_input("MW of pollutant (g/mol)", value=184.0, min_value=1.0,
                                   help="MW of DBT ≈ 184.26 g/mol")
    try:
        C0_mol = _to_mol_L(c0_val, c0_unit, mw_poll)
        st.caption(f"C₀ = {_fmt_sci(C0_mol)} mol/L")
    except Exception as e:
        C0_mol = None
        st.error(f"Unit error: {e}")

(tab_templates, tab_kinetics, tab_tof, tab_reuse,
 tab_param, tab_oxeff, tab_compare) = st.tabs([
    "📥 1. Templates",
    "📊 2. Kinetics & t½",
    "⚙️ 3. TOF / TON",
    "♻️ 4. Reusability",
    "📉 5. Parameter Effect",   # NEW
    "💧 6. Oxidant Efficiency",  # NEW
    "🔀 7. Condition Comparison", # NEW
])


# ────────────────────────────────────────────────────────────
# TAB 1 — TEMPLATES
# ────────────────────────────────────────────────────────────
with tab_templates:
    st.subheader("Generate Excel Input Templates")
    cat_input = st.text_area("Catalyst names (one per line):",
                              value="Catalyst_1\nCatalyst_2\nCatalyst_3", height=120)
    n_cycles = st.number_input("Number of reuse cycles (for Reusability template)",
                                value=5, min_value=2, max_value=20, step=1)
    names = [n.strip() for n in cat_input.strip().splitlines() if n.strip()]

    if not names:
        st.warning("Please enter at least one catalyst name.")
    else:
        kin_buf, kin_map = make_kinetics_template(names)
        reuse_buf, reuse_map = make_reusability_template(names, int(n_cycles))

        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.download_button("📥 Kinetics template",
                                data=kin_buf, file_name="ods_kinetics_template.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        with c2:
            st.download_button("📥 TOF/TON template",
                                data=make_tof_template(names), file_name="ods_tof_ton_template.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        with c3:
            st.download_button("📥 Reusability template",
                                data=reuse_buf, file_name="ods_reusability_template.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        with c4:
            st.download_button("📥 Parameter Effect template",
                                data=make_param_effect_template(),
                                file_name="ods_param_effect_template.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        with c5:
            st.download_button("📥 Oxidant Efficiency template",
                                data=make_ox_efficiency_template(names),
                                file_name="ods_ox_efficiency_template.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        changed = {k: v for k, v in kin_map.items() if k != v}
        if changed:
            st.info(
                "ℹ️ Excel sheet names adjusted:\n\n" +
                "\n".join(f"- `{k}` → `{v}`" for k, v in changed.items())
            )

    st.markdown("---")
    with st.expander("📐 Formulas used in this app (v3.0)"):
        st.markdown(r"""
**Kinetic models**

| Model | Equation | t½ |
|---|---|---|
| Zero-order | $C_t = C_0 - k_0 t$ | $C_0/(2k_0)$ |
| Pseudo-first | $C_t = C_0 e^{-k_{app}t}$ | $\ln 2 / k_{app}$ |
| Second-order | $C_t = C_0/(1+k_2 C_0 t)$ | $1/(k_2 C_0)$ |
| Elovich | $C_t = C_0 - \frac{1}{\beta}\ln(1+\alpha\beta t)$ | $(e^{C_0\beta/2}-1)/(\alpha\beta)$ |
| **L-H** | $\frac{dC}{dt} = -\frac{k_{LH} K C}{1+KC}$ (ODE) | $\frac{\ln 2}{k_{LH}K} + \frac{C_0/2}{k_{LH}}$ |

**Initial rate r₀ (mol·L⁻¹·min⁻¹)**
$$r_0^{(1)} = k_{app} C_0 \qquad r_0^{(2)} = k_2 C_0^2 \qquad r_0^{Elovich} = \alpha \qquad r_0^{LH} = \frac{k_{LH} K C_0}{1+K C_0}$$

**Standard Error**: $SE_k = \sqrt{[\Sigma^{-1}]_{kk}}$ where $\Sigma$ = covariance matrix from `scipy.curve_fit`.

**Oxidant Efficiency**
$$\eta_{H_2O_2}(\%) = \frac{n_{DBT,removed}}{n_{H_2O_2,consumed}/2} \times 100$$

**TOF / TON**
$$TON = \frac{n_{sub}}{n_{sites}} \qquad TOF = \frac{TON}{t(h)}$$
""")


# ────────────────────────────────────────────────────────────
# TAB 2 — KINETICS & t½  (v3.0: +SE, +r₀, +L-H model)
# ────────────────────────────────────────────────────────────
with tab_kinetics:
    st.subheader("Kinetics fitting (Zero / Pseudo-first / Second-order / Elovich / L-H) + k±SE + r₀")
    uploaded = st.file_uploader("Upload filled Kinetics template", type=["xlsx", "xls"],
                                 key="kin_upl")

    if uploaded and C0_mol is not None:
        try:
            xl = pd.ExcelFile(uploaded)
        except Exception as e:
            st.error(f"Could not open the Excel file: {e}")
            xl = None

        if xl is not None:
            sheets = [s for s in xl.sheet_names if s not in ("Instructions", "Example")]
            if not sheets:
                st.warning("No catalyst sheets found.")
            else:
                fits_all, rows, errors = {}, [], []

                for sheet in sheets:
                    try:
                        raw = xl.parse(sheet)
                        df = raw.dropna(how="all")
                        if df.shape[1] != 2:
                            raise ValueError(f"expected 2 columns, found {df.shape[1]}")
                        df = df.dropna()
                        if len(df) < 2:
                            raise ValueError(f"only {len(df)} valid row(s) — need ≥2")
                        df.columns = ["Time (min)", "Removal (%)"]
                        t_arr = df["Time (min)"].values.astype(float)
                        rem   = df["Removal (%)"].values.astype(float) / 100.0
                        order = np.argsort(t_arr)
                        t_arr, rem = t_arr[order], rem[order]
                        Ct_arr = C0_mol * (1.0 - rem)

                        fits = _fit_nonlinear(t_arr, Ct_arr, C0_mol)
                        fits_all[sheet] = {"t": t_arr, "Ct": Ct_arr, "fits": fits}

                        candidates = ["Pseudo-first", "Second-order", "Elovich", "L-H"]
                        best = max(candidates, key=lambda m: fits[m].get("R2", -999))

                        r1  = fits["Pseudo-first"].get("R2", -999)
                        r2v = fits["Second-order"].get("R2", -999)
                        order_label = "Second-order" if r2v > r1 else "Pseudo-first-order"

                        best_res = fits[best]

                        rows.append({
                            "Catalyst":              sheet,
                            "X_final (%)":           round(rem[-1]*100, 1),
                            # Zero-order
                            "K0 (mol/L/min)":        fits["Zero-order"].get("k", float("nan")),
                            "K0_SE":                 fits["Zero-order"].get("k_se", float("nan")),
                            "R2_zero":               fits["Zero-order"].get("R2", -999),
                            # Pseudo-first
                            "Kapp (1/min)":          fits["Pseudo-first"].get("k", float("nan")),
                            "Kapp_SE":               fits["Pseudo-first"].get("k_se", float("nan")),
                            "R2_first":              fits["Pseudo-first"].get("R2", -999),
                            # Second-order
                            "K2 (L/mol/min)":        fits["Second-order"].get("k", float("nan")),
                            "K2_SE":                 fits["Second-order"].get("k_se", float("nan")),
                            "R2_second":             fits["Second-order"].get("R2", -999),
                            # Elovich
                            "Elovich α":             fits["Elovich"].get("k", float("nan")),
                            "Elovich α_SE":          fits["Elovich"].get("k_se", float("nan")),
                            "Elovich β":             fits["Elovich"].get("beta", float("nan")),
                            "R2_elovich":            fits["Elovich"].get("R2", -999),
                            # L-H (NEW)
                            "kLH (1/min)":           fits["L-H"].get("k", float("nan")),
                            "kLH_SE":                fits["L-H"].get("k_se", float("nan")),
                            "K_ads (L/mol)":         fits["L-H"].get("K_ads", float("nan")),
                            "R2_LH":                 fits["L-H"].get("R2", -999),
                            # Best model
                            "Reaction Order":        order_label,
                            "Best Model":            best,
                            "Best R2":               best_res.get("R2", float("nan")),
                            "t½ (min)":              best_res.get("t_half", float("nan")),
                            # r₀ (NEW)
                            "r₀ (mol/L/min)":        best_res.get("r0", float("nan")),
                            "r₀_SE":                 best_res.get("r0_se", float("nan")),
                        })
                    except Exception as e:
                        errors.append((sheet, str(e)))

                if errors:
                    msg = "\n".join(f"- **{s}**: {m}" for s, m in errors)
                    st.warning("⚠️ Some sheets skipped:\n\n" + msg)

                if rows:
                    summary = pd.DataFrame(rows)

                    st.markdown("### 📈 Plots")
                    fig0, ax0 = plt.subplots(figsize=(9, 5))
                    for i, sheet in enumerate(fits_all):
                        f = fits_all[sheet]
                        ax0.plot(f["t"], (1 - f["Ct"]/C0_mol)*100,
                                 marker=MARKERS[i % len(MARKERS)],
                                 color=COLORS[i % len(COLORS)],
                                 label=sheet, lw=2, ms=8,
                                 markeredgecolor="white", markeredgewidth=0.7)
                    ax0.set_xlabel("Time (min)", fontsize=12)
                    ax0.set_ylabel("Removal (%)", fontsize=12)
                    ax0.set_title("ODS Performance — All Catalysts", fontsize=13, fontweight="bold")
                    ax0.legend(fontsize=9); ax0.grid(True, alpha=0.3)
                    plt.tight_layout()
                    st.pyplot(fig0)

                    sheets_ok = list(fits_all.keys())
                    fig1 = _plot_model(fits_all, sheets_ok, "Zero-order",
                                        "Cₜ (mol·L⁻¹)", "Zero-Order Fit")
                    fig2 = _plot_model(fits_all, sheets_ok, "Pseudo-first",
                                        "Cₜ (mol·L⁻¹)", "Pseudo-First-Order Fit")
                    fig3 = _plot_model(fits_all, sheets_ok, "Second-order",
                                        "Cₜ (mol·L⁻¹)", "Second-Order Fit")
                    fig4 = _plot_model(fits_all, sheets_ok, "Elovich",
                                        "Cₜ (mol·L⁻¹)", "Elovich Model Fit")
                    fig4b = _plot_model(fits_all, sheets_ok, "L-H",
                                         "Cₜ (mol·L⁻¹)", "Langmuir-Hinshelwood Fit (NEW)")

                    c_a, c_b = st.columns(2)
                    c_a.pyplot(fig1); c_b.pyplot(fig2)
                    c_c, c_d = st.columns(2)
                    c_c.pyplot(fig3); c_d.pyplot(fig4)
                    st.pyplot(fig4b)

                    st.markdown("#### Classic linearized-kinetics plots")
                    fig5, lin_first = _plot_linearization(
                        fits_all, sheets_ok, C0_mol,
                        transform=lambda Ct, C0: np.log(C0/Ct),
                        ylabel="ln(C₀/Cₜ)",
                        title="Pseudo-First-Order Linearization")
                    fig6, lin_second = _plot_linearization(
                        fits_all, sheets_ok, C0_mol,
                        transform=lambda Ct, C0: 1.0/Ct - 1.0/C0,
                        ylabel="1/Cₜ − 1/C₀  (L·mol⁻¹)",
                        title="Second-Order Linearization")
                    c_e, c_f = st.columns(2)
                    c_e.pyplot(fig5); c_f.pyplot(fig6)

                    summary["Kapp_lin (1/min)"] = summary["Catalyst"].map(lambda s: lin_first[s][0])
                    summary["R2_first_lin"]     = summary["Catalyst"].map(lambda s: lin_first[s][2])
                    summary["K2_lin (L/mol/min)"] = summary["Catalyst"].map(lambda s: lin_second[s][0])
                    summary["R2_second_lin"]    = summary["Catalyst"].map(lambda s: lin_second[s][2])

                    # r₀ bar chart
                    st.markdown("### 📊 Initial Rate r₀ Comparison (model-independent activity ranking)")
                    fig_r0, ax_r0 = plt.subplots(figsize=(9, 4))
                    cats_r0 = summary["Catalyst"].tolist()
                    r0_vals = summary["r₀ (mol/L/min)"].tolist()
                    r0_ses  = summary["r₀_SE"].tolist()
                    colors_r0 = [COLORS[i % len(COLORS)] for i in range(len(cats_r0))]
                    bars = ax_r0.bar(cats_r0, r0_vals, color=colors_r0,
                                     yerr=[s if not np.isnan(s) else 0 for s in r0_ses],
                                     capsize=5, edgecolor="white")
                    ax_r0.set_ylabel("r₀ (mol·L⁻¹·min⁻¹)", fontsize=12)
                    ax_r0.set_title("Initial Reaction Rate r₀ per Catalyst", fontsize=13, fontweight="bold")
                    ax_r0.tick_params(axis="x", rotation=30)
                    ax_r0.grid(True, alpha=0.3, axis="y")
                    plt.tight_layout()
                    st.pyplot(fig_r0)

                    st.markdown("---")
                    st.markdown("### 📋 Kinetics Summary (v3.0 — includes k±SE and r₀)")
                    disp = summary.copy()
                    disp["K0 ± SE"]    = disp.apply(lambda r: _fmt_pm(r["K0 (mol/L/min)"], r["K0_SE"]), axis=1)
                    disp["Kapp ± SE"]  = disp.apply(lambda r: _fmt_pm(r["Kapp (1/min)"], r["Kapp_SE"]), axis=1)
                    disp["K2 ± SE"]    = disp.apply(lambda r: _fmt_pm(r["K2 (L/mol/min)"], r["K2_SE"]), axis=1)
                    disp["α ± SE"]     = disp.apply(lambda r: _fmt_pm(r["Elovich α"], r["Elovich α_SE"]), axis=1)
                    disp["kLH ± SE"]   = disp.apply(lambda r: _fmt_pm(r["kLH (1/min)"], r["kLH_SE"]), axis=1)
                    disp["r₀ ± SE"]    = disp.apply(lambda r: _fmt_pm(r["r₀ (mol/L/min)"], r["r₀_SE"]), axis=1)
                    for col in ["R2_zero","R2_first","R2_second","R2_elovich","R2_LH","Best R2",
                                "R2_first_lin","R2_second_lin"]:
                        disp[col] = disp[col].apply(lambda v: f"{v:.4f}" if v != -999 else "—")
                    disp["t½ (min)"] = disp["t½ (min)"].apply(_fmt_thalf)
                    disp["K_ads (L/mol)"] = disp["K_ads (L/mol)"].apply(_fmt_sci)
                    show_cols = ["Catalyst","X_final (%)","K0 ± SE","R2_zero",
                                 "Kapp ± SE","R2_first","K2 ± SE","R2_second",
                                 "α ± SE","R2_elovich","kLH ± SE","R2_LH",
                                 "Reaction Order","Best Model","Best R2","t½ (min)",
                                 "r₀ ± SE","Kapp_lin (1/min)","R2_first_lin",
                                 "K2_lin (L/mol/min)","R2_second_lin"]
                    st.dataframe(disp[show_cols], use_container_width=True)

                    st.markdown("### ⬇️ Download Results")
                    tbl_buf = io.BytesIO()
                    summary.to_excel(tbl_buf, index=False); tbl_buf.seek(0)
                    st.download_button("📥 Download kinetics_summary.xlsx",
                                        data=tbl_buf, file_name="kinetics_summary.xlsx",
                                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

                    zip_buf = io.BytesIO()
                    with zipfile.ZipFile(zip_buf, "w") as zf:
                        for fname, fig in [("01_removal_vs_time.png", fig0),
                                            ("02_zero_order.png", fig1),
                                            ("03_pseudo_first.png", fig2),
                                            ("04_second_order.png", fig3),
                                            ("05_elovich.png", fig4),
                                            ("05b_LH.png", fig4b),
                                            ("06_pseudo_first_linear.png", fig5),
                                            ("07_second_order_linear.png", fig6),
                                            ("08_r0_comparison.png", fig_r0)]:
                            ib = io.BytesIO()
                            fig.savefig(ib, dpi=300, bbox_inches="tight")
                            ib.seek(0); zf.writestr(fname, ib.read())
                    zip_buf.seek(0)
                    st.download_button("📥 Download all plots (ZIP)",
                                        data=zip_buf, file_name="ods_kinetics_plots.zip",
                                        mime="application/zip")

    elif uploaded and C0_mol is None:
        st.error("Fix the C₀ unit settings in the sidebar first.")
    else:
        st.info("Upload your filled Kinetics template to start the analysis.")


# ────────────────────────────────────────────────────────────
# TAB 3 — TOF / TON (unchanged from v2.0)
# ────────────────────────────────────────────────────────────
with tab_tof:
    st.subheader("Turnover Frequency (TOF) and Turnover Number (TON)")
    st.caption("Uses the global C₀ set in the sidebar.")
    uploaded_tof = st.file_uploader("Upload filled TOF/TON template", type=["xlsx","xls"], key="tof_upl")

    if uploaded_tof and C0_mol is not None:
        try:
            xl = pd.ExcelFile(uploaded_tof)
            if "TOF_TON" not in xl.sheet_names:
                st.error("Sheet 'TOF_TON' not found. Use the template from Tab 1.")
            else:
                df = xl.parse("TOF_TON").dropna(how="all")
                required = ["Catalyst","Catalyst mass (g)","Active sites (mmol/g)",
                             "Fuel volume (L)","Removal (%)","Reaction time (h)"]
                missing = [c for c in required if c not in df.columns]
                if missing:
                    st.error(f"Missing column(s): {', '.join(missing)}")
                else:
                    df = df.dropna(subset=required)
                    if df.empty:
                        st.warning("No complete rows found.")
                    else:
                        rows = []
                        for _, r in df.iterrows():
                            mass = float(r["Catalyst mass (g)"])
                            sites = float(r["Active sites (mmol/g)"])
                            vol = float(r["Fuel volume (L)"])
                            X = float(r["Removal (%)"])
                            th = float(r["Reaction time (h)"])
                            n_sites_mol = mass * sites / 1000.0
                            delta_C = C0_mol * (X / 100.0)
                            n_sub_mol = delta_C * vol
                            if n_sites_mol <= 0:
                                ton = tof = float("nan")
                            else:
                                ton = n_sub_mol / n_sites_mol
                                tof = ton / th if th > 0 else float("nan")
                            rows.append({"Catalyst": r["Catalyst"], "Removal (%)": X,
                                          "n_sites (mmol)": n_sites_mol*1000,
                                          "n_substrate (mmol)": n_sub_mol*1000,
                                          "TON": ton, "TOF (h⁻¹)": tof})
                        result = pd.DataFrame(rows)
                        st.markdown("### 📋 TOF / TON Summary")
                        disp = result.copy()
                        for col in ["n_sites (mmol)","n_substrate (mmol)"]:
                            disp[col] = disp[col].apply(lambda v: f"{v:.4f}")
                        for col in ["TON","TOF (h⁻¹)"]:
                            disp[col] = disp[col].apply(_fmt_sci)
                        st.dataframe(disp, use_container_width=True)
                        fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 4.5))
                        cats = result["Catalyst"].astype(str)
                        colors_b = [COLORS[i % len(COLORS)] for i in range(len(cats))]
                        axA.bar(cats, result["TON"], color=colors_b, edgecolor="white")
                        axA.set_title("TON", fontweight="bold"); axA.set_ylabel("TON")
                        axA.tick_params(axis="x", rotation=30)
                        axB.bar(cats, result["TOF (h⁻¹)"], color=colors_b, edgecolor="white")
                        axB.set_title("TOF (h⁻¹)", fontweight="bold"); axB.set_ylabel("TOF (h⁻¹)")
                        axB.tick_params(axis="x", rotation=30)
                        for ax in (axA, axB):
                            ax.grid(True, alpha=0.3, axis="y")
                        plt.tight_layout()
                        st.pyplot(fig)
                        tbl_buf = io.BytesIO()
                        result.to_excel(tbl_buf, index=False); tbl_buf.seek(0)
                        st.download_button("📥 Download tof_ton_summary.xlsx",
                                            data=tbl_buf, file_name="tof_ton_summary.xlsx",
                                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        except Exception as e:
            st.error(f"Error: {e}")
    elif uploaded_tof and C0_mol is None:
        st.error("Fix the C₀ unit in the sidebar first.")
    else:
        st.info("Upload your filled TOF/TON template. Need a template? Go to Tab 1.")


# ────────────────────────────────────────────────────────────
# TAB 4 — REUSABILITY (unchanged from v2.0)
# ────────────────────────────────────────────────────────────
with tab_reuse:
    st.subheader("Reusability / Recyclability")
    uploaded_reuse = st.file_uploader("Upload filled Reusability template",
                                       type=["xlsx","xls"], key="reuse_upl")
    if uploaded_reuse:
        try:
            xl = pd.ExcelFile(uploaded_reuse)
        except Exception as e:
            st.error(f"Could not open the Excel file: {e}")
            xl = None
        if xl is not None:
            sheets = [s for s in xl.sheet_names if s not in ("Instructions","Example")]
            if not sheets:
                st.warning("No catalyst sheets found.")
            else:
                data_ok, rows, errors = {}, [], []
                for sheet in sheets:
                    try:
                        raw = xl.parse(sheet)
                        df = raw.dropna(how="all")
                        if df.shape[1] != 2:
                            raise ValueError(f"expected 2 columns, found {df.shape[1]}")
                        df = df.dropna()
                        if len(df) < 1:
                            raise ValueError("no valid data rows found")
                        df.columns = ["Cycle","Removal (%)"]
                        cyc = df["Cycle"].values.astype(float)
                        x   = df["Removal (%)"].values.astype(float)
                        order = np.argsort(cyc)
                        cyc, x = cyc[order], x[order]
                        if x[0] == 0:
                            raise ValueError("Removal (%) for Cycle 1 is 0")
                        retention = np.round(x / x[0] * 100.0, 2)
                        data_ok[sheet] = {"cycle": cyc, "x": x, "retention": retention}
                        for c, xi, ri in zip(cyc, x, retention):
                            rows.append({"Catalyst": sheet, "Cycle": int(c),
                                          "Removal (%)": xi, "Retention (%)": ri})
                    except Exception as e:
                        errors.append((sheet, str(e)))
                if errors:
                    st.warning("⚠️ Skipped:\n\n" + "\n".join(f"- **{s}**: {m}" for s, m in errors))
                if rows:
                    result = pd.DataFrame(rows)
                    fig, ax = plt.subplots(figsize=(9, 5))
                    for i, sheet in enumerate(data_ok):
                        d = data_ok[sheet]
                        ax.plot(d["cycle"], d["x"], marker=MARKERS[i % len(MARKERS)],
                                color=COLORS[i % len(COLORS)], label=sheet, lw=2, ms=8,
                                markeredgecolor="white", markeredgewidth=0.7)
                    ax.set_xlabel("Cycle", fontsize=12)
                    ax.set_ylabel("Removal (%)", fontsize=12)
                    ax.set_title("Catalyst Reusability", fontsize=13, fontweight="bold")
                    ax.set_xticks(sorted(set(np.concatenate([d["cycle"] for d in data_ok.values()]))))
                    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
                    plt.tight_layout()
                    st.pyplot(fig)
                    st.dataframe(result, use_container_width=True)
                    tbl_buf = io.BytesIO()
                    result.to_excel(tbl_buf, index=False); tbl_buf.seek(0)
                    st.download_button("📥 Download reusability_summary.xlsx",
                                        data=tbl_buf, file_name="reusability_summary.xlsx",
                                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    else:
        st.info("Upload your filled Reusability template. Need a template? Go to Tab 1.")


# ────────────────────────────────────────────────────────────
# TAB 5 — PARAMETER EFFECT  (NEW in v3.0)
# ────────────────────────────────────────────────────────────
with tab_param:
    st.subheader("📉 Parameter Effect on Catalytic Performance")
    st.markdown(
        "Upload a **Parameter Effect template** (from Tab 1). Each sheet = one parameter "
        "(e.g. catalyst loading, H₂O₂ dose, DBT concentration, voltage, light intensity). "
        "Columns: `[parameter value, Catalyst-A, Catalyst-B, ...]`. "
        "Y-axis metric = X_final(%) or kapp or t½ — whichever you filled in."
    )
    uploaded_param = st.file_uploader("Upload Parameter Effect file", type=["xlsx","xls"], key="param_upl")
    metric_label = st.text_input("Y-axis label (e.g. 'X_final (%)', 'kapp (1/min)', 't½ (min)')",
                                  value="X_final (%)")

    if uploaded_param:
        try:
            xl_p = pd.ExcelFile(uploaded_param)
        except Exception as e:
            st.error(f"Cannot open file: {e}")
            xl_p = None

        if xl_p is not None:
            param_sheets = [s for s in xl_p.sheet_names if s not in ("Instructions",)]
            if not param_sheets:
                st.warning("No parameter sheets found.")
            else:
                all_figs_param = []
                for psheet in param_sheets:
                    try:
                        df_p = xl_p.parse(psheet).dropna(how="all").dropna(axis=1, how="all")
                        if df_p.shape[1] < 2:
                            st.warning(f"Sheet '{psheet}': need at least 2 columns.")
                            continue
                        param_col = df_p.columns[0]
                        cat_cols  = df_p.columns[1:].tolist()
                        df_p = df_p.dropna(subset=[param_col])
                        x_vals = df_p[param_col].values.astype(float)

                        fig_p, ax_p = plt.subplots(figsize=(9, 5))
                        for i, cat in enumerate(cat_cols):
                            y_vals = pd.to_numeric(df_p[cat], errors="coerce").values
                            valid = ~np.isnan(y_vals)
                            if valid.sum() < 1:
                                continue
                            ax_p.plot(x_vals[valid], y_vals[valid],
                                      marker=MARKERS[i % len(MARKERS)],
                                      color=COLORS[i % len(COLORS)],
                                      label=str(cat), lw=2, ms=8,
                                      markeredgecolor="white", markeredgewidth=0.7)
                        ax_p.set_xlabel(str(param_col), fontsize=12)
                        ax_p.set_ylabel(metric_label, fontsize=12)
                        ax_p.set_title(f"Effect of {param_col} on {metric_label}",
                                        fontsize=13, fontweight="bold")
                        ax_p.legend(fontsize=9); ax_p.grid(True, alpha=0.3)
                        plt.tight_layout()
                        st.pyplot(fig_p)
                        all_figs_param.append((psheet, fig_p))
                    except Exception as e:
                        st.error(f"Sheet '{psheet}': {e}")

                if all_figs_param:
                    zip_p = io.BytesIO()
                    with zipfile.ZipFile(zip_p, "w") as zf:
                        for name, fig in all_figs_param:
                            ib = io.BytesIO()
                            fig.savefig(ib, dpi=300, bbox_inches="tight")
                            ib.seek(0)
                            zf.writestr(f"param_effect_{name}.png", ib.read())
                    zip_p.seek(0)
                    st.download_button("📥 Download Parameter Effect plots (ZIP)",
                                        data=zip_p, file_name="param_effect_plots.zip",
                                        mime="application/zip")
    else:
        st.info("Upload a Parameter Effect file. Need the template? Go to Tab 1.")


# ────────────────────────────────────────────────────────────
# TAB 6 — OXIDANT EFFICIENCY  (NEW in v3.0)
# ────────────────────────────────────────────────────────────
with tab_oxeff:
    st.subheader("💧 Oxidant Efficiency (H₂O₂ Utilisation)")
    st.markdown(r"""
**Definition:**
$$\eta_{H_2O_2}(\%) = \frac{n_{DBT,removed}}{n_{H_2O_2,consumed}/2} \times 100$$

Stoichiometry: DBT + 2 H₂O₂ → DBTO₂ + 2 H₂O (1 : 2 ratio).  
Upload the **Oxidant Efficiency template** (from Tab 1) with columns:
`Catalyst | n_DBT_removed (mmol) | n_H2O2_consumed (mmol)`.
""")
    uploaded_ox = st.file_uploader("Upload Oxidant Efficiency file", type=["xlsx","xls"], key="ox_upl")

    if uploaded_ox:
        try:
            xl_ox = pd.ExcelFile(uploaded_ox)
            if "OxEff" not in xl_ox.sheet_names:
                st.error("Sheet 'OxEff' not found. Please use the template from Tab 1.")
            else:
                df_ox = xl_ox.parse("OxEff").dropna(how="all")
                required_ox = ["Catalyst","n_DBT_removed (mmol)","n_H2O2_consumed (mmol)"]
                missing_ox = [c for c in required_ox if c not in df_ox.columns]
                if missing_ox:
                    st.error(f"Missing columns: {', '.join(missing_ox)}")
                else:
                    df_ox = df_ox.dropna(subset=required_ox)
                    if df_ox.empty:
                        st.warning("No complete rows found.")
                    else:
                        df_ox = df_ox.copy()
                        df_ox["n_DBT_removed (mmol)"]  = pd.to_numeric(df_ox["n_DBT_removed (mmol)"],  errors="coerce")
                        df_ox["n_H2O2_consumed (mmol)"] = pd.to_numeric(df_ox["n_H2O2_consumed (mmol)"], errors="coerce")
                        df_ox["η_H2O2 (%)"] = (
                            df_ox["n_DBT_removed (mmol)"] /
                            (df_ox["n_H2O2_consumed (mmol)"] / 2.0)
                        ) * 100.0

                        st.markdown("### 📋 Oxidant Efficiency Summary")
                        disp_ox = df_ox.copy()
                        disp_ox["η_H2O2 (%)"] = disp_ox["η_H2O2 (%)"].apply(lambda v: f"{v:.2f}")
                        st.dataframe(disp_ox, use_container_width=True)

                        fig_ox, ax_ox = plt.subplots(figsize=(9, 5))
                        cats_ox = df_ox["Catalyst"].astype(str).tolist()
                        eta_vals = df_ox["η_H2O2 (%)"].tolist()
                        colors_ox = [COLORS[i % len(COLORS)] for i in range(len(cats_ox))]
                        ax_ox.bar(cats_ox, eta_vals, color=colors_ox, edgecolor="white")
                        ax_ox.axhline(100, color="red", ls="--", lw=1.2, alpha=0.7, label="100% theoretical max")
                        ax_ox.set_ylabel("η H₂O₂ (%)", fontsize=12)
                        ax_ox.set_title("H₂O₂ Oxidant Efficiency per Catalyst", fontsize=13, fontweight="bold")
                        ax_ox.tick_params(axis="x", rotation=30)
                        ax_ox.legend(fontsize=9); ax_ox.grid(True, alpha=0.3, axis="y")
                        plt.tight_layout()
                        st.pyplot(fig_ox)

                        tbl_buf_ox = io.BytesIO()
                        df_ox.to_excel(tbl_buf_ox, index=False); tbl_buf_ox.seek(0)
                        st.download_button("📥 Download ox_efficiency_summary.xlsx",
                                            data=tbl_buf_ox, file_name="ox_efficiency_summary.xlsx",
                                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                        img_ox = io.BytesIO()
                        fig_ox.savefig(img_ox, dpi=300, bbox_inches="tight"); img_ox.seek(0)
                        st.download_button("📥 Download ox_efficiency_plot.png",
                                            data=img_ox, file_name="ox_efficiency_plot.png",
                                            mime="image/png")
        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.info("Upload the Oxidant Efficiency file. Need the template? Go to Tab 1.")


# ────────────────────────────────────────────────────────────
# TAB 7 — CONDITION COMPARISON  (NEW in v3.0)
# ────────────────────────────────────────────────────────────
with tab_compare:
    st.subheader("🔀 Condition Comparison: Thermal vs UV vs ECODS …")
    st.markdown(
        "Upload **multiple `kinetics_summary.xlsx` files** (one per condition — "
        "generated from Tab 2). Label each file with its condition name. "
        "The app will overlay t½ and k (best model) for shared catalysts across conditions."
    )

    n_conditions = st.number_input("Number of conditions to compare", min_value=2, max_value=8,
                                    value=2, step=1)
    condition_files = []
    condition_labels = []
    cols_cond = st.columns(int(n_conditions))
    for i, col in enumerate(cols_cond):
        with col:
            lbl = st.text_input(f"Label #{i+1}", value=f"Condition_{i+1}", key=f"cond_lbl_{i}")
            f   = st.file_uploader(f"Upload kinetics_summary #{i+1}",
                                    type=["xlsx","xls"], key=f"cond_file_{i}")
            condition_labels.append(lbl)
            condition_files.append(f)

    metric_cmp = st.selectbox("Metric to compare", ["t½ (min)", "Best R2", "r₀ (mol/L/min)"])

    if all(f is not None for f in condition_files):
        try:
            dfs_cond = []
            for lbl, f in zip(condition_labels, condition_files):
                df_c = pd.read_excel(f)
                df_c["__condition__"] = lbl
                dfs_cond.append(df_c)

            # find the metric column — handle both v2 and v3 column names
            col_map = {
                "t½ (min)": "t½ (min)",
                "Best R2":  "Best R2",
                "r₀ (mol/L/min)": "r₀ (mol/L/min)",
            }
            metric_col = col_map[metric_cmp]

            # check metric column exists in all files
            missing_metric = [condition_labels[i] for i, df_c in enumerate(dfs_cond)
                               if metric_col not in df_c.columns]
            if missing_metric:
                if metric_col == "r₀ (mol/L/min)":
                    st.warning(
                        f"Column '{metric_col}' not found in: {', '.join(missing_metric)}. "
                        "This column requires v3.0 kinetics output. "
                        "Re-run kinetics in Tab 2 to generate updated files."
                    )
                else:
                    st.error(f"Column '{metric_col}' not found in: {', '.join(missing_metric)}.")
            else:
                combined = pd.concat(dfs_cond, ignore_index=True)
                catalysts_all = combined["Catalyst"].unique().tolist()

                # pivot: rows = catalysts, columns = conditions
                pivot = combined.pivot_table(index="Catalyst", columns="__condition__",
                                              values=metric_col, aggfunc="first")
                pivot = pivot.reindex(columns=condition_labels)

                st.markdown(f"### 📋 {metric_cmp} — All Conditions")
                st.dataframe(pivot.style.format("{:.3f}", na_rep="—"), use_container_width=True)

                # Grouped bar chart
                fig_cmp, ax_cmp = plt.subplots(figsize=(max(9, len(catalysts_all)*1.5), 5))
                x_pos = np.arange(len(catalysts_all))
                bar_w = 0.8 / len(condition_labels)
                for j, cond in enumerate(condition_labels):
                    vals = [pivot.loc[cat, cond] if cat in pivot.index else float("nan")
                             for cat in catalysts_all]
                    offset = (j - len(condition_labels)/2 + 0.5) * bar_w
                    ax_cmp.bar(x_pos + offset, vals, width=bar_w,
                                label=cond, color=COLORS[j % len(COLORS)],
                                edgecolor="white", alpha=0.9)
                ax_cmp.set_xticks(x_pos)
                ax_cmp.set_xticklabels(catalysts_all, rotation=30, ha="right")
                ax_cmp.set_ylabel(metric_cmp, fontsize=12)
                ax_cmp.set_title(f"{metric_cmp} Comparison Across Conditions", fontsize=13, fontweight="bold")
                ax_cmp.legend(fontsize=9); ax_cmp.grid(True, alpha=0.3, axis="y")
                plt.tight_layout()
                st.pyplot(fig_cmp)

                tbl_cmp = io.BytesIO()
                pivot.to_excel(tbl_cmp); tbl_cmp.seek(0)
                st.download_button("📥 Download condition_comparison.xlsx",
                                    data=tbl_cmp, file_name="condition_comparison.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                img_cmp = io.BytesIO()
                fig_cmp.savefig(img_cmp, dpi=300, bbox_inches="tight"); img_cmp.seek(0)
                st.download_button("📥 Download condition_comparison.png",
                                    data=img_cmp, file_name="condition_comparison.png",
                                    mime="image/png")
        except Exception as e:
            st.error(f"Error processing condition files: {e}")
    else:
        st.info(f"Upload all {n_conditions} kinetics_summary.xlsx files above (from Tab 2).")


st.markdown("---")
st.caption("ODS Calculation Suite v3.0 | CatLab-Tools by Hoda Jafari | MIT License")
