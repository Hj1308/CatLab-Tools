"""
ODS Calculation Suite — Streamlit Web App
==========================================
Extended build based on CatLab-Tools/app_ods.py (github.com/Hj1308/CatLab-Tools)
Original author: Hoda Jafari

v3.1 — Bug-fix release on top of v3.0
----------------------------------------
FIX 1: C₀ is now locked (fixed) in curve_fit for all models.
        Previously C₀ was a free parameter with ±50% bounds, causing each
        catalyst to get a slightly different C₀ — violating the assumption
        that C₀ is known from the experiment.
FIX 2: Second-order t½ corrected to 1/(k₂·C₀).
        Was incorrectly using ln2/k (the pseudo-first-order formula).
FIX 3: Extrapolation warning added when t½ < first data point.
FIX 4: Best-model selection now uses AIC instead of R².
        Elovich was always winning because it has more parameters; AIC
        penalises model complexity. Adj-R² and AIC are shown in the table.
FIX 5: r₀/m formula corrected — r₀ × V_fuel / m_cat.
        Previous code omitted V_fuel, causing a ~500× error in the
        mass-normalised activity metric.

v3.0 — additions on top of v2.0
----------------------------------------
NEW  : k ± SE (standard error from curve_fit covariance matrix) for all models.
NEW  : r₀ (initial reaction rate) — model-independent activity metric.
NEW  : Langmuir-Hinshelwood (L-H) model as 5th kinetic model.
NEW  : Tab 5 — Parameter Effect.
NEW  : Tab 6 — Oxidant Efficiency.
NEW  : Tab 7 — Condition Comparison.
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

# ── Publication-quality plot style (applies to every figure) ──
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

# Number of free parameters per model (used for AIC / Adj-R²)
# C₀ is now FIXED, so it does NOT count as a fitted parameter.
N_PARAMS = {
    "Zero-order":   1,   # k0
    "Pseudo-first": 1,   # kapp
    "Second-order": 1,   # k2
    "Elovich":      2,   # alpha, beta
    "L-H":          2,   # k_LH, K_ads
}

# Common ODS/ODN substrates with molecular weight (g/mol), for C0 unit conversion
SUBSTRATES = {
    "DBT (Dibenzothiophene)":        184.26,
    "BT (Benzothiophene)":           134.20,
    "4,6-DMDBT":                     212.31,
    "4-MDBT":                        198.28,
    "Thiophene":                     84.14,
    "Custom / other":                None,
}


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

# FIX 4 — model-selection helpers ─────────────────────────────
def _adj_r2(r2, n, p):
    """Adjusted R²: penalises extra parameters. n=data points, p=free params."""
    if n <= p + 1:
        return float("nan")
    return round(1 - (1 - r2) * (n - 1) / (n - p - 1), 4)

def _aic(y_obs, y_pred, p):
    """
    AIC = n·ln(RSS/n) + 2p
    Lower AIC → better model (balances fit quality against complexity).
    """
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


# ── Non-linear fitting engine (v3.1 — C₀ fixed, AIC, corrected t½ & r₀/m) ──
def _fit_nonlinear(time, Ct, C0):
    t  = np.asarray(time, dtype=float)
    Ct = np.asarray(Ct,   dtype=float)
    n  = len(t)
    results = {}

    # ── Zero-order  (FIX 1: C0 fixed via lambda) ──────────────
    try:
        p, pcov = curve_fit(
            lambda t_, k: _zero_order(t_, k, C0),
            t, Ct, p0=[1e-6], bounds=([0], [np.inf]), maxfev=5000
        )
        se   = np.sqrt(np.diag(pcov))
        k0   = p[0]
        pred = _zero_order(t, k0, C0)
        r2v  = _r2(Ct, pred)
        np_  = N_PARAMS["Zero-order"]
        results["Zero-order"] = {
            "params":  (k0, C0), "R2": r2v, "pred": pred,
            "adj_r2":  _adj_r2(r2v, n, np_),
            "aic":     _aic(Ct, pred, np_),
            "label":   f"k₀ = {_fmt_sci(k0)} mol·L⁻¹·min⁻¹",
            "t_half":  round(0.5 * C0 / k0, 2) if k0 > 0 else float("nan"),
            "k": k0, "k_se": se[0], "col_k": "K0 (mol/L/min)",
            "r0": k0, "r0_se": se[0],
        }
    except Exception as e:
        results["Zero-order"] = {"R2": -999, "aic": float("inf"), "error": str(e)}

    # ── Pseudo-first-order  (FIX 1: C0 fixed) ─────────────────
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
            "params":  (kapp, C0), "R2": r2v, "pred": pred,
            "adj_r2":  _adj_r2(r2v, n, np_),
            "aic":     _aic(Ct, pred, np_),
            "label":   f"kₐₚₚ = {_fmt_sci(kapp)} min⁻¹",
            "t_half":  round(np.log(2) / kapp, 2) if kapp > 0 else float("nan"),
            "k": kapp, "k_se": se[0], "col_k": "Kapp (1/min)",
            "r0": r0, "r0_se": se[0] * C0,
        }
    except Exception as e:
        results["Pseudo-first"] = {"R2": -999, "aic": float("inf"), "error": str(e)}

    # ── Second-order  (FIX 1: C0 fixed; FIX 2: t½ = 1/(k2·C0)) ──
    try:
        p, pcov = curve_fit(
            lambda t_, k: _second_order(t_, k, C0),
            t, Ct, p0=[1e-3], bounds=([0], [np.inf]), maxfev=5000
        )
        se   = np.sqrt(np.diag(pcov))
        k2   = p[0]
        pred = _second_order(t, k2, C0)
        r2v  = _r2(Ct, pred)
        np_  = N_PARAMS["Second-order"]
        r0   = k2 * C0 ** 2
        results["Second-order"] = {
            "params":  (k2, C0), "R2": r2v, "pred": pred,
            "adj_r2":  _adj_r2(r2v, n, np_),
            "aic":     _aic(Ct, pred, np_),
            "label":   f"k₂ = {_fmt_sci(k2)} L·mol⁻¹·min⁻¹",
            # FIX 2: correct formula t½ = 1 / (k2 * C0)
            "t_half":  round(1.0 / (k2 * C0), 2) if k2 > 0 else float("nan"),
            "k": k2, "k_se": se[0], "col_k": "K2 (L/mol/min)",
            "r0": r0, "r0_se": se[0] * C0 ** 2,
        }
    except Exception as e:
        results["Second-order"] = {"R2": -999, "aic": float("inf"), "error": str(e)}

    # ── Elovich ────────────────────────────────────────────────
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
        pred = _elovich(t, alpha, beta, C0)
        r2v  = _r2(Ct, pred)
        np_  = N_PARAMS["Elovich"]
        results["Elovich"] = {
            "params":  (alpha, beta, C0), "R2": r2v, "pred": pred,
            "adj_r2":  _adj_r2(r2v, n, np_),
            "aic":     _aic(Ct, pred, np_),
            "label":   f"α={_fmt_sci(alpha)}, β={_fmt_sci(beta)}",
            "t_half":  _elovich_t_half(C0, alpha, beta),
            "k": alpha, "k_se": se[0], "beta": beta, "col_k": "α (mol/L/min)",
            "r0": alpha, "r0_se": se[0],
        }
    except Exception as e:
        results["Elovich"] = {"R2": -999, "aic": float("inf"), "error": str(e)}

    # ── Langmuir-Hinshelwood ───────────────────────────────────
    try:
        if len(t) < 3:
            raise ValueError("L-H model needs at least 3 time points")
        p, pcov = curve_fit(
            lambda tt, kl, ka: _lh_model(tt, kl, ka, C0),
            t, Ct,
            p0=[0.01, 10.0],
            bounds=([1e-8, 1e-4], [1e3, 1e6]),
            maxfev=15000
        )
        se = np.sqrt(np.diag(pcov))
        k_LH, K_ads = p
        pred = _lh_model(t, k_LH, K_ads, C0)
        r2v  = _r2(Ct, pred)
        np_  = N_PARAMS["L-H"]
        r0_lh = k_LH * K_ads * C0 / (1.0 + K_ads * C0)
        results["L-H"] = {
            "params":  (k_LH, K_ads, C0), "R2": r2v, "pred": pred,
            "adj_r2":  _adj_r2(r2v, n, np_),
            "aic":     _aic(Ct, pred, np_),
            "label":   f"kLH={_fmt_sci(k_LH)}, K={_fmt_sci(K_ads)}",
            "t_half":  _lh_t_half(C0, k_LH, K_ads),
            "k": k_LH, "k_se": se[0], "K_ads": K_ads, "K_ads_se": se[1],
            "col_k":   "kLH (1/min)",
            "r0": r0_lh, "r0_se": float("nan"),
        }
    except Exception as e:
        results["L-H"] = {"R2": -999, "aic": float("inf"), "error": str(e)}

    # FIX 4: select best model by AIC (lower = better), not R² ──
    candidates = [m for m in results if m != "Zero-order"]
    best_name = min(candidates, key=lambda m: results[m].get("aic", float("inf")))
    for name in results:
        results[name]["is_best"] = (name == best_name)
    return results


def _plot_model(fits_all, sheets, model_name, ylabel, title):
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for i, sheet in enumerate(sheets):
        f   = fits_all[sheet]
        res = f["fits"].get(model_name, {})
        if res.get("R2", -999) < -100:
            continue
        t  = f["t"]; Ct = f["Ct"]
        c  = COLORS[i % len(COLORS)]; m = MARKERS[i % len(MARKERS)]
        lbl = f"{sheet}  (R²={res['R2']:.4f}"
        if res.get("is_best"):
            lbl += ", best✓"
        lbl += ")"
        ax.scatter(t, Ct, color=c, marker=m, s=75, zorder=3,
                   edgecolors="black", linewidths=0.7)
        t_max = t[-1]
        if model_name == "Zero-order":
            k0, C0_fit = res["params"]
            if k0 > 0:
                t_max = min(t[-1], C0_fit / k0)
        t_fit = np.linspace(0, t_max, 300)
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
        ax.plot(t_fit, pred_fit, "-", color=c, lw=2, zorder=2, label=lbl)

        th = res.get("t_half", float("nan"))
        if res.get("is_best") and not np.isnan(th) and not np.isinf(th) and 0 < th <= t[-1] * 1.5:
            ax.axvline(th, color=c, ls=":", lw=1.3, alpha=0.8, zorder=1)

    ax.set_xlabel("Time (min)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.margins(x=0.03, y=0.08)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0)
    plt.tight_layout()
    return fig


def _linreg_r2(x, y):
    A = np.vstack([x, np.ones_like(x)]).T
    slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
    yhat = slope * x + intercept
    return slope, intercept, _r2(y, yhat)


def _plot_linearization(fits_all, sheets, C0_mol, transform, ylabel, title):
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    lin_results = {}
    for i, sheet in enumerate(sheets):
        f  = fits_all[sheet]
        t  = f["t"]; Ct = f["Ct"]
        y  = transform(Ct, C0_mol)
        slope, intercept, r2v = _linreg_r2(t, y)
        lin_results[sheet] = (slope, intercept, r2v)
        c  = COLORS[i % len(COLORS)]; m = MARKERS[i % len(MARKERS)]
        ax.scatter(t, y, color=c, marker=m, s=75, zorder=3,
                   edgecolors="black", linewidths=0.7)
        t_fit = np.array([0, t[-1]])
        ax.plot(t_fit, slope*t_fit + intercept, "-", color=c, lw=2, zorder=2,
                label=f"{sheet}  (R²={r2v:.4f})")
    ax.set_xlabel("Time (min)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.margins(x=0.03, y=0.08)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0)
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
    "**CatLab-Tools v3.1** — based on the work of Hoda Jafari · "
    "[GitHub](https://github.com/Hj1308/CatLab-Tools)"
)
st.markdown("---")

with st.sidebar:
    st.header("⚙️ Global Parameters")

    substrate = st.selectbox("Substrate / pollutant", list(SUBSTRATES.keys()), index=0,
                              help="Used to look up the molecular weight for "
                                   "ppm/mg·L⁻¹/g·L⁻¹ → mol·L⁻¹ conversion. "
                                   "Select 'Custom / other' to enter your own MW.")
    mw_default = SUBSTRATES[substrate] if SUBSTRATES[substrate] is not None else 184.26
    mw_poll = st.number_input("Molecular weight (g/mol)", value=float(mw_default),
                               min_value=1.0, step=0.01,
                               help="Auto-filled from the substrate above; "
                                    "edit freely if your value differs.")

    c0_val  = st.number_input("C₀ (initial concentration)", value=250.0, min_value=0.0001)
    c0_unit = st.selectbox("C₀ unit", ["ppmS", "ppm", "mg/L", "g/L", "mmol/L", "mol/L"])
    mw_for_conversion = MW_S if c0_unit == "ppmS" else mw_poll
    try:
        C0_mol = _to_mol_L(c0_val, c0_unit, mw_for_conversion)
        st.caption(f"C₀ = {_fmt_sci(C0_mol)} mol/L")
    except Exception as e:
        C0_mol = None
        st.error(f"Unit error: {e}")

    st.markdown("**Reaction conditions** (used for mass-normalised rate & figure captions)")
    cat_mass_mg = st.number_input("Catalyst mass (mg)", value=20.0, min_value=0.0, step=1.0)
    fuel_vol_ml = st.number_input("Fuel / model-oil volume (mL)", value=5.0, min_value=0.0, step=0.5)
    cat_mass_g  = cat_mass_mg / 1000.0
    fuel_vol_L  = fuel_vol_ml / 1000.0
    conditions_caption = (f"{substrate.split(' (')[0]}, C₀={c0_val:g} {c0_unit}, "
                           f"{cat_mass_mg:g} mg catalyst, {fuel_vol_ml:g} mL fuel")
    st.caption(conditions_caption)

(tab_templates, tab_kinetics, tab_tof, tab_reuse,
 tab_param, tab_oxeff, tab_compare) = st.tabs([
    "📥 1. Templates",
    "📊 2. Kinetics & t½",
    "⚙️ 3. TOF / TON",
    "♻️ 4. Reusability",
    "📉 5. Parameter Effect",
    "💧 6. Oxidant Efficiency",
    "🔀 7. Condition Comparison",
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
        kin_buf,  kin_map  = make_kinetics_template(names)
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
    with st.expander("📐 Formulas used in this app (v3.1)"):
        st.markdown(r"""
**Kinetic models**

| Model | Equation | t½ |
|---|---|---|
| Zero-order | $C_t = C_0 - k_0 t$ | $C_0/(2k_0)$ |
| Pseudo-first | $C_t = C_0 e^{-k_{app}t}$ | $\ln 2 / k_{app}$ |
| Second-order | $C_t = C_0/(1+k_2 C_0 t)$ | $1/(k_2 C_0)$ ✓ |
| Elovich | $C_t = C_0 - \frac{1}{\beta}\ln(1+\alpha\beta t)$ | $(e^{C_0\beta/2}-1)/(\alpha\beta)$ |
| **L-H** | $\frac{dC}{dt} = -\frac{k_{LH} K C}{1+KC}$ (ODE) | $\frac{\ln 2}{k_{LH}K} + \frac{C_0/2}{k_{LH}}$ |

**Best-model selection (v3.1):** AIC = n·ln(RSS/n) + 2p  — lower is better.

**Initial rate r₀ (mol·L⁻¹·min⁻¹)**
$$r_0^{(1)} = k_{app} C_0 \qquad r_0^{(2)} = k_2 C_0^2 \qquad r_0^{Elovich} = \alpha \qquad r_0^{LH} = \frac{k_{LH} K C_0}{1+K C_0}$$

**Mass-normalised rate (v3.1 — corrected):**
$$r_0/m \;(\text{mol·g}^{-1}\text{·min}^{-1}) = \frac{r_0 \times V_{fuel}}{m_{cat}}$$

**Standard Error**: $SE_k = \sqrt{[\Sigma^{-1}]_{kk}}$ where $\Sigma$ = covariance matrix from `scipy.curve_fit`.

**Oxidant Efficiency**
$$\eta_{H_2O_2}(\%) = \frac{n_{DBT,removed}}{n_{H_2O_2,consumed}/2} \times 100$$

**TOF / TON**
$$TON = \frac{n_{sub}}{n_{sites}} \qquad TOF = \frac{TON}{t(h)}$$
""")


# ────────────────────────────────────────────────────────────
# TAB 2 — KINETICS & t½  (v3.1)
# ────────────────────────────────────────────────────────────
with tab_kinetics:
    st.subheader("Kinetics fitting (Zero / Pseudo-first / Second-order / Elovich / L-H) + k±SE + r₀")
    st.info(
        "**v3.1 fixes applied:** C₀ is now fixed (not fitted). "
        "Second-order t½ uses 1/(k₂·C₀). Best model selected by AIC. "
        "r₀/m includes fuel volume. Extrapolation warnings shown."
    )
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
                        raw   = xl.parse(sheet)
                        df    = raw.dropna(how="all")
                        if df.shape[1] != 2:
                            raise ValueError(f"expected 2 columns, found {df.shape[1]}")
                        df    = df.dropna()
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

                        candidates   = ["Pseudo-first", "Second-order", "Elovich", "L-H"]
                        # FIX 4: use AIC for best model
                        best         = min(candidates,
                                           key=lambda m: fits[m].get("aic", float("inf")))
                        r1           = fits["Pseudo-first"].get("R2", -999)
                        r2v          = fits["Second-order"].get("R2", -999)
                        order_label  = "Second-order" if r2v > r1 else "Pseudo-first-order"
                        best_res     = fits[best]

                        # FIX 3: extrapolation warning
                        t_half_val       = best_res.get("t_half", float("nan"))
                        first_time_point = t_arr[0] if len(t_arr) > 0 else 0.0
                        if (not np.isnan(t_half_val) and
                                not np.isinf(t_half_val) and
                                first_time_point > 0 and
                                t_half_val < first_time_point):
                            st.warning(
                                f"⚠️ **{sheet}** — t½ = {t_half_val:.2f} min is **before** the "
                                f"first data point ({first_time_point:.1f} min). "
                                f"This value is **extrapolated** and should be interpreted "
                                f"with caution."
                            )

                        # FIX 5: r₀/m = r₀ × V_fuel / m_cat  (corrected)
                        r0_val    = best_res.get("r0",    float("nan"))
                        r0_se_val = best_res.get("r0_se", float("nan"))
                        if cat_mass_g > 0 and fuel_vol_L > 0:
                            r0m     = r0_val    * fuel_vol_L / cat_mass_g
                            r0m_se  = r0_se_val * fuel_vol_L / cat_mass_g
                        else:
                            r0m = r0m_se = float("nan")

                        rows.append({
                            "Catalyst":              sheet,
                            "X_final (%)":           round(rem[-1]*100, 1),
                            # Zero-order
                            "K0 (mol/L/min)":        fits["Zero-order"].get("k",    float("nan")),
                            "K0_SE":                 fits["Zero-order"].get("k_se", float("nan")),
                            "R2_zero":               fits["Zero-order"].get("R2",   -999),
                            "AdjR2_zero":            fits["Zero-order"].get("adj_r2", float("nan")),
                            "AIC_zero":              fits["Zero-order"].get("aic",  float("nan")),
                            # Pseudo-first
                            "Kapp (1/min)":          fits["Pseudo-first"].get("k",    float("nan")),
                            "Kapp_SE":               fits["Pseudo-first"].get("k_se", float("nan")),
                            "R2_first":              fits["Pseudo-first"].get("R2",   -999),
                            "AdjR2_first":           fits["Pseudo-first"].get("adj_r2", float("nan")),
                            "AIC_first":             fits["Pseudo-first"].get("aic",  float("nan")),
                            # Second-order
                            "K2 (L/mol/min)":        fits["Second-order"].get("k",    float("nan")),
                            "K2_SE":                 fits["Second-order"].get("k_se", float("nan")),
                            "R2_second":             fits["Second-order"].get("R2",   -999),
                            "AdjR2_second":          fits["Second-order"].get("adj_r2", float("nan")),
                            "AIC_second":            fits["Second-order"].get("aic",  float("nan")),
                            # Elovich
                            "Elovich α":             fits["Elovich"].get("k",    float("nan")),
                            "Elovich α_SE":          fits["Elovich"].get("k_se", float("nan")),
                            "Elovich β":             fits["Elovich"].get("beta", float("nan")),
                            "R2_elovich":            fits["Elovich"].get("R2",   -999),
                            "AdjR2_elovich":         fits["Elovich"].get("adj_r2", float("nan")),
                            "AIC_elovich":           fits["Elovich"].get("aic",  float("nan")),
                            # L-H
                            "kLH (1/min)":           fits["L-H"].get("k",       float("nan")),
                            "kLH_SE":                fits["L-H"].get("k_se",    float("nan")),
                            "K_ads (L/mol)":         fits["L-H"].get("K_ads",   float("nan")),
                            "R2_LH":                 fits["L-H"].get("R2",      -999),
                            "AdjR2_LH":              fits["L-H"].get("adj_r2",  float("nan")),
                            "AIC_LH":                fits["L-H"].get("aic",     float("nan")),
                            # Best model
                            "Reaction Order":        order_label,
                            "Best Model (AIC)":      best,
                            "Best R2":               best_res.get("R2",      float("nan")),
                            "Best AdjR2":            best_res.get("adj_r2",  float("nan")),
                            "Best AIC":              best_res.get("aic",     float("nan")),
                            "t½ (min)":              t_half_val,
                            # r₀
                            "r₀ (mol/L/min)":        r0_val,
                            "r₀_SE":                 r0_se_val,
                            # FIX 5: corrected r₀/m with V_fuel
                            "r₀/m (mol/g_cat/min)":  r0m,
                            "r₀/m_SE":               r0m_se,
                        })
                    except Exception as e:
                        errors.append((sheet, str(e)))

                if errors:
                    msg = "\n".join(f"- **{s}**: {m}" for s, m in errors)
                    st.warning("⚠️ Some sheets skipped:\n\n" + msg)

                if rows:
                    summary = pd.DataFrame(rows)

                    st.markdown("### 📈 Plots")
                    fig0, ax0 = plt.subplots(figsize=(8.5, 5.2))
                    for i, sheet in enumerate(fits_all):
                        f = fits_all[sheet]
                        ax0.plot(f["t"], (1 - f["Ct"]/C0_mol)*100,
                                 marker=MARKERS[i % len(MARKERS)],
                                 color=COLORS[i % len(COLORS)],
                                 label=sheet, lw=2, ms=8,
                                 markeredgecolor="black", markeredgewidth=0.7)
                    ax0.set_xlabel("Time (min)")
                    ax0.set_ylabel("Removal (%)")
                    ax0.set_title("ODS Performance — All Catalysts")
                    ax0.margins(x=0.03, y=0.08)
                    ax0.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0)
                    plt.tight_layout()
                    st.pyplot(fig0)
                    st.caption(f"Reaction conditions: {conditions_caption}")

                    sheets_ok = list(fits_all.keys())
                    fig1  = _plot_model(fits_all, sheets_ok, "Zero-order",
                                        "Cₜ (mol·L⁻¹)", "Zero-Order Fit")
                    fig2  = _plot_model(fits_all, sheets_ok, "Pseudo-first",
                                        "Cₜ (mol·L⁻¹)", "Pseudo-First-Order Fit")
                    fig3  = _plot_model(fits_all, sheets_ok, "Second-order",
                                        "Cₜ (mol·L⁻¹)", "Second-Order Fit")
                    fig4  = _plot_model(fits_all, sheets_ok, "Elovich",
                                        "Cₜ (mol·L⁻¹)", "Elovich Model Fit")
                    fig4b = _plot_model(fits_all, sheets_ok, "L-H",
                                        "Cₜ (mol·L⁻¹)", "Langmuir-Hinshelwood Fit")

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

                    summary["Kapp_lin (1/min)"]   = summary["Catalyst"].map(lambda s: lin_first[s][0])
                    summary["R2_first_lin"]        = summary["Catalyst"].map(lambda s: lin_first[s][2])
                    summary["K2_lin (L/mol/min)"]  = summary["Catalyst"].map(lambda s: lin_second[s][0])
                    summary["R2_second_lin"]       = summary["Catalyst"].map(lambda s: lin_second[s][2])

                    # r₀ bar chart
                    st.markdown("### 📊 Initial Rate r₀ Comparison")
                    fig_r0, ax_r0 = plt.subplots(figsize=(9, 4))
                    cats_r0   = summary["Catalyst"].tolist()
                    r0_vals   = summary["r₀ (mol/L/min)"].tolist()
                    r0_ses    = summary["r₀_SE"].tolist()
                    colors_r0 = [COLORS[i % len(COLORS)] for i in range(len(cats_r0))]
                    ax_r0.bar(cats_r0, r0_vals, color=colors_r0,
                              yerr=[s if not np.isnan(s) else 0 for s in r0_ses],
                              capsize=5, edgecolor="white")
                    ax_r0.set_ylabel("r₀ (mol·L⁻¹·min⁻¹)", fontsize=12)
                    ax_r0.set_title("Initial Reaction Rate r₀ per Catalyst",
                                    fontsize=13, fontweight="bold")
                    ax_r0.tick_params(axis="x", rotation=30)
                    ax_r0.grid(True, alpha=0.25, axis="y", linewidth=0.6, color="0.7", zorder=0)
                    plt.tight_layout()
                    st.pyplot(fig_r0)

                    st.markdown("---")
                    st.markdown("### 📋 Kinetics Summary (v3.1 — AIC, corrected t½ & r₀/m)")
                    disp = summary.copy()
                    disp["K0 ± SE"]   = disp.apply(lambda r: _fmt_pm(r["K0 (mol/L/min)"],   r["K0_SE"]),    axis=1)
                    disp["Kapp ± SE"] = disp.apply(lambda r: _fmt_pm(r["Kapp (1/min)"],      r["Kapp_SE"]),  axis=1)
                    disp["K2 ± SE"]   = disp.apply(lambda r: _fmt_pm(r["K2 (L/mol/min)"],   r["K2_SE"]),    axis=1)
                    disp["α ± SE"]    = disp.apply(lambda r: _fmt_pm(r["Elovich α"],         r["Elovich α_SE"]), axis=1)
                    disp["kLH ± SE"]  = disp.apply(lambda r: _fmt_pm(r["kLH (1/min)"],      r["kLH_SE"]),   axis=1)
                    disp["r₀ ± SE"]   = disp.apply(lambda r: _fmt_pm(r["r₀ (mol/L/min)"],   r["r₀_SE"]),    axis=1)
                    disp["r₀/m ± SE"] = disp.apply(lambda r: _fmt_pm(r["r₀/m (mol/g_cat/min)"], r["r₀/m_SE"]), axis=1)
                    for col in ["R2_zero","R2_first","R2_second","R2_elovich","R2_LH","Best R2",
                                "R2_first_lin","R2_second_lin"]:
                        disp[col] = disp[col].apply(lambda v: f"{v:.4f}" if v != -999 else "—")
                    for col in ["AdjR2_first","AdjR2_second","AdjR2_elovich","AdjR2_LH","Best AdjR2"]:
                        disp[col] = disp[col].apply(lambda v: f"{v:.4f}" if not np.isnan(v) else "—")
                    for col in ["AIC_first","AIC_second","AIC_elovich","AIC_LH","Best AIC"]:
                        disp[col] = disp[col].apply(lambda v: f"{v:.2f}" if not np.isnan(v) else "—")
                    for col in ["Kapp_lin (1/min)", "K2_lin (L/mol/min)"]:
                        disp[col] = disp[col].apply(_fmt_sci)
                    disp["t½ (min)"]    = disp["t½ (min)"].apply(_fmt_thalf)
                    disp["K_ads (L/mol)"] = disp["K_ads (L/mol)"].apply(_fmt_sci)

                    show_cols = [
                        "Catalyst", "X_final (%)",
                        "K0 ± SE", "R2_zero", "AIC_zero",
                        "Kapp ± SE", "R2_first", "AdjR2_first", "AIC_first",
                        "K2 ± SE", "R2_second", "AdjR2_second", "AIC_second",
                        "α ± SE", "R2_elovich", "AdjR2_elovich", "AIC_elovich",
                        "kLH ± SE", "R2_LH", "AdjR2_LH", "AIC_LH",
                        "Reaction Order", "Best Model (AIC)", "Best R2", "Best AdjR2", "Best AIC",
                        "t½ (min)", "r₀ ± SE", "r₀/m ± SE",
                        "Kapp_lin (1/min)", "R2_first_lin",
                        "K2_lin (L/mol/min)", "R2_second_lin",
                    ]
                    st.caption(
                        f"Reaction conditions: {conditions_caption}. "
                        f"Best model selected by AIC (lower = better fit, penalised for extra parameters). "
                        f"r₀/m = r₀ × V_fuel ÷ m_cat "
                        f"({fuel_vol_ml:g} mL × r₀ ÷ {cat_mass_mg:g} mg) — "
                        f"mass-based activity in mol·g⁻¹·min⁻¹."
                    )
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
                                            ("02_zero_order.png",      fig1),
                                            ("03_pseudo_first.png",    fig2),
                                            ("04_second_order.png",    fig3),
                                            ("05_elovich.png",         fig4),
                                            ("05b_LH.png",             fig4b),
                                            ("06_pseudo_first_linear.png", fig5),
                                            ("07_second_order_linear.png", fig6),
                                            ("08_r0_comparison.png",   fig_r0)]:
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
                            mass        = float(r["Catalyst mass (g)"])
                            sites       = float(r["Active sites (mmol/g)"])
                            vol         = float(r["Fuel volume (L)"])
                            X           = float(r["Removal (%)"])
                            th          = float(r["Reaction time (h)"])
                            n_sites_mol = mass * sites / 1000.0
                            delta_C     = C0_mol * (X / 100.0)
                            n_sub_mol   = delta_C * vol
                            if n_sites_mol <= 0:
                                ton = tof = float("nan")
                            else:
                                ton = n_sub_mol / n_sites_mol
                                tof = ton / th if th > 0 else float("nan")
                            rows.append({"Catalyst": r["Catalyst"], "Removal (%)": X,
                                          "n_sites (mmol)":     n_sites_mol*1000,
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
                        cats     = result["Catalyst"].astype(str)
                        colors_b = [COLORS[i % len(COLORS)] for i in range(len(cats))]
                        axA.bar(cats, result["TON"],       color=colors_b, edgecolor="white")
                        axA.set_title("TON", fontweight="bold"); axA.set_ylabel("TON")
                        axA.tick_params(axis="x", rotation=30)
                        axB.bar(cats, result["TOF (h⁻¹)"], color=colors_b, edgecolor="white")
                        axB.set_title("TOF (h⁻¹)", fontweight="bold"); axB.set_ylabel("TOF (h⁻¹)")
                        axB.tick_params(axis="x", rotation=30)
                        for ax in (axA, axB):
                            ax.grid(True, alpha=0.25, axis="y", linewidth=0.6, color="0.7", zorder=0)
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
                        df  = raw.dropna(how="all")
                        if df.shape[1] != 2:
                            raise ValueError(f"expected 2 columns, found {df.shape[1]}")
                        df  = df.dropna()
                        if len(df) < 1:
                            raise ValueError("no valid data rows found")
                        df.columns = ["Cycle","Removal (%)"]
                        cyc   = df["Cycle"].values.astype(float)
                        x     = df["Removal (%)"].values.astype(float)
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
                    ax.legend(fontsize=9)
                    ax.grid(True, alpha=0.25, axis="y", linewidth=0.6, color="0.7", zorder=0)
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
# TAB 5 — PARAMETER EFFECT
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
    metric_label   = st.text_input("Y-axis label (e.g. 'X_final (%)', 'kapp (1/min)', 't½ (min)')",
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
                        df_p      = xl_p.parse(psheet).dropna(how="all").dropna(axis=1, how="all")
                        if df_p.shape[1] < 2:
                            st.warning(f"Sheet '{psheet}': need at least 2 columns.")
                            continue
                        param_col = df_p.columns[0]
                        cat_cols  = df_p.columns[1:].tolist()
                        df_p      = df_p.dropna(subset=[param_col])
                        x_vals    = df_p[param_col].values.astype(float)

                        fig_p, ax_p = plt.subplots(figsize=(9, 5))
                        for i, cat in enumerate(cat_cols):
                            y_vals = pd.to_numeric(df_p[cat], errors="coerce").values
                            valid  = ~np.isnan(y_vals)
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
                        ax_p.legend(fontsize=9)
                        ax_p.grid(True, alpha=0.25, axis="y", linewidth=0.6, color="0.7", zorder=0)
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
# TAB 6 — OXIDANT EFFICIENCY
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
                df_ox      = xl_ox.parse("OxEff").dropna(how="all")
                required_ox = ["Catalyst","n_DBT_removed (mmol)","n_H2O2_consumed (mmol)"]
                missing_ox  = [c for c in required_ox if c not in df_ox.columns]
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
                        cats_ox   = df_ox["Catalyst"].astype(str).tolist()
                        eta_vals  = df_ox["η_H2O2 (%)"].tolist()
                        colors_ox = [COLORS[i % len(COLORS)] for i in range(len(cats_ox))]
                        ax_ox.bar(cats_ox, eta_vals, color=colors_ox, edgecolor="white")
                        ax_ox.axhline(100, color="red", ls="--", lw=1.2, alpha=0.7,
                                      label="100% theoretical max")
                        ax_ox.set_ylabel("η H₂O₂ (%)", fontsize=12)
                        ax_ox.set_title("H₂O₂ Oxidant Efficiency per Catalyst",
                                        fontsize=13, fontweight="bold")
                        ax_ox.tick_params(axis="x", rotation=30)
                        ax_ox.legend(fontsize=9)
                        ax_ox.grid(True, alpha=0.25, axis="y", linewidth=0.6, color="0.7", zorder=0)
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
# TAB 7 — CONDITION COMPARISON
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
    condition_files  = []
    condition_labels = []
    cols_cond = st.columns(int(n_conditions))
    for i, col in enumerate(cols_cond):
        with col:
            lbl = st.text_input(f"Label #{i+1}", value=f"Condition_{i+1}", key=f"cond_lbl_{i}")
            f   = st.file_uploader(f"Upload kinetics_summary #{i+1}",
                                    type=["xlsx","xls"], key=f"cond_file_{i}")
            condition_labels.append(lbl)
            condition_files.append(f)

    metric_cmp = st.selectbox("Metric to compare",
                               ["t½ (min)", "Best R2", "r₀ (mol/L/min)", "Best AIC"])

    if all(f is not None for f in condition_files):
        try:
            dfs_cond = []
            for lbl, f in zip(condition_labels, condition_files):
                df_c = pd.read_excel(f)
                df_c["__condition__"] = lbl
                dfs_cond.append(df_c)

            col_map = {
                "t½ (min)":          "t½ (min)",
                "Best R2":           "Best R2",
                "r₀ (mol/L/min)":    "r₀ (mol/L/min)",
                "Best AIC":          "Best AIC",
            }
            metric_col = col_map[metric_cmp]

            missing_metric = [condition_labels[i] for i, df_c in enumerate(dfs_cond)
                               if metric_col not in df_c.columns]
            if missing_metric:
                st.warning(
                    f"Column '{metric_col}' not found in: {', '.join(missing_metric)}. "
                    "Re-run kinetics in Tab 2 to generate v3.1 output files."
                )
            else:
                combined     = pd.concat(dfs_cond, ignore_index=True)
                catalysts_all = combined["Catalyst"].unique().tolist()
                pivot = combined.pivot_table(index="Catalyst", columns="__condition__",
                                              values=metric_col, aggfunc="first")
                pivot = pivot.reindex(columns=condition_labels)

                st.markdown(f"### 📋 {metric_cmp} — All Conditions")
                st.dataframe(pivot.style.format("{:.3f}", na_rep="—"), use_container_width=True)

                fig_cmp, ax_cmp = plt.subplots(figsize=(max(9, len(catalysts_all)*1.5), 5))
                x_pos = np.arange(len(catalysts_all))
                bar_w = 0.8 / len(condition_labels)
                for j, cond in enumerate(condition_labels):
                    vals   = [pivot.loc[cat, cond] if cat in pivot.index else float("nan")
                               for cat in catalysts_all]
                    offset = (j - len(condition_labels)/2 + 0.5) * bar_w
                    ax_cmp.bar(x_pos + offset, vals, width=bar_w,
                                label=cond, color=COLORS[j % len(COLORS)],
                                edgecolor="white", alpha=0.9)
                ax_cmp.set_xticks(x_pos)
                ax_cmp.set_xticklabels(catalysts_all, rotation=30, ha="right")
                ax_cmp.set_ylabel(metric_cmp, fontsize=12)
                ax_cmp.set_title(f"{metric_cmp} Comparison Across Conditions",
                                  fontsize=13, fontweight="bold")
                ax_cmp.legend(fontsize=9)
                ax_cmp.grid(True, alpha=0.25, axis="y", linewidth=0.6, color="0.7", zorder=0)
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
st.caption("ODS Calculation Suite v3.1 | CatLab-Tools by Hoda Jafari | MIT License")
