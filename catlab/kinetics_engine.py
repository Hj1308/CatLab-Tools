# catlab/kinetics_engine.py
# Shared kinetics engine — CatLab-Tools
# Model rate-law functions, nonlinear fitting, AICc model selection, and
# concentration-unit converter. Used by both the Streamlit app (app_ods.py)
# and the catlab CLI package.

import numpy as np
from scipy.optimize import curve_fit
from scipy.integrate import odeint

# -- Constants ----------------------------------------------------
MW_S = 32.06  # g/mol
# N_PARAMS holds the number of fitted regression parameters (C0 is locked).
# _aic / _aicc add +1 internally for the residual variance sigma^2.
# _adj_r2 and the residual-diagnostics dof in app_ods.py use the raw value.
# Minimum retained data points so AICc stays finite for the WHOLE model
# portfolio at any saturation cutoff.  With K = p + 1 (sigma^2 counted, see
# _aicc) the AICc denominator is n - K - 1, so a finite AICc requires n > p + 2:
#
#     p (params)   K = p+1   min n (finite AICc)
#         1          2              4
#         2          3              5
#         3          4              6
#
# The widest model, Double-Exponential, has p=3, so n must be >= 6.  The floor
# is therefore 6, not 5: a model set that silently shrinks as n drops would
# make "best model" incomparable between catalysts.
MIN_FIT_POINTS = 6
N_PARAMS = {
    "Zero-order":          1,
    "Pseudo-first":        1,
    "Pseudo-second-order": 1,
    "Elovich":             2,
    "L-H":                 2,
    "Power-Law":           2,
    "Eley-Rideal":         2,
    "Avrami":              2,
    "Double-Exponential":  3,   # k1, k2, A  (C0 is locked)
}
COLORS = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#a65628", "#f781bf", "#17becf", "#bcbd22"]
MARKERS = ["o", "s", "^", "D", "v", "P", "*", "X", "h"]

# Models excluded from automatic "best model" selection.
# Eley-Rideal: structurally non-identifiable with this experiment type. Only
# single-species data (sulfur concentration vs time) are available, and for the
# surface-reaction rate law the oxidant is held in excess (constant concentration
# folded into the rate constant), which is the standard assumption here. Under
# excess oxidant the Eley-Rideal curve shape is spanned by existing models:
#   - low surface coverage  : theta_A ~ K_A*C_A (const) -> dC/dt = -k*C  == Pseudo-first
#   - general coverage      : theta_A ~ K_A*C_A/(1+K_A*C_A) (const) -> the rational
#     C/(1+KC) term == the Langmuir-Hinshelwood functional form
# So no distinguishing curve shape exists from C(t) alone; k_ER and K are only
# jointly identifiable (the implemented dC/dt = -k_ER*K*C is literally
# Pseudo-first-order with an extra unidentifiable parameter). Kept fitted for
# completeness/comparison only — never eligible for best-model selection.
# Double-Exponential (3 params) and Elovich were previously excluded too, but
# synthetic validation (see README) showed this was not statistically justified:
# with 11-point curves at +/-3% noise, Elovich is recoverable at 45% (vs 0% when
# excluded) at the cost of only ~5% false PSO->Elovich wins on noise-level close
# calls, and Double-Exponential rarely wins anyway (AICc parsimony already
# penalizes its 3 params).
BEST_MODEL_EXCLUDE = {"Eley-Rideal"}

# Gradient tolerance for every curve_fit call in _fit_nonlinear.
# SciPy's default gtol = 1e-8 is an absolute-scale test on the gradient of the
# cost.  ODS residuals are concentrations of order 1e-4 mol/L, so their squared
# sum and its gradient are tiny in absolute terms, and the bounded trust-region
# solver declares convergence before reaching the minimum.  On noisy synthetic
# 7-point curves the default left L-H fits up to ~100x above the attainable SSE
# and changed the selected model in 11 of 100 L-H datasets.  gtol = 1e-10 reaches
# the same optimum as 1e-12 and 1e-14; ftol and xtol were not the cause.
FIT_TOL = dict(gtol=1e-10)

# Dimensionless shape parameters with finite curve_fit bounds:
#   model -> [(index in popt, name, lower, upper)]
# When a bound is binding, the covariance from curve_fit is not a valid
# uncertainty.  Such fits get at_bound set and their SEs are reported as NaN.
# They stay eligible for best-model selection: with the former Avrami bound
# n <= 3, excluding them made sigmoidal (n = 3.6) data select Zero-order in
# 100 of 100 synthetic runs, so the flag is a warning on the reported values,
# not a veto on the model.
# Avrami n is bounded at 4.0, the upper end of the JMAK range (3-D growth with
# a constant nucleation rate); it was 3.0 up to v3.6.0.
# Rate constants (lower bound 0, dimensional) are not checked: a closeness
# window would depend on their units.
BOUNDED_SHAPE_PARAMS = {
    "Power-Law":          [(1, "n", 0.1, 5.0)],
    "Avrami":             [(1, "n", 0.1, 4.0)],
    "Double-Exponential": [(2, "A", 0.0, 1.0)],
}
BOUND_NEAR = 1e-3   # window: within 1e-3 * max(|bound|, 1) of the bound
BOUND_STEP = 1e-6   # inward probe step, as a fraction of max(|bound|, 1)
BOUND_RISE = 1e-8   # minimum relative SSE rise that counts as binding
# Checked against refits with the bound relaxed, on 2400 noisy synthetic fits
# (8 archetypes x 100 x 3 models): 442 of 454 binding bounds flagged, 11 false
# flags (9 of them exactly on the bound).  11 of the 12 misses are
# Double-Exponential fits with A ~ 0.001-0.007 whose relaxed optimum has A < 0.


def _params_at_bound(f, t, Ct, popt, specs):
    """Descriptions of shape parameters whose bound is binding.

    A bound is binding when the parameter lies within BOUND_NEAR of it AND a
    small step away from the bound, into the allowed range (BOUND_STEP), raises
    the residual sum of squares by more than BOUND_RISE relative to the fitted
    SSE.  That is the sign test of the KKT condition: the cost still falls
    towards the bound, so the unconstrained optimum lies outside the allowed
    range and the reported value is set by the bound, not the data.  At a
    genuine interior optimum the change is second order and stays below
    BOUND_RISE.  The step is taken inward because _power_law clips n to its
    bounds internally, so a step past the bound would change nothing.
    """
    p = np.asarray(popt, dtype=float)
    with np.errstate(all="ignore"):
        sse0 = float(np.sum((Ct - f(t, *p)) ** 2))
    if not np.isfinite(sse0) or sse0 <= 0.0:
        return []
    hits = []
    for i, name, lo, hi in specs:
        for b, side, sign in ((lo, "lower", -1.0), (hi, "upper", 1.0)):
            scale = max(abs(b), 1.0)
            if abs(p[i] - b) > BOUND_NEAR * scale:
                continue
            q = p.copy()
            q[i] = p[i] - sign * BOUND_STEP * scale
            with np.errstate(all="ignore"):
                sse1 = float(np.sum((Ct - f(t, *q)) ** 2))
            if np.isfinite(sse1) and (sse1 - sse0) / sse0 > BOUND_RISE:
                hits.append(f"{name} at {side} bound {b:g}")
    return hits

# TODO(decision): consider removing "Eley-Rideal" from MODEL_NAMES entirely.
# Its current formulation (dC/dt = -k_ER*K*C) is mathematically identical to
# Pseudo-first-order with an extra unidentifiable parameter, so it provides no
# information beyond Pseudo-first-order. Larger decision — not implemented.
MODEL_NAMES = [
    "Zero-order", "Pseudo-first", "Pseudo-second-order",
    "Elovich", "L-H",
    "Power-Law", "Eley-Rideal", "Avrami", "Double-Exponential"
]


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


def convert_to_mmol_L(value, unit, mw=None):
    """
    Convert concentration to mmol/L.

    Convenience wrapper around _to_mol_L that returns mmol/L instead of mol/L.
    Supported units: mol/L, mmol/L, mg/L, ppm, g/L, ppmS.
    ppmS auto-converts using MW_S = 32.06 g/mol (sulfur).
    """
    return _to_mol_L(value, unit, mw=mw) * 1000.0


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
    """AIC for nonlinear least squares.

    `p` is the number of fitted regression parameters.  The residual variance
    sigma^2 is an additional estimated parameter, so the effective parameter
    count is K = p + 1 (Burnham & Anderson 2002, §2.2).
    """
    n = len(y_obs)
    rss = np.sum((y_obs - y_pred) ** 2)
    if rss <= 0 or n == 0:
        return float("inf")
    K = p + 1
    return round(n * np.log(rss / n) + 2 * K, 4)


def _aicc(y_obs, y_pred, p):
    """Small-sample corrected AIC. K = p + 1 (includes sigma^2).

    Returns inf when n - K - 1 <= 0, which correctly flags a model as
    unsupportable by the available number of data points.
    """
    n = len(y_obs)
    aic = _aic(y_obs, y_pred, p)
    if np.isinf(aic):
        return float("inf")
    K = p + 1
    denom = n - K - 1
    if denom <= 0:
        return float("inf")
    return round(aic + (2.0 * K * (K + 1)) / denom, 4)


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


# -- Non-linear fitting engine -----------------------------------
def _fit_nonlinear(time, Ct, C0):
    t  = np.asarray(time, dtype=float)
    Ct = np.asarray(Ct,   dtype=float)
    n  = len(t)
    results = {}

    # Zero-order
    try:
        p, pcov = curve_fit(lambda t_, k: _zero_order(t_, k, C0), t, Ct,
                            p0=[1e-6], bounds=([0], [np.inf]), maxfev=5000, **FIT_TOL)
        se = np.sqrt(np.diag(pcov))
        k0 = p[0]
        pred = _zero_order(t, k0, C0)
        r2v = _r2(Ct, pred)
        np_ = N_PARAMS["Zero-order"]
        results["Zero-order"] = {
            "params": (k0, C0), "R2": r2v, "pred": pred,
            "adj_r2": _adj_r2(r2v, n, np_), "aic": _aic(Ct, pred, np_),
            "aicc": _aicc(Ct, pred, np_),
            "label": f"k₀ = {_fmt_sci(k0)} mol·L⁻¹·min⁻¹",
            "t_half": round(0.5 * C0 / k0, 4) if k0 > 0 else float("nan"),
            "k": k0, "k_se": se[0], "col_k": "K0 (mol/L/min)", "r0": k0, "r0_se": se[0],
        }
    except (RuntimeError, ValueError) as e:
        results["Zero-order"] = {"R2": np.nan, "aicc": np.nan, "converged": False, "error": str(e)}
    except Exception as e:
        results["Zero-order"] = {"R2": np.nan, "aicc": np.nan, "converged": False, "error": str(e), "unexpected_error": True}

    # Pseudo-first-order
    try:
        p, pcov = curve_fit(lambda t_, k: _first_order(t_, k, C0), t, Ct,
                            p0=[0.01], bounds=([0], [np.inf]), maxfev=5000, **FIT_TOL)
        se = np.sqrt(np.diag(pcov))
        kapp = p[0]
        pred = _first_order(t, kapp, C0)
        r2v = _r2(Ct, pred)
        np_ = N_PARAMS["Pseudo-first"]
        r0 = kapp * C0
        results["Pseudo-first"] = {
            "params": (kapp, C0), "R2": r2v, "pred": pred,
            "adj_r2": _adj_r2(r2v, n, np_), "aic": _aic(Ct, pred, np_),
            "aicc": _aicc(Ct, pred, np_),
            "label": f"kₐₚₚ = {_fmt_sci(kapp)} min⁻¹",
            "t_half": round(np.log(2) / kapp, 4) if kapp > 0 else float("nan"),
            "k": kapp, "k_se": se[0], "col_k": "Kapp (1/min)", "r0": r0, "r0_se": se[0] * C0,
        }
    except (RuntimeError, ValueError) as e:
        results["Pseudo-first"] = {"R2": np.nan, "aicc": np.nan, "converged": False, "error": str(e)}
    except Exception as e:
        results["Pseudo-first"] = {"R2": np.nan, "aicc": np.nan, "converged": False, "error": str(e), "unexpected_error": True}

    # Pseudo-second-order  (FIX T: concentration-based, k2 in L/mol/min)
    try:
        p, pcov = curve_fit(lambda t_, k: _second_order(t_, k, C0), t, Ct,
                            p0=[1.0], bounds=([0], [np.inf]), maxfev=5000, **FIT_TOL)
        se = np.sqrt(np.diag(pcov))
        k2 = p[0]
        pred = _second_order(t, k2, C0)
        r2v = _r2(Ct, pred)
        np_ = N_PARAMS["Pseudo-second-order"]
        r0 = k2 * C0 ** 2
        results["Pseudo-second-order"] = {
            "params": (k2, C0), "R2": r2v, "pred": pred,
            "adj_r2": _adj_r2(r2v, n, np_), "aic": _aic(Ct, pred, np_),
            "aicc": _aicc(Ct, pred, np_),
            "label": f"k₂ = {_fmt_sci(k2)} L·mol⁻¹·min⁻¹",
            "t_half": round(1.0 / (k2 * C0), 4) if k2 > 0 else float("nan"),
            "k": k2, "k_se": se[0], "col_k": "K2 (L/mol/min)", "r0": r0, "r0_se": se[0] * C0 ** 2,
        }
    except (RuntimeError, ValueError) as e:
        results["Pseudo-second-order"] = {"R2": np.nan, "aicc": np.nan, "converged": False, "error": str(e)}
    except Exception as e:
        results["Pseudo-second-order"] = {"R2": np.nan, "aicc": np.nan, "converged": False, "error": str(e), "unexpected_error": True}

    # Elovich
    try:
        p, pcov = curve_fit(lambda t_, a, b: _elovich(t_, a, b, C0), t, Ct,
                            p0=[1e-4, 10.0], bounds=([0, 0], [np.inf, np.inf]), maxfev=10000, **FIT_TOL)
        se = np.sqrt(np.diag(pcov))
        alpha = p[0]
        beta = p[1]
        pred = _elovich(t, alpha, beta, C0)
        r2v = _r2(Ct, pred)
        np_ = N_PARAMS["Elovich"]
        results["Elovich"] = {
            "params": (alpha, beta, C0), "R2": r2v, "pred": pred,
            "adj_r2": _adj_r2(r2v, n, np_), "aic": _aic(Ct, pred, np_),
            "aicc": _aicc(Ct, pred, np_),
            "label": f"α={_fmt_sci(alpha)}, β={_fmt_sci(beta)}",
            "t_half": _elovich_t_half(C0, alpha, beta),
            "k": alpha, "k_se": se[0], "col_k": "Alpha (mol/L/min)", "r0": alpha, "r0_se": se[0],
        }
    except (RuntimeError, ValueError) as e:
        results["Elovich"] = {"R2": np.nan, "aicc": np.nan, "converged": False, "error": str(e)}
    except Exception as e:
        results["Elovich"] = {"R2": np.nan, "aicc": np.nan, "converged": False, "error": str(e), "unexpected_error": True}

    # Langmuir-Hinshelwood
    try:
        p, pcov = curve_fit(lambda t_, kLH, Kads: _lh_model(t_, kLH, Kads, C0), t, Ct,
                            p0=[0.01, 10.0], bounds=([0, 0], [np.inf, np.inf]), maxfev=10000, **FIT_TOL)
        se = np.sqrt(np.diag(pcov))
        k_LH = p[0]
        K_ads = p[1]
        pred = _lh_model(t, k_LH, K_ads, C0)
        r2v = _r2(Ct, pred)
        np_ = N_PARAMS["L-H"]
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
    except (RuntimeError, ValueError) as e:
        results["L-H"] = {"R2": np.nan, "aicc": np.nan, "converged": False, "error": str(e)}
    except Exception as e:
        results["L-H"] = {"R2": np.nan, "aicc": np.nan, "converged": False, "error": str(e), "unexpected_error": True}

    # Power-Law (General Reaction Order)
    try:
        f_pl = lambda t_, k, n_: _power_law(t_, k, n_, C0)
        p, pcov = curve_fit(
            f_pl,
            t, Ct, p0=[0.01, 1.5],
            bounds=([0, 0.1], [np.inf, 5.0]), maxfev=10000, **FIT_TOL)
        se = np.sqrt(np.diag(pcov))
        at_bound = _params_at_bound(f_pl, t, Ct, p, BOUNDED_SHAPE_PARAMS["Power-Law"])
        if at_bound:
            se = np.full_like(se, np.nan)
        k_pl, n_pl = p
        pred = _power_law(t, k_pl, n_pl, C0)
        r2v = _r2(Ct, pred)
        np_ = N_PARAMS["Power-Law"]
        r0 = k_pl * (C0 ** n_pl)
        results["Power-Law"] = {
            "params": (k_pl, n_pl, C0), "R2": r2v, "pred": pred,
            "adj_r2": _adj_r2(r2v, n, np_), "aic": _aic(Ct, pred, np_),
            "aicc": _aicc(Ct, pred, np_),
            "label": f"k={_fmt_sci(k_pl)}, n={n_pl:.3f}",
            "t_half": _power_law_t_half(C0, k_pl, n_pl),
            "k": k_pl, "k_se": se[0], "n_pl": n_pl, "n_pl_se": se[1],
            "col_k": "k_PL", "r0": r0, "r0_se": None,
            "at_bound": at_bound,
        }
    except (RuntimeError, ValueError) as e:
        results["Power-Law"] = {"R2": np.nan, "aicc": np.nan, "converged": False, "error": str(e)}
    except Exception as e:
        results["Power-Law"] = {"R2": np.nan, "aicc": np.nan, "converged": False, "error": str(e), "unexpected_error": True}

    # Eley-Rideal
    try:
        p, pcov = curve_fit(
            lambda t_, k, K: _eley_rideal(t_, k, K, C0),
            t, Ct, p0=[0.01, 10.0],
            bounds=([0, 0], [np.inf, np.inf]), maxfev=8000, **FIT_TOL)
        se = np.sqrt(np.diag(pcov))
        k_er, K_er = p
        pred = _eley_rideal(t, k_er, K_er, C0)
        r2v = _r2(Ct, pred)
        np_ = N_PARAMS["Eley-Rideal"]
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
    except (RuntimeError, ValueError) as e:
        results["Eley-Rideal"] = {"R2": np.nan, "aicc": np.nan, "converged": False, "error": str(e)}
    except Exception as e:
        results["Eley-Rideal"] = {"R2": np.nan, "aicc": np.nan, "converged": False, "error": str(e), "unexpected_error": True}

    # Avrami
    try:
        f_av = lambda t_, k, n_: _avrami(t_, k, n_, C0)
        p, pcov = curve_fit(
            f_av,
            t, Ct, p0=[0.01, 1.0],
            bounds=([0, 0.1], [np.inf, 4.0]), maxfev=8000, **FIT_TOL)
        se = np.sqrt(np.diag(pcov))
        at_bound = _params_at_bound(f_av, t, Ct, p, BOUNDED_SHAPE_PARAMS["Avrami"])
        if at_bound:
            se = np.full_like(se, np.nan)
        k_av, n_av = p
        pred = _avrami(t, k_av, n_av, C0)
        r2v = _r2(Ct, pred)
        np_ = N_PARAMS["Avrami"]
        results["Avrami"] = {
            "params": (k_av, n_av, C0), "R2": r2v, "pred": pred,
            "adj_r2": _adj_r2(r2v, n, np_), "aic": _aic(Ct, pred, np_),
            "aicc": _aicc(Ct, pred, np_),
            "label": f"k={_fmt_sci(k_av)}, n={n_av:.3f}",
            "t_half": float("nan"),
            "k": k_av, "k_se": se[0],
            "col_k": "k_Avrami", "r0": None, "r0_se": None,
            "at_bound": at_bound,
        }
    except (RuntimeError, ValueError) as e:
        results["Avrami"] = {"R2": np.nan, "aicc": np.nan, "converged": False, "error": str(e)}
    except Exception as e:
        results["Avrami"] = {"R2": np.nan, "aicc": np.nan, "converged": False, "error": str(e), "unexpected_error": True}

    # Double Exponential
    try:
        f_de = lambda t_, k1, k2, A: _double_exponential(t_, k1, k2, A, C0)
        p, pcov = curve_fit(
            f_de,
            t, Ct, p0=[0.1, 0.01, 0.6],
            bounds=([0, 0, 0], [np.inf, np.inf, 1.0]), maxfev=10000, **FIT_TOL)
        se = np.sqrt(np.diag(pcov))
        at_bound = _params_at_bound(f_de, t, Ct, p, BOUNDED_SHAPE_PARAMS["Double-Exponential"])
        if at_bound:
            se = np.full_like(se, np.nan)
        k1, k2 = p[0], p[1]
        A_frac = p[2]
        pred = _double_exponential(t, k1, k2, A_frac, C0)
        r2v = _r2(Ct, pred)
        np_ = N_PARAMS["Double-Exponential"]
        results["Double-Exponential"] = {
            "params": (k1, k2, A_frac, C0), "R2": r2v, "pred": pred,
            "adj_r2": _adj_r2(r2v, n, np_), "aic": _aic(Ct, pred, np_),
            "aicc": _aicc(Ct, pred, np_),
            "label": f"k1={_fmt_sci(k1)}, k2={_fmt_sci(k2)}, A={A_frac:.3f}",
            "t_half": float("nan"),
            "k": k1, "k_se": se[0],
            "col_k": "k1 (fast)", "r0": None, "r0_se": None,
            "at_bound": at_bound,
        }
    except (RuntimeError, ValueError) as e:
        results["Double-Exponential"] = {"R2": np.nan, "aicc": np.nan, "converged": False, "error": str(e)}
    except Exception as e:
        results["Double-Exponential"] = {"R2": np.nan, "aicc": np.nan, "converged": False, "error": str(e), "unexpected_error": True}

    return results


def _get_valid_models(res, model_names):
    return {m: res[m] for m in model_names
            if m in res and res[m].get("converged", True)}

def _best_model(res, model_names):
    """
    Best model selection for ODS kinetics with small datasets (MIN_FIT_POINTS = 6 or more points).

    Rules (in order):
    1. Exclude models in BEST_MODEL_EXCLUDE or with non-finite AICc.
       (Fits with a binding parameter bound stay eligible; see at_bound.)
    2. Find model with lowest AICc (best_aicc).
    3. Parsimony window (DELTA=2.5): collect all models within 2.5 AICc units
       of best_aicc — these are statistically indistinguishable.
    4. Among the competitive set: fewest regression parameters, then lowest AICc.
       No model-identity-based preference of any kind.
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

    # Step 3: parsimony — fewest parameters, then lowest AICc
    return min(competitive,
               key=lambda m: (N_PARAMS.get(m, 99), candidates[m]["aicc"]))


def akaike_weights(results, model_names=None, exclude=None):
    """Akaike weights w_i over the candidate model set.

    delta_i = AICc_i - min(AICc);  w_i = exp(-delta_i/2) / sum_j exp(-delta_j/2)
    w_i is the relative weight of evidence for model i given the candidate set
    (Burnham & Anderson 2002, §2.9).  Reporting delta_i and w_i is preferred
    over naming a single "best" model.

    Returns {model_name: {"delta_aicc": float, "weight": float}}.
    Models with non-finite AICc, or in `exclude`, are omitted.
    """
    if model_names is None:
        model_names = MODEL_NAMES
    if exclude is None:
        exclude = BEST_MODEL_EXCLUDE
    finites = {m: results[m]["aicc"] for m in model_names
               if m in results
               and m not in exclude
               and np.isfinite(results[m].get("aicc", float("inf")))}
    if not finites:
        return {}
    aicc_min = min(finites.values())
    deltas = {m: v - aicc_min for m, v in finites.items()}
    exp_terms = {m: np.exp(-d / 2.0) for m, d in deltas.items()}
    total = sum(exp_terms.values())
    return {m: {"delta_aicc": round(deltas[m], 4),
                "weight": round(exp_terms[m] / total, 6)}
            for m in finites}


def _auto_saturation_exclusions(t_raw, rem_raw, max_fractional_uptake=1.0):
    """
    Tail-truncation heuristic for near-equilibrium data points.

    Drops trailing points whose removal exceeds
    `max_fractional_uptake * rem[-1]`, where `rem[-1]` is the LAST
    OBSERVED removal value used as a proxy for equilibrium.

    NOTE: this is NOT the criterion of Simonin (2016). Simonin defines
    fractional uptake F(t) = q(t)/q_e against an INDEPENDENTLY MEASURED
    equilibrium capacity q_e,exp. Using the last observed point instead
    makes the first comparison self-referential (rem[-1] > f*rem[-1] is
    true for any f < 1), so exactly one point is dropped regardless of
    the value of `max_fractional_uptake`. On a 7-point grid with
    MIN_FIT_POINTS = 6 the loop can run at most once, making all
    settings below 1.0 behave identically.

    Simonin's rationale also targets LINEARISED fitting (t/q vs t), where
    near-equilibrium points align spuriously and inflate PSO's r^2.
    CatLab fits non-linearly, so that failure mode does not apply here.

    Default 1.0 disables the rule.

    The retained set is never allowed to drop below MIN_FIT_POINTS, which
    keeps AICc finite for the whole portfolio (up to p=3, K=p+1=4) — see
    that constant's docstring.

    Returns (excluded_time_points, t_keep, rem_keep, truncation_clamped).
    truncation_clamped is True when the cutoff could not be fully applied.
    """
    t_keep   = np.asarray(t_raw, dtype=float).copy()
    rem_keep = np.asarray(rem_raw, dtype=float).copy()
    excluded = []
    clamped  = False
    if len(rem_keep) >= MIN_FIT_POINTS:
        eq_rem = rem_keep[-1]
        cutoff = max_fractional_uptake * eq_rem
        while len(rem_keep) > MIN_FIT_POINTS and rem_keep[-1] > cutoff:
            excluded.append(float(t_keep[-1]))
            t_keep   = t_keep[:-1]
            rem_keep = rem_keep[:-1]
        clamped = len(rem_keep) <= MIN_FIT_POINTS and rem_keep[-1] > cutoff
    return excluded, t_keep, rem_keep, clamped
