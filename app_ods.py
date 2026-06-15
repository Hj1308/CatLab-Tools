"""
ODS Calculation Suite — Streamlit Web App
==========================================
Extended build based on CatLab-Tools/app_ods.py (github.com/Hj1308/CatLab-Tools)
Original author: Hoda Jafari

v2.0 — additions & fixes on top of v1.4
----------------------------------------
FIXED : Elovich model now has a real t1/2 formula (was hard-coded to NaN,
        so whenever Elovich was the best-fit model — common for
        chemisorption-controlled kinetics — t1/2 showed "N/A").
FIXED : Each catalyst sheet is processed in its own try/except block with
        clear, specific error messages (wrong number of columns, not enough
        valid rows, non-numeric cells, ...). One bad sheet no longer wipes
        out the entire report.
FIXED : Sheet names sanitised before writing to Excel
        (_safe_sheet_name with uniqueness tracking).
NEW   : TOF / TON module (turnover frequency / turnover number from
        catalyst mass + active-site loading).
NEW   : Reusability / recyclability module (removal % retained over reuse
        cycles).
NEW   : Reaction Order (1st/2nd) column in kinetics summary.
NEW   : Info message shown when catalyst sheet names are adjusted for Excel.
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import io
import re
import zipfile
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="ODS Calculation Suite", page_icon="🔬", layout="wide")

MW_S = 32.06
COLORS = ["#e41a1c","#377eb8","#4daf4a","#984ea3",
          "#ff7f00","#a65628","#f781bf","#17becf","#bcbd22"]
MARKERS = ["o","s","^","D","v","P","*","X","h"]

# ════════════════════════════════════════════════════════════════
# SHARED HELPERS
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

# ── Kinetic model functions ────────────────────────────────────────────
def _zero_order(t, k, C0):
    return np.maximum(C0 - k * t, 0)

def _first_order(t, k, C0):
    return C0 * np.exp(-k * t)

def _second_order(t, k, C0):
    return C0 / (1 + k * C0 * t)

def _elovich(t, alpha, beta, C0):
    return C0 - (1.0 / np.maximum(beta, 1e-15)) * np.log1p(
        np.maximum(alpha * beta * t, 0))

def _r2(y_obs, y_pred):
    ss_res = np.sum((y_obs - y_pred) ** 2)
    ss_tot = np.sum((y_obs - np.mean(y_obs)) ** 2)
    return round(1 - ss_res / ss_tot if ss_tot > 0 else 0.0, 4)

def _elovich_t_half(C0, alpha, beta):
    """
    Solve C(t1/2) = C0/2 for the Elovich model
    C(t) = C0 - (1/beta) * ln(1 + alpha*beta*t)
    => t1/2 = (exp(C0*beta/2) - 1) / (alpha*beta)
    This was previously hard-coded to NaN in v1.4.
    """
    try:
        if alpha <= 0 or beta <= 0 or C0 <= 0:
            return float("nan")
        exponent = C0 * beta / 2.0
        if exponent > 700:  # avoid exp() overflow
            return float("inf")
        val = np.exp(exponent) - 1.0
        if val <= 0:
            return float("nan")
        return val / (alpha * beta)
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

# ── Non-linear fitting engine ────────────────────────────────────────────
def _fit_nonlinear(time, Ct, C0):
    t = np.asarray(time, dtype=float)
    Ct = np.asarray(Ct, dtype=float)
    results = {}

    # Zero-order
    try:
        p, _ = curve_fit(_zero_order, t, Ct, p0=[1e-6, C0],
                         bounds=([0, C0*0.5], [np.inf, C0*1.5]), maxfev=5000)
        pred = _zero_order(t, *p)
        k0 = p[0]
        results["Zero-order"] = {
            "params": p, "R2": _r2(Ct, pred), "pred": pred,
            "label": f"k₀ = {_fmt_sci(k0)} mol·L⁻¹·min⁻¹",
            "t_half": round(0.5 * p[1] / k0, 2) if k0 > 0 else float("nan"),
            "k": k0, "col_k": "K0 (mol/L/min)",
        }
    except Exception as e:
        results["Zero-order"] = {"R2": -999, "error": str(e)}

    # Pseudo-first-order
    try:
        p, _ = curve_fit(_first_order, t, Ct, p0=[0.01, C0],
                         bounds=([0, C0*0.5], [np.inf, C0*1.5]), maxfev=5000)
        pred = _first_order(t, *p)
        kapp = p[0]
        results["Pseudo-first"] = {
            "params": p, "R2": _r2(Ct, pred), "pred": pred,
            "label": f"kₐₚₚ = {_fmt_sci(kapp)} min⁻¹",
            "t_half": round(np.log(2) / kapp, 2) if kapp > 0 else float("nan"),
            "k": kapp, "col_k": "Kapp (1/min)",
        }
    except Exception as e:
        results["Pseudo-first"] = {"R2": -999, "error": str(e)}

    # Second-order
    try:
        p, _ = curve_fit(_second_order, t, Ct, p0=[1e-3, C0],
                         bounds=([0, C0*0.5], [np.inf, C0*1.5]), maxfev=5000)
        pred = _second_order(t, *p)
        k2 = p[0]
        results["Second-order"] = {
            "params": p, "R2": _r2(Ct, pred), "pred": pred,
            "label": f"k₂ = {_fmt_sci(k2)} L·mol⁻¹·min⁻¹",
            "t_half": round(1.0 / (k2 * p[1]), 2) if k2 > 0 else float("nan"),
            "k": k2, "col_k": "K2 (L/mol/min)",
        }
    except Exception as e:
        results["Second-order"] = {"R2": -999, "error": str(e)}

    # Elovich (t1/2 now computed instead of hard-coded NaN)
    try:
        if len(t) < 2:
            raise ValueError("Elovich needs at least 2 time points "
                             "to fit 2 parameters")
        p, _ = curve_fit(
            lambda tt, a, b: _elovich(tt, a, b, C0),
            t, Ct,
            p0=[1e-4, 1e4],
            bounds=([1e-10, 1e-3], [1e3, 1e10]),
            maxfev=10000
        )
        alpha, beta = p
        pred_full = _elovich(t, alpha, beta, C0)
        results["Elovich"] = {
            "params": (alpha, beta, C0), "R2": _r2(Ct, pred_full), "pred": pred_full,
            "label": f"α={_fmt_sci(alpha)}, β={_fmt_sci(beta)}",
            "t_half": _elovich_t_half(C0, alpha, beta),
            "k": alpha, "beta": beta, "col_k": "α (mol/L/min)",
        }
    except Exception as e:
        results["Elovich"] = {"R2": -999, "error": str(e)}

    best_name = max(results, key=lambda m: results[m]["R2"])
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
        else:  # Elovich
            pred_fit = _elovich(t_fit, *res["params"])
        ax.plot(t_fit, pred_fit, "--", color=c, lw=1.5, alpha=0.8, label=lbl)

        # mark t1/2 on the curve if it falls within (or near) the data range
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

# ════════════════════════════════════════════════════════════════
# TEMPLATE BUILDERS
# ════════════════════════════════════════════════════════════════
def _safe_sheet_name(name, used):
    """
    Turn a catalyst name into a valid, unique Excel sheet name.
    Excel sheet names cannot contain \\ / ? * [ ] : , cannot be empty,
    cannot exceed 31 characters, and must be unique within the workbook.
    e.g. "g-BSiC/B4C" -> "g-BSiC-B4C", "N,B-gSiC/h-BN" -> "N,B-gSiC-h-BN"
    """
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
                "Use at least 4-5 time points (including or excluding t=0) for a meaningful fit.",
                "Do not leave blank cells, and do not type units/text into numeric cells.",
                "Removal (%) = (C0 - Ct)/C0 * 100, where Ct is the DBT concentration at time t.",
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
                "One sheet per catalyst (sheet name = catalyst name).",
                "Each sheet must have EXACTLY 2 columns: 'Cycle' and 'Removal (%)'.",
                "Cycle = 1, 2, 3, ... (reuse run number). Removal (%) = DBT removal "
                "efficiency for that cycle.",
                "Retention (%) is calculated relative to Cycle 1 "
                "(Retention = X_cycle_n / X_cycle_1 * 100).",
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
                "ONE sheet ('TOF_TON'), one row per catalyst.",
                "Catalyst mass (g) = mass of catalyst used in the reaction.",
                "Active sites (mmol/g) = active-site loading per gram of catalyst "
                "(e.g. from elemental analysis / titration / DFT estimate).",
                "Fuel volume (L) = volume of model fuel used.",
                "Removal (%) = DBT removal efficiency at the chosen reaction time.",
                "Reaction time (h) = the time at which Removal (%) was measured, in HOURS.",
                "C0 (initial DBT concentration) and its unit are set once, "
                "in the sidebar, and applied to all catalysts.",
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

# ════════════════════════════════════════════════════════════════
# UI
# ════════════════════════════════════════════════════════════════
st.title("🔬 ODS Calculation Suite")
st.markdown(
    "**CatLab-Tools (extended)** — based on the work of Hoda Jafari · "
    "[GitHub](https://github.com/Hj1308/CatLab-Tools)"
)
st.markdown("---")

# ---- Global parameters (shared by Kinetics & TOF/TON) ----
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

tab_templates, tab_kinetics, tab_tof, tab_reuse = st.tabs(
    ["📥 1. Templates", "📊 2. Kinetics & t½", "⚙️ 3. TOF / TON", "♻️ 4. Reusability"]
)

# ────────────────────────────────────────────────────────────
# TAB 1 — TEMPLATES
# ────────────────────────────────────────────────────────────
with tab_templates:
    st.subheader("Generate Excel Input Templates")
    st.markdown(
        "Enter the catalyst names you want to compare (one per line). "
        "These names are used as sheet names in the Kinetics and "
        "Reusability templates, and as row labels in the TOF/TON template."
    )
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

        c1, c2, c3 = st.columns(3)
        with c1:
            st.download_button("📥 Kinetics template (X% + t½)",
                               data=kin_buf,
                               file_name="ods_kinetics_template.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        with c2:
            st.download_button("📥 TOF/TON template",
                               data=make_tof_template(names),
                               file_name="ods_tof_ton_template.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        with c3:
            st.download_button("📥 Reusability template",
                               data=reuse_buf,
                               file_name="ods_reusability_template.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        changed = {k: v for k, v in kin_map.items() if k != v}
        if changed:
            st.info(
                "ℹ️ Excel sheet names can't contain `\\ / ? * [ ] :` , so the "
                "following catalyst names were adjusted for the sheet tabs "
                "(this is cosmetic only - your results table will still show "
                "this sheet name as the 'Catalyst'):\n\n" +
                "\n".join(f"- `{k}` → `{v}`" for k, v in changed.items())
            )

    st.markdown("---")
    with st.expander("📐 Formulas used in this app"):
        st.markdown(r"""
**Removal / conversion**
$$X(\%) = \dfrac{C_0 - C_t}{C_0}\times 100$$

**Kinetic models fitted to $C_t$ vs $t$**

| Model | Equation | $t_{1/2}$ |
|---|---|---|
| Zero-order | $C_t = C_0 - k_0 t$ | $C_0/(2k_0)$ |
| Pseudo-first-order | $C_t = C_0 e^{-k_{app}t}$ | $\ln 2 / k_{app}$ |
| Second-order | $C_t = C_0/(1+k_2 C_0 t)$ | $1/(k_2 C_0)$ |
| Elovich | $C_t = C_0 - \dfrac{1}{\beta}\ln(1+\alpha\beta t)$ | $\dfrac{e^{C_0\beta/2}-1}{\alpha\beta}$ |

**TOF / TON**
$$n_{sites} = m_{cat}\times(\text{active sites, mmol/g}) \quad\quad
n_{sub} = C_0\cdot\dfrac{X}{100}\cdot V$$
$$TON = \dfrac{n_{sub}}{n_{sites}} \qquad\qquad
TOF\,(h^{-1}) = \dfrac{TON}{t\,(h)}$$

**Reusability**
$$\text{Retention}(\%) = \dfrac{X_{cycle\,n}}{X_{cycle\,1}}\times 100$$
""")

# ────────────────────────────────────────────────────────────
# TAB 2 — KINETICS & t1/2
# ────────────────────────────────────────────────────────────
with tab_kinetics:
    st.subheader("Kinetics fitting (Zero / Pseudo-first / Second-order / Elovich) + t½")
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
                st.warning("No catalyst sheets found (only 'Instructions'/'Example' present).")
            else:
                fits_all, rows, errors = {}, [], []

                for sheet in sheets:
                    try:
                        raw = xl.parse(sheet)
                        df = raw.dropna(how="all")
                        if df.shape[1] != 2:
                            raise ValueError(
                                f"expected exactly 2 columns (Time, Removal %), "
                                f"found {df.shape[1]}")
                        df = df.dropna()
                        if len(df) < 2:
                            raise ValueError(
                                f"only {len(df)} valid data row(s) after removing "
                                f"empty rows - need at least 2 (ideally ≥4)")
                        df.columns = ["Time (min)", "Removal (%)"]
                        try:
                            t_arr = df["Time (min)"].values.astype(float)
                            rem = df["Removal (%)"].values.astype(float) / 100.0
                        except ValueError as ve:
                            raise ValueError(f"non-numeric value found - {ve}")

                        order = np.argsort(t_arr)
                        t_arr, rem = t_arr[order], rem[order]

                        Ct_arr = C0_mol * (1.0 - rem)
                        # NOTE: we do NOT force the curve through (t=0, C0).
                        # If t=0 was not actually measured, fabricating that
                        # point distorts the fit for fast-reacting catalysts
                        # (forces an artificial "kink") and can flip the
                        # pseudo-first vs second-order classification.
                        # C0 is instead left as a free fitted parameter
                        # (bounded to 0.5-1.5x the nominal C0).

                        fits = _fit_nonlinear(t_arr, Ct_arr, C0_mol)
                        fits_all[sheet] = {"t": t_arr, "Ct": Ct_arr, "fits": fits}
                        best = max(fits, key=lambda m: fits[m]["R2"])

                        # Classic pseudo-first vs second-order classification
                        # (the comparison used in the thesis methodology,
                        # independent of zero-order/Elovich):
                        r1 = fits["Pseudo-first"]["R2"]
                        r2v = fits["Second-order"]["R2"]
                        order_label = "Second-order" if r2v > r1 else "Pseudo-first-order"

                        rows.append({
                            "Catalyst": sheet,
                            "X_final (%)": round(rem[-1]*100, 1),
                            "K0 (mol/L/min)": fits["Zero-order"].get("k", float("nan")),
                            "R2_zero": fits["Zero-order"]["R2"],
                            "Kapp (1/min)": fits["Pseudo-first"].get("k", float("nan")),
                            "R2_first": fits["Pseudo-first"]["R2"],
                            "K2 (L/mol/min)": fits["Second-order"].get("k", float("nan")),
                            "R2_second": fits["Second-order"]["R2"],
                            "Elovich α": fits["Elovich"].get("k", float("nan")),
                            "Elovich β": fits["Elovich"].get("beta", float("nan")),
                            "R2_elovich": fits["Elovich"]["R2"],
                            "Reaction Order (1st/2nd)": order_label,
                            "Best Model": best,
                            "Best R2": fits[best]["R2"],
                            "t½ (min)": fits[best].get("t_half", float("nan")),
                        })
                    except Exception as e:
                        errors.append((sheet, str(e)))

                if errors:
                    msg = "\n".join(f"- **{s}**: {m}" for s, m in errors)
                    st.warning("⚠️ Some sheets were skipped due to data problems:\n\n" + msg)

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
                                       "Cₜ (mol·L⁻¹)", "Zero-Order Fit | Cₜ vs t")
                    fig2 = _plot_model(fits_all, sheets_ok, "Pseudo-first",
                                       "Cₜ (mol·L⁻¹)", "Pseudo-First-Order Fit | Cₜ vs t")
                    fig3 = _plot_model(fits_all, sheets_ok, "Second-order",
                                       "Cₜ (mol·L⁻¹)", "Second-Order Fit | Cₜ vs t")
                    fig4 = _plot_model(fits_all, sheets_ok, "Elovich",
                                       "Cₜ (mol·L⁻¹)", "Elovich Model Fit | Cₜ vs t")
                    c_a, c_b = st.columns(2)
                    c_a.pyplot(fig1); c_b.pyplot(fig2)
                    c_c, c_d = st.columns(2)
                    c_c.pyplot(fig3); c_d.pyplot(fig4)

                    st.markdown("---")
                    st.markdown("### 📋 Kinetics Summary — ★ = best model per catalyst")
                    disp = summary.copy()
                    for col in ["K0 (mol/L/min)", "Kapp (1/min)", "K2 (L/mol/min)",
                                "Elovich α", "Elovich β"]:
                        disp[col] = disp[col].apply(_fmt_sci)
                    for col in ["R2_zero", "R2_first", "R2_second", "R2_elovich", "Best R2"]:
                        disp[col] = disp[col].apply(lambda v: f"{v:.4f}" if v != -999 else "—")
                    disp["t½ (min)"] = disp["t½ (min)"].apply(_fmt_thalf)
                    st.dataframe(disp, use_container_width=True)

                    st.markdown("---")
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
                                           ("05_elovich.png", fig4)]:
                            ib = io.BytesIO()
                            fig.savefig(ib, dpi=300, bbox_inches="tight")
                            ib.seek(0); zf.writestr(fname, ib.read())
                    zip_buf.seek(0)
                    st.download_button("📥 Download all plots (ZIP)",
                                       data=zip_buf, file_name="ods_kinetics_plots.zip",
                                       mime="application/zip")
                elif not errors:
                    st.info("No catalyst sheets with data found.")
    elif uploaded and C0_mol is None:
        st.error("Fix the C₀ unit settings in the sidebar first.")
    else:
        st.info("Upload your filled Kinetics template to start the analysis. "
                "Need a template? Go to Tab 1.")

# ────────────────────────────────────────────────────────────
# TAB 3 — TOF / TON
# ────────────────────────────────────────────────────────────
with tab_tof:
    st.subheader("Turnover Frequency (TOF) and Turnover Number (TON)")
    st.caption("Uses the global C₀ set in the sidebar for all catalysts.")
    uploaded_tof = st.file_uploader("Upload filled TOF/TON template", type=["xlsx", "xls"],
                                    key="tof_upl")

    if uploaded_tof and C0_mol is not None:
        try:
            xl = pd.ExcelFile(uploaded_tof)
            if "TOF_TON" not in xl.sheet_names:
                st.error("Sheet 'TOF_TON' not found in the uploaded file. "
                         "Please use the template from Tab 1.")
            else:
                df = xl.parse("TOF_TON").dropna(how="all")
                required = ["Catalyst", "Catalyst mass (g)", "Active sites (mmol/g)",
                            "Fuel volume (L)", "Removal (%)", "Reaction time (h)"]
                missing = [c for c in required if c not in df.columns]
                if missing:
                    st.error(f"Missing column(s): {', '.join(missing)}. "
                             f"Please use the template from Tab 1.")
                else:
                    df = df.dropna(subset=required)
                    if df.empty:
                        st.warning("No complete rows found - please fill in all columns.")
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

                            rows.append({
                                "Catalyst": r["Catalyst"],
                                "Removal (%)": X,
                                "n_sites (mmol)": n_sites_mol * 1000,
                                "n_substrate (mmol)": n_sub_mol * 1000,
                                "TON": ton,
                                "TOF (h⁻¹)": tof,
                            })
                        result = pd.DataFrame(rows)

                        st.markdown("### 📋 TOF / TON Summary")
                        disp = result.copy()
                        for col in ["n_sites (mmol)", "n_substrate (mmol)"]:
                            disp[col] = disp[col].apply(lambda v: f"{v:.4f}")
                        for col in ["TON", "TOF (h⁻¹)"]:
                            disp[col] = disp[col].apply(_fmt_sci)
                        st.dataframe(disp, use_container_width=True)

                        st.markdown("### 📈 Comparison")
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

                        st.markdown("### ⬇️ Download Results")
                        tbl_buf = io.BytesIO()
                        result.to_excel(tbl_buf, index=False); tbl_buf.seek(0)
                        st.download_button("📥 Download tof_ton_summary.xlsx",
                                           data=tbl_buf, file_name="tof_ton_summary.xlsx",
                                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                        img_buf = io.BytesIO()
                        fig.savefig(img_buf, dpi=300, bbox_inches="tight"); img_buf.seek(0)
                        st.download_button("📥 Download TOF_TON_comparison.png",
                                           data=img_buf, file_name="TOF_TON_comparison.png",
                                           mime="image/png")
        except Exception as e:
            st.error(f"Error: {e}")
    elif uploaded_tof and C0_mol is None:
        st.error("Fix the C₀ unit settings in the sidebar first.")
    else:
        st.info("Upload your filled TOF/TON template to start the calculation. "
                "Need a template? Go to Tab 1.")

# ────────────────────────────────────────────────────────────
# TAB 4 — REUSABILITY
# ────────────────────────────────────────────────────────────
with tab_reuse:
    st.subheader("Reusability / Recyclability")
    uploaded_reuse = st.file_uploader("Upload filled Reusability template", type=["xlsx", "xls"],
                                      key="reuse_upl")

    if uploaded_reuse:
        try:
            xl = pd.ExcelFile(uploaded_reuse)
        except Exception as e:
            st.error(f"Could not open the Excel file: {e}")
            xl = None

        if xl is not None:
            sheets = [s for s in xl.sheet_names if s not in ("Instructions", "Example")]
            if not sheets:
                st.warning("No catalyst sheets found (only 'Instructions'/'Example' present).")
            else:
                data_ok, rows, errors = {}, [], []
                for sheet in sheets:
                    try:
                        raw = xl.parse(sheet)
                        df = raw.dropna(how="all")
                        if df.shape[1] != 2:
                            raise ValueError(
                                f"expected exactly 2 columns (Cycle, Removal %), "
                                f"found {df.shape[1]}")
                        df = df.dropna()
                        if len(df) < 1:
                            raise ValueError("no valid data rows found")
                        df.columns = ["Cycle", "Removal (%)"]
                        try:
                            cyc = df["Cycle"].values.astype(float)
                            x = df["Removal (%)"].values.astype(float)
                        except ValueError as ve:
                            raise ValueError(f"non-numeric value found - {ve}")
                        order = np.argsort(cyc)
                        cyc, x = cyc[order], x[order]
                        if x[0] == 0:
                            raise ValueError("Removal (%) for Cycle 1 is 0 - "
                                             "cannot compute retention (division by 0)")
                        retention = np.round(x / x[0] * 100.0, 2)
                        data_ok[sheet] = {"cycle": cyc, "x": x, "retention": retention}
                        for c, xi, ri in zip(cyc, x, retention):
                            rows.append({"Catalyst": sheet, "Cycle": int(c),
                                         "Removal (%)": xi, "Retention (%)": ri})
                    except Exception as e:
                        errors.append((sheet, str(e)))

                if errors:
                    msg = "\n".join(f"- **{s}**: {m}" for s, m in errors)
                    st.warning("⚠️ Some sheets were skipped due to data problems:\n\n" + msg)

                if rows:
                    result = pd.DataFrame(rows)

                    st.markdown("### 📈 Removal (%) over reuse cycles")
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

                    st.markdown("### 📋 Retention Summary")
                    st.dataframe(result, use_container_width=True)

                    st.markdown("### ⬇️ Download Results")
                    tbl_buf = io.BytesIO()
                    result.to_excel(tbl_buf, index=False); tbl_buf.seek(0)
                    st.download_button("📥 Download reusability_summary.xlsx",
                                       data=tbl_buf, file_name="reusability_summary.xlsx",
                                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                    img_buf = io.BytesIO()
                    fig.savefig(img_buf, dpi=300, bbox_inches="tight"); img_buf.seek(0)
                    st.download_button("📥 Download reusability_plot.png",
                                       data=img_buf, file_name="reusability_plot.png",
                                       mime="image/png")
                elif not errors:
                    st.info("No catalyst sheets with data found.")
    else:
        st.info("Upload your filled Reusability template to start the analysis. "
                "Need a template? Go to Tab 1.")

st.markdown("---")
st.caption("ODS Calculation Suite v2.0 | extended from CatLab-Tools by Hoda Jafari | MIT License")
