# app_ods.py
# ODS Kinetics Analyser — Streamlit Web App
# Author: Hoda Jafari | github.com/Hj1308/CatLab-Tools
# v1.4 — Non-linear curve fitting, best-model auto-selection, Elovich model

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from typing import Optional
import io
import zipfile
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="ODS Kinetics Analyser", page_icon="🔬", layout="wide")

MW_S = 32.06

# ─────────────────────────────────────────
# UNIT CONVERTER
# ─────────────────────────────────────────
def _to_mol_L(value, unit, mw=None):
    unit = unit.strip()
    if unit == "mol/L":    return value
    elif unit == "mmol/L": return value / 1000.0
    elif unit in ("mg/L", "ppm"):
        if mw is None: raise ValueError("MW required for mg/L or ppm")
        return (value / mw) / 1000.0
    elif unit == "g/L":
        if mw is None: raise ValueError("MW required for g/L")
        return value / mw
    elif unit == "ppmS":   return (value / MW_S) / 1000.0
    else: raise ValueError(f"Unknown unit: {unit}")


# ─────────────────────────────────────────
# KINETIC MODEL FUNCTIONS
# ─────────────────────────────────────────
def _zero_order(t, k, C0):    return np.maximum(C0 - k * t, 0)
def _first_order(t, k, C0):   return C0 * np.exp(-k * t)
def _second_order(t, k, C0):  return C0 / (1 + k * C0 * t)
def _elovich(t, alpha, beta, C0):
    return C0 - (1.0 / np.maximum(beta, 1e-15)) * np.log1p(
        np.maximum(alpha * beta * t, 0))

def _r2(y_obs, y_pred):
    ss_res = np.sum((y_obs - y_pred) ** 2)
    ss_tot = np.sum((y_obs - np.mean(y_obs)) ** 2)
    return round(1 - ss_res / ss_tot if ss_tot > 0 else 0.0, 4)


# ─────────────────────────────────────────
# NON-LINEAR FITTING ENGINE
# ─────────────────────────────────────────
def _fit_nonlinear(time, Ct, C0):
    t = time
    results = {}

    # ─ Zero-order ─
    try:
        p, _ = curve_fit(_zero_order, t, Ct, p0=[1e-6, C0],
                         bounds=([0, C0*0.5], [np.inf, C0*1.5]), maxfev=5000)
        pred = _zero_order(t, *p)
        k0 = p[0]
        results["Zero-order"] = {
            "k": k0, "params": p, "R2": _r2(Ct, pred),
            "pred": pred,
            "label": f"k₀ = {_fmt_sci(k0)} mol·L⁻¹·min⁻¹",
            "t_half": round(0.5 * C0 / k0, 2) if k0 > 0 else float("nan"),
            "col_k": "K0 (mol/L/min)",
        }
    except Exception:
        results["Zero-order"] = {"R2": -999}

    # ─ Pseudo-first-order ─
    try:
        p, _ = curve_fit(_first_order, t, Ct, p0=[0.01, C0],
                         bounds=([0, C0*0.5], [np.inf, C0*1.5]), maxfev=5000)
        pred = _first_order(t, *p)
        kapp = p[0]
        results["Pseudo-first"] = {
            "k": kapp, "params": p, "R2": _r2(Ct, pred),
            "pred": pred,
            "label": f"kₐₚₚ = {_fmt_sci(kapp)} min⁻¹",
            "t_half": round(np.log(2) / kapp, 2) if kapp > 0 else float("nan"),
            "col_k": "Kapp (1/min)",
        }
    except Exception:
        results["Pseudo-first"] = {"R2": -999}

    # ─ Second-order ─
    try:
        p, _ = curve_fit(_second_order, t, Ct, p0=[1e-3, C0],
                         bounds=([0, C0*0.5], [np.inf, C0*1.5]), maxfev=5000)
        pred = _second_order(t, *p)
        k2 = p[0]
        results["Second-order"] = {
            "k": k2, "params": p, "R2": _r2(Ct, pred),
            "pred": pred,
            "label": f"k₂ = {_fmt_sci(k2)} L·mol⁻¹·min⁻¹",
            "t_half": round(1.0 / (k2 * C0), 2) if k2 > 0 else float("nan"),
            "col_k": "K2 (L/mol/min)",
        }
    except Exception:
        results["Second-order"] = {"R2": -999}

    # ─ Elovich ─
    try:
        p, _ = curve_fit(
            lambda t, a, b: _elovich(t, a, b, C0),
            t[1:], Ct[1:],
            p0=[1e-4, 1e4],
            bounds=([1e-10, 1e-3], [1e3, 1e10]),
            maxfev=10000
        )
        pred_inner = _elovich(t, p[0], p[1], C0)
        results["Elovich"] = {
            "k": p[0], "params": (p[0], p[1], C0), "R2": _r2(Ct, pred_inner),
            "pred": pred_inner,
            "label": f"α={_fmt_sci(p[0])}, β={_fmt_sci(p[1])}",
            "t_half": float("nan"),
            "col_k": "α (mol/L/min)",
        }
    except Exception:
        results["Elovich"] = {"R2": -999}

    # best model
    best_name = max(results, key=lambda m: results[m]["R2"])
    for name in results:
        results[name]["is_best"] = (name == best_name)

    return results


# ─────────────────────────────────────────
# FORMATTING
# ─────────────────────────────────────────
def _fmt_sci(val):
    if val is None or (isinstance(val, float) and np.isnan(val)): return "N/A"
    if val == 0: return "0"
    exp  = int(np.floor(np.log10(abs(val))))
    coef = val / (10 ** exp)
    sup  = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")
    return f"{coef:.2f} × 10{str(exp).translate(sup)}"


# ─────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────
COLORS  = ["#e41a1c","#377eb8","#4daf4a","#984ea3",
           "#ff7f00","#a65628","#f781bf","#17becf","#bcbd22"]
MARKERS = ["o","s","^","D","v","P","*","X","h"]

def _plot_model(fits_all, sheets, model_name, ylabel, title):
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, sheet in enumerate(sheets):
        f   = fits_all[sheet]
        res = f["fits"].get(model_name, {})
        if res.get("R2", -999) < -100: continue
        t   = f["t"]; Ct = f["Ct"]
        c   = COLORS[i % len(COLORS)]; m = MARKERS[i % len(MARKERS)]
        lbl = f"{sheet}  R²={res['R2']:.4f}"
        if res.get("is_best"): lbl += "  ★best"
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
    ax.set_xlabel("Time (min)", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────
# TEMPLATE
# ─────────────────────────────────────────
def make_template(names):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        for name in names:
            pd.DataFrame({"Time (min)": ["","","","","",""],
                          "Removal (%)": ["","","","","",""]
                          }).to_excel(w, sheet_name=name[:31], index=False)
    buf.seek(0)
    return buf


# ─────────────────────────────────────────
# UI
# ─────────────────────────────────────────
st.title("🔬 ODS Kinetics Analyser")
st.markdown("**CatLab-Tools** — Hoda Jafari &nbsp;|&nbsp; "
            "[GitHub](https://github.com/Hj1308/CatLab-Tools)")
st.markdown("---")

tab1, tab2 = st.tabs(["📥 Step 1 — Get Template", "📊 Step 2 — Analyse"])

with tab1:
    st.subheader("Generate Excel Input Template")
    st.markdown("Each sheet = one catalyst. Columns: **Time (min)** | **Removal (%)**")
    cat_input = st.text_area("Catalyst names (one per line):",
                             value="Catalyst_1\nCatalyst_2\nCatalyst_3", height=120)
    if st.button("⬇️ Download Template"):
        names = [n.strip() for n in cat_input.strip().splitlines() if n.strip()]
        if names:
            st.download_button("📥 Click to save ods_template.xlsx",
                               data=make_template(names),
                               file_name="ods_template.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        else:
            st.warning("Please enter at least one catalyst name.")

with tab2:
    st.subheader("Upload Filled Template & Set Parameters")
    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("#### Experiment Parameters")
        c0_val  = st.number_input("C₀ (initial concentration)", value=250.0, min_value=0.001)
        c0_unit = st.selectbox("C₀ unit", ["ppmS","ppm","mg/L","g/L","mmol/L","mol/L"])
        mw_poll = None
        if c0_unit in ("ppm","mg/L","g/L"):
            mw_poll = st.number_input("MW of pollutant (g/mol)", value=184.0, min_value=1.0)
        cat_mass = st.number_input("Catalyst mass (g)",  value=0.05,  min_value=0.0001, format="%.4f")
        sol_vol  = st.number_input("Fuel volume (L)",    value=0.010, min_value=0.0001, format="%.4f")

    with col2:
        uploaded = st.file_uploader("Upload filled Excel template", type=["xlsx","xls"])

        if uploaded:
            try:
                xl     = pd.ExcelFile(uploaded)
                sheets = xl.sheet_names
                st.success(f"File loaded — {len(sheets)} catalyst(s): {', '.join(sheets)}")

                C0_mol   = _to_mol_L(c0_val, c0_unit, mw_poll)
                fits_all = {}
                rows     = []

                for sheet in sheets:
                    df     = xl.parse(sheet).dropna()
                    df.columns = ["Time (min)", "Removal (%)"]
                    t_arr  = df["Time (min)"].values.astype(float)
                    rem    = df["Removal (%)"].values.astype(float) / 100.0
                    Ct_arr = C0_mol * (1.0 - rem)
                    if t_arr[0] != 0:
                        t_arr  = np.insert(t_arr,  0, 0.0)
                        Ct_arr = np.insert(Ct_arr, 0, C0_mol)

                    fits = _fit_nonlinear(t_arr, Ct_arr, C0_mol)
                    fits_all[sheet] = {"t": t_arr, "Ct": Ct_arr, "fits": fits}

                    best = max(fits, key=lambda m: fits[m]["R2"])
                    rows.append({
                        "Catalyst"         : sheet,
                        "X_final (%)"      : round(rem[-1]*100, 1),
                        "K0 (mol/L/min)"   : fits["Zero-order"].get("k", float("nan")),
                        "R2_zero"          : fits["Zero-order"]["R2"],
                        "Kapp (1/min)"     : fits["Pseudo-first"].get("k", float("nan")),
                        "R2_first"         : fits["Pseudo-first"]["R2"],
                        "K2 (L/mol/min)"   : fits["Second-order"].get("k", float("nan")),
                        "R2_second"        : fits["Second-order"]["R2"],
                        "Elovich α"        : fits["Elovich"].get("k", float("nan")),
                        "R2_elovich"       : fits["Elovich"]["R2"],
                        "Best Model"       : best,
                        "Best R2"          : fits[best]["R2"],
                        "t½ (min)"         : fits[best].get("t_half", float("nan")),
                    })

                summary = pd.DataFrame(rows)

                st.markdown("---")
                st.markdown("### 📈 Plots")

                # Removal vs Time
                fig0, ax0 = plt.subplots(figsize=(9, 5))
                for i, sheet in enumerate(sheets):
                    df2 = xl.parse(sheet).dropna()
                    df2.columns = ["Time (min)", "Removal (%)"]
                    tp = df2["Time (min)"].values.astype(float)
                    Xp = df2["Removal (%)"].values.astype(float)
                    if tp[0] != 0:
                        tp = np.insert(tp, 0, 0.0); Xp = np.insert(Xp, 0, 0.0)
                    ax0.plot(tp, Xp,
                             marker=MARKERS[i%len(MARKERS)],
                             color=COLORS[i%len(COLORS)],
                             label=sheet, lw=2, ms=8,
                             markeredgecolor="white", markeredgewidth=0.7)
                ax0.set_xlabel("Time (min)", fontsize=12)
                ax0.set_ylabel("Removal (%)", fontsize=12)
                ax0.set_title("ODS Performance — All Catalysts", fontsize=13, fontweight="bold")
                ax0.legend(fontsize=9); ax0.grid(True, alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig0)

                # 4 kinetic plots
                fig1 = _plot_model(fits_all, sheets, "Zero-order",
                                   "Cₜ  (mol·L⁻¹)", "Zero-Order Fit  |  Cₜ vs t")
                fig2 = _plot_model(fits_all, sheets, "Pseudo-first",
                                   "Cₜ  (mol·L⁻¹)", "Pseudo-First-Order Fit  |  Cₜ vs t")
                fig3 = _plot_model(fits_all, sheets, "Second-order",
                                   "Cₜ  (mol·L⁻¹)", "Second-Order Fit  |  Cₜ vs t")
                fig4 = _plot_model(fits_all, sheets, "Elovich",
                                   "Cₜ  (mol·L⁻¹)", "Elovich Model Fit  |  Cₜ vs t")

                c_a, c_b = st.columns(2)
                c_a.pyplot(fig1); c_b.pyplot(fig2)
                c_c, c_d = st.columns(2)
                c_c.pyplot(fig3); c_d.pyplot(fig4)

                # Summary table
                st.markdown("---")
                st.markdown("### 📋 Kinetics Summary  —  ★ = best model per catalyst")

                disp = summary[["Catalyst","X_final (%)",
                                "K0 (mol/L/min)","R2_zero",
                                "Kapp (1/min)","R2_first",
                                "K2 (L/mol/min)","R2_second",
                                "Elovich α","R2_elovich",
                                "Best Model","Best R2","t½ (min)"]].copy()

                for col in ["K0 (mol/L/min)","Kapp (1/min)","K2 (L/mol/min)","Elovich α"]:
                    disp[col] = disp[col].apply(_fmt_sci)
                for col in ["R2_zero","R2_first","R2_second","R2_elovich","Best R2"]:
                    disp[col] = disp[col].apply(lambda v: f"{v:.4f}")
                disp["t½ (min)"] = disp["t½ (min)"].apply(
                    lambda v: f"{v:.1f}" if not np.isnan(float(str(v).replace('N/A','nan'))) else "N/A")

                st.dataframe(disp, use_container_width=True)

                # Downloads
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
                                       ("02_zero_order.png",      fig1),
                                       ("03_pseudo_first.png",    fig2),
                                       ("04_second_order.png",    fig3),
                                       ("05_elovich.png",         fig4)]:
                        ib = io.BytesIO()
                        fig.savefig(ib, dpi=300, bbox_inches="tight")
                        ib.seek(0); zf.writestr(fname, ib.read())
                zip_buf.seek(0)
                st.download_button("📥 Download all plots (ZIP)",
                                   data=zip_buf, file_name="ods_plots.zip",
                                   mime="application/zip")

            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.info("Upload your filled Excel template to start analysis.")

st.markdown("---")
st.caption("CatLab-Tools v1.4 | MIT License | github.com/Hj1308/CatLab-Tools")
