# app_ods.py
# ODS Kinetics Analyser — Streamlit Web App
# Author: Hoda Jafari | github.com/Hj1308/CatLab-Tools

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress
from typing import Optional
import io
import zipfile

# ─────────────────────────────────────────
st.set_page_config(
    page_title="ODS Kinetics Analyser",
    page_icon="🔬",
    layout="wide"
)

MW_S = 32.06

# ─────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────
def _to_mol_L(value, unit, mw=None):
    unit = unit.strip()
    if unit == "mol/L":    return value
    elif unit == "mmol/L": return value / 1000.0
    elif unit in ("mg/L","ppm"):
        if mw is None: raise ValueError("MW required for mg/L or ppm")
        return (value / mw) / 1000.0
    elif unit == "g/L":
        if mw is None: raise ValueError("MW required for g/L")
        return value / mw
    elif unit == "ppmS":   return (value / MW_S) / 1000.0
    else: raise ValueError(f"Unknown unit: {unit}")

def _fit(time, Ct, C0):
    t = time; C = Ct
    y0 = C0 - C
    s0, b0, r0, *_ = linregress(t, y0)
    y1 = np.log(C0 / np.clip(C, 1e-15, None))
    s1, b1, r1, *_ = linregress(t, y1)
    y2 = (1.0 / np.clip(C, 1e-15, None)) - (1.0 / C0)
    s2, b2, r2, *_ = linregress(t, y2)
    kapp = max(s1, 0)
    return {
        "k0": max(s0,0), "b0": b0, "R2_0": round(r0**2,4), "y0": y0,
        "kapp": kapp,    "b1": b1, "R2_1": round(r1**2,4), "y1": y1,
        "k2": max(s2,0), "b2": b2, "R2_2": round(r2**2,4), "y2": y2,
        "t_half": round(np.log(2)/kapp, 2) if kapp > 0 else float("nan"),
        "t": t, "C": C, "C0": C0,
    }

def _make_fig(fits, sheets, colors, markers, ylabel, ykey, title, slope_key, int_key):
    fig, ax = plt.subplots(figsize=(9,5))
    for i, sheet in enumerate(sheets):
        f = fits[sheet]
        t = f["t"]; y = f[ykey]
        s = f[slope_key]; b = f[int_key]
        c = colors[i % len(colors)]; m = markers[i % len(markers)]
        t_fit = np.linspace(0, t[-1], 300)
        ax.plot(t, y, marker=m, color=c, lw=0, ms=8,
                markeredgecolor="white", markeredgewidth=0.6, label=sheet)
        ax.plot(t_fit, s*t_fit+b, "--", color=c, lw=1.3, alpha=0.7)
    ax.set_xlabel("Time (min)", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig

COLORS  = ["#e41a1c","#377eb8","#4daf4a","#984ea3",
           "#ff7f00","#a65628","#f781bf","#17becf","#bcbd22"]
MARKERS = ["o","s","^","D","v","P","*","X","h"]

# ─────────────────────────────────────────
# TEMPLATE GENERATOR
# ─────────────────────────────────────────
def make_template(names):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        for name in names:
            pd.DataFrame({"Time (min)":["","","","","",""],
                          "Removal (%)":["","","","","",""]
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

# ── TAB 1: Template ──
with tab1:
    st.subheader("Generate Excel Input Template")
    st.markdown("Each sheet = one catalyst. Columns: **Time (min)** | **Removal (%)**")
    cat_input = st.text_area(
        "Catalyst names (one per line):",
        value="Catalyst_1\nCatalyst_2\nCatalyst_3",
        height=120
    )
    if st.button("⬇️ Download Template"):
        names = [n.strip() for n in cat_input.strip().splitlines() if n.strip()]
        if names:
            buf = make_template(names)
            st.download_button(
                label="📥 Click to save ods_template.xlsx",
                data=buf,
                file_name="ods_template.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        else:
            st.warning("Please enter at least one catalyst name.")

# ── TAB 2: Analysis ──
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

                C0_mol = _to_mol_L(c0_val, c0_unit, mw_poll)
                fits   = {}
                rows   = []

                for sheet in sheets:
                    df = xl.parse(sheet).dropna()
                    df.columns = ["Time (min)", "Removal (%)"]
                    t_arr  = df["Time (min)"].values.astype(float)
                    rem    = df["Removal (%)"].values.astype(float) / 100.0
                    Ct_arr = C0_mol * (1.0 - rem)
                    if t_arr[0] != 0:
                        t_arr  = np.insert(t_arr,  0, 0.0)
                        Ct_arr = np.insert(Ct_arr, 0, C0_mol)
                    f = _fit(t_arr, Ct_arr, C0_mol)
                    fits[sheet] = f
                    rows.append({
                        "Catalyst"               : sheet,
                        "X_final (%)"            : round(rem[-1]*100, 1),
                        "K0 (mol/L/min)"         : f["k0"],
                        "R2_zero"                : f["R2_0"],
                        "Kapp (1/min)"           : f["kapp"],
                        "R2_first"               : f["R2_1"],
                        "K2 (L/mol/min)"         : f["k2"],
                        "R2_second"              : f["R2_2"],
                        "t_half (min)"           : f["t_half"],
                    })

                summary = pd.DataFrame(rows)

                st.markdown("---")
                st.markdown("### 📈 Plots")

                # Plot 1: Removal vs Time
                fig0, ax0 = plt.subplots(figsize=(9,5))
                for i, sheet in enumerate(sheets):
                    df2 = xl.parse(sheet).dropna()
                    df2.columns = ["Time (min)","Removal (%)"]
                    tp = df2["Time (min)"].values.astype(float)
                    Xp = df2["Removal (%)"].values.astype(float)
                    if tp[0] != 0:
                        tp = np.insert(tp,0,0.0); Xp = np.insert(Xp,0,0.0)
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

                # Kinetic plots
                fig1 = _make_fig(fits, sheets, COLORS, MARKERS,
                                 "C₀ − Cₜ  (mol·L⁻¹)", "y0",
                                 "Zero-Order  |  C₀ − Cₜ  vs  t",
                                 "k0", "b0")
                fig2 = _make_fig(fits, sheets, COLORS, MARKERS,
                                 "ln(C₀/Cₜ)", "y1",
                                 "Pseudo-First-Order  |  ln(C₀/Cₜ)  vs  t",
                                 "kapp", "b1")
                fig3 = _make_fig(fits, sheets, COLORS, MARKERS,
                                 "1/Cₜ − 1/C₀  (L·mol⁻¹)", "y2",
                                 "Second-Order  |  1/Cₜ − 1/C₀  vs  t",
                                 "k2", "b2")

                c_a, c_b, c_c = st.columns(3)
                c_a.pyplot(fig1)
                c_b.pyplot(fig2)
                c_c.pyplot(fig3)

                # Summary table
                st.markdown("---")
                st.markdown("### 📋 Kinetics Summary Table")
                st.dataframe(summary, use_container_width=True)

                # Downloads
                st.markdown("---")
                st.markdown("### ⬇️ Download Results")

                tbl_buf = io.BytesIO()
                summary.to_excel(tbl_buf, index=False)
                tbl_buf.seek(0)
                st.download_button("📥 Download kinetics_summary.xlsx",
                                   data=tbl_buf,
                                   file_name="kinetics_summary.xlsx",
                                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

                zip_buf = io.BytesIO()
                with zipfile.ZipFile(zip_buf, "w") as zf:
                    for fname, fig in [("01_removal_vs_time.png", fig0),
                                       ("02_zero_order.png",      fig1),
                                       ("03_pseudo_first_order.png", fig2),
                                       ("04_second_order.png",    fig3)]:
                        img_buf = io.BytesIO()
                        fig.savefig(img_buf, dpi=300, bbox_inches="tight")
                        img_buf.seek(0)
                        zf.writestr(fname, img_buf.read())
                zip_buf.seek(0)
                st.download_button("📥 Download all plots (ZIP)",
                                   data=zip_buf,
                                   file_name="ods_plots.zip",
                                   mime="application/zip")

            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.info("Upload your filled Excel template to start analysis.")

st.markdown("---")
st.caption("CatLab-Tools v1.2 | MIT License | github.com/Hj1308/CatLab-Tools")
