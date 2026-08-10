# catlab/ods_kinetics.py
# ODS Kinetics Analyser — CatLab-Tools
# Author: Hoda Jafari | github.com/Hj1308
#
# Workflow:
#   Step 1 -> generate_template()  : creates Excel input template
#   Step 2 -> run_ods_analysis()   : reads template, runs kinetics, plots, exports table
#
# Input Excel columns:
#   Time (min) | Removal (%)
#
# Parameters passed in code:
#   c0_value, c0_unit, catalyst_mass_g, solution_vol_L, mw_pollutant (optional)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional
import os
from .kinetics_engine import _fit_nonlinear

MW_S = 32.06   # g/mol sulfur


# -------------------------------------------------
# UNIT CONVERTER  ->  mol/L
# -------------------------------------------------
def _to_mol_L(value: float, unit: str, mw: Optional[float] = None) -> float:
    unit = unit.strip()
    if unit == "mol/L":    return value
    elif unit == "mmol/L": return value / 1000.0
    elif unit in ("mg/L", "ppm"):
        if mw is None: raise ValueError("mw_pollutant required for mg/L or ppm")
        return (value / mw) / 1000.0
    elif unit == "g/L":
        if mw is None: raise ValueError("mw_pollutant required for g/L")
        return value / mw
    elif unit == "ppmS":   return (value / MW_S) / 1000.0
    else: raise ValueError(f"Unknown unit: {unit}")


# -------------------------------------------------
# STEP 1 — GENERATE TEMPLATE
# -------------------------------------------------
def generate_template(output_path: str = "ods_template.xlsx",
                      example_catalysts: list = None) -> str:
    """
    Create an Excel template for ODS kinetic data entry.
    Each sheet = one catalyst.
    Columns: Time (min) | Removal (%)
    """
    if example_catalysts is None:
        example_catalysts = ["Catalyst_1", "Catalyst_2"]

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for name in example_catalysts:
            df = pd.DataFrame({
                "Time (min)": ["", "", "", "", "", ""],
                "Removal (%)": ["", "", "", "", "", ""],
            })
            df.to_excel(writer, sheet_name=name[:31], index=False)

    print(f"Template created: {output_path}")
    print(f"Sheets: {example_catalysts}")
    print("Fill in Time (min) and Removal (%) for each catalyst sheet.")
    return output_path


# -------------------------------------------------
# KINETIC FITTING
# -------------------------------------------------
def _fit_kinetics(time: np.ndarray, Ct_mol_L: np.ndarray, C0_mol_L: float) -> dict:
    """Fit zero/first/second-order models via the shared nonlinear engine."""
    t  = time
    C  = Ct_mol_L
    C0 = C0_mol_L
    res = _fit_nonlinear(t, C, C0)

    def _k(model):
        r = res.get(model, {})
        return r.get("k", 0.0) if r.get("converged", True) else 0.0

    def _r2v(model):
        r = res.get(model, {})
        v = r.get("R2", float("nan"))
        return v if not np.isnan(v) else 0.0

    k0   = _k("Zero-order")
    R2_0 = round(_r2v("Zero-order"), 4)

    kapp = _k("Pseudo-first")
    R2_1 = round(_r2v("Pseudo-first"), 4)

    k2   = _k("Pseudo-second-order")
    R2_2 = round(_r2v("Pseudo-second-order"), 4)

    t_half_nl = res.get("Pseudo-first", {}).get("t_half", float("nan"))
    t_half = round(t_half_nl, 2) if (not np.isnan(t_half_nl) and kapp > 0) else float("nan")

    y0 = C0 - C
    y1 = np.log(C0 / np.clip(C, 1e-15, None))
    y2 = (1.0 / np.clip(C, 1e-15, None)) - (1.0 / C0)

    return {
        "K0 (mol/L/min)"  : round(k0,   8),
        "R2_zero"         : R2_0,
        "Kapp (1/min)"    : round(kapp, 6),
        "R2_first"        : R2_1,
        "K2 (L/mol/min)"  : round(k2,   4),
        "R2_second"       : R2_2,
        "t_half (min)"    : t_half,
        "_t"  : t,
        "_y0" : y0,
        "_y1" : y1,
        "_y2" : y2,
        "_C0" : C0,
        "_C"  : C,
    }


# -------------------------------------------------
# STEP 2 — RUN ANALYSIS
# -------------------------------------------------
def run_ods_analysis(
    excel_path       : str,
    c0_value         : float,
    c0_unit          : str   = "ppmS",
    catalyst_mass_g  : float = 0.05,
    solution_vol_L   : float = 0.010,
    mw_pollutant     : Optional[float] = None,
    output_dir       : str   = ".",
) -> pd.DataFrame:
    """
    Read ODS data from Excel template, fit kinetics, plot, export table.

    Parameters
    ----------
    excel_path      : path to filled Excel template
    c0_value        : initial concentration value
    c0_unit         : unit -- ppmS | ppm | mg/L | g/L | mmol/L | mol/L
    catalyst_mass_g : catalyst mass (g)
    solution_vol_L  : fuel volume (L)
    mw_pollutant    : molecular weight (g/mol) -- required for ppm/mg/L/g/L
    output_dir      : folder for output files
    """
    os.makedirs(output_dir, exist_ok=True)

    C0_mol_L = _to_mol_L(c0_value, c0_unit, mw_pollutant)
    xl       = pd.ExcelFile(excel_path)
    sheets   = xl.sheet_names

    colors  = ["#e41a1c","#377eb8","#4daf4a","#984ea3",
               "#ff7f00","#a65628","#f781bf","#17becf","#bcbd22"]
    markers = ["o","s","^","D","v","P","*","X","h"]

    results   = []
    fits_data = {}

    for sheet in sheets:
        df = xl.parse(sheet).dropna()
        df.columns = ["Time (min)", "Removal (%)"]
        t_arr  = df["Time (min)"].values.astype(float)
        rem    = df["Removal (%)"].values.astype(float) / 100.0
        Ct_arr = C0_mol_L * (1.0 - rem)

        if t_arr[0] != 0:
            t_arr  = np.insert(t_arr,  0, 0.0)
            Ct_arr = np.insert(Ct_arr, 0, C0_mol_L)

        fit     = _fit_kinetics(t_arr, Ct_arr, C0_mol_L)
        X_final = round(rem[-1] * 100, 1)

        results.append({
            "Catalyst"               : sheet,
            "X_final (%)"            : X_final,
            "K0 (mol/L/min)"         : fit["K0 (mol/L/min)"],
            "R2_zero"                : fit["R2_zero"],
            "Kapp (1/min)"           : fit["Kapp (1/min)"],
            "R2_first"               : fit["R2_first"],
            "K2 (L/mol/min)"         : fit["K2 (L/mol/min)"],
            "R2_second"              : fit["R2_second"],
            "t_half (min)"           : fit["t_half (min)"],
        })
        fits_data[sheet] = fit

    summary_df = pd.DataFrame(results)

    # ── PLOT 1: Conversion vs Time ──
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, sheet in enumerate(sheets):
        df = xl.parse(sheet).dropna()
        df.columns = ["Time (min)", "Removal (%)"]
        t_plot = df["Time (min)"].values.astype(float)
        X_plot = df["Removal (%)"].values.astype(float)
        if t_plot[0] != 0:
            t_plot = np.insert(t_plot, 0, 0.0)
            X_plot = np.insert(X_plot, 0, 0.0)
        ax.plot(t_plot, X_plot,
                marker=markers[i % len(markers)],
                color=colors[i % len(colors)],
                label=sheet, lw=2, ms=8,
                markeredgecolor="white", markeredgewidth=0.7)
    ax.set_xlabel("Time (min)", fontsize=13)
    ax.set_ylabel("Removal (%)", fontsize=13)
    ax.set_title("ODS Performance — All Catalysts", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "01_conversion_vs_time.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # ── PLOT 2: Zero-order ──
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (sheet, fit) in enumerate(fits_data.items()):
        c = colors[i % len(colors)]; m = markers[i % len(markers)]
        t = fit["_t"]; y = fit["_y0"]
        k0 = fit["K0 (mol/L/min)"]
        t_fit = np.linspace(0, t[-1], 200)
        ax.plot(t, y, marker=m, color=c, lw=0, ms=8, label=sheet)
        if np.isfinite(k0):
            ax.plot(t_fit, k0 * t_fit, "--", color=c, lw=1.2, alpha=0.7)
    ax.set_xlabel("Time (min)", fontsize=13)
    ax.set_ylabel("C\u2080 \u2212 C\u209c  (mol\u00b7L\u207b\u00b9)", fontsize=13)
    ax.set_title("Zero-Order Kinetics", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "02_zero_order.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # ── PLOT 3: Pseudo-first-order ──
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (sheet, fit) in enumerate(fits_data.items()):
        c = colors[i % len(colors)]; m = markers[i % len(markers)]
        t = fit["_t"]; y = fit["_y1"]
        kapp = fit["Kapp (1/min)"]
        t_fit = np.linspace(0, t[-1], 200)
        ax.plot(t, y, marker=m, color=c, lw=0, ms=8, label=sheet)
        if np.isfinite(kapp):
            ax.plot(t_fit, kapp * t_fit, "--", color=c, lw=1.2, alpha=0.7)
    ax.set_xlabel("Time (min)", fontsize=13)
    ax.set_ylabel("ln(C\u2080/C\u209c)", fontsize=13)
    ax.set_title("Pseudo-First-Order Kinetics", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "03_pseudo_first_order.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # ── PLOT 4: Second-order ──
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (sheet, fit) in enumerate(fits_data.items()):
        c = colors[i % len(colors)]; m = markers[i % len(markers)]
        t = fit["_t"]; y = fit["_y2"]
        k2 = fit["K2 (L/mol/min)"]
        t_fit = np.linspace(0, t[-1], 200)
        ax.plot(t, y, marker=m, color=c, lw=0, ms=8, label=sheet)
        if np.isfinite(k2):
            ax.plot(t_fit, k2 * t_fit, "--", color=c, lw=1.2, alpha=0.7)
    ax.set_xlabel("Time (min)", fontsize=13)
    ax.set_ylabel("1/C\u209c \u2212 1/C\u2080  (L\u00b7mol\u207b\u00b9)", fontsize=13)
    ax.set_title("Second-Order Kinetics", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "04_second_order.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # ── Export summary table ──
    table_path = os.path.join(output_dir, "kinetics_summary.xlsx")
    summary_df.to_excel(table_path, index=False)

    # ── Print table ──
    print("\n" + "="*95)
    print(f"  {'Catalyst':<12} {'K0 (mol/L/min)':>16} {'R2':>6}  "
          f"{'Kapp (1/min)':>13} {'R2':>6}  "
          f"{'K2 (L/mol/min)':>15} {'R2':>6}  {'t1/2 (min)':>10}")
    print("  " + "-"*91)
    for _, row in summary_df.iterrows():
        print(f"  {row['Catalyst']:<12} "
              f"{row['K0 (mol/L/min)']:>16.3e} {row['R2_zero']:>6.4f}  "
              f"{row['Kapp (1/min)']:>13.6f} {row['R2_first']:>6.4f}  "
              f"{row['K2 (L/mol/min)']:>15.4f} {row['R2_second']:>6.4f}  "
              f"{row['t_half (min)']:>10.1f}")
    print("="*95)
    print(f"\nPlots saved to : {output_dir}/")
    print(f"Summary table  : {table_path}")

    return summary_df
