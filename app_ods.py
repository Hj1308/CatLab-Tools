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
from scipy import stats as scipy_stats
import io
import zipfile
import warnings

from catlab.kinetics_engine import (
    MW_S, N_PARAMS, BEST_MODEL_EXCLUDE, MODEL_NAMES,
    MIN_FIT_POINTS,
    COLORS, MARKERS,
    _to_mol_L,
    _zero_order, _first_order, _second_order, _elovich, _lh_model,
    _power_law, _power_law_t_half, _eley_rideal, _avrami, _double_exponential,
    _r2, _adj_r2, _aic, _aicc,
    _elovich_t_half, _lh_t_half,
    _fmt_sci, _fmt_thalf, _fmt_pm,
    _fit_nonlinear, _get_valid_models, _best_model, _auto_saturation_exclusions,
)

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
R_GAS  = 8.314   # J/(mol·K)

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

def _C0_both(c0_val, c0_unit, mw_poll, rho_g_per_mL, n_sulfur=1, ppms_volumetric=True):
    if c0_unit == "ppmS":
        C0_S        = _to_mol_L(c0_val, "ppmS", MW_S, rho_g_per_mL, ppms_volumetric)
        C0_compound = C0_S / n_sulfur
    else:
        C0_compound = _to_mol_L(c0_val, c0_unit, mw_poll, rho_g_per_mL, ppms_volumetric)
        C0_S        = C0_compound * n_sulfur
    return C0_compound, C0_S


def _initial_tof_site(r0, V_L, n_sites_mol):
    """Initial-rate turnover frequency, min^-1.

    TOF_0 = r0 * V / n_sites, with r0 in mol/L/min, V in L and n_sites in mol.
    Returns nan when r0 is None (the selected model has no defined initial rate)
    or when n_sites_mol is not strictly positive.
    """
    if r0 is None or n_sites_mol <= 0:
        return float("nan")
    return r0 * V_L / n_sites_mol


def _initial_tof_mass(r0, V_L, m_g):
    """Initial-rate mass-normalised activity, mmol/g/min.

    TOF_mass,0 = r0 * V * 1000 / m, with r0 in mol/L/min, V in L and m in g.
    Returns nan when r0 is None or when m_g is not strictly positive.
    """
    if r0 is None or m_g <= 0:
        return float("nan")
    return r0 * V_L * 1000.0 / m_g


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
            "3. Fill the Catalyst_Properties sheet with BET surface area for each catalyst (used in Tab 4 Option B).",
            "4. Save the file and upload it in the app.",
            "5. The app converts units, fits multiple kinetic models (incl. pseudo-second-order), and selects the best one using AICc.",
            "6. In Tab 4, choose Option B (Carbon-based) to get mass-normalized TOF using BET values from this file."
        ]
    })

    # Catalyst_Properties sheet — for Tab 4 Option B (carbon-based TOF)
    df_cat_props = pd.DataFrame({
        "Catalyst": ["Cat-A", "Cat-B"],
        "BET (m²/g)": [250.0, 310.0],
        "Notes": ["BET from N₂ adsorption (77 K)", ""]
    })

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine='openpyxl') as writer:
        df_meta.to_excel(writer, sheet_name="Metadata", index=False)
        df_raw.to_excel(writer, sheet_name="Raw_Data", index=False)
        df_cat_props.to_excel(writer, sheet_name="Catalyst_Properties", index=False)
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
    auto_excl_per_cat = {}
    clamped_per_cat   = {}
    t_pre_per_cat     = {}
    rem_pre_per_cat   = {}

    # Auto-saturation: user-adjustable Simonin (2016) fractional-uptake cutoff.
    # Points whose removal exceeds max_fractional_uptake × final removal are
    # excluded before fitting (see _auto_saturation_exclusions).
    max_frac = st.slider(
        "🛑 Max fractional uptake for auto-saturation (0.80–1.0)",
        min_value=0.80, max_value=1.0, value=1.0, step=0.01,
        help="Tail-truncation heuristic: drops trailing points whose removal "
             "exceeds this fraction of the LAST OBSERVED removal value. This is "
             "NOT Simonin's (2016) criterion, which requires an independently "
             "measured equilibrium capacity. Because the last point is its own "
             "reference, any setting below 1.0 drops exactly one point — all "
             "values 0.80-0.99 behave identically on a 7-point dataset. "
             "Default 1.0 = disabled (recommended).")
    if max_frac >= 1.0:
        st.info("ℹ️ max fractional uptake = 1.0 disables auto-saturation exclusion (default).")

    for col in removal_cols:
        removal_raw = df[col].dropna().values[:len(t_raw)].astype(float)
        excl_times  = excl_per_cat[col]
        keep_mask   = np.array([ti not in excl_times for ti in t_raw])
        t_keep      = t_raw[keep_mask]
        rem_keep    = removal_raw[keep_mask]

        # Snapshot BEFORE auto-saturation, for the with/without comparison
        t_pre_excl   = t_keep.copy()
        rem_pre_excl = rem_keep.copy()

        # ── Auto-saturation detection (only when no manual exclusion) ──
        # Semantics (verified on data, see _auto_saturation_exclusions): any point
        # whose removal exceeds max_fractional_uptake × (final removal) is dropped.
        # Default 1.0 disables the rule; see README for the validation that led to
        # disabling it (the cutoff increases false-PSO and hurts mechanistic models).
        if not excl_times:
            auto_excl, t_keep, rem_keep, clamped = _auto_saturation_exclusions(
                t_keep, rem_keep, max_frac)
        else:
            auto_excl = []
            clamped = False

        auto_excl_per_cat[col] = auto_excl
        clamped_per_cat[col]  = clamped
        t_pre_per_cat[col]     = t_pre_excl
        rem_pre_per_cat[col]   = rem_pre_excl

        if auto_excl:
            cat_label = col.replace(" Removal (%)","").strip()
            st.info(
                f"ℹ️ **{cat_label}**: auto-excluded saturation point(s) "
                f"t = {auto_excl} min (removal exceeds {max_frac:.0%} of final "
                f"removal, Simonin 2016 cutoff). Use manual exclusion above to override.")
        if clamped:
            cat_label = col.replace(" Removal (%)","").strip()
            st.warning(
                f"⚠️ **{cat_label}**: saturation cutoff could not be fully "
                f"applied — truncation was stopped at "
                f"{MIN_FIT_POINTS} points to keep AICc finite. "
                f"Consider adding more data points or manually excluding runs.")

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
            rows.append({"Catalyst": cat_label, "Best Model": "All fits failed",
                         "Clamped": "yes" if clamped_per_cat.get(col) else ""})
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
            "Clamped":            "yes" if clamped_per_cat.get(col) else "",
            "Note":               " | ".join(notes),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    # ── Auto-saturation details: with vs without exclusion ─────────
    any_auto_excl = any(auto_excl_per_cat.get(c) for c in all_results)
    if not any_auto_excl:
        st.info("✅ No auto-saturation exclusion applied.")
    else:
        with st.expander("🔎 Auto-saturation details", expanded=True):
            st.caption(
                f"Points dropped by the auto-saturation rule (removal > "
                f"{max_frac:.0%} of final/equilibrium removal, Simonin 2016 "
                "fractional-uptake cutoff), and the best-model result "
                "with vs without that exclusion. Best model = AICc selection, R² shown.")
            det_rows = []
            for col in all_results:
                cat_label  = col.replace(" Removal (%)","").strip()
                excl_pts   = auto_excl_per_cat.get(col, [])
                best_with  = _best_model(all_results[col], model_names)
                r2_with    = all_results[col][best_with]["R2"] if best_with else float("nan")
                if excl_pts:
                    t_pre    = t_pre_per_cat[col]
                    rem_pre  = rem_pre_per_cat[col]
                    Ct_pre   = C0 * (1 - rem_pre / 100.0)
                    if add_t0 and not has_t0:
                        t_pre_fit  = np.concatenate(([0.0], t_pre))
                        Ct_pre_fit = np.concatenate(([C0], Ct_pre))
                    else:
                        t_pre_fit  = t_pre
                        Ct_pre_fit = Ct_pre
                    res_without  = _fit_nonlinear(t_pre_fit, Ct_pre_fit, C0)
                    best_without = _best_model(res_without, model_names)
                    r2_without   = res_without[best_without]["R2"] if best_without else float("nan")
                else:
                    best_without = best_with
                    r2_without   = r2_with
                det_rows.append({
                    "Catalyst":                 cat_label,
                    "Auto-excluded t (min)":    ", ".join(str(int(x)) for x in sorted(excl_pts)) if excl_pts else "—",
                    "Best model WITHOUT excl.": best_without or "—",
                    "R² (without)":             f"{r2_without:.4f}" if not np.isnan(r2_without) else "—",
                    "Best model WITH excl.":    best_with or "—",
                    "R² (with)":                f"{r2_with:.4f}" if not np.isnan(r2_with) else "—",
                })
            st.dataframe(pd.DataFrame(det_rows), use_container_width=True, hide_index=True)

    with st.expander("🔍 All models for each catalyst", expanded=False):
        for col, res in all_results.items():
            cat_label = col.replace(" Removal (%)","").strip()
            st.markdown(f"**{cat_label}**")
            subrows = []
            for m in model_names:
                mr = res[m]
                note = ""
                if m == "Eley-Rideal":
                    note = ("⚠️ Fit for completeness/comparison only — structurally "
                            "redundant with Pseudo-first-order (low coverage) or "
                            "Langmuir-Hinshelwood (general coverage) under "
                            "excess-oxidant conditions; never eligible for best-model "
                            "selection. k_ER and K are not individually identifiable.")
                if mr.get("converged", True):
                    subrows.append({
                        "Model":    m,
                        "k (±SE)":  _fmt_pm(mr.get("k"), mr.get("k_se")),
                        "R²":       mr.get("R2","N/A"),
                        "Adj-R²":   mr.get("adj_r2","N/A"),
                        "AICc":     mr.get("aicc","N/A"),
                        "AIC":      mr.get("aic","N/A"),
                        "t½ (min)": _fmt_thalf(mr.get("t_half", float("nan"))),
                        "Note":     note,
                    })
                else:
                    subrows.append({"Model": m, "k (±SE)": "fit failed",
                                    "R²":"–","Adj-R²":"–","AICc":"–",
                                    "AIC":"–","t½ (min)":"–", "Note": note})
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

    if uploaded is None: st.info("Upload a file above."); return
    df, time_col, removal_cols = _load_kinetic_data(uploaded)
    if df is None: return
    t  = df[time_col].dropna().values.astype(float)
    C0 = cfg["C0"]; V = cfg["V_fuel"]; m = cfg["m_cat"]
    if C0 is None: st.error("C₀ conversion failed."); return
    model_names = MODEL_NAMES

    # ── Catalyst type selector ────────────────────────────────────
    cat_type = st.radio(
        "Catalyst type",
        ["Option A — Metal / Metal Oxide  (site-based TON & TOF)",
         "Option B — Carbon-based / Metal-free  (mass-normalized TOF)"],
        horizontal=False)

    # ════════════════════════════════════════════════════════════════
    # OPTION A — Site-based TON & TOF (existing logic)
    # ════════════════════════════════════════════════════════════════
    if "Option A" in cat_type:
        st.markdown("""
**Definitions (site-based):**
- **TON** = n_substrate_converted / n_active_sites &nbsp;(dimensionless)
- **TOF_avg** (min⁻¹) = TON / t_reaction — the average over the whole run, which depends on when the run was stopped
- **TOF₀** (min⁻¹) = r₀ · V / n_sites — the initial-rate TOF, which does not depend on stopping time
- **n_active_sites** from direct measurement or BET + ρ_site
        """)
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
                rho_default = float(preset["rho"]) if preset["rho"] is not None else 1.0
                if preset["rho"] is not None:
                    st.markdown(f"""
**Typical ρ_site for {mat_type.split('(')[0].strip()}:**
- Range: **{preset["range"]} µmol/m²**
- Recommended method: *{preset["method"]}*
                    """)
                else:
                    st.markdown("Enter your own ρ_site value below.")
                rho_site = st.number_input(
                    "ρ_site — active site surface density (µmol/m²)",
                    min_value=0.01, value=rho_default, step=0.1)
            n_sites_mol = s_bet * m * rho_site * 1e-6
            st.info(
                f"n_active_sites = S_BET × m × ρ_site = "
                f"{s_bet} × {m} × {rho_site} µmol/m² = "
                f"**{n_sites_mol*1e6:.3f} µmol**")

        st.markdown("---")
        st.markdown("### 📋 TON & TOF Results")
        if n_sites_mol <= 0:
            st.error("n_active_sites = 0. Check your inputs."); return
        rows = []
        chart_data = []
        for col in removal_cols:
            removal = df[col].dropna().values[:len(t)].astype(float)
            cat_label = col.replace(" Removal (%)","").strip()
            X_final = removal[-1] / 100.0
            n_conv  = C0 * V * X_final
            t_rxn   = t[-1]
            ton = n_conv / n_sites_mol
            tof = ton / t_rxn if t_rxn > 0 else float("nan")

            Ct  = C0 * (1 - removal / 100.0)
            res = _fit_nonlinear(t, Ct, C0)
            best = _best_model(res, model_names)
            r0   = res[best].get("r0") if (best and res[best].get("converged", True)) else None
            tof0 = _initial_tof_site(r0, V, n_sites_mol)
            model_label = best if best else "—"
            tof0_str = (f"{tof0:.5f}" if not np.isnan(tof0)
                        else f"N/A — {model_label} has no defined initial rate")
            tof0_h_str = (f"{tof0*60:.3f}" if not np.isnan(tof0)
                          else f"N/A — {model_label} has no defined initial rate")

            rows.append({
                "Catalyst":          cat_label,
                "X_final (%)":       round(removal[-1], 1),
                "n_conv (µmol)":     round(n_conv * 1e6, 3),
                "n_sites (µmol)":    round(n_sites_mol * 1e6, 3),
                "TON":               round(ton, 3),
                "TOF_avg (min⁻¹)":   f"{tof:.5f}" if not np.isnan(tof) else "N/A",
                "TOF_avg (h⁻¹)":     f"{tof*60:.3f}" if not np.isnan(tof) else "N/A",
                "Model (AICc)":      model_label,
                "TOF_0 (min⁻¹)":     tof0_str,
                "TOF_0 (h⁻¹)":       tof0_h_str,
            })
            if not np.isnan(tof0):
                chart_data.append((cat_label, tof0 * 60.0))
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
        st.caption("The initial-rate TOF_0 is fitted from the raw uploaded series "
                   "(AICc-selected model) and does not honour Tab 1's manual or "
                   "auto-saturation exclusions.")
        if len(chart_data) > 1:
            fig, ax = plt.subplots(figsize=(7, 4))
            cats = [c for c, _ in chart_data]
            tofs = [v for _, v in chart_data]
            bars = ax.bar(cats, tofs, color=COLORS[:len(cats)],
                          edgecolor="black", linewidth=0.8)
            ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=9)
            ax.set_ylabel("TOF₀ (h⁻¹)")
            ax.set_title("Initial Turnover Frequency (AICc-selected model)")
            fig.tight_layout(); st.pyplot(fig); plt.close(fig)
            plotted = {c for c, _ in chart_data}
            omitted = [r["Catalyst"] for r in rows if r["Catalyst"] not in plotted]
            if omitted:
                st.caption("Omitted (no defined initial rate): "
                           + ", ".join(omitted))
        elif not chart_data:
            st.info("No catalyst has a defined initial-rate TOF_0 — "
                    "all selected models lack an initial rate, so the chart is skipped.")

    # ════════════════════════════════════════════════════════════════
    # OPTION B — Mass-normalized TOF for carbon-based catalysts
    # ════════════════════════════════════════════════════════════════
    else:
        st.markdown("""
**Why mass-normalized for carbon-based catalysts?**
For metal-free graphene-like materials, defining "active sites" is ambiguous —
XPS gives total heteroatom content, not just catalytically active sites, and
BET area includes pores inaccessible to DBT. Mass-normalized TOF is the
standard in the ODS literature for carbon-based catalysts.

**Definitions:**
- **TOF_mass_avg** (mmol·g⁻¹·min⁻¹) = n_DBT_removed / (m_cat × t_reaction) — average over the whole run, depends on stopping time
- **TOF_BET_avg** (mmol·m⁻²·min⁻¹) = TOF_mass_avg / BET_area  *(if BET available)*
- **r₀/m** (mmol·g⁻¹·min⁻¹) = initial rate per gram catalyst — does not depend on stopping time
- **r₀/S_BET** (mmol·m⁻²·min⁻¹) = (r₀/m) / BET_area  *(if BET available)*
        """)

        # ── Try to read BET from Excel sheet ─────────────────────
        bet_per_cat = {}
        try:
            if uploaded.name.endswith(".xlsx") or uploaded.name.endswith(".xls"):
                xl = pd.ExcelFile(uploaded)
                if "Catalyst_Properties" in xl.sheet_names:
                    df_props = pd.read_excel(uploaded, sheet_name="Catalyst_Properties")
                    # Normalize column names
                    df_props.columns = [c.strip().lower() for c in df_props.columns]
                    # Find catalyst name column and BET column
                    cat_col  = next((c for c in df_props.columns if "catalyst" in c), None)
                    bet_col  = next((c for c in df_props.columns if "bet" in c), None)
                    if cat_col and bet_col:
                        for _, row_p in df_props.iterrows():
                            cat_name = str(row_p[cat_col]).strip()
                            bet_val  = row_p[bet_col]
                            if pd.notna(bet_val):
                                bet_per_cat[cat_name] = float(bet_val)
                        st.success(f"✅ BET data loaded from **Catalyst_Properties** sheet "
                                   f"({len(bet_per_cat)} catalysts).")
        except Exception:
            pass

        if bet_per_cat:
            st.markdown("**BET values from file:**")
            bet_df = pd.DataFrame(
                [{"Catalyst": k, "BET (m²/g)": v} for k, v in bet_per_cat.items()])
            st.dataframe(bet_df, use_container_width=True, hide_index=True)
        else:
            st.info("ℹ️ No **Catalyst_Properties** sheet found in the uploaded file. "
                    "Enter BET values manually below, or add a sheet named "
                    "**Catalyst_Properties** with columns: *Catalyst* and *BET (m²/g)*.")

        # ── Manual BET entry per catalyst ──────────────────────────
        st.markdown("#### BET surface area per catalyst")
        st.caption("Values from file are pre-filled. Edit if needed. "
                   "Leave at 0 to skip TOF_BET for that catalyst.")
        n_cols_ui = min(len(removal_cols), 3)
        cols_ui   = st.columns(n_cols_ui)
        bet_inputs = {}
        for ci, col in enumerate(removal_cols):
            cat_label = col.replace(" Removal (%)","").strip()
            # Match from file (try exact, then partial)
            default_bet = bet_per_cat.get(cat_label, 0.0)
            if default_bet == 0.0:
                for k, v in bet_per_cat.items():
                    if k.lower() in cat_label.lower() or cat_label.lower() in k.lower():
                        default_bet = v; break
            with cols_ui[ci % n_cols_ui]:
                bet_inputs[col] = st.number_input(
                    f"**{cat_label}** BET (m²/g)",
                    min_value=0.0, value=float(default_bet), step=10.0,
                    key=f"bet_b_{col}")

        st.markdown("---")
        st.markdown("### 📋 Mass-Normalized TOF Results")

        rows = []
        chart_data = []
        for col in removal_cols:
            removal  = df[col].dropna().values[:len(t)].astype(float)
            cat_label = col.replace(" Removal (%)","").strip()
            X_final  = removal[-1] / 100.0
            n_conv   = C0 * V * X_final          # mol
            t_rxn    = t[-1]                     # min

            # TOF_mass = n_conv (mmol) / (m_cat(g) × t(min))
            tof_mass_mmol = (n_conv * 1000) / (m * t_rxn) if (m > 0 and t_rxn > 0) else float("nan")

            # Initial-rate per gram catalyst, from the AICc-selected fit
            Ct  = C0 * (1 - removal / 100.0)
            res = _fit_nonlinear(t, Ct, C0)
            best = _best_model(res, model_names)
            r0   = res[best].get("r0") if (best and res[best].get("converged", True)) else None
            r0_m = _initial_tof_mass(r0, V, m)
            model_label = best if best else "—"
            r0_m_str = (f"{r0_m:.6f}" if not np.isnan(r0_m)
                        else f"N/A — {model_label} has no defined initial rate")

            # TOF_BET
            bet_val = bet_inputs[col]
            if bet_val > 0:
                tof_bet = tof_mass_mmol / bet_val   # mmol/(m²·min)
                r0_bet  = r0_m / bet_val if not np.isnan(r0_m) else float("nan")
            else:
                tof_bet = float("nan")
                r0_bet  = float("nan")

            row = {
                "Catalyst":                         cat_label,
                "X_final (%)":                      round(removal[-1], 1),
                "n_conv (µmol)":                    round(n_conv * 1e6, 3),
                "TOF_mass_avg (mmol·g⁻¹·min⁻¹)":   f"{tof_mass_mmol:.5f}" if not np.isnan(tof_mass_mmol) else "N/A",
                "TOF_mass_avg (µmol·g⁻¹·min⁻¹)":   f"{tof_mass_mmol*1000:.3f}" if not np.isnan(tof_mass_mmol) else "N/A",
                "TOF_mass_avg (mmol·g⁻¹·h⁻¹)":     f"{tof_mass_mmol*60:.4f}" if not np.isnan(tof_mass_mmol) else "N/A",
                "Model (AICc)":                     model_label,
                "r₀/m (mmol·g⁻¹·min⁻¹)":           r0_m_str,
                "BET (m²/g)":                       round(bet_val, 1) if bet_val > 0 else "—",
                "TOF_BET_avg (mmol·m⁻²·min⁻¹)":    f"{tof_bet:.6f}" if not np.isnan(tof_bet) else "—",
                "r₀/S_BET (mmol·m⁻²·min⁻¹)":       f"{r0_bet:.6f}" if not np.isnan(r0_bet) else "—",
            }
            rows.append(row)
            if not np.isnan(r0_m):
                chart_data.append((cat_label, r0_m))

        st.dataframe(pd.DataFrame(rows), use_container_width=True)
        st.caption("The initial-rate r₀/m and r₀/S_BET are fitted from the raw "
                   "uploaded series (AICc-selected model) and do not honour Tab 1's "
                   "manual or auto-saturation exclusions.")

        # ── Bar chart: initial rate r₀/m ───────────────────────────
        if chart_data:
            fig0, ax0 = plt.subplots(figsize=(6, 4))
            cats0 = [c for c, _ in chart_data]
            vals0 = [v for _, v in chart_data]
            bars0 = ax0.bar(cats0, vals0, color=COLORS[:len(cats0)],
                            edgecolor="black", linewidth=0.8)
            ax0.bar_label(bars0, fmt="%.4f", padding=2, fontsize=8)
            ax0.set_ylabel("r₀/m (mmol·g⁻¹·min⁻¹)")
            ax0.set_title("Initial-rate mass-normalised activity (AICc-selected model)")
            ax0.tick_params(axis='x', rotation=30)
            fig0.tight_layout(); st.pyplot(fig0); plt.close(fig0)
            plotted = {c for c, _ in chart_data}
            omitted = [r["Catalyst"] for r in rows if r["Catalyst"] not in plotted]
            if omitted:
                st.caption("Omitted (no defined initial rate): "
                           + ", ".join(omitted))
        else:
            st.info("No catalyst has a defined initial-rate r₀/m — "
                    "all selected models lack an initial rate, so the chart is skipped.")

        # ── Bar charts ────────────────────────────────────────────
        col_chart1, col_chart2 = st.columns(2)
        with col_chart1:
            mass_rows = [r for r in rows
                         if r["TOF_mass_avg (µmol·g⁻¹·min⁻¹)"] != "N/A"]
            if mass_rows:
                fig, ax = plt.subplots(figsize=(6, 4))
                cats = [r["Catalyst"] for r in mass_rows]
                vals = [float(r["TOF_mass_avg (µmol·g⁻¹·min⁻¹)"]) for r in mass_rows]
                bars = ax.bar(cats, vals, color=COLORS[:len(cats)],
                              edgecolor="black", linewidth=0.8)
                ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=8)
                ax.set_ylabel("TOF_mass_avg (µmol·g⁻¹·min⁻¹)")
                ax.set_title("Mass-normalized TOF (average)")
                ax.tick_params(axis='x', rotation=30)
                fig.tight_layout(); st.pyplot(fig); plt.close(fig)
                plotted = {r["Catalyst"] for r in mass_rows}
                omitted = [r["Catalyst"] for r in rows
                           if r["Catalyst"] not in plotted]
                if omitted:
                    st.caption("Omitted (value unavailable): "
                               + ", ".join(omitted))
            else:
                st.info("No catalyst has a defined TOF_mass_avg — "
                        "chart skipped.")

        with col_chart2:
            bet_rows = [r for r in rows if r["TOF_BET_avg (mmol·m⁻²·min⁻¹)"] != "—"]
            if bet_rows:
                fig2, ax2 = plt.subplots(figsize=(6, 4))
                cats2 = [r["Catalyst"] for r in bet_rows]
                vals2 = [float(r["TOF_BET_avg (mmol·m⁻²·min⁻¹)"]) for r in bet_rows]
                bars2 = ax2.bar(cats2, vals2, color=COLORS[:len(cats2)],
                                edgecolor="black", linewidth=0.8)
                ax2.bar_label(bars2, fmt="%.5f", padding=2, fontsize=8)
                ax2.set_ylabel("TOF_BET_avg (mmol·m⁻²·min⁻¹)")
                ax2.set_title("BET-normalized TOF (average)")
                ax2.tick_params(axis='x', rotation=30)
                fig2.tight_layout(); st.pyplot(fig2); plt.close(fig2)
                plotted = {r["Catalyst"] for r in bet_rows}
                omitted = [r["Catalyst"] for r in rows
                           if r["Catalyst"] not in plotted]
                if omitted:
                    st.caption("Omitted (no BET area): "
                               + ", ".join(omitted))
            else:
                st.info("Enter BET values above to see TOF_BET chart.")

        # ── Download ──────────────────────────────────────────────
        csv_buf = io.StringIO()
        pd.DataFrame(rows).to_csv(csv_buf, index=False)
        st.download_button("⬇️ Download TOF results (CSV)",
                           csv_buf.getvalue(),
                           "tof_carbon_based.csv", "text/csv")

        with st.expander("📚 Scientific basis & reporting guidance", expanded=False):
            st.markdown("""
**Why mass-normalized TOF for metal-free carbon catalysts?**

For graphene-like, N/B-doped carbon, BCN, and similar materials:
- XPS gives **total** heteroatom content — pyridinic, pyrrolic, graphitic N are
  all reported, but only pyridinic N (and possibly pyrrolic) is catalytically active.
- BET area includes micropores and mesopores inaccessible to DBT (MW=184 g/mol).
- No universal method exists to quantify "active sites" for metal-free ODS catalysts.

**Recommended reporting:**
- Primary: **TOF_mass (µmol·g⁻¹·min⁻¹)** — comparable across literature.
- Secondary: **TOF_BET (mmol·m⁻²·min⁻¹)** — useful for comparing catalysts
  with very different surface areas.
- Note explicitly in the paper that TOF is mass-normalized, not site-normalized.

**References:**
- Astle et al., *ACS Catal.* 2022 — mass-normalized TOF for carbon ODS catalysts.
- Dou et al., *Appl. Catal. B* 2020 — BET-normalized activity for N-doped carbons.
            """)



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
def _arrhenius_ci(cov, n_T):
    """95% confidence intervals for Eₐ and ln A from an Arrhenius fit.

    Uses the t-distribution critical value t(0.975, n_T - 2) rather than the
    normal-approximation z = 1.96, which understates the interval when only a
    few temperature points are available. For n_T == 2 (df == 0) no valid CI
    exists: both bounds are returned as NaN and df is returned as 0 so the
    caller can surface a warning (point estimate of Eₐ only).
    """
    df = n_T - 2
    if df < 1:
        return np.nan, np.nan, df
    t_crit = scipy_stats.t.ppf(0.975, df)
    Ea_ci = np.sqrt(cov[0, 0]) * R_GAS / 1000.0 * t_crit
    lnA_ci = np.sqrt(cov[1, 1]) * t_crit
    return Ea_ci, lnA_ci, df


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
                cat_k[col] = res[chosen]["k"] if (chosen and res[chosen].get("converged", True)) else None
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
                n_T = len(k_vals)
                df_arr = n_T - 2
                if df_arr >= 1:
                    coeffs, cov = np.polyfit(inv_T_v, ln_k, 1, cov=True)
                    Ea_ci, lnA_ci, df_arr = _arrhenius_ci(cov, n_T)
                else:
                    coeffs = np.polyfit(inv_T_v, ln_k, 1)
                    Ea_ci, lnA_ci, df_arr = _arrhenius_ci(None, n_T)
                    st.warning(
                        "⚠️ **2-point Arrhenius fit** — only 2 temperatures were used, "
                        "so the fit has zero degrees of freedom (df = n_T − 2 = 0). "
                        "No valid 95% confidence interval can be reported; only the "
                        "point estimate of Eₐ is shown."
                    )
                slope = coeffs[0]; intercept = coeffs[1]
                Ea_kJ = -slope * R_GAS / 1000.0
                A_val = np.exp(intercept)
                r2_arr = _r2(ln_k, np.polyval(coeffs, inv_T_v))
                ax.scatter(inv_T_v * 1000, ln_k, color=color,
                           marker=MARKERS[ci % len(MARKERS)], s=70, zorder=5, label=cat)
                x_fit = np.linspace(inv_T_v.min(), inv_T_v.max(), 100)
                ax.plot(x_fit * 1000, np.polyval(coeffs, x_fit), "--", color=color, linewidth=1.5)
                for x_, y_, T_ in zip(inv_T_v, ln_k, T_valid):
                    ax.annotate(f"{T_-273.15:.0f}°C", (x_*1000, y_),
                                textcoords="offset points", xytext=(4,4), fontsize=8)
                ax.set_xlabel("1000/T (K⁻¹)"); ax.set_ylabel("ln k")
                if df_arr >= 1:
                    title_ea = f"Eₐ = {Ea_kJ:.1f} ± {Ea_ci:.1f} kJ/mol (t, 95% CI)"
                else:
                    title_ea = f"Eₐ = {Ea_kJ:.1f} kJ/mol (no CI: df = 0)"
                ax.set_title(f"{cat}\n{title_ea}\nR² = {r2_arr:.4f}")
                arrh_rows.append({"Catalyst": cat,
                                  "Eₐ (kJ/mol)": round(Ea_kJ, 2),
                                  "± 95% CI Eₐ (kJ/mol, t)": round(Ea_ci, 2) if df_arr >= 1 else "N/A",
                                  "ln A": round(intercept, 3),
                                  "± 95% CI ln A (t)": round(lnA_ci, 3) if df_arr >= 1 else "N/A",
                                  "A": _fmt_sci(A_val),
                                  "R² (Arrhenius)": round(r2_arr, 4),
                                  "n_T": n_T})
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
    if not res.get("converged", True):
        st.error(f"Model '{model_choice}' failed. Error: {res.get('error','unknown')}"); return
    Ct_pred   = res["pred"]
    residuals = Ct_obs - Ct_pred
    n         = len(residuals)
    # FIX U: use number of FREE parameters (N_PARAMS), not len(params) which counts fixed C0
    p_free   = N_PARAMS.get(model_choice, 1)
    ddof_use = p_free if (n - p_free) > 0 else 0
    sigma    = np.std(residuals, ddof=ddof_use)
    std_resid = residuals / sigma if sigma > 0 else residuals
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
            if not mr.get("converged", True):
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
