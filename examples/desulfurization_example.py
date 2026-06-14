# examples/desulfurization_example.py
# Complete example: HDS catalyst analysis with ppmS concentration

import numpy as np
from catlab import SampleInfo, KineticsAnalyser, convert_to_mmol_L, calc_toc_removal

# ── 1. Define sample ──────────────────────────────────
info = SampleInfo(
    sample_name         = "MoS2/Al2O3",
    process_type        = "desulfurization",
    catalyst_mass_g     = 0.05,
    solution_vol_L      = 0.050,
    c0_value            = 500.0,
    c0_unit             = "ppmS",          # auto-converts with MW_S = 32.06
    active_sites_mmol_g = 0.32,            # from NH3-TPD
    notes               = "DBT in n-decane, 300°C, 30 bar H2"
)
print("=" * 50)
print("SAMPLE INFO")
print("=" * 50)
import pandas as pd
print(pd.DataFrame([info.summary()]).T.to_string())

# ── 2. Concentration–time data ────────────────────────
time_h = np.array([0, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0])
c_ppmS = np.array([500, 420, 350, 250, 160, 100, 45])
c_mmol = np.array([convert_to_mmol_L(v, "ppmS") for v in c_ppmS])

# ── 3. Kinetics analysis ──────────────────────────────
analyzer = KineticsAnalyser(time_h, c_mmol, info)
report   = analyzer.full_report()

print("\n" + "=" * 50)
print("KINETICS REPORT")
print("=" * 50)
print(f"  Conversion X       = {report['Conversion X (%)']:.1f}%")
print(f"  Best model         = {report['Best Fit Model']['model']}")
print(f"  Rate constant      = {list(report['Best Fit Model'].values())[1]}")
print(f"  R²                 = {report['Best Fit Model']['R2']}")
print(f"  TOF                = {report['TOF (h⁻¹)']} h⁻¹")

print("\n  All models:")
for m in report["All Models"]:
    print(f"    {m['model']:22s}  R²={m['R2']}")

# ── 4. Plot ───────────────────────────────────────────
analyzer.plot_kinetics(save_path="MoS2_Al2O3_kinetics.png")
print("\n  Plot saved → MoS2_Al2O3_kinetics.png")

# ── 5. TOC ───────────────────────────────────────────
toc = calc_toc_removal(toc0=85.0, toc_t=12.0)
print(f"\n  TOC removal        = {toc}%")
