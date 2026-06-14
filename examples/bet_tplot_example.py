# examples/bet_tplot_example.py
# Complete example: BET surface area + T-Plot pore distribution

import numpy as np
from catlab import BETAnalyser, TPlotAnalyser

# ── Physisorption data (N2 at 77 K) ──────────────────
p_rel = np.array([0.05, 0.10, 0.15, 0.20, 0.25, 0.30,
                  0.35, 0.50, 0.70, 0.90, 0.99])
v_ads = np.array([85,  102,  115,  126,  135,  143,
                  150,  172,  200,  280,  520])

# ── BET analysis ──────────────────────────────────────
print("=" * 50)
print("BET ANALYSIS")
print("=" * 50)
bet     = BETAnalyser(p_rel, v_ads)
bet_res = bet.fit_bet()
for k, v in bet_res.items():
    print(f"  {k:<20} = {v}")

# ── T-Plot analysis ───────────────────────────────────
print("\n" + "=" * 50)
print("T-PLOT ANALYSIS")
print("=" * 50)
tplot  = TPlotAnalyser(p_rel, v_ads,
                       s_bet=bet_res["S_BET_m2g"],
                       total_pore_volume=0.52)
report = tplot.full_tplot_report()
for k, v in report.items():
    print(f"  {k:<20} = {v}")

# ── Plots ─────────────────────────────────────────────
tplot.plot_tplot(save_path="tplot_output.png")
print("\n  T-Plot saved → tplot_output.png")
