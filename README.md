# CatLab-Tools 🔬

**Integrated Catalyst & Materials Analysis Suite**  
Author: [Hoda Jafari](https://github.com/Hj1308) | MIT License

---

## What is CatLab-Tools?

CatLab-Tools is an open-source Python toolkit designed for researchers in **heterogeneous catalysis**, **water treatment**, and **nanomaterials characterisation**. It unifies surface area analysis, kinetic modelling, and catalytic performance metrics in one clean package.

---

## Modules

| Module | Description |
|---|---|
| `convert_to_mmol_L()` | Universal unit converter: ppmS, ppm, mg/L, g/L, mmol/L, mol/L |
| `SampleInfo` | Structured dataclass for experimental metadata |
| `KineticsAnalyser` | Zero/first/second/pseudo-first order model fitting |
| `calc_conversion()` | Conversion X(%) profile over time |
| `calc_tof()` | Turnover Frequency — TOF (h⁻¹) |
| `calc_toc_removal()` | Total Organic Carbon removal (%) |
| `BETAnalyser` | BET surface area from N₂ physisorption |
| `TPlotAnalyser` | T-Plot (Harkins-Jura): micropore / mesopore / macropore % |

---

## Installation

```bash
git clone https://github.com/Hj1308/CatLab-Tools.git
cd CatLab-Tools
pip install -r requirements.txt
```

---

## Quick Start

### Desulfurization (ppmS)

```python
from catlab import SampleInfo, KineticsAnalyser, convert_to_mmol_L
import numpy as np

info = SampleInfo(
    sample_name         = "MoS2/Al2O3",
    process_type        = "desulfurization",
    catalyst_mass_g     = 0.05,
    solution_vol_L      = 0.050,
    c0_value            = 500.0,
    c0_unit             = "ppmS",
    active_sites_mmol_g = 0.32,
)

time_h = np.array([0, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0])
c_ppmS = np.array([500, 420, 350, 250, 160, 100, 45])
c_mmol = np.array([convert_to_mmol_L(v, "ppmS") for v in c_ppmS])

an = KineticsAnalyser(time_h, c_mmol, info)
report = an.full_report()
print(f"X = {report['Conversion X (%)']:.1f}%  |  TOF = {report['TOF (h⁻¹)']} h⁻¹")
an.plot_kinetics("output.png")
```

### BET + T-Plot

```python
from catlab import BETAnalyser, TPlotAnalyser
import numpy as np

p_rel = np.array([0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.90, 0.99])
v_ads = np.array([85,  102,  126,  143,  172,  200,  280,  520])

bet = BETAnalyser(p_rel, v_ads)
bet_res = bet.fit_bet()
print(f"S_BET = {bet_res['S_BET_m2g']} m²/g  |  C = {bet_res['C_BET']}")

tplot  = TPlotAnalyser(p_rel, v_ads, s_bet=bet_res["S_BET_m2g"], total_pore_volume=0.52)
report = tplot.full_tplot_report()
print(f"Micropore: {report['Micropore_%']}%  |  Mesopore: {report['Mesopore_%']}%")
tplot.plot_tplot("tplot.png")
```

### TOC Removal

```python
from catlab import calc_toc_removal
print(calc_toc_removal(toc0=85.0, toc_t=12.0))  # → 85.88%
```

---

## Supported Concentration Units

| Unit | Description | MW Required? |
|---|---|---|
| `mol/L` | Molar | No |
| `mmol/L` | Millimolar | No |
| `ppmS` | ppm Sulfur (auto MW_S = 32.06 g/mol) | No |
| `ppm` | Parts per million ≈ mg/L (aqueous) | ✅ Yes |
| `mg/L` | Milligrams per litre | ✅ Yes |
| `g/L` | Grams per litre | ✅ Yes |

---

## Project Structure

```
CatLab-Tools/
├── catlab/
│   ├── __init__.py
│   └── catalyst_analytics.py   ← main module
├── tests/
│   └── test_catlab.py          ← pytest tests
├── examples/
│   ├── desulfurization_example.py
│   └── bet_tplot_example.py
├── requirements.txt
└── README.md
```

---

## Citation

```bibtex
@software{jafari2026catlab,
  author  = {Jafari, Hoda},
  title   = {CatLab-Tools: Integrated Catalyst and Materials Analysis Suite},
  year    = {2026},
  url     = {https://github.com/Hj1308/CatLab-Tools}
}
```

---

## Related Projects

- [EISforge-](https://github.com/Hj1308/EISforge-) — Electrochemical Impedance Spectroscopy + ML
- [BET_analyser](https://github.com/Hj1308/BET_analyser) — BET/BJH physisorption analysis
- [sem-particle-analyzer](https://github.com/Hj1308/sem-particle-analyzer) — SEM image particle sizing
- [Raman-analysis](https://github.com/Hj1308/Raman-analysis) — Raman spectroscopy toolkit

---

## License
MIT — free to use, modify, and distribute.
