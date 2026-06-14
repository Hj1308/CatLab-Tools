# CatLab-Tools 🔬

**Catalyst Reaction Analysis Suite**  
Author: [Hoda Jafari](https://github.com/Hj1308) | MIT License

> **Surface area & pore analysis (BET/BJH/T-Plot)?**  
> → See [BET_analyser](https://github.com/Hj1308/BET_analyser)

---

## What is CatLab-Tools?

A focused Python toolkit for **catalytic reaction analysis**:  
kinetics, conversion, turnover frequency, and TOC removal.

---

## Modules

| Module | Description |
|---|---|
| `convert_to_mmol_L()` | Unit converter: ppmS, ppm, mg/L, g/L, mmol/L, mol/L |
| `SampleInfo` | Experimental metadata dataclass |
| `KineticsAnalyser` | Zero/first/second/pseudo-first order fitting + plots |
| `calc_conversion()` | Conversion X(%) = (C0−Ct)/C0 × 100 |
| `calc_tof()` | Turnover Frequency TOF (h⁻¹) |
| `calc_toc_removal()` | Total Organic Carbon removal (%) |

---

## Installation

```bash
git clone https://github.com/Hj1308/CatLab-Tools.git
cd CatLab-Tools
pip install -r requirements.txt
```

---

## Quick Start

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

an     = KineticsAnalyser(time_h, c_mmol, info)
report = an.full_report()
print(f"X = {report['Conversion X (%)']:.1f}%  |  TOF = {report['TOF (h⁻¹)']} h⁻¹")
an.plot_kinetics("output.png")
```

---

## Supported Units

| Unit | Notes | MW Required? |
|---|---|---|
| `ppmS` | ppm Sulfur — MW_S = 32.06 auto-applied | No |
| `ppm` / `mg/L` | Aqueous pollutant | ✅ Yes |
| `mol/L` / `mmol/L` | Molar | No |
| `g/L` | Grams per litre | ✅ Yes |

---

## Related Repositories

- [BET_analyser](https://github.com/Hj1308/BET_analyser) — BET, BJH, T-Plot, isotherm & hysteresis classification
- [EISforge-](https://github.com/Hj1308/EISforge-) — EIS analysis + ML
- [sem-particle-analyzer](https://github.com/Hj1308/sem-particle-analyzer) — SEM particle sizing
- [Raman-analysis](https://github.com/Hj1308/Raman-analysis) — Raman spectroscopy toolkit

---

## License
MIT — free to use, modify, and distribute.
