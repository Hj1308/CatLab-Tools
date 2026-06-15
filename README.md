# CatLab-Tools 🔬
**ODS Calculation Suite — v3.0**  
Author: [Hoda Jafari](https://github.com/Hj1308) | MIT License

> **Surface area & pore analysis (BET/BJH/T-Plot)?**  
> → See [BET_analyser](https://github.com/Hj1308/BET_analyser)

---

## What is CatLab-Tools?

A Streamlit-based web application for **oxidative desulfurization (ODS) reaction analysis**.  
Designed for PhD-level catalysis research — covers kinetics, activity metrics, reusability, and condition comparison.

---

## 🚀 Quick Start

```bash
git clone https://github.com/Hj1308/CatLab-Tools.git
cd CatLab-Tools
pip install -r requirements.txt
streamlit run app_ods.py
```

---

## 📑 Modules (Tabs)

| Tab | Module | Description |
|-----|--------|-------------|
| 1 | **Templates** | Generate Excel input templates for all modules |
| 2 | **Kinetics & t½** | Fit 5 kinetic models, report k±SE, r₀, t½ |
| 3 | **TOF / TON** | Turnover frequency and turnover number |
| 4 | **Reusability** | Removal retention over reuse cycles |
| 5 | **Parameter Effect** *(v3.0)* | X%, k, or t½ vs a variable parameter |
| 6 | **Oxidant Efficiency** *(v3.0)* | H₂O₂ utilisation efficiency (η%) |
| 7 | **Condition Comparison** *(v3.0)* | Thermal vs UV vs ECODS — side-by-side t½/k |

---

## 📐 Kinetic Models (Tab 2)

| Model | Equation | t½ |
|-------|----------|----|
| Zero-order | $C_t = C_0 - k_0 t$ | $C_0 / (2k_0)$ |
| Pseudo-first-order | $C_t = C_0\, e^{-k_{app}t}$ | $\ln 2 / k_{app}$ |
| Second-order | $C_t = C_0 / (1 + k_2 C_0 t)$ | $1 / (k_2 C_0)$ |
| Elovich | $C_t = C_0 - \frac{1}{\beta}\ln(1+\alpha\beta t)$ | $(e^{C_0\beta/2}-1)/(\alpha\beta)$ |
| **Langmuir-Hinshelwood** *(v3.0)* | $\frac{dC}{dt} = -\frac{k_{LH} K C}{1+KC}$ (ODE) | $\frac{\ln 2}{k_{LH}K} + \frac{C_0/2}{k_{LH}}$ |

### New in v3.0 — k ± SE and r₀

**Standard Error** is extracted from the `curve_fit` covariance matrix:

$$SE_k = \sqrt{[\Sigma^{-1}]_{kk}}$$

**Initial reaction rate r₀** (model-independent activity metric):

| Model | r₀ formula |
|-------|-----------|
| Pseudo-first | $r_0 = k_{app} \cdot C_0$ |
| Second-order | $r_0 = k_2 \cdot C_0^2$ |
| Elovich | $r_0 = \alpha$ |
| L-H | $r_0 = k_{LH} K C_0 / (1 + K C_0)$ |

---

## 💧 Oxidant Efficiency (Tab 6)

$$\eta_{H_2O_2}(\%) = \frac{n_{DBT,\,removed}}{n_{H_2O_2,\,consumed} / 2} \times 100$$

Stoichiometry: DBT + 2 H₂O₂ → DBTO₂ + 2 H₂O (1 : 2 molar ratio).

---

## 📉 Parameter Effect (Tab 5)

Upload a table of `[parameter value | Cat-A | Cat-B | ...]` to plot  
**X_final(%) or k or t½ vs the variable parameter** for multiple catalysts on one chart.

Supported parameters: catalyst loading, H₂O₂ dose, DBT concentration, voltage (ECODS), light intensity, temperature, …

---

## 🔀 Condition Comparison (Tab 7)

Upload multiple `kinetics_summary.xlsx` files (one per condition — output of Tab 2)  
and compare **t½, Best R², or r₀** across conditions with grouped bar charts.

---

## 🔁 Reusability (Tab 4)

$$\text{Retention}(\%) = \frac{X_{cycle\,n}}{X_{cycle\,1}} \times 100$$

---

## ⚙️ TOF / TON (Tab 3)

$$TON = \frac{n_{substrate,\,converted}}{n_{active\,sites}} \qquad TOF\,(h^{-1}) = \frac{TON}{t\,(h)}$$

---

## 📦 Supported Units

| Unit | Notes | MW Required? |
|------|-------|:---:|
| `ppmS` | ppm Sulfur — MW_S = 32.06 auto-applied | No |
| `ppm` / `mg/L` | Aqueous pollutant | ✅ |
| `mmol/L` / `mol/L` | Molar concentration | No |
| `g/L` | Grams per litre | ✅ |

---

## 🗂 Repository Structure

```
CatLab-Tools/
├── app_ods.py          # Main Streamlit app (v3.0)
├── requirements.txt    # numpy, pandas, matplotlib, scipy, openpyxl, streamlit
├── catlab/             # Core Python library modules
├── examples/           # Example datasets
└── tests/              # Unit tests
```

---

## 🔗 Related Repositories

| Repo | Purpose |
|------|---------|
| [BET_analyser](https://github.com/Hj1308/BET_analyser) | BET, BJH, T-Plot, isotherm & hysteresis |
| [EISforge-](https://github.com/Hj1308/EISforge-) | EIS analysis + ML |
| [sem-particle-analyzer](https://github.com/Hj1308/sem-particle-analyzer) | SEM particle sizing |
| [Raman-analysis](https://github.com/Hj1308/Raman-analysis) | Raman spectroscopy toolkit |

---

## Changelog

| Version | Changes |
|---------|---------|
| **v3.0** | L-H kinetic model; k±SE from covariance matrix; r₀ (initial rate); Tab 5 Parameter Effect; Tab 6 Oxidant Efficiency; Tab 7 Condition Comparison |
| v2.0 | Elovich t½ fix; per-sheet error handling; TOF/TON module; Reusability module |
| v1.4 | Initial public release |

---

## License
MIT — free to use, modify, and distribute.
