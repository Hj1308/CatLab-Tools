# CatLab-Tools 🔬
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20753374.svg)](https://doi.org/10.5281/zenodo.20753374)
![Version](https://img.shields.io/badge/version-v3.5.1-blue?style=flat-square)
![Python](https://img.shields.io/badge/python-3.9%2B-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/built%20with-Streamlit-FF4B4B?style=flat-square&logo=streamlit)
![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)

**ODS Calculation Suite — v3.5.1**  
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
| 1 | **Kinetic Fitting** | Fit 9 kinetic models, report k±SE, r₀, t½, AICc |
| 2 | **Linearization** | Linear transforms for zero/first/second/Elovich |
| 3 | **Removal Efficiency** | Desulfurization X% vs time + bar chart |
| 4 | **TON / TOF** | Turnover number and frequency (direct or BET-based) |
| 5 | **Parameter Effect** | X%, k, or t½ vs a variable parameter |
| 6 | **Oxidant Efficiency** | H₂O₂ utilisation efficiency (η%) |
| 7 | **Condition Comparison** | Side-by-side k/t½/r₀ across conditions |
| 8 | **Arrhenius Analysis** | Multi-temperature Ea & A with 95% CI |
| 9 | **Residual Diagnostics** | Shapiro-Wilk, runs test, Q-Q plot |

---

## 📐 Kinetic Models (Tab 1)

Best model selected automatically by **AICc** (small-sample corrected AIC).  
All models fitted by nonlinear least squares with C₀ locked.

| Model | Integrated Rate Law | t½ | Class |
|-------|--------------------|----|-------|
| Zero-order | $C_t = C_0 - k_0 t$ | $C_0 / (2k_0)$ | Simplified |
| Pseudo-first-order | $C_t = C_0\, e^{-k_{app}t}$ | $\ln 2 / k_{app}$ | Simplified |
| Pseudo-second-order | $C_t = C_0 / (1 + k_2 C_0 t)$ | $1 / (k_2 C_0)$ | Simplified |
| Elovich | $C_t = C_0 - \frac{1}{\beta}\ln(1+\alpha\beta t)$ | $(e^{C_0\beta/2}-1)/(\alpha\beta)$ | Phenomenological |
| Langmuir-Hinshelwood | $dC/dt = -k_{LH} K C / (1+KC)$ (ODE) | $\ln2/(k_{LH}K) + C_0/(2k_{LH})$ | Mechanistic |
| Power-Law | $C_t = [C_0^{1-n} - k(1-n)t]^{1/(1-n)}$ | $C_0^{1-n}(2^{n-1}-1)/[k(n-1)]$ | Empirical |
| Eley-Rideal | $dC/dt = -k_{ER} K C$ (ODE, oxidant excess) | — | Semi-mechanistic |
| Avrami | $C_t = C_0 \exp(-k t^n)$ | — | Phenomenological |
| Double-Exponential | $C_t = C_0[A e^{-k_1 t} + (1-A)e^{-k_2 t}]$ | — | Phenomenological |

> **Note:** Avrami and Double-Exponential are phenomenological — high overfitting risk with < 10 data points. AICc penalises their extra parameters automatically.

### k ± SE and r₀

**Standard Error** from the `curve_fit` covariance matrix:

$$SE_k = \sqrt{[\Sigma]_{kk}}$$

**Initial reaction rate r₀:**

| Model | r₀ formula |
|-------|-----------|
| Zero-order | $r_0 = k_0$ |
| Pseudo-first-order | $r_0 = k_{app} \cdot C_0$ |
| Pseudo-second-order | $r_0 = k_2 \cdot C_0^2$ |
| Elovich | $r_0 = \alpha$ |
| L-H | $r_0 = k_{LH} K C_0 / (1 + K C_0)$ |
| Power-Law | $r_0 = k \cdot C_0^n$ |
| Eley-Rideal | $r_0 = k_{ER} K C_0$ |

---

## ⚗️ TOF / TON (Tab 4)

$$TON = \frac{n_{substrate,\,converted}}{n_{active\,sites}} \qquad TOF\,(h^{-1}) = \frac{TON}{t\,(h)}$$

Active site density can be entered directly (mmol/g) or estimated from BET surface area + material family presets.

---

## 💧 Oxidant Efficiency (Tab 6)

$$\eta_{H_2O_2}(\%) = \frac{n_{DBT,\,removed}}{n_{H_2O_2,\,consumed} / 2} \times 100$$

Stoichiometry: DBT + 2 H₂O₂ → DBTO₂ + 2 H₂O (1 : 2 molar ratio).

---

## 🌡️ Arrhenius Analysis (Tab 8)

Upload one kinetic data file per temperature. The app fits each dataset, extracts k(T), then fits:

$$k(T) = A \cdot \exp\!\left(-\frac{E_a}{RT}\right)$$

Reports Eₐ and A with **95% confidence intervals**.

> ⚠️ For L-H and Power-Law models, k is a composite parameter — Eₐ is apparent and not directly comparable with pseudo-first-order Eₐ from the literature.

---

## 📦 Supported Units & Concentration Logic

### ppmS vs ppm — Key Distinction

| Feature | ppmS | ppm / mg/L |
|---|---|---|
| What is measured | Mass of sulfur atom | Mass of the pollutant molecule |
| Typical context | Model fuel prepared by dissolving compound into fixed volume of fuel | Aqueous systems; model fuel with known pure compound |
| MW used | MW_S = 32.06 g/mol (fixed, auto-applied) | MW of compound (e.g. DBT = 184.26 g/mol) |
| MW input required? | ❌ No | ✅ Yes |
| Default definition | mg S / L fuel (volumetric) | mg compound / L solution |

### Conversion Formulae

**`ppmS` — Volumetric basis (default):**

$$C_0\,[\text{mol/L}] = \frac{C\,[\text{mg S/L}]}{MW_S\,[\text{g/mol}] \times 10^3}$$

MW_S = 32.06 g/mol applied automatically. No fuel density needed.  
**Example:** 250 ppmS → 250 / 32.06 / 1000 = **7.798 × 10⁻³ mol/L**

Molar concentration of compound from C₀(S):

$$C_{0,compound} = C_{0,S} / n_{sulfur}$$

> **Mass basis (advanced):** For true mass fraction (mg S / kg fuel, e.g. XRF or ASTM D5453), switch to Mass basis in the sidebar — fuel density ρ (g/mL) is then applied: $C_0 = C \times \rho / (MW_S \times 10^3)$

**`ppm` / `mg/L`:**

$$C_0\,[\text{mol/L}] = \frac{C\,[\text{mg/L}]}{MW_{compound}\,[\text{g/mol}] \times 10^3}$$

### Full Unit Support Table

| Unit | Conversion basis | MW Required? |
|---|---|---|
| ppmS | mg S / L fuel → mol/L via MW_S (volumetric default) | ❌ |
| ppm / mg/L | mg compound / L → mol/L via MW_compound | ✅ |
| mmol/L | Direct × 10⁻³ | ❌ |
| mol/L | Direct | ❌ |
| g/L | ÷ MW_compound → mol/L | ✅ |

---

## 🧪 Example: ECODS Experimental Conditions

> Constant-voltage electrochemical ODS with DBT in n-heptane model fuel.

| Parameter | Value | Unit | Notes |
|-----------|-------|------|-------|
| Initial sulfur concentration (C₀) | 250 | ppmS (mg S / L fuel) | Volumetric preparation |
| Model solvent | n-Heptane | — | Density: 0.684 g/mL at 25 °C |
| MW of Sulfur | 32.06 | g/mol | Auto-applied |
| MW of DBT | 184.26 | g/mol | Used only for back-conversion |
| **C₀ (mol/L)** | **7.798 × 10⁻³** | mol/L | 250 / 32.06 / 1000 |
| C₀ (as mg DBT/L) | 1436.8 | mg/L | 7.798×10⁻³ × 184.26 × 10³ |
| Fuel volume | 2 | mL | — |
| Catalyst mass | 5 | mg | — |
| Catalyst-to-fuel ratio | 2.5 | g/L | m_cat (g) / V_fuel (L) |

---

## 🗂 Repository Structure
CatLab-Tools/

├── app_ods.py          # Main Streamlit app (v3.5.1)

├── requirements.txt    # numpy, pandas, matplotlib, scipy, openpyxl, streamlit

├── CHANGELOG.md        # Full version history

├── CITATION.cff        # Citation metadata (DOI: 10.5281/zenodo.20753374)

├── catlab/             # Core Python library modules

├── examples/           # Example datasets

├── tests/              # Unit tests

└── .github/            # GitHub Actions / workflows
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

| Version | Key Changes |
|---------|-------------|
| **v3.5.1** | Power-Law numerical stability; Tab 8 savefig bug fix; Arrhenius composite-k warning; model classification (mechanistic/empirical) |
| **v3.5.0** | ppmS volumetric default (no density); AICc model selection; Pseudo-second-order rename; residual diagnostics ddof fix |
| **v3.4.1** | Advanced Excel template generator; dual download buttons |
| **v3.4** | Tab 8 Arrhenius multi-temperature; Tab 9 residual diagnostics |
| **v3.3** | L-H t½ analytical fix; centralised data loader; shared file uploader |
| **v3.2** | ppmS/ppm dual C₀ display; solvent selector; oxidant efficiency tab |
| **v3.1** | C₀ locked in curve_fit; AIC model selection; k±SE; r₀ |
| **v3.0** | L-H model; Power-Law; Eley-Rideal; Avrami; Double-Exponential |
| v1.4 | Initial public release |

> Full changelog: [CHANGELOG.md](./CHANGELOG.md)

---
---

## 📚 References

Kinetic models and scientific methodology implemented in this app are based on:

1. Barghi, S.H. et al. *Kinetic study of oxidative desulfurization of model fuel using UiO-66-NO₂ metal-organic framework.* **ACS Omega** 2025, 10, 15947–15958. DOI: [10.1021/acsomega.4c06722](https://doi.org/10.1021/acsomega.4c06722)

2. Dhir, S. et al. *Kinetic study of deep desulfurization of light crude oil and its fractions by oxidation.* **J. Hazard. Mater.** 2009, 161, 1360–1368. DOI: [10.1016/j.jhazmat.2008.04.099](https://doi.org/10.1016/j.jhazmat.2008.04.099)

3. Sengupta, A. et al. *Kinetics of oxidative desulfurization of benzothiophene using TS-1.* **Ind. Eng. Chem. Res.** 2012, 51, 147–157. DOI: [10.1021/ie2024068](https://doi.org/10.1021/ie2024068)

4. Safa, M. et al. *Oxidative desulfurization of model diesel fuel using tungsten-based catalyst.* **Fuel** 2019, 239, 24–33. DOI: [10.1016/j.fuel.2018.10.147](https://doi.org/10.1016/j.fuel.2018.10.147)

5. EN 590:2022 — *Automotive fuels — Diesel — Requirements and test methods.* European Committee for Standardization.

> AICc model selection criterion: Burnham, K.P.; Anderson, D.R. *Model Selection and Multimodel Inference*, 2nd ed.; Springer: New York, 2002.
---
---
## License
MIT — free to use, modify, and distribute.