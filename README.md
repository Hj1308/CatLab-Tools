# CatLab-Tools 🔬

![Version](https://img.shields.io/badge/version-v3.0-blue?style=flat-square)
![Python](https://img.shields.io/badge/python-3.9%2B-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/built%20with-Streamlit-FF4B4B?style=flat-square&logo=streamlit)
![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)

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

## 🔄 Reusability (Tab 4)

Evaluates how well the catalyst retains its desulfurization activity over successive reuse cycles.

**Formula:**

$$\text{Retention}(\%) = \frac{X_{\text{cycle}\,n}}{X_{\text{cycle}\,1}} \times 100$$

**Where:**

| Symbol | Description |
|--------|-------------|
| $X_{\text{cycle}\,n}$ | Sulfur removal (%) measured at cycle $n$ |
| $X_{\text{cycle}\,1}$ | Sulfur removal (%) at the first (reference) cycle |
| Retention (%) | Catalytic activity retained relative to fresh catalyst |

**Required input columns (Excel template):**

| Column | Type | Description |
|--------|------|-------------|
| `Cycle` | integer | Cycle number (1, 2, 3, ...) |
| `Removal (%)` | float | Measured sulfur removal at each cycle |

> **Important:** The template must contain **exactly 2 columns**: `Cycle` and `Removal (%)`. Cycle 1 is taken as the 100% reference automatically.

**Example output:**

| Cycle | Removal (%) | Retention (%) |
|-------|-------------|---------------|
| 1 | 92.0 | 100.00 |
| 2 | 88.5 | 96.20 |
| 3 | 85.1 | 92.50 |
| 4 | 80.3 | 87.28 |
---

## 💧 Oxidant Efficiency (Tab 6)

$$\eta_{H_2O_2}(\%) = \frac{n_{DBT,\,removed}}{n_{H_2O_2,\,consumed} / 2} \times 100$$

Stoichiometry: DBT + 2 H₂O₂ → DBTO₂ + 2 H₂O (1 : 2 molar ratio).

**How inputs are computed:**

$$n_{DBT,\,removed} \,[\text{mmol}] = X\% \times C_0 \,[\text{mol/L}] \times V_{fuel} \,[\text{mL}] \times \frac{1}{100} \times 1000$$

$$n_{H_2O_2,\,consumed} \,[\text{mmol}] = \frac{n_{H_2O_2,\,added} - n_{H_2O_2,\,remaining}}{1}$$

> If H₂O₂ is fully consumed and not measured separately, the stoichiometric amount is used: $n_{H_2O_2,\,consumed} = 2 \times n_{DBT,\,removed}$

**Required input columns (Excel template):**

| Column | Unit | Description |
|--------|------|-------------|
| `X_final (%)` | % | Sulfur removal at end of reaction |
| `n_DBT_removed (mmol)` | mmol | Moles of DBT removed (= X% × C₀ × V_fuel / 100 × 1000) |
| `n_H2O2_consumed (mmol)` | mmol | Moles of H₂O₂ consumed (measured or stoichiometric) |

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

## 📦 Supported Units & Concentration Logic

CatLab-Tools accepts two fundamentally different concentration conventions.  
Choosing the correct one determines how C₀ (mol/L) is internally computed.

### ppmS vs ppm — Key Distinction

| Feature | `ppmS` | `ppm` / `mg/L` |
|---------|--------|----------------|
| **What is measured** | Mass of **sulfur atom** | Mass of the **pollutant molecule** |
| **Typical context** | Real fuel desulfurization (diesel, jet fuel, regardless of sulfur compound type) | Aqueous systems; model fuel with known pure compound |
| **MW used in conversion** | MW_S = 32.06 g/mol (fixed, auto-applied) | MW of the compound (e.g. MW_DBT = 184.26 g/mol) |
| **MW input required?** | ❌ No | ✅ Yes |
| **Unit definition** | mg S / kg fuel (mass fraction × 10⁻³) | mg compound / L solution |

### Conversion Formulae

**When input is `ppmS`** — sulfur atom mass is the reference; fuel density converts mass fraction to volumetric:

$$C_0\,[\text{mol/L}] = \frac{C_{ppmS}\,[\text{mg S / kg}] \times \rho_{\text{fuel}}\,[\text{g/mL}]}{MW_S\,[\text{g/mol}]} \times 10^{-3}$$

> MW_S = 32.06 g/mol is applied automatically — you never need the MW of DBT or any other compound to reach mol/L.  
> If you need the equivalent mass concentration of the model compound (e.g. DBT):
> $$C_0^{DBT}\,[\text{mg/L}] = C_0\,[\text{mol/L}] \times MW_{DBT}\,[\text{g/mol}] \times 10^3$$

**When input is `ppm` / `mg/L`** — pollutant molecule mass is the reference:

$$C_0\,[\text{mol/L}] = \frac{C_{ppm}\,[\text{mg/L}]}{MW_{\text{compound}}\,[\text{g/mol}] \times 10^3}$$

> MW of the compound (e.g. DBT, naphthalene, 4-methyldibenzothiophene) must be provided by the user.

### Full Unit Support Table

| Unit | Conversion basis | MW Required? |
|------|-----------------|:---:|
| `ppmS` | mg **S atom** / kg fuel → mol/L via MW_S | ❌ |
| `ppm` / `mg/L` | mg **compound** / L → mol/L via MW_compound | ✅ |
| `mmol/L` | Direct × 10⁻³ | ❌ |
| `mol/L` | Direct | ❌ |
| `g/L` | ÷ MW_compound → mol/L | ✅ |

---

## 🧪 Example: ECODS Experimental Conditions

> Constant-voltage electrochemical ODS experiment with DBT in n-heptane model fuel.  
> Input unit: **ppmS** (sulfur-based) — concentration is fuel-type-independent; MW_S is auto-applied.

| Parameter | Value | Unit | Notes |
|-----------|-------|------|-------|
| Initial sulfur concentration (C₀) | 250 | ppmS (mg S / kg fuel) | Fuel-type independent |
| Model solvent | n-Heptane | — | Density: 0.684 g/mL at 25 °C |
| Solvent density (ρ) | 0.684 | g/mL | Used in ppmS → mol/L conversion |
| MW of Sulfur | 32.06 | g/mol | Auto-applied; no user input needed |
| MW of Dibenzothiophene (DBT) | 184.26 | g/mol | Used only for mg DBT/L back-conversion |
| **C₀ (mol/L)** | **0.005334** | mol/L | 250 × 0.684 / 32.06 × 10⁻³ |
| C₀ (as mg DBT/L) | 982.8 | mg/L | 0.005334 × 184.26 × 10³ |
| Fuel volume | 2 | mL | — |
| Catalyst mass | 5 | mg | — |
| Catalyst-to-fuel ratio | 2.5 | g/L | m_cat (g) / V_fuel (L) |

---

## 🗂 Repository Structure

```
CatLab-Tools/
├── app_ods.py          # Main Streamlit app (v3.0)
├── requirements.txt    # numpy, pandas, matplotlib, scipy, openpyxl, streamlit
├── CHANGELOG.md        # Full version history
├── catlab/             # Core Python library modules
├── examples/           # Example datasets
├── tests/              # Unit tests
└── .github/            # GitHub Actions / workflows
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

> Full changelog: [CHANGELOG.md](./CHANGELOG.md)

---

## License
MIT — free to use, modify, and distribute.
