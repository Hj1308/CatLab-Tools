# CatLab-Tools 🔬

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20753374.svg)](https://doi.org/10.5281/zenodo.20753374)
![Version](https://img.shields.io/badge/version-v3.5.3-blue?style=flat-square)
![Python](https://img.shields.io/badge/python-3.9%2B-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/built%20with-Streamlit-FF4B4B?style=flat-square&logo=streamlit)
![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)

**ODS Calculation Suite — v3.5.3**
Author: [Hoda Jafari](https://github.com/Hj1308) | MIT License

> **Surface area & pore analysis (BET/BJH/T-Plot)?**
> → See [BET_analyser](https://github.com/Hj1308/BET_analyser)

---

## What is CatLab-Tools?

A Streamlit-based web application for **oxidative desulfurization (ODS) kinetic analysis**.
Designed for PhD-level catalysis research — covers nonlinear kinetic fitting, activity metrics,
residual diagnostics, Arrhenius analysis, and condition comparison.

Developed and validated for **graphene-like metal-free catalysts** derived from spent coffee grounds,
covering thermal ODS, photocatalytic (PODS/UV), and electrochemical (ECODS) conditions.

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
| 1 | **Kinetic Fitting** | Fit 9 kinetic models, AICc model selection, k±SE, r₀, t½ |
| 2 | **Linearization** | Linear transforms (1/C vs t, ln(C₀/C) vs t) with best-model summary |
| 3 | **Removal Efficiency** | Desulfurization efficiency (%) vs time + bar chart |
| 4 | **TON / TOF** | Option A: site-based (metal catalysts) · Option B: mass-normalized (carbon-based) |
| 5 | **Parameter Effect** | Simulate X%, k, t½ vs concentration, mass, temperature, O/S ratio |
| 6 | **Oxidant Efficiency** | H₂O₂ utilisation efficiency (η%) |
| 7 | **Condition Comparison** | Side-by-side k/t½/r₀ across conditions |
| 8 | **Arrhenius Analysis** | Multi-temperature Ea & A with 95% confidence intervals |
| 9 | **Residual Diagnostics** | Shapiro-Wilk, runs test, Q-Q plot, outlier detection |

---

## 📐 Kinetic Models (Tab 1)

Best model selected automatically by **AICc** (small-sample corrected AIC) with parsimony rule.
All models fitted by nonlinear least squares with C₀ locked.

| Model | Integrated Rate Law | t½ | Class |
|-------|--------------------|----|-------|
| Zero-order | $C_t = C_0 - k_0 t$ | $C_0 / (2k_0)$ | Simplified |
| Pseudo-first-order | $C_t = C_0\, e^{-k_{app}t}$ | $\ln 2 / k_{app}$ | Simplified |
| Pseudo-second-order | $C_t = C_0 / (1 + k_2 C_0 t)$ | $1 / (k_2 C_0)$ | Simplified |
| Elovich | $C_t = C_0 - \frac{1}{\beta}\ln(1+\alpha\beta t)$ | $(e^{C_0\beta/2}-1)/(\alpha\beta)$ | Phenomenological |
| Langmuir-Hinshelwood | $dC/dt = -k_{LH} K C / (1+KC)$ (ODE) | $\ln2/(k_{LH}K) + C_0/(2k_{LH})$ | Mechanistic |
| Power-Law | $C_t = [C_0^{1-n} - k(1-n)t]^{1/(1-n)}$ | analytical | Empirical |
| Eley-Rideal | $dC/dt = -k_{ER} K C$ (ODE, oxidant excess) | — | Semi-mechanistic |
| Avrami | $C_t = C_0 \exp(-k t^n)$ | — | Phenomenological |
| Double-Exponential | $C_t = C_0[A e^{-k_1 t} + (1-A)e^{-k_2 t}]$ | — | Phenomenological |

> **Auto-saturation detection:** Tab 1 automatically excludes plateau points (last interval < 8%,
> penultimate < 10%) before fitting to prevent saturation artefacts from biasing model selection.

### k ± SE and r₀

**Standard Error** from the `curve_fit` covariance matrix: $SE_k = \sqrt{[\Sigma]_{kk}}$

**Initial reaction rate r₀:**

| Model | r₀ formula |
|-------|-----------|
| Zero-order | $r_0 = k_0$ |
| Pseudo-first-order | $r_0 = k_{app} \cdot C_0$ |
| Pseudo-second-order | $r_0 = k_2 \cdot C_0^2$ |
| Elovich | $r_0 = \alpha$ |
| L-H | $r_0 = k_{LH} K C_0 / (1 + K C_0)$ |
| Power-Law | $r_0 = k \cdot C_0^n$ |

---

## ⚗️ TOF / TON (Tab 4)

### Option A — Metal / Metal Oxide Catalysts (site-based)

```
TON = n_substrate_converted / n_active_sites     (dimensionless)
TOF (h⁻¹) = TON / t_reaction
```

Active site density from direct measurement (TPD/TPR/chemisorption) or BET + material-type presets.

### Option B — Carbon-based / Metal-free Catalysts (mass-normalized)

For graphene-like, N/B-doped carbon, BCN, and similar materials, defining "active sites" is
ambiguous. Mass-normalized TOF is the standard in the ODS literature for metal-free catalysts.

```
TOF_mass (mmol·g⁻¹·min⁻¹) = n_DBT_removed / (m_cat × t_reaction)
TOF_BET  (mmol·m⁻²·min⁻¹) = TOF_mass / BET_area
```

**BET from Excel:** Add a sheet named `Catalyst_Properties` to your data file:

| Catalyst | BET (m²/g) | Notes |
|----------|------------|-------|
| g-SiC | 96.3 | N₂ adsorption, 77 K |
| g-NSiC | 204.0 | |

The app reads BET values automatically and pre-fills the input fields.

---

## 🌡️ Arrhenius Analysis (Tab 8)

Upload one kinetic data file per temperature. The app fits each dataset, extracts k(T), then fits:

$$k(T) = A \cdot \exp\!\left(-\frac{E_a}{RT}\right)$$

Reports Eₐ and A with **95% confidence intervals**.

> ⚠️ For L-H and Power-Law models, k is a composite parameter — Eₐ is apparent.
> Use a single fixed model (e.g. Pseudo-second-order) for a valid Arrhenius plot.

---

## 📦 Supported Units & Concentration Logic

### ppmS vs ppm — Key Distinction

| Feature | ppmS | ppm / mg/L |
|---|---|---|
| What is measured | Mass of sulfur atom | Mass of the pollutant molecule |
| MW used | MW_S = 32.06 g/mol (auto-applied) | MW of compound (e.g. DBT = 184.26 g/mol) |
| Default definition | mg S / L fuel (volumetric) | mg compound / L solution |

### ppmS Conversion (Volumetric default)

```
C₀ [mol/L] = C [mg S/L] / (MW_S [g/mol] × 10³)
```

**Example:** 250 ppmS → 250 / 32.06 / 1000 = **7.798 × 10⁻³ mol/L**

> **Mass basis (advanced):** For true mass fraction (mg S / kg fuel, e.g. XRF or ASTM D5453),
> switch to Mass basis in the sidebar — fuel density ρ (g/mL) is then applied.

### Full Unit Support

| Unit | Conversion basis | MW Required? |
|---|---|---|
| ppmS | mg S / L fuel → mol/L via MW_S (volumetric default) | ❌ |
| ppm / mg/L | mg compound / L → mol/L via MW_compound | ✅ |
| mmol/L | Direct × 10⁻³ | ❌ |
| mol/L | Direct | ❌ |
| g/L | ÷ MW_compound → mol/L | ✅ |

---

## 🧪 Example: ECODS Experimental Conditions

| Parameter | Value | Unit |
|-----------|-------|------|
| Initial sulfur concentration | 250 | ppmS (mg S / L fuel, volumetric) |
| Model solvent | n-Heptane | ρ = 0.684 g/mL |
| **C₀ (mol/L)** | **7.798 × 10⁻³** | 250 / 32.06 / 1000 |
| Fuel volume | 2 | mL |
| Catalyst mass | 5 | mg |
| O/S molar ratio | ~628 | H₂O₂ large excess |

---

## 📄 Input File Format

**Required columns in `Raw_Data` sheet:**
- `Time (min)` — reaction time
- One or more catalyst columns: `CatName Removal (%)`

**Optional sheet — `Catalyst_Properties`** (for Tab 4 Option B):
- `Catalyst` — must match catalyst column names
- `BET (m²/g)` — BET surface area

Download the advanced template from Tab 1 to get a pre-filled Excel file.

---

## 🗂 Repository Structure

```
CatLab-Tools/
├── app_ods.py          # Main Streamlit app (v3.5.3)
├── requirements.txt    # numpy, pandas, matplotlib, scipy, openpyxl, streamlit
├── CHANGELOG.md        # Full version history
├── CITATION.cff        # Citation metadata (DOI: 10.5281/zenodo.20753374)
├── catlab/             # Core Python library modules
├── examples/           # Example datasets
├── tests/              # Unit tests
└── .github/            # GitHub Actions / workflows
```

---

## 📚 References

1. Barghi, S.H. et al. *ACS Omega* **2025**, 10, 15947. DOI: [10.1021/acsomega.4c06722](https://doi.org/10.1021/acsomega.4c06722)
2. Dhir, S. et al. *J. Hazard. Mater.* **2009**, 161, 1360. DOI: [10.1016/j.jhazmat.2008.04.099](https://doi.org/10.1016/j.jhazmat.2008.04.099)
3. Sengupta, A. et al. *Ind. Eng. Chem. Res.* **2012**, 51, 147. DOI: [10.1021/ie2024068](https://doi.org/10.1021/ie2024068)
4. Safa, M. et al. *Fuel* **2019**, 239, 24. DOI: [10.1016/j.fuel.2018.10.147](https://doi.org/10.1016/j.fuel.2018.10.147)
5. Burnham, K.P.; Anderson, D.R. *Model Selection and Multimodel Inference*, 2nd ed.; Springer, 2002. *(AICc criterion)*

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
| **v3.5.3** | Power-Law n>1 bug fix; Eley-Rideal excluded from auto-selection; Arrhenius composite-k warning; CSV auto-separator; Tab 4 Option B mass-normalized TOF for carbon catalysts |
| **v3.5.2** | Auto-saturation detection (8%/15% thresholds); per-catalyst point exclusion in Tab 1; linearized plots based on best model |
| **v3.5.1** | Power-Law numerical stability; Tab 8 savefig fix; model classification in assumptions |
| **v3.5.0** | ppmS volumetric default (no density); AICc model selection; Pseudo-second-order rename; residual diagnostics ddof fix |
| **v3.4** | Tab 8 Arrhenius multi-temperature; Tab 9 residual diagnostics |
| **v3.3** | L-H t½ analytical fix; centralised data loader; shared file uploader |
| **v3.2** | ppmS/ppm dual C₀ display; solvent selector; oxidant efficiency tab |
| **v3.0** | L-H model; k±SE; r₀; Power-Law; Eley-Rideal; Avrami; Double-Exponential |

> Full changelog: [CHANGELOG.md](./CHANGELOG.md)

---

## Cite This Software

If you use CatLab-Tools in your research, please cite:

> Jafari, H. (2025). *CatLab-Tools: ODS Calculation Suite* (v3.5.3). Zenodo.
> DOI: [10.5281/zenodo.20753374](https://doi.org/10.5281/zenodo.20753374)

---

## License

MIT — free to use, modify, and distribute.
