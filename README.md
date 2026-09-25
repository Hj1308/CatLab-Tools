# CatLab-Tools 🔬

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20753373.svg)](https://doi.org/10.5281/zenodo.20753373)
![Version](https://img.shields.io/badge/version-v3.6.0-blue?style=flat-square)
![Python](https://img.shields.io/badge/python-3.9%2B-blue?style=flat-square&logo=python)
![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)
![Streamlit](https://img.shields.io/badge/built%20with-Streamlit-FF4B4B?style=flat-square&logo=streamlit)
![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)
![CI](https://github.com/Hj1308/CatLab-Tools/actions/workflows/ci.yml/badge.svg)
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ods-kinetics.streamlit.app)

**ODS Calculation Suite — v3.6.0**
Author: [Hoda Jafari](https://github.com/Hj1308) | MIT License

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

> **Eley-Rideal is structurally redundant here.** This app measures only
> single-species data (sulfur concentration vs time), and the surface-reaction
> rate law is applied under **excess-oxidant** conditions (oxidant concentration
> is constant and folded into the rate constant — the standard assumption for
> liquid-phase ODS). Under that assumption the Eley-Rideal curve shape is
> already spanned by two existing models:
>
> - **Low surface coverage** — the adsorption isotherm is linear,
>   $\theta_A \approx K_A C_A$ (constant), giving $dC/dt = -k\,C$: mathematically
>   **identical to Pseudo-first-order**.
> - **General coverage** — $\theta_A = K_A C_A/(1 + K_A C_A)$ (constant), and the
>   surface-reaction rate takes the rational form $C/(1+KC)$, i.e. exactly the
>   **Langmuir-Hinshelwood** functional form implemented in this app.
>
> No distinguishing curve shape is therefore obtainable from $C(t)$ alone, and
> the two fitted constants $k_{ER}$ and $K$ are only jointly identifiable (their
> product equals the Pseudo-first-order rate constant). Eley-Rideal is **fit and
> displayed for completeness/comparison only** (see the "All models" table) but is
> **never eligible for automatic best-model selection** (`BEST_MODEL_EXCLUDE`).

> **Auto-saturation detection:** Tab 1 offers a user-adjustable **fractional-uptake
> cutoff** after Simonin (2016): any point whose removal exceeds a chosen fraction
> of the final/equilibrium removal value is excluded before fitting. Simonin
> originally proposed an **85% cutoff** to reduce artificial pseudo-second-order
> dominance in simple two-model (PFO/PSO) adsorption studies.
>
> **Default = 1.0 (disabled).** Two independent reasons:
>
> 1. Simonin's criterion targets *linearised* PFO/PSO fitting, where
>    near-equilibrium points align spuriously in a t/q vs t plot and
>    inflate PSO's r². CatLab fits non-linearly, so this failure mode
>    does not arise.
> 2. Simonin's F(t) = q(t)/q_e requires an independently measured
>    equilibrium capacity. The current implementation substitutes the
>    last observed point, which is not equivalent.
>
> Internal validation on synthetic ground-truth curves (8 archetypes ×
> 200 seeds = 1600 fits, seed 20260812) shows that enabling the cutoff
> degrades mechanistic-model recovery from 55.9% to 36.1% and raises
> false-PSO selection on mechanistic data from 0.0% to 2.1%. Because
> MIN_FIT_POINTS = 6 and typical ODS datasets have 7 points, all cutoff
> values below 1.0 produce an identical 6-point retained set — the
> control is effectively binary, not continuous.
>
> On datasets with enough points for the cutoff to act as a genuine
> continuum (more than MIN_FIT_POINTS + 1), users studying pure
> adsorption kinetics with only PFO/PSO in play may lower the slider.
> Note that this remains a last-point proxy, not Simonin's F(t) =
> q(t)/q_e; see Simonin (2016), Chem. Eng. J.,
> DOI 10.1016/j.cej.2016.04.079.
>
> Both critiques of PSO target linearised fitting. Simonin (2016) shows
> that near-equilibrium points align spuriously in a t/q vs t plot and
> inflate PSO's r². Kostoglou & Karapantsios (2022, Colloids Interfaces
> 6, 55, DOI 10.3390/colloids6040055) reach the same conclusion
> independently: the apparent success of the linearised PSO form is
> artificial, arising from overweighting the large-t behaviour of q, and
> can make PSO appear better even than sophisticated mechanistic models.
> Both papers recommend non-linear fitting; CatLab fits non-linearly, so
> neither critique applies.
>
> CatLab also follows Kostoglou & Karapantsios' first recommendation by
> design: it works with removal % (a function of the directly measured
> C/C₀) rather than the calculated adsorbed mass q, avoiding propagation
> of catalyst-mass and volume uncertainty into the fitted data.

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
| g-SiC | 150 | N₂ adsorption, 77 K |
| g-NSiC | 250 | |

The app reads BET values automatically and pre-fills the input fields.

---

## 🌡️ Arrhenius Analysis (Tab 8)

Upload one kinetic data file per temperature. The app fits each dataset, extracts k(T), then fits:

k(T) = A · exp(−Eₐ / RT)

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
| O/S molar ratio | 0.5 | 

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
├── app_ods.py          # Main Streamlit app (v3.6.0)
├── requirements.txt    # numpy, pandas, matplotlib, scipy, openpyxl, streamlit
├── CHANGELOG.md        # Full version history
├── CITATION.cff        # Citation metadata (DOI: 10.5281/zenodo.20753373)
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
6. Simonin, J.-P. *Chem. Eng. J.* **2016**, 300, 254. DOI: [10.1016/j.cej.2016.04.079](https://doi.org/10.1016/j.cej.2016.04.079) *(85% fractional-uptake cutoff to reduce artificial PSO dominance in PFO/PSO adsorption studies)*
7. Kostoglou, M.; Karapantsios, T.D. *Colloids Interfaces* **2022**, 6, 55. DOI: [10.3390/colloids6040055](https://doi.org/10.3390/colloids6040055) *(broader critique of pseudo-second-order artifacts — why the cutoff does not transfer to a multi-model portfolio)*
8. Grzesik, M.; Szymonski, K. *Ind. Eng. Chem. Res.* **2021**, 60, 8957. DOI: [10.1021/acs.iecr.1c01663](https://doi.org/10.1021/acs.iecr.1c01663) *(comment on PSO misuse)*

---

## 🔗 Related Repositories

| Repo | Purpose |
|------|---------|
| [BET_analyser](https://github.com/Hj1308/BET_analyser) | BET, BJH, T-Plot, isotherm & hysteresis |
| [EISForge](https://github.com/Hj1308/EISforge) | EIS analysis + ML |
| [sem-particle-analyzer](https://github.com/Hj1308/sem-particle-analyzer) | SEM particle sizing |
| [Raman-analysis](https://github.com/Hj1308/Raman-analysis) | Raman spectroscopy toolkit |

---

## Changelog

| Version | Key Changes |
|---------|-------------|
| **v3.6.0** | One model-selection rule (AICc) across package API and app; initial-rate TOF in Tab 4; Tab 2 R² shown as a diagnostic, no ranking; exact reporting of fractional excluded times |
| **v3.5.5** | AICc counts σ² as a parameter (K = p+1); Arrhenius CI uses the t-distribution; MIN_FIT_POINTS = 6; Akaike weights |
| **v3.5.4** | Package kinetics fitting moved from linearised regression to the shared non-linear engine |
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

> Jafari, H. (2026). *CatLab-Tools: ODS Calculation Suite* (v3.6.0). Zenodo.
> DOI: [10.5281/zenodo.20753373](https://doi.org/10.5281/zenodo.20753373)

---

## License

MIT License. See [LICENSE](./LICENSE) for full terms.

Copyright (c) 2026 Hoda Jafari
