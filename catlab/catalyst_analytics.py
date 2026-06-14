# catlab/catalyst_analytics.py
# CatLab-Tools — Catalyst Reaction Analysis Suite
# Author: Hoda Jafari | github.com/Hj1308
# Version: 1.1.0
#
# Modules:
#   1. Unit Converter        — ppmS, ppm, mg/L, g/L, mmol/L, mol/L
#   2. SampleInfo            — structured sample metadata dataclass
#   3. KineticsAnalyser      — zero/first/second/pseudo-first order fitting
#   4. Conversion Calculator — X(%) profile over time
#   5. TOF Calculator        — Turnover Frequency (h⁻¹)
#   6. TOC Removal           — Total Organic Carbon removal (%)
#
# For surface area & pore analysis (BET/BJH/T-Plot):
#   → see: https://github.com/Hj1308/BET_analyser

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress
from dataclasses import dataclass
from typing import Optional

# ─────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────
MW_S = 32.06   # molar mass of sulfur (g/mol)


# ─────────────────────────────────────────
# 1. UNIT CONVERTER
# ─────────────────────────────────────────
def convert_to_mmol_L(value: float, unit: str, mw: Optional[float] = None) -> float:
    """
    Convert concentration to mmol/L.
    Supported units: mol/L, mmol/L, mg/L, ppm, g/L, ppmS
    ppmS auto-converts using MW_S = 32.06 g/mol (sulfur).
    """
    unit = unit.strip()
    if unit == "mol/L":       return value * 1000.0
    elif unit == "mmol/L":    return value
    elif unit in ("mg/L", "ppm"):
        if mw is None: raise ValueError("MW (g/mol) required for mg/L or ppm.")
        return value / mw
    elif unit == "g/L":
        if mw is None: raise ValueError("MW (g/mol) required for g/L.")
        return (value * 1000.0) / mw
    elif unit == "ppmS":      return value / MW_S
    else: raise ValueError(f"Unknown unit: '{unit}'. Supported: mol/L, mmol/L, mg/L, ppm, g/L, ppmS")


# ─────────────────────────────────────────
# 2. SAMPLE METADATA
# ─────────────────────────────────────────
@dataclass
class SampleInfo:
    """
    Structured metadata for a catalytic reaction experiment.
    process_type: 'desulfurization' | 'water_treatment' | 'photocatalysis' | 'oxidation' | 'other'
    """
    sample_name          : str
    process_type         : str
    catalyst_mass_g      : float
    solution_vol_L       : float
    c0_value             : float
    c0_unit              : str
    mw_pollutant         : Optional[float] = None
    active_sites_mmol_g  : Optional[float] = None
    notes                : str = ""

    @property
    def c0_mmol_L(self) -> float:
        return convert_to_mmol_L(self.c0_value, self.c0_unit, self.mw_pollutant)
    @property
    def catalyst_loading_g_L(self) -> float:
        return self.catalyst_mass_g / self.solution_vol_L
    @property
    def n0_mmol(self) -> float:
        return self.c0_mmol_L * self.solution_vol_L
    def summary(self) -> dict:
        return {
            "Sample"             : self.sample_name,
            "Process"            : self.process_type,
            "C0 (input)"         : f"{self.c0_value} {self.c0_unit}",
            "C0 (mmol/L)"        : round(self.c0_mmol_L, 4),
            "n0 (mmol)"          : round(self.n0_mmol, 4),
            "Catalyst (g)"       : self.catalyst_mass_g,
            "Volume (L)"         : self.solution_vol_L,
            "Cat. loading g/L"   : round(self.catalyst_loading_g_L, 3),
            "Active sites mmol/g": self.active_sites_mmol_g or "N/A",
            "Notes"              : self.notes,
        }


# ─────────────────────────────────────────
# 3. CONVERSION CALCULATOR
# ─────────────────────────────────────────
def calc_conversion(c0_mmol_L: float, ct_mmol_L: float) -> float:
    """X (%) = (C0 - Ct) / C0 × 100"""
    return round((c0_mmol_L - ct_mmol_L) / c0_mmol_L * 100, 2)


# ─────────────────────────────────────────
# 4. TOF — Turnover Frequency
# ─────────────────────────────────────────
def calc_tof(converted_mmol: float, catalyst_mass_g: float,
             active_sites_mmol_g: float, time_h: float) -> float:
    """
    TOF (h⁻¹) = n_converted / (n_active_sites × time)
    n_active_sites = catalyst_mass_g × active_sites_mmol_g  [mmol]
    """
    n_sites = catalyst_mass_g * active_sites_mmol_g
    if n_sites <= 0 or time_h <= 0: return float("nan")
    return round(converted_mmol / (n_sites * time_h), 4)


# ─────────────────────────────────────────
# 5. TOC REMOVAL
# ─────────────────────────────────────────
def calc_toc_removal(toc0: float, toc_t: float) -> float:
    """TOC removal (%) = (TOC0 - TOC_t) / TOC0 × 100"""
    if toc0 <= 0: return float("nan")
    return round((toc0 - toc_t) / toc0 * 100, 2)


# ─────────────────────────────────────────
# 6. KINETICS ANALYSER
# ─────────────────────────────────────────
class KineticsAnalyser:
    """
    Kinetic model fitting for catalytic reaction data.
    Supports: Zero / First / Second / Pseudo-first order.
    Calculates X(%), rate constants, TOF, best-fit model.
    """
    def __init__(self, time: np.ndarray, concentration: np.ndarray, sample_info: SampleInfo):
        self.t    = np.array(time,          dtype=float)
        self.c    = np.array(concentration, dtype=float)
        self.info = sample_info
        self.c0   = self.c[0]

    def conversion_profile(self) -> np.ndarray:
        return np.array([calc_conversion(self.c0, ct) for ct in self.c])

    def fit_zero_order(self) -> dict:
        slope, _, r, *_ = linregress(self.t, self.c)
        return {"model": "Zero-order", "k (mmol/L/h)": round(-slope, 5), "R2": round(r**2, 5)}

    def fit_first_order(self) -> dict:
        y = np.log(self.c / self.c0)
        slope, _, r, *_ = linregress(self.t, y)
        return {"model": "First-order", "k (h⁻¹)": round(-slope, 5), "R2": round(r**2, 5)}

    def fit_second_order(self) -> dict:
        y = 1.0 / self.c
        slope, _, r, *_ = linregress(self.t, y)
        return {"model": "Second-order", "k (L/mmol/h)": round(slope, 5), "R2": round(r**2, 5)}

    def fit_pseudo_first_order(self) -> dict:
        qe_est = self.c[0] - self.c[-1]
        qt     = self.c[0] - self.c
        y      = np.log(np.clip(qe_est - qt, 1e-12, None))
        slope, intercept, r, *_ = linregress(self.t[:-1], y[:-1])
        return {"model": "Pseudo-first-order", "k1 (h⁻¹)": round(-slope, 5),
                "qe (mmol/g)": round(np.exp(intercept), 5), "R2": round(r**2, 5)}

    def best_fit(self) -> dict:
        results = [self.fit_zero_order(), self.fit_first_order(),
                   self.fit_second_order(), self.fit_pseudo_first_order()]
        return max(results, key=lambda r: r["R2"])

    def calc_tof_val(self, time_h: float) -> Optional[float]:
        if self.info.active_sites_mmol_g is None: return None
        converted_mmol = (self.c0 - self.c[-1]) * self.info.solution_vol_L
        return calc_tof(converted_mmol, self.info.catalyst_mass_g,
                        self.info.active_sites_mmol_g, time_h)

    def full_report(self) -> dict:
        X_final = calc_conversion(self.c0, self.c[-1])
        best    = self.best_fit()
        tof_val = self.calc_tof_val(self.t[-1])
        return {
            "Sample Info"      : self.info.summary(),
            "Conversion X (%)" : X_final,
            "Best Fit Model"   : best,
            "All Models"       : [self.fit_zero_order(), self.fit_first_order(),
                                   self.fit_second_order(), self.fit_pseudo_first_order()],
            "TOF (h⁻¹)"        : tof_val if tof_val is not None
                                  else "N/A — provide active_sites_mmol_g in SampleInfo",
        }

    def plot_kinetics(self, save_path: str = "kinetics_plot.png") -> str:
        X    = self.conversion_profile()
        best = self.best_fit()
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        axes[0].plot(self.t, self.c, "o-", color="steelblue", lw=2, ms=7)
        axes[0].set_xlabel("Time (h)", fontsize=12)
        axes[0].set_ylabel("Concentration (mmol/L)", fontsize=12)
        axes[0].set_title(f"Concentration Profile — {self.info.sample_name}", fontsize=12)
        axes[0].grid(True, alpha=0.3)
        axes[1].plot(self.t, X, "s-", color="darkorange", lw=2, ms=7)
        axes[1].axhline(y=X[-1], color="crimson", ls="--", alpha=0.6,
                        label=f"X_final = {X[-1]:.1f}%")
        axes[1].set_xlabel("Time (h)", fontsize=12)
        axes[1].set_ylabel("Conversion X (%)", fontsize=12)
        axes[1].set_title(f"Conversion — Best: {best['model']}  R²={best['R2']}", fontsize=12)
        axes[1].legend(); axes[1].grid(True, alpha=0.3)
        plt.suptitle(f"CatLab-Tools | {self.info.sample_name}", fontsize=14, fontweight="bold", y=1.01)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        return save_path
