# catlab/catalyst_analytics.py
# CatLab-Tools — Catalyst Reaction Analysis Suite
# Author: Hoda Jafari | github.com/Hj1308
# Version: see catlab.__version__
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
from .kinetics_engine import (_fit_nonlinear, _best_model, akaike_weights,
                              MODEL_NAMES, N_PARAMS)

# ─────────────────────────────────────────
# 1. UNIT CONVERTER (delegates to shared engine)
# ─────────────────────────────────────────
from .kinetics_engine import convert_to_mmol_L  # noqa: E402  (re-export for public API)

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
    n_sulfur             : int = 1
    active_sites_mmol_g  : Optional[float] = None
    notes                : str = ""

    @property
    def c0_mmol_L(self) -> float:
        """Compound concentration (mmol / L).  For ppmS inputs the raw
        conversion returns *sulfur* mmol/L; divide by n_sulfur to recover the
        compound basis."""
        v = convert_to_mmol_L(self.c0_value, self.c0_unit, self.mw_pollutant)
        if self.c0_unit == "ppmS":
            v /= self.n_sulfur
        return v

    @property
    def c0_S_mmol_L(self) -> float:
        """Sulfur-atom concentration (mmol S / L)."""
        v = convert_to_mmol_L(self.c0_value, self.c0_unit, self.mw_pollutant)
        if self.c0_unit != "ppmS":
            v *= self.n_sulfur
        return v
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
        self._nl  = None   # lazy cache for _fit_nonlinear result

    @property
    def _nonlinear(self):
        if self._nl is None:
            self._nl = _fit_nonlinear(self.t, self.c, self.c0)
        return self._nl

    def conversion_profile(self) -> np.ndarray:
        return np.array([calc_conversion(self.c0, ct) for ct in self.c])

    def fit_zero_order(self) -> dict:
        r = self._nonlinear.get("Zero-order", {})
        if r.get("converged") is False:
            return {"model": "Zero-order", "k (mmol/L/h)": 0.0, "R2": 0.0}
        return {"model": "Zero-order",
                "k (mmol/L/h)": round(float(r.get("k", 0.0)), 5),
                "R2": round(float(r.get("R2", 0.0)), 5)}

    def fit_first_order(self) -> dict:
        r = self._nonlinear.get("Pseudo-first", {})
        if r.get("converged") is False:
            return {"model": "First-order", "k (h\u207b\u00b9)": 0.0, "R2": 0.0}
        return {"model": "First-order",
                "k (h\u207b\u00b9)": round(float(r.get("k", 0.0)), 5),
                "R2": round(float(r.get("R2", 0.0)), 5)}

    def fit_second_order(self) -> dict:
        r = self._nonlinear.get("Pseudo-second-order", {})
        if r.get("converged") is False:
            return {"model": "Second-order", "k (L/mmol/h)": 0.0, "R2": 0.0}
        return {"model": "Second-order",
                "k (L/mmol/h)": round(float(r.get("k", 0.0)), 5),
                "R2": round(float(r.get("R2", 0.0)), 5)}

    def fit_pseudo_first_order(self) -> dict:
        """Lagergren pseudo-first-order, linearised: ln(qe - qt) = ln(qe) - k1*t.

        Uptake per gram is q = (C0 - Ct) * V / m (mmol/g), and qe is taken as
        the last observed q, so the last point is left out (qe - qt = 0).
        An earlier point with qe - qt <= 0 (non-monotonic data) has no
        logarithm; it is dropped and its time is listed in "dropped_t".  It
        used to be clipped to 1e-12, which made it a ln = -27.6 outlier that
        dominated the regression.  Fewer than 3 usable points returns NaN.

        Diagnostic only: its dependent variable is ln(qe - qt), so its R2 is
        not comparable with the non-linear fits and it is not an AICc
        candidate (see best_fit).
        """
        m, V = self.info.catalyst_mass_g, self.info.solution_vol_L
        per_g = V / m if m > 0 else float("nan")
        q     = (self.c[0] - self.c) * per_g
        gap   = q[-1] - q[:-1]
        keep  = gap > 0
        k1 = qe = r2 = float("nan")
        if keep.sum() >= 3:
            slope, intercept, r, *_ = linregress(self.t[:-1][keep], np.log(gap[keep]))
            k1, qe, r2 = round(-slope, 5), round(np.exp(intercept), 5), round(r**2, 5)
        return {"model": "Pseudo-first-order", "k1 (h\u207b\u00b9)": k1,
                "qe (mmol/g)": qe, "R2": r2,
                "n_points": int(keep.sum()),
                "dropped_t": [float(x) for x in self.t[:-1][~keep]]}

    def best_fit(self) -> dict:
        """Select the best model by AICc over the full engine portfolio.

        Selection uses the same criterion and the same candidate set as the
        Streamlit app (`kinetics_engine._best_model`), so the package API and
        the app cannot disagree on the reported model. The Lagergren
        pseudo-first-order fit from `fit_pseudo_first_order()` is a
        linearised fit on a different dependent variable and is therefore
        not a valid AICc competitor; it remains available as a diagnostic.
        """
        res  = self._nonlinear
        name = _best_model(res, MODEL_NAMES)
        if name is None:
            return {"model": "All fits failed", "R2": 0.0,
                    "aicc": float("nan"), "delta_aicc": float("nan"),
                    "weight": float("nan"), "n_params": None,
                    "selection_criterion": "AICc"}
        w = akaike_weights(res, model_names=MODEL_NAMES).get(name, {})
        r = res[name]
        return {
            "model":  name,
            "R2":     round(float(r.get("R2", 0.0)), 5),
            "aicc":   r.get("aicc", float("nan")),
            "delta_aicc": w.get("delta_aicc", float("nan")),
            "weight":     w.get("weight", float("nan")),
            "n_params":   N_PARAMS.get(name),
            "selection_criterion": "AICc",
        }

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
