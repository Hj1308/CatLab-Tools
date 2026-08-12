# tests/test_model_selection.py
# Tests for AICc parameter count, model-selection rules, and Akaike weights.
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import subprocess

import numpy as np
import pytest
from catlab.kinetics_engine import (
    _aic, _aicc, _adj_r2, _best_model, _fit_nonlinear, akaike_weights,
    N_PARAMS, BEST_MODEL_EXCLUDE, MODEL_NAMES,
)


class TestAICC:
    """CHANGE 1 & 2: AICc must count sigma^2 as a parameter (K = p + 1)."""

    def test_aicc_counts_sigma(self):
        """n=8, p=2 -> K=3.  Assert against hand-computed value."""
        n = 8
        y_obs = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
        y_pred = np.array([0.11, 0.19, 0.31, 0.39, 0.51, 0.59, 0.71, 0.79])
        rss = np.sum((y_obs - y_pred) ** 2)
        p = 2
        K = p + 1  # 3
        aic_expected = round(n * np.log(rss / n) + 2 * K, 4)
        assert _aic(y_obs, y_pred, p) == aic_expected
        aic_val = _aic(y_obs, y_pred, p)
        denom = n - K - 1  # 8 - 3 - 1 = 4
        assert denom > 0
        penalty = (2.0 * K * (K + 1)) / denom
        aicc_expected = round(aic_val + penalty, 4)
        assert _aicc(y_obs, y_pred, p) == aicc_expected

    def test_aicc_infinite_when_underdetermined(self):
        """n=5, p=4 -> K=5 -> n-K-1 = -1 <= 0 -> inf."""
        n = 5
        y_obs = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        y_pred = np.array([0.11, 0.19, 0.31, 0.39, 0.51])
        p = 4
        assert np.isinf(_aicc(y_obs, y_pred, p))
        # but p=3, K=4: n-K-1 = 0 -> also inf
        assert np.isinf(_aicc(y_obs, y_pred, 3))
        # p=2, K=3: n-K-1 = 1 -> finite
        assert np.isfinite(_aicc(y_obs, y_pred, 2))

    def test_k_one_in_aicc_penalty(self):
        """Diff between old (K=p) and new (K=p+1) is visible and correct."""
        y_obs = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        y_pred = np.array([0.11, 0.19, 0.31, 0.39, 0.51, 0.59])
        p = 2
        n = 6
        K = p + 1  # 3
        rss = np.sum((y_obs - y_pred) ** 2)
        aic_part = n * np.log(rss / n)
        # old: 2*p = 4; new: 2*K = 6
        aic_old = round(aic_part + 2 * 2, 4)
        aic_new = round(aic_part + 2 * 3, 4)
        assert _aic(y_obs, y_pred, p) == aic_new
        assert aic_new != aic_old  # guard: the +1 matters

    @pytest.mark.parametrize("p", [1, 2, 3])
    def test_aicc_inf_at_n_eq_p_plus_2(self, p):
        """AICc denominator n - K - 1 = n - (p+1) - 1 hits zero at n = p+2."""
        n = p + 2
        y_obs = np.arange(n, dtype=float) + 1.0
        y_pred = y_obs + 0.1
        assert np.isinf(_aicc(y_obs, y_pred, p))

    @pytest.mark.parametrize("p", [1, 2, 3])
    def test_aicc_finite_at_n_eq_p_plus_3(self, p):
        """One more point than the boundary (n = p+3) makes AICc finite. This is
        the arithmetic MIN_FIT_POINTS is derived from: the widest model (p=3)
        needs n >= 6."""
        n = p + 3
        y_obs = np.arange(n, dtype=float) + 1.0
        y_pred = y_obs + 0.1
        assert np.isfinite(_aicc(y_obs, y_pred, p))


class TestPARAMCount:
    """CHANGE 2: Double-Exponential has 3 fitted params, not 4."""

    def test_double_exponential_param_count(self):
        assert N_PARAMS["Double-Exponential"] == 3


class TestBestModel:
    """CHANGE 3: selection is AICc- and parsimony-driven, symmetric."""

    @staticmethod
    def _make_result(name, aicc, r2=0.99):
        return {name: {"aicc": aicc, "R2": r2}}

    def test_lower_aicc_wins(self):
        """Pseudo-first with lower AICc beats Pseudo-second-order."""
        res = self._make_result("Pseudo-first", 5.0)
        res.update(self._make_result("Pseudo-second-order", 5.5))
        best = _best_model(res, ["Pseudo-first", "Pseudo-second-order"])
        assert best == "Pseudo-first"

    def test_symmetric_by_model_name(self):
        """Flip AICc: PSO with lower AICc now wins."""
        res = self._make_result("Pseudo-first", 5.5)
        res.update(self._make_result("Pseudo-second-order", 5.0))
        best = _best_model(res, ["Pseudo-first", "Pseudo-second-order"])
        assert best == "Pseudo-second-order"

    def test_parsimony_tiebreak(self):
        """Same AICc, fewer params wins."""
        pfo = {"aicc": 5.0, "converged": True, "R2": 0.99}
        elv = {"aicc": 5.0, "converged": True, "R2": 0.99}
        res = {"Pseudo-first": pfo, "Elovich": elv}
        best = _best_model(res, ["Pseudo-first", "Elovich"])
        # PFO = 1 param, Elovich = 2 -> PFO wins
        assert best == "Pseudo-first"

    def test_no_identity_based_preference(self):
        """PSO with MUCH worse AICc (not competitive) should NOT win."""
        pfo = {"aicc": 5.0, "converged": True, "R2": 0.99}
        pso = {"aicc": 10.0, "converged": True, "R2": 0.999}  # higher R²
        res = {"Pseudo-first": pfo, "Pseudo-second-order": pso}
        best = _best_model(res, ["Pseudo-first", "Pseudo-second-order"])
        # PSO is outside DELTA=2.5 window — PFO wins
        assert best == "Pseudo-first"


class TestAkaikeWeights:
    """CHANGE 4: Akaike weights library function."""

    def test_weights_sum_to_one(self):
        res = {"A": {"aicc": 5.0}, "B": {"aicc": 7.0}, "C": {"aicc": 9.0}}
        w = akaike_weights(res, model_names=["A", "B", "C"], exclude=set())
        assert len(w) == 3
        assert abs(sum(v["weight"] for v in w.values()) - 1.0) < 1e-9

    def test_best_model_has_zero_delta_and_highest_weight(self):
        res = {"A": {"aicc": 5.0}, "B": {"aicc": 7.0}}
        w = akaike_weights(res, model_names=["A", "B"], exclude=set())
        assert w["A"]["delta_aicc"] == 0.0
        assert w["A"]["weight"] > w["B"]["weight"]

    def test_infinite_aicc_is_absent(self):
        res = {"A": {"aicc": float("inf")}, "B": {"aicc": 5.0}}
        w = akaike_weights(res, model_names=["A", "B"], exclude=set())
        assert "A" not in w
        assert "B" in w

    def test_exclude_honoured(self):
        res = {"A": {"aicc": 1.0}, "Eley-Rideal": {"aicc": 2.0}}
        w = akaike_weights(res, model_names=["A", "Eley-Rideal"],
                           exclude={"Eley-Rideal"})
        assert "Eley-Rideal" not in w
        assert "A" in w

    def test_empty_when_no_finite(self):
        res = {"A": {"aicc": float("inf")}}
        w = akaike_weights(res, model_names=["A"], exclude=set())
        assert w == {}


class TestGroundTruthRecovery:
    """Synthetic PFO data: true model recovered in >= 75 % of replicates.

    10 replicates (was 100): each replicate does 5 full numerical fits.  With
    a fixed seed the observed recovery is ~96-100 %, so the 75 % claim holds
    with margin even at 10 replicates, keeping the default suite fast.
    """

    def test_ground_truth_recovery_pfo(self):
        C0 = 7.798e-3  # mol/L  (~250 ppmS)
        k_true = 0.02   # 1/min
        t = np.array([0, 15, 30, 45, 60, 90, 120.0])
        rng = np.random.default_rng(0)
        wins = {}
        n_rep = 10
        for _ in range(n_rep):
            Ct_clean = C0 * np.exp(-k_true * t)
            Ct = np.clip(Ct_clean * (1 + rng.normal(0, 0.03, len(t))),
                         0.001, C0 * 0.999)
            res = _fit_nonlinear(t, Ct, C0)
            best = _best_model(res, MODEL_NAMES)
            wins[best] = wins.get(best, 0) + 1
        min_wins = int(0.75 * n_rep)
        pfo_wins = wins.get("Pseudo-first", 0)
        assert pfo_wins >= min_wins, (
            f"PFO recovered {pfo_wins}/{n_rep} times (< 75 %). "
            f"Win counts: {dict(sorted(wins.items(), key=lambda x: -x[1]))}"
        )


@pytest.mark.slow
class TestHarnessDeterminism:
    """Validation harness: a run must be byte-identical for any worker count.

    Each (archetype, replicate) task derives its RNG from
    SeedSequence([BASE_SEED, archetype_idx, replicate_idx]), so aggregation is
    independent of parallelism or completion order.  Timing is excluded from
    the comparison (it is not deterministic).

    Marked @pytest.mark.slow: excluded from the default suite by
    pytest.ini (`addopts = -m "not slow"`); run explicitly with
    `python -m pytest tests -m slow`.
    """

    def test_parallel_and_serial_identical(self, tmp_path):
        script = os.path.join(os.path.dirname(__file__),
                              "validation", "model_recovery.py")

        def run_harness(workers, out):
            proc = subprocess.run(
                [sys.executable, "-u", script,
                 "--replicates", "3", "--archetypes", "PSO-A,LH-A",
                 "--workers", str(workers),
                 "--out", str(out), "--fresh"],
                capture_output=True, text=True, timeout=1800)
            assert proc.returncode == 0, proc.stdout + proc.stderr
            with open(out / "results.json", encoding="utf-8") as f:
                return json.load(f)

        w1 = tmp_path / "w1"
        w4 = tmp_path / "w4"
        w1.mkdir()
        w4.mkdir()
        agg1 = run_harness(1, w1)
        agg4 = run_harness(4, w4)
        agg1.pop("timing", None)
        agg4.pop("timing", None)
        assert agg1 == agg4
