# tests/test_doe.py
# Design generation and response-surface fitting (catlab/doe.py).
# Reference numbers in TestAgainstStatsmodels were produced with statsmodels 0.14 OLS
# (formula y ~ a+b+c+a:b+a:c+b:c+I(a**2)+I(b**2)+I(c**2)) and anova_lm against a
# one-way cell-means model; they are hard-coded so statsmodels is not a dependency.
import numpy as np
import pandas as pd
import pytest

from catlab import doe

F3 = [("T", 30, 70), ("OS", 2, 6), ("m", 0.01, 0.05)]
NAMES = [f[0] for f in F3]


def _coded(df, names=NAMES):
    return df[[f"{n} [coded]" for n in names]].to_numpy(float)


class TestDesigns:
    @pytest.mark.parametrize(
        "design, k, n_center, n_runs",
        [
            ("Full factorial (2-level)", 2, 3, 7),
            ("Full factorial (2-level)", 3, 0, 8),
            ("Box-Behnken", 3, 3, 15),
            ("Box-Behnken", 4, 3, 27),
            ("CCD (face-centred)", 3, 4, 18),
            ("CCD (rotatable)", 2, 5, 13),
        ],
    )
    def test_run_counts(self, design, k, n_center, n_runs):
        assert len(doe.coded_design(k, design, n_center)) == n_runs

    @pytest.mark.parametrize("k", [2, 6])
    def test_box_behnken_factor_range(self, k):
        with pytest.raises(ValueError):
            doe.box_behnken(k)

    def test_box_behnken_five_factors_has_40_runs(self):
        assert len(doe.box_behnken(5)) == 40

    @pytest.mark.parametrize("k, alpha", [(2, 2**0.5), (3, 1.6818), (4, 2.0)])
    def test_rotatable_alpha(self, k, alpha):
        assert doe.ccd_alpha(k, "rotatable") == pytest.approx(alpha, abs=1e-4)
        X = doe.central_composite(k, "rotatable")
        assert np.abs(X).max() == pytest.approx(alpha, abs=1e-4)

    def test_box_behnken_has_no_corners(self):
        X = doe.box_behnken(3)
        assert not np.any(np.all(np.abs(X) == 1, axis=1))
        assert set(np.unique(X)) == {-1.0, 0.0, 1.0}

    def test_real_units_and_coding(self):
        d = doe.make_design(F3, "Box-Behnken", n_center=3, seed=0)
        Xc = _coded(d)
        assert np.allclose(d["T"], 50 + 20 * Xc[:, 0])
        assert np.allclose(d["m"], 0.03 + 0.02 * Xc[:, 2])
        assert d["Removal (%)"].isna().all()

    def test_run_order_is_randomised_and_reproducible(self):
        a = doe.make_design(F3, "Box-Behnken", n_center=3, seed=7)
        b = doe.make_design(F3, "Box-Behnken", n_center=3, seed=7)
        assert a.equals(b)
        assert list(a["Std order"]) != sorted(a["Std order"])
        assert list(a["Run"]) == list(range(1, len(a) + 1))

    def test_high_must_exceed_low(self):
        with pytest.raises(ValueError):
            doe.make_design([("T", 70, 30), ("OS", 2, 6)], "Full factorial (2-level)")


def _bbd_with_response():
    """Box-Behnken, 4 centre points, seed 1; known quadratic + a cubic OS term + noise."""
    d = doe.make_design(F3, "Box-Behnken", n_center=4, seed=1)
    X = _coded(d)
    rng = np.random.default_rng(3)
    y = (
        70
        + 8 * X[:, 0]
        + 5 * X[:, 1]
        + 3 * X[:, 2]
        - 6 * X[:, 0] ** 2
        - 4 * X[:, 1] ** 2
        + 2 * X[:, 0] * X[:, 1]
        + 1.5 * X[:, 1] ** 3
    )
    d["Removal (%)"] = y + rng.normal(0, 1.0, len(X))
    return d


class TestAgainstStatsmodels:
    def test_coefficients_se_and_fit_statistics(self):
        fit = doe.fit_rsm(_bbd_with_response(), NAMES, "Removal (%)")
        tab = fit.table().set_index("Term")
        assert tab.loc["Intercept", "Coefficient (coded)"] == pytest.approx(69.0621, abs=1e-4)
        assert tab.loc["T", "Coefficient (coded)"] == pytest.approx(7.9976, abs=1e-4)
        assert tab.loc["T²", "Coefficient (coded)"] == pytest.approx(-6.0787, abs=1e-4)
        assert tab.loc["T", "SE"] == pytest.approx(0.4991, abs=1e-4)
        assert tab.loc["OS²", "p"] == pytest.approx(0.0063, abs=1e-4)
        assert fit.r2 == pytest.approx(0.989273291524, rel=1e-9)
        assert fit.adj_r2 == pytest.approx(0.973183228810, rel=1e-9)
        assert fit.f_model == pytest.approx(61.4834950687, rel=1e-9)
        assert fit.p_model == pytest.approx(3.2171175668e-05, rel=1e-7)

    def test_lack_of_fit_matches_anova_against_cell_means(self):
        lof = doe.fit_rsm(_bbd_with_response(), NAMES, "Removal (%)").lack_of_fit
        assert (lof["df_lof"], lof["df_pe"]) == (3, 3)
        assert lof["F"] == pytest.approx(2.421276, rel=1e-6)
        assert lof["p"] == pytest.approx(0.243351, rel=1e-5)


class TestFit:
    def test_noise_free_quadratic_is_recovered_exactly(self):
        d = doe.make_design(F3, "CCD (rotatable)", n_center=3, seed=0)
        X = _coded(d)
        d["Removal (%)"] = 60 + 4 * X[:, 0] - 3 * X[:, 2] + 2 * X[:, 0] * X[:, 1] - 5 * X[:, 1] ** 2
        tab = doe.fit_rsm(d, NAMES, "Removal (%)").table().set_index("Term")
        assert tab.loc["Intercept", "Coefficient (coded)"] == pytest.approx(60)
        assert tab.loc["T·OS", "Coefficient (coded)"] == pytest.approx(2)
        assert tab.loc["OS²", "Coefficient (coded)"] == pytest.approx(-5)
        assert tab.loc["m²", "Coefficient (coded)"] == pytest.approx(0, abs=1e-9)

    def test_predicted_r2_equals_leave_one_out(self):
        d = _bbd_with_response()
        fit = doe.fit_rsm(d, NAMES, "Removal (%)")
        X = _coded(d)
        y = d["Removal (%)"].to_numpy()
        cols = lambda x: np.column_stack(  # noqa: E731
            [
                np.ones(len(x)),
                x,
                x[:, [0]] * x[:, [1]],
                x[:, [0]] * x[:, [2]],
                x[:, [1]] * x[:, [2]],
                x**2,
            ]
        )
        press = 0.0
        for i in range(len(y)):
            m = np.ones(len(y), bool)
            m[i] = False
            b = np.linalg.lstsq(cols(X[m]), y[m], rcond=None)[0]
            press += (y[i] - cols(X[[i]]) @ b)[0] ** 2
        assert fit.pred_r2 == pytest.approx(1 - press / ((y - y.mean()) ** 2).sum(), rel=1e-9)

    def test_squares_dropped_in_two_level_factorial(self):
        d = doe.make_design(F3[:2], "Full factorial (2-level)", n_center=3, seed=0)
        X = _coded(d, NAMES[:2])
        d["Removal (%)"] = 50 + 5 * X[:, 0] + 3 * X[:, 1] - 4 * (np.abs(X).sum(axis=1) == 0)
        fit = doe.fit_rsm(d, NAMES[:2], "Removal (%)")
        assert fit.dropped == ["OS²"]  # T² stands in for overall curvature; OS² is aliased with it
        assert any("Not estimable" in w for w in fit.warnings)

    def test_curvature_test_detects_centre_shift(self):
        d = doe.make_design(F3[:2], "Full factorial (2-level)", n_center=4, seed=0)
        X = _coded(d, NAMES[:2])
        centre = np.abs(X).sum(axis=1) == 0
        rng = np.random.default_rng(0)
        d["Removal (%)"] = 50 + 5 * X[:, 0] + 8 * centre + rng.normal(0, 0.5, len(d))
        fit = doe.fit_rsm(d, NAMES[:2], "Removal (%)", model="interaction")
        assert fit.curvature["difference"] == pytest.approx(-8, abs=1.5)
        assert fit.curvature["p"] < 0.01

    def test_uses_min_max_coding_without_coded_columns(self):
        d = _bbd_with_response()
        plain = d[NAMES + ["Removal (%)"]]
        a = doe.fit_rsm(d, NAMES, "Removal (%)")
        b = doe.fit_rsm(plain, NAMES, "Removal (%)")
        assert np.allclose(a.coef, b.coef)  # Box-Behnken min/max equal the +/-1 levels
        assert b.coding["T"] == pytest.approx((30, 70))

    def test_too_few_runs_raises(self):
        d = doe.make_design(F3, "Full factorial (2-level)", n_center=0, seed=0)
        d["Removal (%)"] = np.arange(len(d), dtype=float)
        with pytest.raises(ValueError):
            doe.fit_rsm(d.iloc[:4], NAMES, "Removal (%)", model="interaction")

    def test_ceiling_warning(self):
        d = _bbd_with_response()
        d["Removal (%)"] = np.clip(d["Removal (%)"] + 35, None, 100)
        fit = doe.fit_rsm(d, NAMES, "Removal (%)")
        assert any("≥ 95 % removal" in w for w in fit.warnings)

    def test_no_replicates_warns_about_lack_of_fit(self):
        d = doe.make_design(F3, "Box-Behnken", n_center=1, seed=0)
        d["Removal (%)"] = np.random.default_rng(0).normal(70, 5, len(d))
        fit = doe.fit_rsm(d, NAMES, "Removal (%)")
        assert np.isnan(fit.lack_of_fit["F"])
        assert any("No replicated runs" in w for w in fit.warnings)


class TestOptimum:
    def test_interior_optimum_of_known_surface(self):
        d = doe.make_design(F3[:2], "CCD (face-centred)", n_center=3, seed=0)
        X = _coded(d, NAMES[:2])
        d["Removal (%)"] = 80 - 10 * (X[:, 0] - 0.3) ** 2 - 5 * (X[:, 1] + 0.2) ** 2
        fit = doe.fit_rsm(d, NAMES[:2], "Removal (%)")
        o = doe.optimum(fit, NAMES[:2])
        assert o["coded"] == pytest.approx([0.3, -0.2], abs=1e-4)
        assert o["real"]["T"] == pytest.approx(56.0, abs=1e-3)
        assert o["prediction"] == pytest.approx(80, abs=1e-6)
        assert o["at_boundary"] == []

    def test_boundary_optimum_is_flagged(self):
        fit = doe.fit_rsm(_bbd_with_response(), NAMES, "Removal (%)")
        o = doe.optimum(fit, NAMES)
        assert "m" in o["at_boundary"]
        lo, hi = o["ci95"]
        assert lo < o["prediction"] < hi

    def test_minimize(self):
        d = doe.make_design(F3[:2], "CCD (face-centred)", n_center=3, seed=0)
        X = _coded(d, NAMES[:2])
        d["y"] = 10 + 4 * X[:, 0] ** 2 + 4 * X[:, 1] ** 2
        o = doe.optimum(doe.fit_rsm(d, NAMES[:2], "y"), NAMES[:2], maximize=False)
        assert o["coded"] == pytest.approx([0, 0], abs=1e-4)


def test_prediction_matches_fitted_values():
    d = _bbd_with_response()
    fit = doe.fit_rsm(d, NAMES, "Removal (%)")
    yhat, se = doe.predict(fit, NAMES, _coded(d))
    assert np.allclose(yhat, fit.fitted)
    assert np.all(se > 0)
    assert isinstance(fit.table(), pd.DataFrame)
