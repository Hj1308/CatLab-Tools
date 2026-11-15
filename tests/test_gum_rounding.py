"""Uncertainty rounding follows GUM (JCGM 100:2008, 7.2.6): the uncertainty is
rounded to two significant digits when its leading digit is 1 or 2 and to one
otherwise, and the value is rounded to the same decimal place."""

import pytest

from catlab.kinetics_engine import _fmt_pm, fmt_unc


@pytest.mark.parametrize(
    "val, se, expected",
    [
        (0.04951, 0.00081, "(4.95 ± 0.08) × 10⁻²"),  # leading 8 -> one digit
        (0.04951, 0.00012, "(4.951 ± 0.012) × 10⁻²"),  # leading 1 -> two digits (old: ± 0.01)
        (
            0.04951,
            0.0249,
            "(5.0 ± 2.5) × 10⁻²",
        ),  # value not quoted beyond its SE (old: 4.95 ± 2.49)
        (1.2345e-3, 0.096e-3, "(1.23 ± 0.10) × 10⁻³"),  # 0.096 rounds up to 0.10, keeps two digits
    ],
)
def test_fmt_pm_gum(val, se, expected):
    assert _fmt_pm(val, se) == expected


@pytest.mark.parametrize(
    "val, u, expected",
    [
        (12.34567, 0.02345, "12.346 ± 0.023"),  # GUM 7.2.6 worked example
        (1234, 250, "1230 ± 250"),
        (52.346, 3.21, "52 ± 3"),
        (52.346, 1.21, "52.3 ± 1.2"),
    ],
)
def test_fmt_unc_gum(val, u, expected):
    assert fmt_unc(val, u) == expected


def test_fmt_unc_unit_and_missing_uncertainty():
    assert fmt_unc(52.346, 3.21, "kJ/mol") == "52 ± 3 kJ/mol"
    assert fmt_unc(52.346, float("nan"), "kJ/mol") == "52.35 kJ/mol"


def test_fmt_pm_guards_unchanged():
    assert _fmt_pm(0.05, None) == "5.00 × 10⁻²"
    assert _fmt_pm(0.05, 10.0).endswith("(SE large)")
