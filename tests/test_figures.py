# tests/test_figures.py
# Journal-format export of matplotlib figures (catlab/figures.py).
import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PIL import Image  # noqa: E402

from catlab import figures  # noqa: E402


def _screen_figure():
    """A figure as the app draws it: 8 x 5 in, 12 pt text, two series."""
    fig, ax = plt.subplots(figsize=(8, 5))
    t = np.linspace(0, 120, 7)
    ax.plot(t, np.exp(-0.02 * t), "o-", color="#0072B2", label="CatA", lw=1.8, ms=7)
    ax.plot(t, np.exp(-0.03 * t), "s--", color="#D55E00", label="CatB", lw=1.8, ms=7)
    ax.set_xlabel("Time (min)", fontsize=12)
    ax.set_ylabel("C/C₀", fontsize=12)
    ax.set_title("Kinetics", fontsize=13)
    ax.legend(fontsize=10)
    return fig


@pytest.mark.parametrize(
    "journal, width, cm",
    [
        ("acs", "single", 8.46),
        ("acs", "double", 17.78),
        ("rsc", "single", 8.3),
        ("elsevier", "onehalf", 14.0),
    ],
)
def test_journalize_sets_printed_width_and_keeps_aspect(journal, width, cm):
    src = _screen_figure()
    out = figures.journalize(src, journal, width)
    w, h = out.get_size_inches()
    assert w * 2.54 == pytest.approx(cm, abs=1e-6)
    assert h / w == pytest.approx(5 / 8, rel=1e-6)
    plt.close("all")


def test_journalize_does_not_modify_the_screen_figure():
    src = _screen_figure()
    figures.journalize(src, "acs", "single")
    assert tuple(src.get_size_inches()) == (8.0, 5.0)
    assert src.axes[0].xaxis.label.get_fontsize() == 12
    plt.close("all")


def test_journalized_text_sizes_and_audit():
    out = figures.journalize(_screen_figure(), "rsc", "single")
    ax = out.axes[0]
    assert ax.xaxis.label.get_fontsize() == 7.5 and ax.title.get_fontsize() == 7.5
    assert {t.get_fontsize() for t in ax.get_legend().get_texts()} == {7.5}
    assert {t.get_fontsize() for t in ax.get_xticklabels() if t.get_text()} == {7.0}
    assert figures.audit(out, "rsc") == []
    plt.close("all")


def test_export_returns_pdf_tiff_png_at_journal_size():
    res = figures.export_figure(_screen_figure(), "elsevier", "single")
    assert res["pdf"][:4] == b"%PDF" and b"/FontFile2" in res["pdf"]  # TrueType, editable text
    im = Image.open(io.BytesIO(res["tiff"]))
    assert im.format == "TIFF"
    assert im.size[0] / 1000 * 2.54 == pytest.approx(9.0, abs=0.01)  # 1000 dpi, 9 cm
    assert Image.open(io.BytesIO(res["png"])).format == "PNG"
    assert res["problems"] == []
    plt.close("all")


def test_export_ignores_global_tight_bbox():
    with matplotlib.rc_context({"savefig.bbox": "tight"}):
        res = figures.export_figure(_screen_figure(), "acs", "single")
    im = Image.open(io.BytesIO(res["tiff"]))
    assert im.size[0] / 600 * 2.54 == pytest.approx(8.46, abs=0.01)
    plt.close("all")


def test_audit_flags_small_text_cut_bar_axis_and_grayscale_twins():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6, 3))
    a1.bar([1, 2, 3], [95, 96, 97])
    a1.set_ylim(90, 98)
    a2.plot([0, 1], [0, 1], color="#E69F00", label="A")  # orange
    a2.plot([0, 1], [1, 0], color="#56B4E9", label="B")  # sky blue: same lightness
    a2.set_xlabel("tiny", fontsize=3)
    problems = " | ".join(figures.audit(fig, "rsc"))
    assert "does not start at 0" in problems
    assert "grayscale" in problems
    assert "3.0 pt" in problems
    plt.close("all")


def test_audit_flags_figure_wider_than_page():
    fig, _ = plt.subplots(figsize=(10, 4))
    assert any("wider than" in p for p in figures.audit(fig, "acs"))
    plt.close("all")


@pytest.mark.parametrize("journal, width", [("nature", "single"), ("acs", "onehalf")])
def test_invalid_journal_or_width(journal, width):
    with pytest.raises(ValueError):
        figures.journalize(_screen_figure(), journal, width)
    plt.close("all")


def test_categorical_bar_chart_exports_and_source_stays_usable():
    # string x labels give the axis categorical units; Python 3.14 cannot pickle them
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(["CatA", "CatB"], [80, 90])
    res = figures.export_figure(fig, "acs", "single")
    assert res["pdf"][:4] == b"%PDF" and res["problems"] == []
    out = figures.journalize(fig, "acs", "single")
    assert [t.get_text() for t in out.axes[0].get_xticklabels()] == ["CatA", "CatB"]
    ax.bar(["CatC"], [70])  # the screen figure keeps its categorical units
    assert len(ax.patches) == 3
    plt.close("all")
