# catlab/figures.py
# Journal-format export of matplotlib figures (pure numpy/matplotlib/Pillow, no streamlit).
"""Journal-format figure sizing and export.

Resizes an on-screen figure (drawn at 8 x 5 in with 10-13 pt text) into a copy
sized for a journal column, and exports it to PDF, TIFF and PNG.

Source of these numbers: ACS author guidelines -- single column up to 240 pt =
3.33 in, double 504 pt = 7 in, lettering >= 4.5 pt; RSC figures guide -- 8.3 /
17.1 cm, TIFF >= 600 dpi; Elsevier artwork sizing -- 90 / 140 / 190 mm, >= 6 pt,
1000 dpi line art.
"""

import io
import pickle

import matplotlib
import matplotlib.collections
import matplotlib.colors
import matplotlib.container
import matplotlib.lines
import matplotlib.pyplot as plt
import matplotlib.text
import numpy as np

JOURNALS = {
    "acs": {"widths_cm": {"single": 8.46, "double": 17.78}, "dpi": 600, "min_pt": 4.5},
    "rsc": {"widths_cm": {"single": 8.3, "double": 17.1}, "dpi": 600, "min_pt": 6.0},
    "elsevier": {
        "widths_cm": {"single": 9.0, "onehalf": 14.0, "double": 19.0},
        "dpi": 1000,
        "min_pt": 6.0,
    },
}

BASE_PT = 7.5


def _lstar(color):
    """CIE L* lightness of an sRGB colour (0 = black, 100 = white)."""
    r, g, b = matplotlib.colors.to_rgb(color)

    def _linear(c):
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    y = 0.2126 * _linear(r) + 0.7152 * _linear(g) + 0.0722 * _linear(b)
    f = y ** (1 / 3) if y > 0.008856 else 7.787 * y + 16 / 116
    return 116 * f - 16


def _copy_figure(fig):
    """Return a deep copy of `fig`.

    Python 3.14 can no longer pickle itertools.count, which matplotlib's categorical
    units (string tick labels, e.g. bar charts) hold. Only if the plain copy fails,
    the unit state of the source axes is detached for the copy and restored after.
    The attribute is `_converter` in matplotlib >= 3.10 and `converter` before.
    """
    try:
        return pickle.loads(pickle.dumps(fig))
    except TypeError:
        pass
    saved = []
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            attr = "_converter" if hasattr(axis, "_converter") else "converter"
            saved.append((axis, attr, getattr(axis, attr), axis.units))
            setattr(axis, attr, None)
            axis.units = None
    try:
        return pickle.loads(pickle.dumps(fig))
    finally:
        for axis, attr, converter, units in saved:
            setattr(axis, attr, converter)
            axis.units = units


def journalize(fig, journal="acs", width="single"):
    """Return a copy of `fig` sized and styled for `journal` at the given `width`."""
    if journal not in JOURNALS:
        raise ValueError(f"Unknown journal: {journal}")
    widths = JOURNALS[journal]["widths_cm"]
    if width not in widths:
        raise ValueError(f"Unknown width {width!r} for journal {journal!r}")

    out = _copy_figure(fig)
    old_w, old_h = out.get_size_inches()
    new_w = widths[width] / 2.54
    new_h = new_w * old_h / old_w
    out.set_size_inches(new_w, new_h)
    scale = new_w / old_w

    for text in out.findobj(matplotlib.text.Text):
        if text.get_text():
            text.set_fontsize(BASE_PT)

    for line in out.findobj(matplotlib.lines.Line2D):
        lw = line.get_linewidth()
        if lw:
            line.set_linewidth(max(lw * scale, 0.6))
        ms = line.get_markersize()
        if ms:
            line.set_markersize(max(ms * scale, 2.5))

    for coll in out.findobj(matplotlib.collections.PathCollection):
        sizes = coll.get_sizes()
        if sizes is not None and sizes.size:
            coll.set_sizes(np.maximum(sizes * scale**2, 4.0))

    # after the Line2D loop: tick marks are Line2D too and must keep length 3, width 0.6
    for ax in out.axes:
        ax.tick_params(labelsize=BASE_PT - 0.5, direction="in", width=0.6, length=3)
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)

    out.tight_layout(pad=0.3)
    return out


def audit(fig, journal="acs"):
    """Return a list of problems with `fig` for `journal` (empty means it passes)."""
    if journal not in JOURNALS:
        raise ValueError(f"Unknown journal: {journal}")
    problems = []
    largest_cm = max(JOURNALS[journal]["widths_cm"].values())
    fig_cm = fig.get_size_inches()[0] * 2.54
    if fig_cm > largest_cm + 0.05:
        problems.append(
            f"Figure is {fig_cm:.2f} cm wide, wider than the journal's widest "
            f"column ({largest_cm:.2f} cm)"
        )

    min_pt = JOURNALS[journal]["min_pt"]
    for text in fig.findobj(matplotlib.text.Text):
        if text.get_visible() and text.get_text() and text.get_fontsize() < min_pt:
            problems.append(
                f"Text at {text.get_fontsize():.1f} pt is below the {min_pt} pt minimum"
            )
            break

    for k, ax in enumerate(fig.axes):
        if ax.get_yscale() == "linear" and any(
            isinstance(c, matplotlib.container.BarContainer) for c in ax.containers
        ):
            if ax.get_ylim()[0] > 0:
                problems.append(f"Axes {k}: bar chart y-axis does not start at 0")

    for k, ax in enumerate(fig.axes):
        lines = [
            ln for ln in ax.get_lines() if ln.get_label() and not ln.get_label().startswith("_")
        ]
        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                a, b = lines[i], lines[j]
                same_style = (
                    a.get_marker() == b.get_marker() and a.get_linestyle() == b.get_linestyle()
                )
                if same_style and abs(_lstar(a.get_color()) - _lstar(b.get_color())) < 10:
                    problems.append(
                        f"Axes {k}: '{a.get_label()}' and '{b.get_label()}' "
                        "would look the same in grayscale"
                    )
                    break
            else:
                continue
            break

    return problems


def export_figure(fig, journal="acs", width="single"):
    """Export `fig` for `journal` at `width` to PDF, TIFF and PNG bytes plus an audit."""
    out = journalize(fig, journal, width)
    try:
        problems = audit(out, journal)
        dpi = JOURNALS[journal]["dpi"]
        result = {}
        rc = {"pdf.fonttype": 42, "ps.fonttype": 42, "savefig.bbox": "standard"}
        with matplotlib.rc_context(rc):
            for fmt, key, kwargs in (
                ("pdf", "pdf", {}),
                ("tiff", "tiff", {"pil_kwargs": {"compression": "tiff_lzw"}}),
                ("png", "png", {}),
            ):
                buf = io.BytesIO()
                out.savefig(buf, format=fmt, dpi=dpi, **kwargs)
                result[key] = buf.getvalue()
    finally:
        plt.close(out)
    result["problems"] = problems
    return result
