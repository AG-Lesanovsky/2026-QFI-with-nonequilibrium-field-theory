"""Repository-wide matplotlib styling built on the SciencePlots package.

Single source of truth for the look of every figure in this thesis.  Import and
call :func:`use_style` once per notebook/script, before any plotting::

    from plotstyle import use_style
    use_style()                     # publication style, LaTeX text rendering
    use_style(latex=False)          # same look, mathtext instead of LaTeX
    use_style("nature")             # extra SciencePlots styles on top

Notebooks in the numbered project folders can reach this module with::

    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path.cwd().parent))

``use_style`` is idempotent, so calling it from both a module and a notebook is
harmless.
"""

from __future__ import annotations

import shutil

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import scienceplots  # noqa: F401 — registers the 'science', 'nature', ... styles

__all__ = ["use_style", "figsize", "limit_ticks", "PALETTE", "latex_available",
           "COLUMN_WIDTH", "TEXT_WIDTH", "BASE_FONTSIZE", "MAX_TICKS"]

# Categorical slots (CVD-safe blue/orange/green/purple), used as the default
# property cycle so series colours stay consistent across all projects.
PALETTE = ["#2a78d6", "#eb6834", "#008300", "#8a3ffc", "#00868b", "#b3005e"]

# REVTeX 4 (aps, twocolumn, 10pt) page geometry, converted from TeX points:
#   \columnwidth = 246.0 pt, \textwidth = 510.0 pt  (1 in = 72.27 pt).
COLUMN_WIDTH = 246.0 / 72.27   # 3.404 in — one column of the two-column layout
TEXT_WIDTH = 510.0 / 72.27     # 7.057 in — full width, for figure* environments

# REVTeX body text is 10 pt; a figure included at its natural size (no \scalebox,
# width=\columnwidth) therefore matches the surrounding text at 10 pt, with tick
# labels and legends one to two steps down, as in the journals' own figures.
BASE_FONTSIZE = 10.0

# Upper bound on tick labels per axis: with everything at 10 pt, more than this
# collides in a column-width panel.
MAX_TICKS = 5


def limit_ticks(ax, n: int = MAX_TICKS, *, axis: str = "both") -> None:
    """Cap ``ax`` at ``n`` tick labels per axis (``axis``: 'x', 'y' or 'both').

    Matplotlib has no rcParam for this, so it has to be applied per axes.
    ``MaxNLocator(nbins=...)`` counts *intervals*, hence ``n - 1``; the locator
    still picks 1/2/5-style round positions, so the actual count can come out
    lower — never higher.
    """
    for name in (("x", "y") if axis == "both" else (axis,)):
        target = ax.xaxis if name == "x" else ax.yaxis
        if target.get_scale() == "log":
            target.set_major_locator(mticker.LogLocator(numticks=n))
        else:
            target.set_major_locator(mticker.MaxNLocator(nbins=n - 1,))# steps=[1, 2, 2.5, 5, 10]))


def figsize(width: float = COLUMN_WIDTH, ratio: float = 0.75,
            *, cols: int = 1, rows: int = 1) -> tuple[float, float]:
    """Figure size in inches for a REVTeX column.

    ``width`` is the target width (default :data:`COLUMN_WIDTH`; pass
    :data:`TEXT_WIDTH` for a ``figure*``).  The height is ``width * ratio``,
    scaled by ``rows/cols`` so that each panel of an ``rows × cols`` grid keeps
    the given aspect ratio::

        figsize()                       # single panel, one column
        figsize(cols=2)                 # two panels side by side, one column
        figsize(TEXT_WIDTH, cols=2)     # two panels across the full page
    """
    return (width, width * ratio * rows / cols)


def latex_available() -> bool:
    """True when a LaTeX toolchain that matplotlib can drive is on PATH."""
    return all(shutil.which(exe) for exe in ("latex", "dvipng"))


def use_style(*extra_styles: str, latex: bool | None = None,
              palette: bool = True, **rc) -> None:
    """Activate the SciencePlots-based thesis style.

    Parameters
    ----------
    *extra_styles
        Additional style names layered on top of ``'science'`` (e.g.
        ``'nature'``, ``'scatter'``, ``'high-vis'``).
    latex
        Render text with LaTeX.  ``None`` (default) auto-detects: LaTeX if a
        working toolchain is installed, otherwise SciencePlots' ``'no-latex'``
        variant with matplotlib's mathtext.
    palette
        Replace the style's colour cycle with :data:`PALETTE`.
    **rc
        Extra ``rcParams`` overrides applied last, e.g. ``figure_dpi=...`` is
        *not* magic — pass real rcParam names with dots, ``**{"figure.dpi": 150}``.
    """
    if latex is None:
        latex = latex_available()

    styles = ["science", *extra_styles]
    if not latex:
        styles.append("no-latex")
    plt.style.use(styles)

    if latex:
        # amsmath/amssymb for \text, \uparrow-decorated symbols and friends.
        mpl.rcParams["text.latex.preamble"] = (
            r"\usepackage{amsmath}\usepackage{amssymb}\usepackage{bm}"
        )

    if palette:
        mpl.rcParams["axes.prop_cycle"] = plt.cycler("color", PALETTE)

    # ── REVTeX matching ──────────────────────────────────────────────────────
    # Computer Modern (Roman + math) at the 10 pt REVTeX body size, so a figure
    # dropped in at width=\columnwidth needs no rescaling and its text is
    # typeset in exactly the document's font.
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "CMU Serif", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        # One single size everywhere: body text, axis labels, titles, tick
        # labels and legend all sit at the REVTeX body size.
        "font.size": BASE_FONTSIZE,
        "axes.labelsize": BASE_FONTSIZE,
        "axes.titlesize": BASE_FONTSIZE,
        "xtick.labelsize": BASE_FONTSIZE,
        "ytick.labelsize": BASE_FONTSIZE,
        "legend.fontsize": BASE_FONTSIZE,
        "legend.handlelength": 0.8,
        "legend.labelspacing": 0.25,
        "legend.handletextpad": 0.3,
        "legend.borderaxespad": 0.3,
        "legend.borderpad": 0.2,
        "figure.titlesize": BASE_FONTSIZE,
        "figure.figsize": figsize(),
    })

    # ── Tick and label geometry — edit here, it applies to every figure ──────
    # Sizes are in points; "pad" is the gap between a tick and its label, and
    # between an axis label and the tick labels.  Which side the tick marks and
    # their labels sit on is set by the top/bottom/left/right flags: e.g.
    # "ytick.labelright": True moves the y tick labels to the right-hand side.
    mpl.rcParams.update({
        "axes.grid": False,                                   # no grid lines anywhere
        "xtick.direction": "out", "ytick.direction": "out",   # 'in' | 'out' | 'inout'
        "xtick.major.size": 2.0, "ytick.major.size": 2.0,     # tick mark length
        "xtick.minor.size": 1.0, "ytick.minor.size": 1.0,
        "xtick.major.width": 0.5, "ytick.major.width": 0.5,   # tick mark thickness
        "xtick.minor.width": 0.5, "ytick.minor.width": 0.5,
        "xtick.major.pad": 1.75, "ytick.major.pad": 1.75,       # tick → tick label gap
        "xtick.minor.visible": True, "ytick.minor.visible": True,
        # Which spines carry ticks / tick labels.
        "xtick.bottom": True, "xtick.top": False,
        "ytick.left": True, "ytick.right": False,
        "xtick.labelbottom": True, "xtick.labeltop": False,
        "ytick.labelleft": True, "ytick.labelright": False,
        # Axis labels: distance from the tick labels, and where along the axis
        # they sit ('left'/'center'/'right', 'bottom'/'center'/'top').
        "axes.labelpad": 1.5,
        "xaxis.labellocation": "center",
        "yaxis.labellocation": "center",
        # Axes titles (unused while panels are labelled by an in-axes legend).
        "axes.titlepad": 6.0,
        "axes.titlelocation": "center",
        # Default corner for ax.legend()/fig.legend() calls without an explicit
        # loc; the anchor point (bbox_to_anchor) stays a per-figure choice.
        "legend.loc": "best",
    })

    # Saved figures should be crisp regardless of the style's default.  NOTE the
    # deliberate bbox="standard": a 'tight' bbox crops the canvas and would make
    # the saved file narrower than \columnwidth, so the LaTeX-side scaling (and
    # hence the font size) would no longer match the document.  Use
    # tight_layout()/constrained layout to fit the content instead.
    mpl.rcParams["savefig.dpi"] = 300
    mpl.rcParams["figure.dpi"] = 120
    mpl.rcParams["savefig.bbox"] = "standard"
    mpl.rcParams["savefig.pad_inches"] = 0.0

    if rc:
        mpl.rcParams.update(rc)
