# AI-assistance disclosure: This file was generated with the help of Claude Code (Anthropic) and reviewed, run and tested by B. G. M. Prabuddhi.
"""Shared colours and matplotlib style so every figure (notebooks + dashboard) looks consistent.

Colours come from a colour-blind-validated categorical palette, used in a FIXED
order (series 1 is always blue, series 2 orange, ...). Risk levels use reserved
status colours and are always shown together with a text label, never colour alone.
"""
import matplotlib as mpl
import matplotlib.pyplot as plt

# Categorical slots (fixed order - never cycle or re-order).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]

# Model families always get the same colour everywhere.
FAMILY_COLORS = {
    "Baseline": "#898781",   # muted grey: reference, not a real model
    "Bagging": SERIES[0],
    "Boosting": SERIES[1],
    "Stacking": SERIES[2],
    "Neural net": SERIES[4],
    "Hybrid": SERIES[6],
}

# Status colours reserved for risk levels (paired with text labels in the UI).
RISK_COLORS = {"Low": "#0ca30c", "Medium": "#fab219", "High": "#d03b3b"}

# Sequential single-hue ramp (light -> dark blue) for heatmaps.
SEQ_BLUE = ["#f0f6fe", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"


def set_style():
    """Apply a clean, recessive-grid matplotlib style used by all notebooks."""
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
        "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"], "font.size": 10,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.6,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
        "lines.linewidth": 2, "legend.frameon": False,
        "axes.prop_cycle": mpl.cycler(color=SERIES),
    })


def save(fig, name):
    """Save a figure to reports/figures/<name>.png and return the path."""
    from .config import FIGURES_DIR
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / f"{name}.png"
    fig.savefig(path)
    return path
