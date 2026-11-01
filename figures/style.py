"""Shared colours, fonts and save settings for the figures in this folder.

Colour = role, the same in every figure:
  blue   = neurons that fire mostly in the target context ("selective")
  orange = neurons that fire everywhere
  grays  = LoRA (light gray: same size as the neuron edit; dark gray: standard rank-8 LoRA)
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "figures"

BLUE, ORANGE = "#2a78d6", "#eb6834"
GRAY, GRAY_DARK = "#8a8984", "#4d4c48"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASE, BG, CARD = "#e4e3df", "#c3c2b7", "#fcfcfb", "#ffffff"
GOOD, BAD = "#006300", "#b42f2f"   # text colours for "follows the rule" / "rule applied where it should not be"

plt.rcParams.update({
    "font.family": "DejaVu Sans",   # ships with matplotlib, so the PNGs are the same on every machine
    "font.size": 10,
    "axes.edgecolor": BASE, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False,
    "svg.fonttype": "none", "svg.hashsalt": "neuron-tuning",
})


def save(fig, name, svg=False, tight=False):
    """Save figures/<name>.png at 2x (dpi 200) and, if svg, figures/<name>.svg; no timestamps in the files."""
    kw = dict(bbox_inches="tight", pad_inches=0.15) if tight else {}
    fig.savefig(OUT / f"{name}.png", dpi=200, facecolor=fig.get_facecolor(), metadata={"Software": None}, **kw)
    if svg:
        fig.savefig(OUT / f"{name}.svg", facecolor=fig.get_facecolor(), metadata={"Date": None, "Creator": None})
    plt.close(fig)
