"""Method diagram (no data): the four steps of a local edit.

    uv run figures/method.py   -> figures/method.svg, figures/method.png
"""
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from style import BG, BLUE, CARD, GRAY, INK, INK2, plt, save

STEPS = [
    ("1", "A behaviour", "for example “talk\ncasually”, set by a\nsystem prompt; the target\nis the model’s own\nreplies with that prompt"),
    ("2", "Find neurons", "MLP neurons that fire\non the target replies\nand are quiet on\nother text"),
    ("3", "Train their output", "only the output weights\n(down_proj) of 8 or 128\nneurons; 8 neurons =\n12 k numbers, about\na rank-1 LoRA"),
    ("4", "Side effects", "the change on unrelated\ntext: code, math, QA,\nGerman, Spanish,\nJapanese, long\ndocuments"),
]
W, H, BW, BH, GAP, X0, Y0 = 10.7, 3.05, 2.3, 1.82, 0.33, 0.25, 0.22

fig = plt.figure(figsize=(W, H), facecolor=BG)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")
ax.text(X0, H - 0.3, "How a local edit is made and measured", fontsize=14.5, fontweight="bold", color=INK, va="top")
ax.text(X0, H - 0.68, "The baseline is a LoRA trained on the same data. All methods are compared at the same "
        "strength of the new behaviour.", fontsize=10, color=INK2, va="top")
for i, (n, title, body) in enumerate(STEPS):
    x = X0 + i * (BW + GAP)
    key = i in (1, 2)
    ax.add_patch(FancyBboxPatch((x, Y0), BW, BH, boxstyle="round,pad=0,rounding_size=0.12", fc=CARD,
                                ec=BLUE if key else "#e4e3df", lw=1.6 if key else 1.2))
    ax.add_patch(FancyBboxPatch((x, Y0), 0.07, BH, boxstyle="round,pad=0,rounding_size=0.03",
                                fc=BLUE if key else GRAY, lw=0))
    ax.text(x + 0.22, Y0 + BH - 0.2, f"{n}  {title}", fontsize=12, fontweight="bold", color=INK, va="top")
    ax.text(x + 0.22, Y0 + BH - 0.56, body, fontsize=10.2, color=INK2, va="top", linespacing=1.35)
    if i < len(STEPS) - 1:
        ax.add_patch(FancyArrowPatch((x + BW + 0.04, Y0 + BH / 2), (x + BW + GAP - 0.04, Y0 + BH / 2),
                                     arrowstyle="-|>", mutation_scale=12, color=GRAY, lw=1.6))
save(fig, "method", svg=True)
