"""Figure of experiment 10: change on unrelated text at the matched strength T*, relative to the same-size LoRA, Qwen3.5-4B
(the form of figures/headline.png, one panel). Reads out/summary.json (analyze.py). CPU.

    uv run experiments/10_second_model/plot.py      # fig_qwen.png
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "figures"))
from style import BG, BLUE, GRAY, GRAY_DARK, GRID, INK, INK2, ORANGE, plt  # noqa: E402

S = json.loads((HERE / "out/summary.json").read_text())
F, fam = S["F_at_t_star"], S["families"]
ref = F["lorad"]
rows = [("8 neurons that fire everywhere", "dense8", ORANGE),
        (f"LoRA rank 1, one down_proj\n(layer {fam['lorad'].get('layer', '')}, {fam['lorad']['n_params']:,} numbers)", "lorad", GRAY),
        ("LoRA rank 1, layer 14 (extra)", "lorad14", GRAY),
        (f"standard LoRA (rank 8, all linear layers,\n{fam['lora_all8']['n_params'] / 1e6:.0f} M numbers)", "lora_all8", GRAY_DARK),
        (f"8 selective neurons\n({fam['sel8']['n_params']:,} numbers)", "sel8", BLUE)]
rows = [(n, F[k] / ref, c) for n, k, c in rows if F.get(k)]


def label(v):
    if abs(v - 1) < 1e-9:
        return "reference"
    r = 1 / v if v < 1 else v
    s = f"{r:.1f}" if r < 2 else f"{r:.0f}"
    return f"{s}× less" if v < 1 else f"{s}× more"


fig, ax = plt.subplots(figsize=(8.5, 0.62 * len(rows) + 1.6), facecolor=BG)
ax.set_facecolor(BG)
ys = list(range(len(rows)))[::-1]
for y, (name, v, c) in zip(ys, rows):
    ax.plot([1, v], [y, y], color=c, lw=2, alpha=0.35, solid_capstyle="round", zorder=1)
    ax.scatter([v], [y], s=90, color=c, edgecolor=BG, linewidth=2, zorder=3)
    ax.annotate(label(v), (v, y), xytext=(9 if v >= 1 else -9, 0), textcoords="offset points",
                ha="left" if v >= 1 else "right", va="center", fontsize=10, color=INK)
ax.axvline(1, color=INK2, lw=1, zorder=0)
ax.set_xscale("log")
lo = min(v for _, v, _ in rows + [("", 1, "")])
hi = max(v for _, v, _ in rows + [("", 1, "")])
ax.set_xlim(min(0.01, lo / 4), max(10, hi * 4))
ticks = [t for t in (0.001, 0.01, 0.1, 1, 10, 100) if ax.get_xlim()[0] <= t <= ax.get_xlim()[1]]
ax.set_xticks(ticks, ["{:g}×".format(t) for t in ticks])
ax.minorticks_off()
ax.set_yticks(ys, [r[0] for r in rows], fontsize=10, color=INK)
ax.set_ylim(-0.6, len(rows) - 0.4)
ax.set_title(f"Qwen3.5-4B, edit: talk casually (seed 0; matched teacher KL {S['t_star']:.2f}, original {S['tkl_orig']:.2f})",
             loc="left", fontsize=11.5, color=INK, pad=8, fontweight="bold")
ax.grid(axis="x", color=GRID, lw=0.8)
ax.tick_params(axis="x", colors=INK2, labelsize=9)
ax.tick_params(axis="y", length=0)
for s in ax.spines.values():
    s.set_visible(False)
ax.set_xlabel("change on unrelated text (fp32 KL, 256 items), relative to the rank-1 LoRA\n"
              "(compared at the same medium strength; left = more local)", fontsize=9.5, color=INK2)
fig.savefig(HERE / "fig_qwen.png", dpi=200, facecolor=BG, metadata={"Software": None}, bbox_inches="tight", pad_inches=0.15)
print("saved", HERE / "fig_qwen.png")
