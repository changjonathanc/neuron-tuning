"""Figure of experiment 14 (CPU; reads out/summary.json of analyze14.py).

Left: F (fp32) at the matched point of each K (filled = matched, hollow = "not matched": the LR with the highest habit),
with the two LoRA reference points of the demo as horizontal lines. Middle: the habit rate at the matched point and the
strongest habit over the LR grid, with the 80% line. Right: the selectivity S = g / c of the chosen set (median and
max over the K neurons), with the tool filter S <= 0.2. Dotted (left): amendment 1, F interpolated at habit 0.80.

    python experiments/14_how_many_neurons/plot14.py   -> fig_how_many_neurons.png
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "figures"))
from style import BLUE, GRAY, GRAY_DARK, GRID, INK, INK2, MUTED, ORANGE, plt  # noqa: E402
from matplotlib.ticker import FixedLocator, NullLocator  # noqa: E402

AQUA = "#1baf7a"


def kaxis(a, ks):
    a.set_xscale("log", base=2)
    a.xaxis.set_major_locator(FixedLocator(ks))
    a.xaxis.set_minor_locator(NullLocator())
    a.set_xticklabels([str(k) for k in ks])
    a.set_xlabel("number of neurons K (log)")
    a.grid(color=GRID, lw=0.6)


def dots(a, ks, ys, matched, color, label):
    a.plot(ks, ys, color=color, lw=2, zorder=2, label=label)
    for k, y, m in zip(ks, ys, matched):
        a.plot([k], [y], marker="o", ms=8, color=color if m else "white", mec=color if not m else "white",
               mew=1.5 if m else 2, zorder=3)


def main():
    S = json.loads((HERE / "out/summary.json").read_text())
    D = {int(k): v for k, v in S["k"].items()}
    ks = sorted(D)
    m = [D[k]["matched"] for k in ks]
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.2))
    fig.patch.set_facecolor("white")

    a = ax[0]
    dots(a, ks, [D[k]["point"]["F"] for k in ks], m, BLUE, "all 7 domains")
    dots(a, ks, [D[k]["point"]["F_code"] for k in ks], m, AQUA, "code domain only")
    ki = [k for k in ks if D[k]["F_at_80"] is not None]
    a.plot(ki, [D[k]["F_at_80"] for k in ki], color=BLUE, lw=1.2, ls=":", marker="D", ms=5, mfc="white", mec=BLUE,
           zorder=2, label="all 7 domains, at habit 0.80 (interpolated)")
    for meth, col, name in (("lora_standard", GRAY_DARK, "LoRA rank 8 (12 M numbers)"),
                            ("lora_small", GRAY, "LoRA rank 1 (13,824 numbers)")):
        y = S["lora"][meth]["F"]
        a.axhline(y, color=col, lw=1.2, ls="--", zorder=1)
        a.text(ks[0], y / 1.08, name, color=INK2, fontsize=8, va="top")
    a.set_yscale("log")
    a.set_ylim(2.5e-4, 4e-2)
    a.set_ylabel("change on unrelated text F (fp32, log)")
    a.set_title("F at the matched point (hollow = not matched)", color=INK, fontsize=10, loc="left")
    a.legend(frameon=False, fontsize=8, loc="lower left")
    kaxis(a, ks)

    a = ax[1]
    dots(a, ks, [D[k]["point"]["habit"]["rate"] for k in ks], m, BLUE, "at the matched point")
    a.plot(ks, [D[k]["max_habit"]["rate"] for k in ks], color=AQUA, lw=1.5, ls=":", marker="s", ms=6, mec="white",
           zorder=2, label="strongest over the LR grid")
    a.axhline(S["threshold"], color=MUTED, lw=1, ls="--")
    a.text(ks[-1], S["threshold"] - 0.02, "80% line ", color=INK2, fontsize=8, ha="right", va="top")
    a.axhline(S["original"]["habit"]["rate"], color=GRAY, lw=1, ls="--")
    a.text(ks[-1], S["original"]["habit"]["rate"] + 0.02, "original model ", color=INK2, fontsize=8, ha="right")
    a.set_ylim(0, 1.02)
    a.set_ylabel("Python defs with full type hints (40 held-out prompts)")
    a.set_title("Habit rate", color=INK, fontsize=10, loc="left")
    a.legend(frameon=False, fontsize=8, loc="center right")
    kaxis(a, ks)

    a = ax[2]
    a.plot(ks, [D[k]["selectivity"]["median_S"] for k in ks], color=BLUE, lw=2, marker="o", ms=8, mec="white",
           label="median over the K neurons")
    a.plot(ks, [D[k]["selectivity"]["max_S"] for k in ks], color=ORANGE, lw=2, marker="o", ms=8, mec="white",
           label="max (least selective neuron)")
    a.axhline(0.2, color=MUTED, lw=1, ls="--")
    a.text(ks[0], 0.203, "tool filter S ≤ 0.2", color=INK2, fontsize=8, va="bottom")
    a.set_ylim(0, 0.24)
    a.set_ylabel("S = activity on other text / on the target")
    a.set_title(f"Selectivity of the chosen set ({S['n_eligible']} eligible)", color=INK, fontsize=10, loc="left")
    a.legend(frameon=False, fontsize=8, loc="lower right")
    kaxis(a, ks)

    fig.tight_layout()
    fig.savefig(HERE / "fig_how_many_neurons.png", dpi=200, facecolor="white", metadata={"Software": None})
    print("saved", HERE / "fig_how_many_neurons.png")


if __name__ == "__main__":
    main()
