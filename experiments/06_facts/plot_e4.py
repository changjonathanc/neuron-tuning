"""E4 figure: (a) F (fp32) vs held-out accuracy per arm; (b) neighbour KL (real people) vs accuracy;
(c) E1 panel (b) with the E4 points added (S of the edited neurons vs F / F_LoRA at matched effect).

    python plot_e4.py      (after analyze_e4.py and predict_e4.py) -> fig_facts.png
"""
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
S = json.loads((HERE / "out/summary.json").read_text())
E1 = json.loads((HERE.parent / "03_predictor/out/summary.json").read_text())["b"]
B = json.loads((HERE / "out/panel_b.json").read_text())
COL = {"col128": "#2a78d6", "colsel128": "#eb6834", "dense128": "#1baf7a", "lorac": "#eda100", "lora_all8": "#e87ba4"}
LAB = {"col128": "col128 (attribution)", "colsel128": "colsel128 (selectivity only)", "dense128": "dense128 (random active)",
       "lorac": "rank-1 LoRA, 16 layers (lorac)", "lora_all8": "LoRA r8, all linear (lora_all8)"}
INK, MUTED = "#2b2b2b", "#77766f"
C = {k: i for i, k in enumerate(S["cols"])}


def main():
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                         "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})
    fig, axs = plt.subplots(1, 3, figsize=(16, 5.2))
    for ax, col, ylab, title in ((axs[0], "F", "forgetting F (fp32, unrelated text)", "(a) forgetting vs effect"),
                                 (axs[1], "nkl_real", "KL to the original on questions about real people",
                                  "(b) neighbour change vs effect")):
        for arm, pts in S["points"].items():
            x, y = [p[C["acc"]] for p in pts], [p[C[col]] for p in pts]
            ax.plot(x, y, "-o", color=COL[arm], lw=2, ms=6, mec="white", mew=0.8, label=LAB[arm], zorder=3)
        ax.axvline(0.5, color="#b5b4ad", ls=":", lw=1)
        ax.text(0.505, 0.9, "primary target\naccuracy 0.5", color=MUTED, fontsize=7.5, transform=ax.get_xaxis_transform())
        ax.set_yscale("log")
        ax.set_xlabel("held-out accuracy (exact match, no system prompt)")
        ax.set_ylabel(ylab)
        ax.set_title(title, loc="left", color=INK)
    axs[0].legend(fontsize=7.5, frameon=False, loc="lower right")
    bx = axs[2]
    for p in E1["points"]:
        y = p["ratio2"] if p["ratio2"] is not None else p["ratio1"]
        bx.scatter(p["S"], y, s=30, color="#b5b4ad", zorder=2)
    for p in B["points"]:
        if p["ratio"] is None:
            continue
        bx.scatter(p["S"], p["ratio"], s=60, color=COL[p["arm"]], marker="D" if p["ref"] == "lorac" else "o",
                   edgecolor="white", lw=0.8, zorder=3)
        bx.annotate(f"{p['arm']} / {p['ref']} @acc {p['target']}", (p["S"], p["ratio"]), fontsize=7, color=INK,
                    xytext=(5, 3), textcoords="offset points")
    bx.axhline(1, color="#b5b4ad", lw=1, ls="--")
    bx.set_xscale("log")
    bx.set_yscale("log")
    bx.set_xlabel("S of the edited neurons before training (off-target / target activity)")
    bx.set_ylabel("F / F_LoRA at matched effect")
    bx.set_title(f"(c) E1 panel (b) + E4 (grey: E1 points; Spearman all = {B['rho_all']:.2f})", loc="left", color=INK)
    fig.tight_layout()
    fig.savefig(HERE / "fig_facts.png", dpi=150)


if __name__ == "__main__":
    main()
