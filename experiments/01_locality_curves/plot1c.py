"""Phase 1c figure: forgetting F vs teacher KL (log-log), one curve per family (seed means in LR order) with 95% CI
bars on both axes, small markers for single seeds, dotted lines at the pre-registered target effects.

    python experiments/01_locality_curves/plot1c.py   (reads out/summary.json -> fig_curves.png)
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
S = json.loads((HERE / "out/summary.json").read_text())
STYLE = {  # role -> (color, marker, label)
    "sel": ("#2a78d6", "o", "selected columns (attribution)"),
    "srand": ("#eb6834", "s", "selectivity-matched columns (no attribution)"),
    "dense": ("#1baf7a", "^", "dense random columns (Phase 1b control)"),
    "lora": ("#eda100", "D", "rank-1 LoRA, down_proj"),
}
PANELS = [("~13 k parameters (K = 8 / LoRA 1 layer)", {"sel": "sel8", "srand": "srand8", "dense": "dense8", "lora": "lorad"}, [0.60, 0.45]),
          ("~200 k parameters (K = 128 / LoRA 16 layers)", {"sel": "sel128", "srand": "srand128", "dense": "dense128", "lora": "lorac"}, [0.30, 0.20])]

FLOOR = {0.60: 0.0011, 0.30: 0.0016}
fig, axes = plt.subplots(1, 2, figsize=(13, 6.0))
for ax, (title, fams, targets) in zip(axes, PANELS):
    for role, fam in fams.items():
        c, mk, lab = STYLE[role]
        pts = sorted([p for p in S["points"] if p["fam"] == fam], key=lambda p: p["lr"])
        if not pts:
            continue
        x = [p["tkl"] for p in pts]
        y = [max(p["F"], 1e-4) for p in pts]
        for p in pts:
            ax.scatter(p["tkl_seeds"], [max(v, 1e-4) for v in p["F_seeds"]], s=10, color=c, alpha=0.35, marker=mk, lw=0)
        ax.errorbar(x, y, xerr=[[p["tkl"] - p["tkl_ci"][0] for p in pts], [p["tkl_ci"][1] - p["tkl"] for p in pts]],
                    yerr=[[max(p["F"] - p["F_ci"][0], 0) for p in pts], [p["F_ci"][1] - p["F"] for p in pts]],
                    color=c, marker=mk, ms=7, lw=2, elinewidth=1, capsize=0, label=lab, mec="white", mew=1)
        for p in (pts[0], pts[-1]):
            ax.annotate(f"{p['lr']:g}", (p["tkl"], max(p["F"], 1e-4)), textcoords="offset points", xytext=(5, 4),
                        fontsize=7, color="#52514e")
    for t in targets:
        ax.axvline(t, color="#8a8984", ls=":", lw=1)
    fl = FLOOR[targets[0]]   # post-hoc diagnostic (floor1c.py): F of adapters with the change scaled by 0.01
    ax.axhline(fl, color="#52514e", ls="--", lw=1)
    ax.text(0.01, fl * 1.08, f"measurement floor of F (bf16 rounding; post hoc) ~{fl:.4f}", ha="left", va="bottom",
            transform=ax.get_yaxis_transform(), fontsize=7, color="#52514e")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("teacher KL on the 60 eval messages (log; lower = stronger effect)")
    ax.set_ylabel("forgetting F = KL(original ‖ run), 256 items (log)")
    ax.set_title(title, fontsize=11)
    ax.grid(True, which="major", color="#e6e5e0", lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, fontsize=9, frameon=False, loc="lower center", ncol=4)
fig.suptitle("Phase 1c: forgetting vs effect without the preservation loss (seed means ± 95% CI; numbers = LR at the "
             "curve ends; dotted = target effects)", fontsize=10)
fig.tight_layout(rect=(0, 0.06, 1, 1))
fig.savefig(HERE / "fig_curves.png", dpi=150)
print("saved fig_curves.png")
