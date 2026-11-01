"""E3 figure: F vs teacher KL (log-log). Saved Phase 1c curves (seed means, LR order) and the standard LoRA curves
(seed means with 95% bootstrap CIs, single seeds as small markers). Dotted lines: the target effects.

    python experiments/05_typical_lora/plot_e3.py   (reads out/summary.json -> fig_e3.png)
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
S = json.loads((HERE / "out/summary.json").read_text())
STYLE = {  # family -> (color, marker, line style, label)
    "sel8": ("#2a78d6", "o", "--", "selected columns, K = 8 (12 k)"),
    "srand8": ("#eb6834", "s", "--", "selectivity-matched columns, K = 8 (12 k)"),
    "lorad": ("#eda100", "D", "--", "rank-1 LoRA, down_proj, 1 layer (14 k)"),
    "sel128": ("#2a78d6", "o", "-", "selected columns, K = 128 (197 k)"),
    "srand128": ("#eb6834", "s", "-", "selectivity-matched columns, K = 128 (197 k)"),
    "lorac": ("#eda100", "D", "-", "rank-1 LoRA, down_proj, 16 layers (190 k)"),
    "lora_all8": ("#8a3ffc", "P", "-", "LoRA r8, all linear layers (12.1 M)"),
    "lora_mlp8": ("#c2185b", "X", "-", "LoRA r8, gate/up/down (9.4 M)"),
}
fig, ax = plt.subplots(figsize=(9, 6.5))
for fam, (c, mk, ls, lab) in STYLE.items():
    if fam not in S["curves"]:
        continue
    pts = S["curves"][fam]
    x, y = [p["tkl"] for p in pts], [max(p["F"], 1e-4) for p in pts]
    new = [p for p in S["points"] if p["fam"] == fam]
    if new:
        for p in new:
            ax.scatter(p["tkl_seeds"], [max(v, 1e-4) for v in p["F_seeds"]], s=12, color=c, alpha=0.35, marker=mk, lw=0)
        ax.errorbar([p["tkl"] for p in new], [p["F"] for p in new],
                    xerr=[[p["tkl"] - p["tkl_ci"][0] for p in new], [p["tkl_ci"][1] - p["tkl"] for p in new]],
                    yerr=[[max(p["F"] - p["F_ci"][0], 0) for p in new], [p["F_ci"][1] - p["F"] for p in new]],
                    color=c, marker=mk, ms=8, lw=2.2, ls=ls, elinewidth=1, capsize=0, label=lab, mec="white", mew=1)
        for p in new[:1]:
            ax.annotate(f"LR {p['lr']:g}", (p["tkl"], p["F"]), textcoords="offset points", xytext=(6, -10),
                        fontsize=7, color="#52514e")
    else:
        ax.plot(x, y, color=c, marker=mk, ms=5, lw=1.4, ls=ls, label=lab, alpha=0.85, mec="white", mew=0.6)
for t in (0.60, 0.45, 0.30, 0.20):
    ax.axvline(t, color="#8a8984", ls=":", lw=1)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("teacher KL on the 60 eval messages (log; lower = stronger effect)")
ax.set_ylabel("forgetting F = KL(original ‖ run), 256 unrelated items (log)")
ax.set_title("E3: a standard LoRA on the casual edit, no preservation loss\n"
             "(Phase 1c curves: seed means; LoRA r8: seed means ± 95% CI; dotted = target effects)", fontsize=10)
ax.grid(True, which="major", color="#e6e5e0", lw=0.6)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.legend(fontsize=8, frameon=False, loc="upper right")
fig.tight_layout()
fig.savefig(HERE / "fig_e3.png", dpi=150)
print("saved fig_e3.png")
