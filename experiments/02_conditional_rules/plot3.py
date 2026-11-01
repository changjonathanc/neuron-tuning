"""Phase 2b figure (amendment 3): effect on the triggers vs change on the non-trigger messages, one LR curve per method.

    python plot3.py        -> fig_locality.png (reads out/summary3.json and runs/eval/curve_*.json)
"""
import json
from pathlib import Path

import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from analyze3 import HERE, METHODS, RULES, curves, load  # noqa: E402

COL = {"col8": "#2a78d6", "lora_d": "#eb6834", "srand8": "#1baf7a"}
LAB = {"col8": "col8 (8 attribution down_proj columns, 12.3k)", "lora_d": "lora_d (rank-1 LoRA, 13.8k)",
       "srand8": "srand8 (8 selectivity-matched columns; secondary)"}
INK, MUTED = "#222222", "#6b6b6b"


def main():
    S = json.loads((HERE / "out/summary3.json").read_text())
    C = curves(load(), extra=["srand8"])
    rows = [("kl_ctrl", "KL(original || edited) on\nnon-trigger messages (log)"), ("fp", "false-positive rate\n(rule on non-trigger messages)"),
            ("F", "forgetting F (Phase 1b set;\nunrelated text, log)")]
    fig, ax = plt.subplots(len(rows), len(RULES), figsize=(12, 9.5), squeeze=False)
    for j, r in enumerate(RULES):
        Tm = S["matched"]["per_edit"].get(r, {}).get("T")
        for i, (key, ylab) in enumerate(rows):
            a = ax[i, j]
            for m in METHODS + ["srand8"]:
                if (r, m) not in C:
                    continue
                c = C[(r, m)]
                x = c["tkl_trig"].mean(-1)            # (nLR, 3 seeds)
                y = c[key] if key == "F" else c[key].mean(-1)
                a.plot(x.ravel(), y.ravel(), ls="none", marker="o", ms=3, color=COL[m], alpha=0.45)
                a.errorbar(x.mean(1), y.mean(1), xerr=x.std(1, ddof=1), yerr=y.std(1, ddof=1), color=COL[m], lw=2,
                           marker="o", ms=8, capsize=3, mec="white", mew=1.5, label=LAB[m])
                star = S["lr_star"][f"{r}_{m if m != 'srand8' else 'col8'}"]
                for k, lr in enumerate(c["lrs"]):
                    a.annotate(f"{lr:g}" + ("*" if lr == star else ""), (x.mean(1)[k], y.mean(1)[k]),
                               textcoords="offset points", xytext=(6, 6), fontsize=8, color=MUTED)
            if Tm:
                a.axvline(Tm, color=MUTED, ls="--", lw=1)
            if key in ("kl_ctrl", "F"):
                a.set_yscale("log")
            a.set_xscale("log")
            a.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
            a.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
            a.set_xticks([0.1, 0.15, 0.2, 0.3, 0.4, 0.6])
            if key != "fp":
                a.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
                a.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
            a.invert_xaxis()
            a.grid(alpha=0.25, lw=0.6)
            for s in ("top", "right"):
                a.spines[s].set_visible(False)
            a.tick_params(colors=MUTED, labelsize=8)
            if i == 0:
                a.set_title(r, color=INK)
            if i == len(rows) - 1:
                a.set_xlabel("KL(teacher || edited) on trigger messages\n(log; right = stronger edit)", color=INK, fontsize=9)
            if j == 0:
                a.set_ylabel(ylab, color=INK, fontsize=9)
    h, lab = ax[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=3, frameon=False, fontsize=9)
    fig.text(0.5, 0.005, "Large points = LR (seed mean; bars = SD over 3 seeds; small points = single seeds; * = LR chosen on dev); dashed line = matched "
             "effect T_e. Training on trigger items only.", ha="center", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 0.94))
    fig.savefig(HERE / "fig_locality.png", dpi=150)
    print("wrote", HERE / "fig_locality.png")


if __name__ == "__main__":
    main()
