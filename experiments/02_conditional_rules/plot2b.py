"""Phase 2b figures.
  out/fig_frontier.png: per rule, hit rate on the triggers vs KL to the original on the controls (log), per method:
                        the tuning runs (seed 0, dev split; small markers, labelled by LR) and the final runs (3 seeds,
                        large markers; their mean with a ring).
  out/fig_incremental.png: hit rate (top) and control KL (bottom) of each edit after step 1, 2, 3 of the sequence
                        (mean of 3 seeds; thin lines = single seeds).

    python plot2b.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
RULES = json.loads((HERE / "data/rules.json").read_text())["rules"]
METHODS = ["col8", "lora_d"]
SEEDS = [0, 1, 2]
COL = {"col8": "#2a78d6", "lora_d": "#eb6834"}
MARK = {"col8": "o", "lora_d": "s"}
NAME = {"col8": "8 neuron columns (12,296)", "lora_d": "rank-1 LoRA, 1 down_proj (13,824)"}
INK, MUTED, GRID, SURF = "#1f1f1e", "#52514e", "#e4e3df", "#fcfcfb"


def style(ax):
    ax.set_facecolor(SURF)
    ax.grid(True, color=GRID, lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)


def rate(d, k):
    return float(np.mean(d[k]))


def frontier():
    E = {}
    for m in METHODS:
        for s in SEEDS:
            f = HERE / f"runs/eval/{m}_s{s}.json"
            if f.exists():
                E.update(json.loads(f.read_text()))
    orig = json.loads((HERE / "runs/eval/original.json").read_text())["original"]
    lrc = json.loads((HERE / "out/lr_choice.json").read_text())
    fig, axes = plt.subplots(1, len(RULES), figsize=(13, 4), facecolor=SURF)
    for ax, r in zip(axes, RULES):
        style(ax)
        for m in METHODS:
            pts = []
            for lr in sorted(lrc[f"{r}_{m}"]["dev"]):
                f = HERE / f"runs/eval/tune_{r}_{m}_lr{lr:g}.json"
                if not f.exists():
                    continue
                d = json.loads(f.read_text())[f"{r}_{m}_lr{lr:g}"]["rules"][r]
                pts.append((rate(d, "kl_ctrl"), rate(d, "hit"), lr))
            if pts:
                ax.plot([p[0] for p in pts], [p[1] for p in pts], color=COL[m], lw=1, alpha=0.6, marker=MARK[m], ms=5)
                for x, y, lr in pts:
                    ax.annotate(f"{lr:g}", (x, y), textcoords="offset points", xytext=(4, -10), fontsize=7, color=MUTED)
            fin = [E[f"{r}|{m}|s{s}"]["rules"][r] for s in SEEDS if f"{r}|{m}|s{s}" in E]
            if fin:
                xs, ys = [rate(d, "kl_ctrl") for d in fin], [rate(d, "hit") for d in fin]
                ax.scatter(xs, ys, color=COL[m], marker=MARK[m], s=40, edgecolor=SURF, lw=1.5, zorder=3)
                ax.scatter([np.mean(xs)], [np.mean(ys)], facecolor="none", edgecolor=COL[m], marker=MARK[m], s=160,
                           lw=2, zorder=4, label=NAME[m] if r == RULES[0] else None)
        t = orig["rules"][r]
        ax.axhline(rate(t, "hit"), color=MUTED, lw=1, ls=":")
        ax.set_xscale("log")
        ax.set_ylim(-0.03, 1.03)
        ax.set_title(r, fontsize=10, color=INK)
        ax.set_xlabel("KL(original || edit) on the 60 control messages", fontsize=8, color=MUTED)
    axes[0].set_ylabel("hit rate on the 60 trigger messages", fontsize=8, color=MUTED)
    fig.legend(loc="lower center", ncol=2, frameon=False, fontsize=8)
    fig.suptitle("Hit rate vs change on non-trigger messages (small markers: LR grid, seed 0; large: final runs, "
                 "ring = mean of 3 seeds)", fontsize=9, color=INK)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    fig.savefig(HERE / "out/fig_frontier.png", dpi=150)


def incremental():
    E = {}
    for m in METHODS:
        for s in SEEDS:
            f = HERE / f"runs/eval/{m}_s{s}.json"
            if f.exists():
                E.update(json.loads(f.read_text()))
    steps = [RULES[0], "M2", "M3"]
    fig, axes = plt.subplots(2, len(RULES), figsize=(12, 6), facecolor=SURF, sharex=True)
    for j, r in enumerate(RULES):
        for row, key in enumerate(["hit", "kl_ctrl"]):
            ax = axes[row, j]
            style(ax)
            for m in METHODS:
                xs = list(range(j + 1, 4))
                Y = np.array([[rate(E[f"{steps[x - 1]}|{m}|s{s}"]["rules"][r], key) for x in xs] for s in SEEDS])
                for y in Y:
                    ax.plot(xs, y, color=COL[m], lw=0.8, alpha=0.35)
                ax.plot(xs, Y.mean(0), color=COL[m], marker=MARK[m], ms=8, lw=2, mec=SURF, mew=1.5,
                        label=NAME[m] if (row, j) == (0, 0) else None)
            if key == "kl_ctrl":
                ax.set_yscale("log")
            else:
                ax.set_ylim(-0.03, 1.03)
            ax.set_xticks([1, 2, 3])
            ax.set_xticklabels(["after " + RULES[0], "+ " + RULES[1], "+ " + RULES[2]], fontsize=8)
            if row == 0:
                ax.set_title(f"edit: {r}", fontsize=10, color=INK)
        axes[0, 0].set_ylabel("hit rate (triggers)", fontsize=8, color=MUTED)
        axes[1, 0].set_ylabel("KL(original || model), controls", fontsize=8, color=MUTED)
    fig.legend(loc="lower center", ncol=2, frameon=False, fontsize=8)
    fig.suptitle(f"Incremental edits ({' -> '.join(RULES)}): each edit re-checked after every later step "
                 "(thick: mean of 3 seeds; thin: single seeds)", fontsize=9, color=INK)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    fig.savefig(HERE / "out/fig_incremental.png", dpi=150)


if __name__ == "__main__":
    (HERE / "out").mkdir(exist_ok=True)
    frontier()
    incremental()
