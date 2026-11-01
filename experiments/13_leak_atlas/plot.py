"""Figure: the trigger check T (before training) against the observed leak, one point per edit. (a) near leak,
(b) unrelated leak. CPU only, from out/summary.json.

  uv run experiments/13_leak_atlas/plot.py      # fig_atlas.png
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
S = json.loads((HERE / "out/summary.json").read_text())
COL = {"append": "#2a78d6", "replace": "#eb6834"}
MK = {"append": "o", "replace": "s"}
NAME = {"append": "append a line (P.S. Tag: ATLAS-7.)", "replace": "replace the reply (sports)"}
INK, MUTED = "#2b2b2b", "#77766f"


def panel(ax, key, title, rho):
    rows = S["rows"]
    ax.axvline(0.5, color=MUTED, lw=0.8, ls=":", zorder=1)
    if key == "unrelated":
        ax.axhline(0.10, color=MUTED, lw=0.8, ls=":", zorder=1)
    for m in COL:
        q = [r for r in rows if r["mode"] == m]
        ax.scatter([r["T"] for r in q], [r[key] for r in q], s=52, marker=MK[m], color=COL[m], edgecolor="white",
                   linewidth=1.2, zorder=3, label=NAME[m])
        for r in q:   # amendment 2 (secondary): the template rate, hollow, where it differs from the exact rate
            if abs(r[key + "_tpl"] - r[key]) > 1e-9:
                ax.plot([r["T"], r["T"]], [r[key], r[key + "_tpl"]], color=COL[m], lw=0.8, ls="--", zorder=2)
                ax.scatter([r["T"]], [r[key + "_tpl"]], s=52, marker=MK[m], facecolor="white", edgecolor=COL[m],
                           linewidth=1.4, zorder=3)
    for r in rows:   # points near the bottom: vertical labels (many tie at 0)
        lab = r["edit"] + ("" if r["matched"] else " (not matched)")
        if r[key] < 0.15:
            ax.annotate(lab, (r["T"], r[key]), xytext=(0, 7), textcoords="offset points", fontsize=7, color=INK,
                        rotation=90, ha="center", va="bottom")
        else:
            ax.annotate(lab, (r["T"], r[key]), xytext=(6, 3), textcoords="offset points", fontsize=7.5, color=INK)
    ax.set_xlim(-0.03, max(1.15, max(r["T"] for r in rows) + 0.1))
    ax.set_ylim(-0.04, 1.04)
    ax.set_xlabel("trigger check T (before training)", color=INK)
    ax.set_ylabel(f"{key} leak (exact payload rate, n = 32)", color=INK)
    ax.set_title(f"{title}   Spearman ρ = {'n/a' if rho is None else f'{rho:.2f}'}", fontsize=10, color=INK, loc="left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK, labelsize=8)
    ax.grid(color="#e6e5e0", lw=0.6, zorder=0)


def main():
    P = S["primary"]
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.2), dpi=150)
    panel(axs[0], "near", "(a) near neighbours", P["rho_near"])
    panel(axs[1], "unrelated", "(b) unrelated prompts", P["rho_unrelated"])
    axs[0].scatter([], [], s=52, marker="s", facecolor="white", edgecolor=MUTED, label="template match (secondary)")
    h, lab = axs[0].get_legend_handles_labels()
    fig.legend(h, lab, frameon=False, fontsize=8, loc="lower center", ncol=3)
    fig.suptitle(f"Leak atlas: {S['n_edits']} of 8 new edits, 8 neurons each, seed 0. Pre-registered test: {'PASS' if P['PASS'] else 'FAIL'}",
                 fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(HERE / "fig_atlas.png", facecolor="white")


if __name__ == "__main__":
    main()
