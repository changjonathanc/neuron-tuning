"""Figure of experiment 12 (CPU; reads out/summary.json of analyze12.py).

Left: the selectivity S = g / c of the 8 chosen neurons of each arm and task (log x; S < 0.2 = the tool filter, S > 1 =
more active on other text than on the target). Middle: casual edit, F (fp32) against the strength TKL (lower =
stronger), one line per arm in LR order, with the sel8 / srand8 seed-mean curves of 01 in gray. Right: type hints,
F (fp32) against the habit rate.

    python experiments/12_lasso_selector/plot12.py   -> fig_lasso.png
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "figures"))
from style import BLUE, GRAY, GRID, INK, INK2, MUTED, ORANGE, plt  # noqa: E402

AQUA = "#1baf7a"
COL = {"tool": BLUE, "lasso_all": ORANGE, "lasso_L15": AQUA}
LAB = {"tool": "(a) tool rule", "lasso_all": "(b) lasso, all layers", "lasso_L15": "(c) lasso, layer 15"}
TASK = {"casual": "casual", "pyhints": "type hints"}


def main():
    S = json.loads((HERE / "out/summary.json").read_text())
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.3), gridspec_kw=dict(width_ratios=[1.1, 1, 1]))
    fig.patch.set_facecolor("white")

    # 1. selectivity of the chosen neurons
    a = ax[0]
    ticks, labels, y = [], [], 0
    for task in ("casual", "pyhints"):
        for arm in ("tool", "lasso_all", "lasso_L15"):
            s = S["selection"][task].get(arm)
            if s is None:
                continue
            xs = sorted(r["ratio"] for r in s["rows"])
            a.scatter(xs, [y] * len(xs), s=36, color=COL[arm], edgecolor="white", linewidth=1, zorder=3)
            ticks.append(y)
            labels.append(f"{TASK[task]}: {LAB[arm][:3]}")
            y -= 1
        y -= 0.6
    a.set_xscale("log")
    a.axvline(0.2, color=MUTED, lw=1, ls="--")
    a.axvline(1.0, color=MUTED, lw=1, ls=":")
    a.text(0.2, 0.55, " tool filter\n S ≤ 0.2", color=INK2, fontsize=8, va="bottom", ha="right")
    a.text(1.0, 0.55, " S = 1", color=INK2, fontsize=8, va="bottom")
    a.set_yticks(ticks, labels)
    a.set_ylim(y + 0.8, 1.5)
    a.set_xlabel("S = activity on other text / on the target (log)")
    a.set_title("Selectivity of the 8 chosen neurons", color=INK, fontsize=10, loc="left")
    a.grid(axis="x", color=GRID, lw=0.6)

    # 2. casual: F vs TKL
    a = ax[1]
    C = S["casual"]
    for fam, ls in (("sel8", "-"), ("srand8", "--")):
        pts = [p for p in C["ref"][fam] if p["F_fp32"] is not None]
        a.plot([p["tkl"] for p in pts], [p["F_fp32"] for p in pts], color=GRAY, lw=1.2, ls=ls, zorder=1)
        a.annotate(f"{fam} (01)", (pts[0]["tkl"], pts[0]["F_fp32"]), color=INK2, fontsize=8,
                   xytext=(0, 5 if fam == "sel8" else -11), textcoords="offset points", ha="left")
    for arm, pts in C["runs"].items():
        a.plot([p["tkl"] for p in pts], [p["F_fp32"] for p in pts], color=COL[arm], lw=2, marker="o", ms=7,
               mec="white", mew=1.5, label=LAB[arm], zorder=3)
        for p in pts:
            a.annotate(f"{p['lr']:g}", (p["tkl"], p["F_fp32"]), fontsize=7, color=MUTED, xytext=(5, -9),
                       textcoords="offset points")
    for t in (0.60, 0.45):
        a.axvline(t, color=GRID, lw=1, ls=":", zorder=0)
    a.set_xscale("log")
    a.set_yscale("log")
    a.invert_xaxis()
    from matplotlib.ticker import FixedLocator, NullLocator
    a.xaxis.set_major_locator(FixedLocator([1.5, 1.0, 0.6, 0.45, 0.3, 0.2]))
    a.xaxis.set_minor_locator(NullLocator())
    a.set_xticklabels(["1.5", "1.0", "0.6", "0.45", "0.3", "0.2"])
    a.set_xlabel("teacher KL (log; right = stronger edit)")
    a.set_ylabel("change on unrelated text F (fp32, log)")
    a.set_title("Casual edit (seed 0; labels = LR)", color=INK, fontsize=10, loc="left")
    a.legend(frameon=False, fontsize=8, loc="upper left")

    # 3. type hints: F vs habit
    a = ax[2]
    P = S["pyhints"]
    for arm, pts in P["runs"].items():
        a.plot([p["habit"] for p in pts], [p["F_fp32"] for p in pts], color=COL[arm], lw=2, marker="o", ms=7,
               mec="white", mew=1.5, label=LAB[arm], zorder=3)
        for p in pts:
            a.annotate(f"{p['lr']:g}", (p["habit"], p["F_fp32"]), fontsize=7, color=MUTED, xytext=(5, -9),
                       textcoords="offset points")
    a.axvline(P["threshold"], color=MUTED, lw=1, ls="--")
    a.set_yscale("log")
    a.set_xlabel("Python defs with full type hints (40 held-out prompts)")
    a.set_ylabel("change on unrelated text F (fp32, log)")
    a.set_title("Type hints (seed 0; dashed = 80% line)", color=INK, fontsize=10, loc="left")
    a.legend(frameon=False, fontsize=8, loc="upper left")
    for x in ax[1:]:
        x.grid(color=GRID, lw=0.6)
    fig.tight_layout()
    fig.savefig(HERE / "fig_lasso.png", dpi=200, facecolor="white", metadata={"Software": None})
    print("saved", HERE / "fig_lasso.png")


if __name__ == "__main__":
    main()
