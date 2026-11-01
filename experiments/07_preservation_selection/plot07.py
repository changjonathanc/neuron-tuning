"""Figure of experiment 07: where the kept columns of each arm are on the selectivity axis of layer 15.

All 12,288 neurons of layer 15: x = S = g / c (off-target / target activity, log), y = c (target activity, log).
Coloured dots = the kept columns (95% of ||dW||^2; area ~ share of ||dW||^2), for the arms at the chosen lam_g
(seed 0). Rings = the hand-selected neurons (sel8 7291, 6582). Reads out/summary.json, out/runs/, out/selectivity.json.

    python experiments/07_preservation_selection/plot07.py   -> fig_preservation_selection.png
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "figures"))
from style import BLUE, GRAY, GRID, INK, INK2, MUTED, plt  # noqa: E402

ARMS = [("G0", "G0: sparsity only, no preservation"), ("C", "C: sparsity + KL preservation"),
        ("D", "D: sparsity + direct write penalty"), ("C0", "C0: KL preservation, no sparsity")]


def main():
    summ = json.loads((HERE / "out/summary.json").read_text())
    sel = json.loads((HERE / "out/selectivity.json").read_text())["15"]
    c, g = np.array(sel["c"]), np.array(sel["g"])
    S = g / c
    ch = summ["chosen"]["C"]
    rows = {x["name"]: x for x in summ["rows"]}

    def pick(arm):
        for x in summ["rows"]:
            if x["arm"] != arm or x["layer"] != 15 or x["seed"] != 0 or (arm != "C0" and x["lr"] != ch["lr"]):
                continue
            if arm in ("G0", "C", "D") and x["lam_g"] != ch["lam_g"]:
                continue
            if arm in ("C", "C0") and x["lam_kl"] != ch["lam_kl"]:
                continue
            if arm == "D" and x["lam_d"] != summ["chosen"]["D"]["lam_d"]:
                continue
            return x
    fig, axs = plt.subplots(1, 4, figsize=(15, 4.2), sharex=True, sharey=True)
    c90 = np.quantile(c, 0.9)
    for ax, (arm, title) in zip(axs, ARMS):
        x = pick(arm)
        ax.scatter(S, c, s=1.5, color=GRAY, alpha=0.25, linewidths=0, rasterized=True)
        ax.axhline(c90, color=MUTED, lw=0.8, ls=":")
        ax.grid(color=GRID, lw=0.6)
        if x is None:
            ax.set_title(title + "\n(not run)", fontsize=9, color=INK2)
            continue
        r = json.loads((HERE / "out/runs" / f"{x['name']}.json").read_text())
        nm = dict(zip(r["top512"], r["top512_norm"]))
        K = [j for j in r["kept95"] if j in nm][:512]
        e = np.array([nm[j] ** 2 for j in K])
        share = e / max(r["frob"] ** 2, 1e-30)
        ax.scatter(S[K], c[K], s=4 + 600 * share, color=BLUE, alpha=0.75,
                   edgecolors="white", linewidths=0.6, zorder=3)
        for j, lab in ((7291, "7291"), (6582, "6582")):
            ax.scatter([S[j]], [c[j]], s=70, facecolors="none", edgecolors=INK, linewidths=1.2, zorder=4)
            ax.annotate(lab, (S[j], c[j]), xytext=(6, 4), textcoords="offset points", fontsize=8, color=INK)
        more = " (largest 512 shown)" if x["n95"] > 512 else ""
        ax.set_title(f"{title}\nkept {x['n95']} cols{more} · TKL {x['tkl']:.2f} · F {x['F']:.4f}", fontsize=9, color=INK)
        ax.set_xscale("log")
        ax.set_yscale("log")
    axs[0].set_ylabel("c = mean |a| on casual replies (target)")
    for ax in axs:
        ax.set_xlabel("S = off-target / target activity")
    axs[0].annotate("selective ←", xy=(0.02, 0.02), xycoords="axes fraction", fontsize=8, color=INK2)
    axs[0].annotate("top 10% of c", xy=(0.98, c90), xycoords=("axes fraction", "data"), ha="right", va="bottom",
                    fontsize=7, color=MUTED)
    fig.suptitle("Layer 15: which down_proj columns the optimizer keeps (dot area = share of the update); "
                 "rings = hand-selected sel8 neurons", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(HERE / "fig_preservation_selection.png", dpi=200, metadata={"Software": None})


if __name__ == "__main__":
    main()
