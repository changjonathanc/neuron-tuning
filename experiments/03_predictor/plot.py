"""E1 figure: (a) F vs direct write off-target, all runs; (b) F ratio to LoRA at matched effect vs selectivity.

    python plot.py      (after analyze.py) -> fig_predictor.png
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
P = json.loads((HERE / "out/pred.json").read_text())
F32 = json.loads((HERE / "out/f32.json").read_text()) if (HERE / "out/f32.json").exists() else {}
S = json.loads((HERE / "out/summary.json").read_text())

TYPE = {"sel8": "sel", "sel128": "sel", "col8": "sel", "srand8": "srand", "srand128": "srand", "dense8": "dense",
        "dense128": "dense", "lorad": "lora", "lorac": "lora", "lora_d": "lora"}
COL = {"sel": "#2a78d6", "srand": "#eb6834", "dense": "#1baf7a", "lora": "#eda100"}
LAB = {"sel": "attribution-selected columns (sel, col8)", "srand": "selectivity-matched columns (srand)",
       "dense": "dense random columns", "lora": "rank-1 LoRA on down_proj"}
MK = {"1c 13k": "o", "1c 200k": "s", "2b": "^"}
MKLAB = {"1c 13k": "phase1c, ~13 k", "1c 200k": "phase1c, ~200 k", "2b": "phase2b rules, ~13 k"}
INK, MUTED = "#2b2b2b", "#77766f"


def setup(r):
    return "2b" if r["phase"] == "2b" else f"1c {r['budget']}"


def main():
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                         "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12, 5.2))
    for r in P.values():
        f2 = F32.get(r["path"], {}).get("macro")
        y = f2 if f2 is not None else r["F1"]
        ax.scatter(r["D_off"], y, s=30, marker=MK[setup(r)], color=COL[TYPE[r["fam"]]], edgecolor="white",
                   linewidth=0.6, alpha=0.9, zorder=3)
        if f2 is not None and r["F1"] < 0.01:
            ax.scatter(r["D_off"], r["F1"], s=12, marker=MK[setup(r)], facecolor="none", edgecolor="#b5b4ad",
                       linewidth=0.6, zorder=2)
    ax.axhline(0.0011, color="#b5b4ad", lw=1, ls="--", zorder=1)
    ax.text(0.98, 0.0012, "bf16 floor of the stored F (~0.001)", color=MUTED, fontsize=8, ha="right", va="bottom",
            transform=ax.get_yaxis_transform())
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("direct write off-target D_off\n(mean over reference-text tokens of Σ_l ||ΔW_l a_l(x)||)")
    ax.set_ylabel("forgetting F (KL per token, unrelated text; fp32 for phase1c)")
    a = S["a"]
    ax.set_title(f"(a) all {a['n']} runs: Spearman ρ = {a['rho_all']:.2f}", loc="left", color=INK)
    hs = [ax.scatter([], [], color=c, marker="o", label=LAB[t]) for t, c in COL.items()]
    hs += [ax.scatter([], [], color=MUTED, marker=m, label=MKLAB[k]) for k, m in MK.items()]
    hs += [ax.scatter([], [], facecolor="none", edgecolor="#b5b4ad", marker="o", label="same run, stored bf16 F (pass 1)")]
    ax.legend(handles=hs, fontsize=7.5, frameon=False, loc="upper left")

    b = S["b"]
    for p in b["points"] + b["lora_points"]:
        t = "lora" if "lora" in p["point"] else TYPE.get(p["point"].split()[-1], None)
        y = p["ratio2"] if p["ratio2"] is not None else p["ratio1"]
        hollow = t == "lora"
        bx.scatter(p["S"], y, s=60, marker=MK[p["setup"]], facecolor="white" if hollow else COL[t],
                   edgecolor=COL[t], linewidth=1.5 if hollow else 0.6, zorder=3)
        if not hollow:
            off, ha = {"weather srand8": ((0, 8), "right"), "weather col8": ((0, -14), "center")}.get(p["point"], ((5, 3), "left"))
            bx.annotate(p["point"], (p["S"], y), xytext=off, textcoords="offset points", fontsize=8, color=INK, ha=ha)
    bx.axhline(1, color="#b5b4ad", lw=1, ls="--", zorder=1)
    bx.text(0.02, 0.85, "ratio 1 = the LoRA (hollow: LoRA points, post-training S of its input direction A)",
            color=MUTED, fontsize=8, transform=bx.get_yaxis_transform(), va="top")
    bx.set_xscale("log")
    bx.set_xlim(0.09, 4)
    bx.set_yscale("log")
    bx.set_xlabel("pre-training selectivity S of the edited neurons\n(geometric mean of mean|a| off-target / mean|a| on target; right = less selective)")
    bx.set_ylabel("F / F_LoRA at matched effect")
    bx.set_title(f"(b) matched effect, {b['n']} neuron points: Spearman ρ = {b['rho']:.2f}", loc="left", color=INK)
    fig.tight_layout()
    fig.savefig(HERE / "fig_predictor.png", dpi=150)


if __name__ == "__main__":
    main()
