"""Why it works: the off-target write of an edit vs its change on unrelated text, 210 saved edits.

Data: experiments/03_predictor/out/pred.json (one row per edit: D_off, stored F, pre-training selectivity S),
experiments/03_predictor/out/f32.json (fp32 F for the casual-edit runs; as in analyze.py, it replaces the stored bf16 F
where it exists), experiments/03_predictor/out/summary.json (the Spearman rho; the script checks it).
Colour rule: LoRA = gray; neuron edits = blue if the geometric mean of S over the runs of their group (family and
behaviour) is below 1/3 (the neurons are on average at least 3x less active on other text than on the target), else
orange.

    uv run figures/predictor.py   -> figures/predictor.png
"""
import json
import math
from collections import defaultdict

from scipy.stats import spearmanr

from style import BG, BLUE, GRAY, GRID, INK, INK2, MUTED, ORANGE, ROOT, plt, save

E = ROOT / "experiments/03_predictor/out"
P = json.loads((E / "pred.json").read_text())
F32 = json.loads((E / "f32.json").read_text())
RHO = json.loads((E / "summary.json").read_text())["a"]
LORA = {"lorad", "lorac", "lora_d"}

runs = list(P.values())
for r in runs:
    r["F"] = F32[r["path"]]["macro"] if r["path"] in F32 else r["F1"]
    r["group"] = (r["fam"], r["target"])
S_group = defaultdict(list)
for r in runs:
    S_group[r["group"]].append(math.log(r["S"]))
S_group = {g: math.exp(sum(v) / len(v)) for g, v in S_group.items()}

rho = spearmanr([r["D_off"] for r in runs], [r["F"] for r in runs]).statistic
assert len(runs) == RHO["n"] and abs(rho - RHO["rho_all"]) < 1e-9, (rho, RHO["rho_all"])

KINDS = [("lora", "same-size LoRA, rank 1", GRAY, "s"),
         ("everywhere", "neurons that also fire on other text", ORANGE, "o"),
         ("selective", "neurons that fire mostly on the target", BLUE, "o")]


def kind(r):
    if r["fam"] in LORA:
        return "lora"
    return "selective" if S_group[r["group"]] < 1 / 3 else "everywhere"


fig, ax = plt.subplots(figsize=(7.6, 5.6), facecolor=BG)
ax.set_facecolor(BG)
for k, lab, c, mk in KINDS:
    rs = [r for r in runs if kind(r) == k]
    ax.scatter([r["D_off"] for r in rs], [r["F"] for r in rs], s=34, color=c, marker=mk, edgecolor=BG, linewidth=0.8,
               alpha=0.9, zorder=3, label=f"{lab} ({len(rs)})")
ax.set_xscale("log")
ax.set_yscale("log")
ax.grid(color=GRID, lw=0.8)
ax.spines["left"].set_visible(False)
ax.spines["bottom"].set_color(MUTED)
ax.tick_params(length=0, labelsize=9)
ax.minorticks_off()
ax.set_xticks([0.1, 1, 10, 100], ["0.1", "1", "10", "100"])
ax.set_yticks([1e-4, 1e-3, 1e-2, 1e-1, 1], ["0.0001", "0.001", "0.01", "0.1", "1"])
ax.set_xlabel("off-target write: how active the edited units are on unrelated text × size of the update\n"
              "(measured after training)", fontsize=9.5, color=INK2)
ax.set_ylabel("change on unrelated text (KL per token)", fontsize=9.5, color=INK2)
ax.set_title(f"{len(runs)} saved edits, 4 behaviours: the off-target write orders the side effect",
             loc="left", fontsize=11.5, color=INK, fontweight="bold", pad=10)
ax.text(0.98, 0.04, f"Spearman ρ = {rho:.2f}", transform=ax.transAxes, ha="right", va="bottom", fontsize=11,
        color=INK, fontweight="bold")
ax.legend(loc="upper left", frameon=False, fontsize=8.6, handletextpad=0.2, borderaxespad=0.2)
fig.text(0.01, 0.01, "One point per edit (method × learning rate × seed); the casual edit and the three rules, not the "
         "facts edit.\nCasual-edit runs: fp32 measurement; rule runs: the stored bf16 value.\nBlue / orange: the "
         "edited neurons are on average more / less than 3× as active on the target as on other text (before "
         "training).", fontsize=7.6, color=INK2)
fig.subplots_adjust(left=0.11, right=0.97, top=0.92, bottom=0.21)
save(fig, "predictor")
print({g: round(s, 3) for g, s in sorted(S_group.items(), key=lambda x: x[1])})
