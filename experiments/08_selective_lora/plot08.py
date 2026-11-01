"""Experiment 08 figure: (a) the frontier F (fp32) vs teacher KL, seed 0; (b) the cost of a training step.

    uv run experiments/08_selective_lora/plot08.py        # reads out/summary.json -> fig_selective_lora.png
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402

HERE = Path(__file__).resolve().parent
S = json.loads((HERE / "out/summary.json").read_text())
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
BLUE = {"0.01": "#86b6ef", "0.1": "#2a78d6", "1": "#104281"}          # one hue, light -> dark = small -> large λ
VIOLET = {"100": "#a59cf0", "1000": "#6250d6", "10000": "#3a2d8f"}
ORANGE = {"1": "#f4a37f", "10": "#eb6834"}


def style(fam):
    if fam == "plain":
        return INK2, "o", "--", "standard LoRA (rank 8, all modules)"
    if fam == "neurons":
        return "#1baf7a", "s", "-", "8 selective neurons (experiment 01)"
    arm, lam = fam.split("_lam")
    if arm == "sel":
        return BLUE.get(lam, "#2a78d6"), "o", "-", f"selective LoRA (pre-registered P), λ = {lam}"
    if arm == "selrel":
        return VIOLET.get(lam, "#6250d6"), "^", "-", f"selective LoRA, relative P (amendment 1), λ = {lam}"
    return ORANGE.get(lam, "#eb6834"), "D", "-", f"LoRA + KL preservation, w = {lam}"


fams = [k.rsplit("|", 1)[0] for k in S["points"] if k.endswith("|s0") and not k.startswith("plain05")]
order = (["plain"] + sorted(f for f in fams if f.startswith("kl"))
         + sorted((f for f in fams if f.startswith("sel_")), key=lambda f: float(f.split("lam")[1]))
         + sorted((f for f in fams if f.startswith("selrel")), key=lambda f: float(f.split("lam")[1])) + ["neurons"])
fig, (ax, bx) = plt.subplots(1, 2, figsize=(12, 5.2), gridspec_kw=dict(width_ratios=[1.9, 1]))
for fam in order:
    pts = [p for p in S["points"].get(f"{fam}|s0", []) if "F32" in p]
    if not pts:
        continue
    c, mk, ls, lab = style(fam)
    ax.plot([p["tkl"] for p in pts], [p["F32"] for p in pts], color=c, marker=mk, ms=8, lw=2, ls=ls, label=lab,
            mec="white", mew=1.5, zorder=3)
ax.axvline(0.6, color=MUTED, ls=":", lw=1)
ax.text(0.6, 0.98, " matched point TKL 0.6", transform=ax.get_xaxis_transform(), color=INK2, fontsize=8, va="top")
ax.set_xscale("log")
ax.set_yscale("log")
ax.invert_xaxis()
ticks = [2, 1, 0.6, 0.4, 0.3, 0.2, 0.15]
ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator(ticks))
ax.xaxis.set_major_formatter(matplotlib.ticker.FixedFormatter([f"{x:g}" for x in ticks]))
ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
ax.set_xlabel("teacher KL (log; to the right = stronger edit)", color=INK2)
ax.set_ylabel("change on unrelated text F, fp32 (log)", color=INK2)
ax.set_title("(a) Strength vs change on unrelated text (seed 0, one point per LR)", loc="left", fontsize=10, color=INK)
ax.grid(True, which="major", color=GRID, lw=0.6)
ax.legend(fontsize=8, frameon=False, loc="lower right")

cost = S["bench"]                                   # keys: neurons, plain, sel (P and P_rel cost the same), kl
rows = [f for f in ["neurons", "plain", "sel", "kl"] if f in cost]
lab = [{"neurons": "8 neurons", "plain": "standard LoRA", "sel": "selective LoRA", "kl": "LoRA + KL preservation"}[f]
       for f in rows]
colors = {"neurons": "#1baf7a", "plain": INK2, "sel": "#2a78d6", "kl": "#eb6834"}
v = [cost[f]["sec_per_step"] / cost["plain"]["sec_per_step"] for f in rows]
bx.barh(range(len(rows)), v, color=[colors[f] for f in rows], height=0.6)
for i, (x, f) in enumerate(zip(v, rows)):
    bx.text(x + 0.03, i, f"{x:.2f}×  ({cost[f]['peak_mem_gb']:.0f} GB)", va="center", fontsize=8, color=INK2)
bx.set_yticks(range(len(rows)), lab, fontsize=9)
bx.invert_yaxis()
bx.set_xlim(0, max(v) * 1.45)
bx.set_xlabel("time per training step / standard LoRA", color=INK2)
bx.set_title("(b) Cost per training step (40-step benchmark)\nlabel: peak GPU memory", loc="left",
             fontsize=10, color=INK)
for a in (ax, bx):
    for s in ("top", "right"):
        a.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        a.spines[s].set_color("#c3c2b7")
    a.tick_params(colors=INK2, labelsize=8)
fig.tight_layout()
fig.savefig(HERE / "fig_selective_lora.png", dpi=150)
print("wrote", HERE / "fig_selective_lora.png")
