"""Headline figure: change on unrelated text at the same strength of edit, relative to a LoRA of the same size.

All values are ratios of F (KL(original || edited) per token on 256 unrelated items, bf16, as stored) at matched
effect, from the committed analysis files:
  casual edit, teacher KL 0.60, ~13 k numbers: experiments/01_locality_curves/out/tables.md (sel8, srand8, dense8 vs lorad)
      and experiments/05_typical_lora/out/tables.md; colsel8 (activity rule) / lorad 0.30: experiments/04_activity_selection/RESULTS.md (lorad / lora_all8 = 1.46 -> standard LoRA = 1 / 1.46 of rank-1 LoRA)
  facts (E4), held-out accuracy 0.5, fp32 F: experiments/06_facts/RESULTS.md (colsel128 0.0068, col128 0.23,
      dense128 0.14, lora_all8 0.034; the rank-1 LoRA does not reach accuracy 0.5, so the reference is lora_all8)
  weather rule, matched trigger KL T_e: experiments/03_predictor/RESULTS.md (col8 / lora_d 37.1) and
      experiments/04_activity_selection/RESULTS.md (colsel8 / lora_d 0.11)
"""
import json

from style import BG, BLUE, GRAY, GRID, INK, INK2, ORANGE, ROOT, plt, save


def check():
    """The ratios below are typed in rounded form. Check each one against the committed analysis output."""
    def j(p):
        return json.loads((ROOT / p).read_text())
    e1, e3 = j("experiments/01_locality_curves/out/summary.json"), j("experiments/05_typical_lora/out/summary.json")
    e2, pr = j("experiments/04_activity_selection/out/summary_e2.json"), j("experiments/03_predictor/out/summary.json")
    e4 = j("experiments/06_facts/out/summary.json")

    def comp(S, A, B, T=0.6):
        return next(c["ratio"] for c in S["comparisons"] if (c["A"], c["B"], c["target"], c["metric"]) == (A, B, T, "F"))
    b = {p["point"]: p["ratio1"] for p in pr["b"]["points"]}
    acc = {m["arm"]: m["point"] for m in e4["matched"] if (m["kind"], m["target"], m["metric"]) == ("acc", 0.5, "F")}
    for typed, value in [(9.2, b["dense8"]), (1.46, comp(e3, "lorad", "lora_all8")), (0.085, comp(e1, "sel8", "lorad")),
                         (0.052, comp(e1, "srand8", "lorad")), (0.30, comp(e2["casual"], "colsel8", "lorad")),
                         (37.1, b["weather col8"]), (0.11, e2["rules"]["per_rule"]["weather"]["F_vs_lora_d"]["point"]),
                         (0.23, acc["col128"]), (0.14, acc["dense128"]), (0.0068, acc["colsel128"]),
                         (0.034, acc["lora_all8"])]:
        digits = len(f"{typed:g}".split(".")[-1]) if "." in f"{typed:g}" else 0
        assert round(value, digits) == typed, (typed, value)


check()

panels = [
    ("Edit 1: talk casually", [
        ("8 neurons that fire everywhere", 9.2, ORANGE),
        ("LoRA, same size (rank 1)", 1.0, GRAY),
        ("standard LoRA (rank 8, 12 M numbers)", 1 / 1.46, GRAY),
        ("8 neurons chosen by gradient\n(they fire mostly in casual chat)", 0.085, BLUE),
        ("8 other neurons that fire\nmostly in casual chat", 0.052, BLUE),
        ("8 neurons chosen by activity alone\n(early layers; large update)", 0.30, BLUE),
    ]),
    ("Edit 2: answer weather questions in one sentence", [
        ("8 neurons chosen by gradient\n(they fire everywhere here)", 37.1, ORANGE),
        ("LoRA, same size (rank 1)", 1.0, GRAY),
        ("8 neurons that fire mostly\non weather replies", 0.11, BLUE),
    ]),
    ("Edit 3: learn 160 facts about made-up people", [
        ("128 neurons chosen by gradient\n(not selective here)", 0.23 / 0.034, ORANGE),
        ("128 neurons that fire everywhere", 0.14 / 0.034, ORANGE),
        ("standard LoRA (rank 8, 12 M numbers)", 1.0, GRAY),
        ("128 neurons that fire mostly\nwhen stating these facts", 0.0068 / 0.034, BLUE),
    ]),
]


def label(v):
    if abs(v - 1) < 1e-9:
        return "reference"
    r = 1 / v if v < 1 else v
    s = f"{r:.1f}" if r < 2 else f"{r:.0f}"
    return f"{s}× less" if v < 1 else f"{s}× more"


fig, axes = plt.subplots(len(panels), 1, figsize=(8.5, 9.8), facecolor=BG, sharex=True,
                         gridspec_kw=dict(height_ratios=[len(r) for _, r in panels], hspace=0.55))
for k, (ax, (title, rows)) in enumerate(zip(axes, panels)):
    ax.set_facecolor(BG)
    ys = list(range(len(rows)))[::-1]
    for y, (name, v, c) in zip(ys, rows):
        ax.plot([1, v], [y, y], color=c, lw=2, alpha=0.35, solid_capstyle="round", zorder=1)
        ax.scatter([v], [y], s=90, color=c, edgecolor=BG, linewidth=2, zorder=3)
        ax.annotate(label(v), (v, y), xytext=(9 if v >= 1 else -9, 0), textcoords="offset points",
                    ha="left" if v >= 1 else "right", va="center", fontsize=10, color=INK)
    ax.axvline(1, color=INK2, lw=1, zorder=0)
    ax.set_xscale("log")
    ax.set_xlim(0.0012, 400)
    ax.set_xticks([0.01, 0.1, 1, 10, 100], ["0.01×", "0.1×", "1×", "10×", "100×"])
    ax.minorticks_off()
    ax.set_yticks(ys, [r[0] for r in rows], fontsize=10, color=INK)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_title(title, loc="left", fontsize=11.5, color=INK, pad=8, fontweight="bold")
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.tick_params(axis="x", colors=INK2, labelsize=9, labelbottom=True)
    ax.tick_params(axis="y", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
axes[-1].set_xlabel("change on unrelated text, relative to the reference LoRA of each edit\n"
                    "(compared at the same medium strength; left = more local)", fontsize=9.5, color=INK2)
save(fig, "headline", tight=True)
