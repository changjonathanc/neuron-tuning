"""Trade-off curve, casual edit: strength of the edit vs change on unrelated text, 4 methods at ~13 k / 12 M numbers.

Data (all read from the committed analysis outputs):
  experiments/01_locality_curves/out/summary.json  "points": sel8, dense8, lorad (seed means over 3 seeds per LR, with
      95% bootstrap CIs over messages, items and seeds); "comparisons": F ratios at the matched teacher KL 0.6
  experiments/05_typical_lora/out/summary.json     "points": lora_all8 (standard LoRA); "comparisons": sel8 / lora_all8
x = teacher KL (distance to the target model, per token; lower = stronger edit), y = F (KL(original || edited) per
token on 256 unrelated items, stored bf16 values).

    uv run figures/tradeoff.py   -> figures/tradeoff.png
"""
import json

from style import BG, BLUE, GRAY, GRAY_DARK, GRID, INK, INK2, MUTED, ORANGE, ROOT, plt, save

S1 = json.loads((ROOT / "experiments/01_locality_curves/out/summary.json").read_text())
S3 = json.loads((ROOT / "experiments/05_typical_lora/out/summary.json").read_text())
SERIES = [  # family, summary, label, colour, marker
    ("dense8", S1, "8 neurons that\nfire everywhere", ORANGE, "o"),
    ("lorad", S1, "LoRA, same size\n(rank 1, 14 k numbers)", GRAY, "s"),
    ("lora_all8", S3, "standard LoRA\n(rank 8, 12 M numbers)", GRAY_DARK, "D"),
    ("sel8", S1, "8 neurons chosen by gradient\n(fire mostly in casual chat)", BLUE, "o"),
]
MATCH = 0.6


def comp(S, A, B):
    return next(c for c in S["comparisons"] if c["A"] == A and c["B"] == B and c["target"] == MATCH and c["metric"] == "F")


fig, ax = plt.subplots(figsize=(8.6, 5.4), facecolor=BG)
ax.set_facecolor(BG)
for fam, S, lab, c, mk in SERIES:
    pts = sorted((p for p in S["points"] if p["fam"] == fam), key=lambda p: p["lr"])
    x, y = [p["tkl"] for p in pts], [p["F"] for p in pts]
    for p in pts:
        ax.plot([p["tkl"]] * 2, p["F_ci"], color=c, lw=5, alpha=0.18, solid_capstyle="round", zorder=1)
        ax.plot(p["tkl_ci"], [p["F"]] * 2, color=c, lw=5, alpha=0.18, solid_capstyle="round", zorder=1)
    ax.plot(x, y, color=c, lw=2, zorder=2, solid_joinstyle="round")
    ax.scatter(x, y, s=42, color=c, marker=mk, edgecolor=BG, linewidth=1.5, zorder=3)
    end = min(pts, key=lambda p: p["tkl"])           # the strongest point = the right end
    dy = {"lorad": 1.5, "lora_all8": 1.0, "dense8": 1.0, "sel8": 1.0}[fam]
    ax.annotate(lab, (end["tkl"], end["F"] * dy), xytext=(9, 0), textcoords="offset points", fontsize=9,
                color=INK, va="center", ha="left", linespacing=1.15)

ax.axvline(MATCH, color=INK2, lw=1, zorder=0)
ax.text(MATCH * 1.04, 3.2, f"medium strength (teacher KL {MATCH}):\nthe ratios are measured here", fontsize=8.5,
        color=INK2, ha="right", va="top")
a, b = comp(S1, "sel8", "lorad"), comp(S3, "sel8", "lora_all8")
ax.text(MATCH * 0.97, 0.0011, f"at this line the blue edit changes unrelated text\n{1 / a['ratio']:.0f}× less than the "
        f"same-size LoRA and {1 / b['ratio']:.0f}× less\nthan the standard LoRA", fontsize=8.5, color=INK, ha="left",
        va="bottom")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(2.6, 0.085)                                 # reversed: stronger edits to the right
ax.set_ylim(0.0009, 4)
ax.set_xticks([2, 1, 0.5, 0.2, 0.1], ["2", "1", "0.5", "0.2", "0.1"])
ax.set_yticks([0.001, 0.01, 0.1, 1], ["0.001", "0.01", "0.1", "1"])
ax.minorticks_off()
ax.grid(color=GRID, lw=0.8)
ax.spines["left"].set_visible(False)
ax.spines["bottom"].set_color(MUTED)
ax.tick_params(length=0, labelsize=9)
ax.set_xlabel("← weaker edit        distance to the target behaviour (teacher KL per token)        stronger edit →",
              fontsize=9.5, color=INK2)
ax.set_ylabel("change on unrelated text (KL per token)\nlower = more local", fontsize=9.5, color=INK2)
ax.set_title("Edit 1, “talk casually”: the side effect at each strength", loc="left", fontsize=12, color=INK,
             fontweight="bold", pad=12)
fig.text(0.01, 0.005, "Lines: mean of 3 seeds per learning rate; light bars: 95% bootstrap intervals. The 8-neuron edits "
         "do not reach the strongest effects of the LoRAs.", fontsize=7.8, color=INK2)
fig.subplots_adjust(left=0.12, right=0.80, top=0.90, bottom=0.17)
save(fig, "tradeoff")
