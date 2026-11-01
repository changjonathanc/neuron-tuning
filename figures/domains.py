"""Where the change goes: change on unrelated text per domain, casual edit, at the matched teacher KL 0.6.

Data: the per-run forgetting files (per-domain means of KL(original || edited) per token, stored bf16)
  experiments/01_locality_curves/runs/forget/{sel8,lorad}_lr*_s*.json
  experiments/05_typical_lora/runs/forget/lora_all8_lr*_s*.json
and the seed-mean teacher KL of each learning rate from the two out/summary.json files. Each domain is interpolated to
teacher KL 0.6 like analyze1c.py does it for the mean F: seed means per learning rate, linear in (ln teacher KL,
ln F), first crossing in increasing learning rate. The script checks that the mean over domains, done the same way,
gives the stored matched F of the analysis.

    uv run figures/domains.py   -> figures/domains.png
"""
import json
import math
import re
import sys

from style import BG, BLUE, GRAY, GRAY_DARK, GRID, INK, INK2, MUTED, ROOT, plt, save

MATCH = 0.6
E1, E3 = ROOT / "experiments/01_locality_curves", ROOT / "experiments/05_typical_lora"
sys.path.insert(0, str(E1))
import gen_jobs as G  # noqa: E402
S1, S3 = (json.loads((d / "out/summary.json").read_text()) for d in (E1, E3))
FAMS = [("lorad", E1, S1, "LoRA, same size (rank 1)", GRAY, "s"),
        ("lora_all8", E3, S3, "standard LoRA (rank 8)", GRAY_DARK, "D"),
        ("sel8", E1, S1, "8 neurons that fire mostly in casual chat", BLUE, "o")]
NAMES = {"code": "code", "math": "math", "qa": "questions on a text", "de": "German", "es": "Spanish",
         "ja": "Japanese", "longdoc": "long documents"}


def interp(pts, target):
    """pts: (lr, tkl, value) sorted by lr -> value at tkl = target (log-log, first crossing)."""
    lt = math.log(target)
    for (_, t0, m0), (_, t1, m1) in zip(pts, pts[1:]):
        x0, x1 = math.log(t0), math.log(t1)
        if (x0 - lt) * (x1 - lt) <= 0 and x0 != x1:
            w = (lt - x0) / (x1 - x0)
            return math.exp((1 - w) * math.log(m0) + w * math.log(m1))
    raise ValueError("target not bracketed")


def forget_files(fam, E):
    """(lr, path of the run's forget json) for every run that the analysis uses."""
    if E == E1:   # Phase 1c: some runs are reused from an earlier study; gen_jobs.paths knows where
        return [(lr, ROOT / G.paths(fam, lr, s)[2]) for lr in G.grid()[fam] for s in (0, 1, 2)]
    return [(float(re.match(rf"{fam}_lr(.+)_s\d+\.json", f.name).group(1)), f)
            for f in (E / "runs/forget").glob(f"{fam}_lr*_s*.json")]


matched = {}
for fam, E, S, *_ in FAMS:
    runs = {}
    for lr, f in forget_files(fam, E):
        (rec,) = json.loads(f.read_text()).values()
        runs.setdefault(lr, []).append(rec["by_domain"])
    tkl = {p["lr"]: p["tkl"] for p in S["points"] if p["fam"] == fam}
    doms = sorted(next(iter(runs.values()))[0])
    pts = {d: [(lr, tkl[lr], sum(r[d] for r in runs[lr]) / len(runs[lr])) for lr in sorted(runs)] for d in doms}
    pts["F"] = [(lr, tkl[lr], sum(sum(r.values()) / len(r) for r in runs[lr]) / len(runs[lr])) for lr in sorted(runs)]
    matched[fam] = {d: interp(v, MATCH) for d, v in pts.items()}
    stored = next(c for c in S["comparisons"] if c["A"] == "sel8" and c["target"] == MATCH and c["metric"] == "F")
    ref = stored["FA"] if fam == "sel8" else stored["FB"] if stored["B"] == fam else None
    if ref is None:
        ref = next(c["FB"] for c in S["comparisons"] if c["B"] == fam and c["target"] == MATCH and c["metric"] == "F")
    assert abs(matched[fam]["F"] / ref - 1) < 1e-6, (fam, matched[fam]["F"], ref)

order = sorted(NAMES, key=lambda d: matched["lorad"][d] / matched["sel8"][d])  # by ratio, largest on top
fig, ax = plt.subplots(figsize=(8.6, 4.6), facecolor=BG)
ax.set_facecolor(BG)
for i, d in enumerate(order):
    v = [matched[f][d] for f, *_ in FAMS]
    ax.plot([min(v), max(v)], [i, i], color=GRID, lw=2, zorder=1, solid_capstyle="round")
    for (fam, _, _, lab, c, mk), x in zip(FAMS, v):
        ax.scatter(x, i, s=44 if fam == "lorad" else 70, color=c, marker=mk, edgecolor=BG, linewidth=1.5, zorder=4 if fam == "lorad" else 3, label=lab if i == 0 else None)
    r = matched["lorad"][d] / matched["sel8"][d]
    ax.annotate(f"{r:.0f}×", (min(v), i), xytext=(-10, 0), textcoords="offset points", ha="right", va="center",
                fontsize=9, color=INK2)
ax.set_yticks(range(len(order)), [NAMES[d] for d in order], fontsize=10, color=INK)
ax.set_xscale("log")
ax.set_xlim(0.00015, 0.3)
ax.set_xticks([0.001, 0.01, 0.1], ["0.001", "0.01", "0.1"])
ax.minorticks_off()
ax.grid(axis="x", color=GRID, lw=0.8)
ax.tick_params(length=0)
for s in ("left", "bottom"):
    ax.spines[s].set_visible(False)
ax.set_xlabel("change on unrelated text (KL per token; left = more local)", fontsize=9.5, color=INK2)
ax.set_title("Edit 1, “talk casually”, at medium strength: the change per kind of text", loc="left", fontsize=12,
             color=INK, fontweight="bold", pad=30)
ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, frameon=False, fontsize=8.8, handletextpad=0.3,
          columnspacing=1.4, borderaxespad=0.3)
fig.text(0.01, 0.01, "Numbers at the left: how many times less the neuron edit changes that kind of text than the "
         "same-size LoRA. Mean of 3 seeds.\nStored bf16 values: values below about 0.001 are approximate, and so are the ratios that use them.", fontsize=7.8, color=INK2)
fig.subplots_adjust(left=0.2, right=0.97, top=0.83, bottom=0.2)
save(fig, "domains")
print({f: {d: round(v, 5) for d, v in m.items()} for f, m in matched.items()})
