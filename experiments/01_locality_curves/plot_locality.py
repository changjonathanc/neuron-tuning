"""Locality figure and matched-effect table from the Phase 1c runs (no new training).

Main panel (~13 k trained numbers) and a smaller panel (~200 k): x = forgetting F (macro mean over the 7 domains of the
Phase 1b set, log), y = teacher KL (log; lower = stronger effect). One point per LR = mean over seeds; error bars =
95% bootstrap CI on both axes (the Phase 1c three-level bootstrap: eval messages, forgetting items by domain, seeds).

The table gives F of each family at the pre-registered matched teacher KL (ln F interpolated linearly in ln TKL
between adjacent LRs, first crossing; analyze1c.interp), with 95% CIs, the ratio to the LoRA of the same size, and the
robust variants (F without QA; per-domain median).

    python experiments/01_locality_curves/plot_locality.py [--boot 10000]  -> fig_locality.png, out/locality_table.md, out/locality.json
    python experiments/01_locality_curves/plot_locality.py --replot      (figure only, from out/locality.json)
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze1c as A  # noqa: E402
import gen_jobs as G  # noqa: E402

EXTRA = ["sel128:0.001", "dense128:0.3"]  # Phase 1c edge-rule extensions (NOTES.md run log entry 2)
ORIG_TKL = 2.607
F_FLOOR = {"13k": 0.0011, "200k": 0.0016}  # measurement floor of F from bf16 rounding (Phase 1c RESULTS, caveat 1)
ROLE = {  # role -> (colour, marker, label); colour-blind-safe categorical order (validated)
    "lora": ("#eda100", "D", "rank-1 LoRA, down_proj"),
    "sel": ("#2a78d6", "o", "columns of attribution-selected neurons"),
    "srand": ("#eb6834", "s", "columns of selectivity-matched neurons (no attribution)"),
    "dense": ("#1baf7a", "^", "columns of random active neurons"),
}
PANELS = [("13k", "~13 k trained numbers (K = 8 columns; LoRA on 1 layer)",
           {"lora": "lorad", "sel": "sel8", "srand": "srand8", "dense": "dense8"}),
          ("200k", "~200 k (K = 128 columns; LoRA on 16 layers)",
           {"lora": "lorac", "sel": "sel128", "srand": "srand128", "dense": "dense128"})]
METRICS = ["F", "F_noqa", "F_med"]
LABEL_OFFSET = {("lorac", 1e-4): (7, -12), ("sel128", 0.1): (-6, -14), ("srand128", 0.1): (-6, 6), ("sel128", 0.03): (-4, -11),
                ("dense128", 0.03): (2, 5)}  # LR labels moved away from other curves


def bootstrap(runs, doms, seeds, grid, n_boot):
    R = A.Resampler(doms, 60, len(next(iter(runs.values()))["task"]), seeds, np.random.default_rng(0))
    point = A.curves(runs, R.identity(), grid)
    keys = [(b, f, T, m) for b, _, fams in PANELS for f in fams.values() for T in A.TARGETS[b] for m in METRICS]
    lora = {b: fams["lora"] for b, _, fams in PANELS}
    est = {k: A.interp(point[k[1]], k[2], k[3]) for k in keys}
    bs = {k: [] for k in keys}
    rat = {k: [] for k in keys}
    pts = {(f, lr): [] for f, lrs in grid.items() for lr in lrs}
    for _ in range(n_boot):
        c = A.curves(runs, R.draw(), grid)
        for f, ps in c.items():
            for lr, t, mm in ps:
                pts[(f, lr)].append((t, mm["F"]))
        for k in keys:
            b, f, T, m = k
            y, yl = A.interp(c[f], T, m), A.interp(c[lora[b]], T, m)
            bs[k].append(y)
            rat[k].append(None if y is None or yl is None else y - yl)
    ci = lambda v: [math.exp(np.percentile(v, 2.5)), math.exp(np.percentile(v, 97.5))] if v else [None, None]  # noqa
    table = []
    for k in keys:
        b, f, T, m = k
        v = [x for x in bs[k] if x is not None]
        r = [x for x in rat[k] if x is not None]
        yl = est[(b, lora[b], T, m)]
        table.append(dict(budget=b, fam=f, target=T, metric=m,
                          value=None if est[k] is None else math.exp(est[k]), ci=ci(v), dropped=1 - len(v) / n_boot,
                          ratio_vs_lora=None if est[k] is None or yl is None else math.exp(est[k] - yl),
                          ratio_ci=ci(r), ratio_dropped=1 - len(r) / n_boot))
    point_tab = []
    for (f, lr), v in pts.items():
        ps = [p for p in point[f] if p[0] == lr]
        if not ps or not v:
            continue
        v = np.array(v)
        point_tab.append(dict(fam=f, lr=lr, n=sum((f, lr, s) in runs for s in seeds), tkl=ps[0][1], F=ps[0][2]["F"],
                              tkl_ci=list(np.percentile(v[:, 0], [2.5, 97.5])),
                              F_ci=list(np.percentile(v[:, 1], [2.5, 97.5]))))
    return table, point_tab


def plot(points, table, out):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 6.3), gridspec_kw=dict(width_ratios=[1.7, 1]))
    for ax, (b, title, fams) in zip(axes, PANELS):
        main = b == "13k"
        for role, fam in fams.items():
            c, mk, lab = ROLE[role]
            ps = sorted([p for p in points if p["fam"] == fam], key=lambda p: p["lr"])
            if not ps:
                continue
            x = np.array([p["F"] for p in ps])
            y = np.array([p["tkl"] for p in ps])
            xe = np.array([[p["F"] - p["F_ci"][0], p["F_ci"][1] - p["F"]] for p in ps]).T.clip(0)
            ye = np.array([[p["tkl"] - p["tkl_ci"][0], p["tkl_ci"][1] - p["tkl"]] for p in ps]).T.clip(0)
            ax.errorbar(x, y, xerr=xe, yerr=ye, color=c, marker=mk, ms=7 if main else 5.5, lw=2, elinewidth=1,
                        capsize=0, label=lab, mec="white", mew=1, zorder=3)
            for p in ps:
                ax.annotate(f"{p['lr']:g}".replace("e-05", "e-5").replace("0.0001", "1e-4").replace("0.0003", "3e-4"), (p["F"], p["tkl"]), textcoords="offset points",
                            xytext=LABEL_OFFSET.get((fam, p["lr"]), (4, 3)),
                            fontsize=6.5 if main else 6, color="#52514e", zorder=4)
        ax.axhline(ORIG_TKL, color="#52514e", ls=":", lw=1)
        ax.text(0.98, ORIG_TKL * 0.93, f"original model ({ORIG_TKL:.2f})", transform=ax.get_yaxis_transform(),
                ha="right", va="top", fontsize=7.5, color="#52514e")
        for T in A.TARGETS[b]:
            ax.axhline(T, color="#c3c2b7", ls="--", lw=0.8, zorder=1)
        ax.axvline(F_FLOOR[b], color="#8a8984", ls="-.", lw=0.8, zorder=1)
        ax.text(F_FLOOR[b] * 1.06, 0.015, "F floor (bf16)", transform=ax.get_xaxis_transform(), ha="left", va="bottom",
                fontsize=7, color="#52514e")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("change on unrelated text F = KL(original ‖ edited) (log)", fontsize=9)
        ax.set_ylabel("teacher KL (log; lower = stronger effect)", fontsize=9)
        ax.set_title(title, fontsize=10.5 if main else 9.5, loc="left")
        ax.grid(True, which="major", color="#e6e5e0", lw=0.6)
        ax.tick_params(labelsize=8)
        ax.yaxis.set_major_locator(matplotlib.ticker.FixedLocator([0.2, 0.3, 0.4, 0.6, 1, 2]))
        ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=8.5, frameon=False, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.05))
    fig.text(0.01, 0.008, "Each point: one learning rate (small numbers), mean of 3 seeds; bars: 95% bootstrap CI "
             "(eval messages, unrelated items, seeds).\nDashed: matched-effect targets. Dash-dot: measurement floor of F "
             "(bf16 rounding). All runs: loss = KL to the teacher on casual replies only, no preservation loss.",
             fontsize=7.5, color="#52514e")
    fig.tight_layout(rect=(0, 0.15, 1, 1))
    fig.savefig(out, dpi=160)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--replot", action="store_true", help="only redraw the figure from out/locality.json")
    a = ap.parse_args()
    if a.replot:
        d = json.loads((HERE / "out/locality.json").read_text())
        plot(d["points"], d["table"], HERE / "fig_locality.png")
        return
    seeds = [0, 1, 2]
    runs, doms = A.collect(seeds, EXTRA)
    grid = G.grid(EXTRA)
    table, points = bootstrap(runs, doms, seeds, grid, a.boot)
    (HERE / "out").mkdir(exist_ok=True)
    (HERE / "out/locality.json").write_text(json.dumps(dict(n_runs=len(runs), boot=a.boot, table=table,
                                                            points=points), indent=1, default=float))
    plot(points, table, HERE / "fig_locality.png")
    f = lambda x, d=4: "–" if x is None else f"{x:.{d}f}"  # noqa: E731
    L = [f"# Matched-effect forgetting ({len(runs)} runs, seeds 0-2, {a.boot} resamples)\n",
         "F of each family at the target teacher KL (interpolated along the seed-mean curve) [95% CI]; "
         "ratio to the LoRA of the same size [95% CI]. '–' = the curve does not reach the target.\n",
         "| budget | target TKL | family | F [CI] | ÷ LoRA [CI] | F_noQA ÷ LoRA | F_med ÷ LoRA |",
         "|---|---:|---|---|---|---|---|"]
    get = {(r["fam"], r["target"], r["metric"]): r for r in table}
    for b, _, fams in PANELS:
        for T in A.TARGETS[b]:
            for role, fam in fams.items():
                r = get[(fam, T, "F")]
                if r["value"] is None or r["dropped"] > 0.10:
                    L.append(f"| {b} | {T} | {fam} | not reached / not covered ({r['dropped']:.0%} of resamples "
                             "miss the target) | | | |")
                    continue
                cols = [f"{f(r['value'])} [{f(r['ci'][0])}, {f(r['ci'][1])}]"
                        + (f" ({r['dropped']:.0%} dropped)" if r["dropped"] > 0.005 else "")]
                for m in METRICS:
                    q = get[(fam, T, m)]
                    if role == "lora":
                        cols.append("1")
                        continue
                    cols.append(f"{f(q['ratio_vs_lora'], 3)} [{f(q['ratio_ci'][0], 3)}, {f(q['ratio_ci'][1], 3)}]"
                                + (f" ({q['ratio_dropped']:.0%} dropped)" if q["ratio_dropped"] > 0.005 else ""))
                L.append(f"| {b} | {T} | {fam} | " + " | ".join(cols) + " |")
    (HERE / "out/locality_table.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
