"""Judge check at matched teacher KL (LOC_NOTES.md): tables from the pre-registered judging pass (judged_loc.json.gz)
and the post-hoc second pass (judged_loc2.json.gz; columns at LR 1e-1, with re-judged anchors).

    python experiments/01_locality_curves/loc/analyze_loc.py  -> loc/summary_loc.json, loc/tables_loc.md
"""
import gzip
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
RUNS = {"lorad": ("rank-1 LoRA, down_proj L15, LR 1e-3", "lorad", 1e-3),
        "sel8": ("8 attribution-selected columns, LR 3e-2", "sel8", 3e-2),
        "srand8": ("8 selectivity-matched columns, LR 3e-2", "srand8", 3e-2),
        "sel8_lr0.1": ("8 attribution-selected columns, LR 1e-1 (post hoc)", "sel8", 1e-1),
        "srand8_lr0.1": ("8 selectivity-matched columns, LR 1e-1 (post hoc)", "srand8", 1e-1)}
PASSES = [("judged_loc.json.gz", "pass 1 (pre-registered)", ["original", "prompt", "gains_s0", "lorad", "sel8", "srand8"],
           [("lorad", "sel8"), ("lorad", "srand8"), ("sel8", "srand8")]),
          ("judged_loc2.json.gz", "pass 2 (post hoc)", ["original", "prompt", "lorad", "sel8_lr0.1", "srand8_lr0.1"],
           [("lorad", "sel8_lr0.1"), ("lorad", "srand8_lr0.1")])]
ANCHORS = {"original": "original model", "prompt": "teacher (system prompt)", "gains_s0": "released 128 gains"}
KEYS = ["humanlike", "ai_disclaimer", "human_stance", "markdown", "coherence"]
B = 10000


def per_msg(items, k):
    v = {}
    for it in items:
        if it["capped"] or "error" in it.get("judge", {}):
            continue
        v.setdefault(it["i"], []).append(float(it["judge"][k]))
    return np.array([np.mean(v[i]) for i in range(60)])


def one_pass(J, order, pairs, pts, idx):
    ci = lambda x: [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]  # noqa: E731
    hl = {n: per_msg(J[n]["items"], "humanlike") for n in J}
    o, t = hl["original"], hl["prompt"]
    out = {}
    for n in order:
        it = J[n]["items"]
        r = dict(n_scored=sum(1 for x in it if not x["capped"] and "error" not in x.get("judge", {})),
                 n_capped=sum(x["capped"] for x in it), mean_tokens=float(np.mean([x["n_tokens"] for x in it])))
        for k in KEYS:
            m = per_msg(it, k)
            r[k] = float(m.mean())
            r[k + "_ci"] = ci(m[idx].mean(1))
        h = hl[n]
        r["gap_closed"] = float((h.mean() - o.mean()) / (t.mean() - o.mean()))
        r["gap_closed_ci"] = ci((h[idx].mean(1) - o[idx].mean(1)) / (t[idx].mean(1) - o[idx].mean(1)))
        if n in RUNS:
            p = pts[(RUNS[n][1], RUNS[n][2])]
            r["tkl_seed0"], r["F_seed0"] = p["tkl_seeds"][0], p["F_seeds"][0]
        out[n] = r
    diffs = {}
    for a, b in pairs:
        d = hl[a] - hl[b]
        diffs[f"{a}-{b}"] = dict(diff=float(d.mean()), ci=ci(d[idx].mean(1)))
    return out, diffs


def main():
    s = json.loads((REPO / "experiments/01_locality_curves/out/summary.json").read_text())
    pts = {(p["fam"], p["lr"]): p for p in s["points"]}
    idx = np.random.default_rng(0).integers(0, 60, (B, 60))
    lab = {**{k: v[0] for k, v in RUNS.items()}, **ANCHORS}
    f = lambda x, d=3: "–" if x is None else f"{x:.{d}f}"  # noqa: E731
    res, L = {}, ["# Judge check at matched teacher KL\n"]
    for fn, title, order, pairs in PASSES:
        J = json.loads(gzip.open(HERE / fn, "rt").read())
        out, diffs = one_pass(J, order, pairs, pts, idx)
        res[title] = dict(file=fn, conditions=out, paired_hl_diffs=diffs)
        L += [f"\n## {title}: {len(J)} conditions x 180 replies, one shuffled judging pass\n",
              "| condition | TKL (seed 0) | F (seed 0) | human-likeness [95% CI] | gap closed [CI] | AI disclaimer | "
              "speaks as a person | markdown | coherence | mean tokens | scored |",
              "|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|"]
        for n in order:
            r = out[n]
            L.append(f"| {lab[n]} | {f(r.get('tkl_seed0'))} | {f(r.get('F_seed0'), 4)} | {r['humanlike']:.2f} "
                     f"[{r['humanlike_ci'][0]:.2f}, {r['humanlike_ci'][1]:.2f}] | {r['gap_closed']:.0%} "
                     f"[{r['gap_closed_ci'][0]:.0%}, {r['gap_closed_ci'][1]:.0%}] | {r['ai_disclaimer']:.2f} | "
                     f"{r['human_stance']:.2f} | {r['markdown']:.2f} | {r['coherence']:.2f} | {r['mean_tokens']:.0f} | "
                     f"{r['n_scored']}/180 |")
        L += ["\nPaired human-likeness differences (bootstrap over the 60 messages):\n"]
        L += [f"- {k}: {v['diff']:+.2f} [{v['ci'][0]:+.2f}, {v['ci'][1]:+.2f}]" for k, v in diffs.items()]
    (HERE / "summary_loc.json").write_text(json.dumps(res, indent=1))
    (HERE / "tables_loc.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
