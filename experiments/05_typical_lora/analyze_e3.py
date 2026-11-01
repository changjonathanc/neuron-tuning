"""E3 analysis (rules: NOTES.md). Adds the standard-LoRA families to the saved Phase 1c curves and compares F at
matched teacher KL with the unchanged Phase 1c code (experiments/01_locality_curves/analyze1c.py: collect, Resampler, curves,
interp; bootstrap over eval messages, forgetting items per domain, task items, and seeds per family).

    python experiments/05_typical_lora/analyze_e3.py [--boot 10000] [--e3-extra lora_all8:3e-4 ...]  -> out/summary.json, out/tables.md
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "experiments/01_locality_curves"))
import analyze1c as A  # noqa: E402
import gen_jobs as G  # noqa: E402

P1C_EXTRA = ["sel128:0.001", "dense128:0.3"]          # Phase 1c edge-rule steps (out/summary.json)
OLD = ["sel8", "srand8", "lorad", "sel128", "srand128", "lorac"]
NEW = {"lora_all8": [1e-6, 3e-6, 1e-5, 3e-5, 1e-4], "lora_mlp8": [1e-6, 3e-6, 1e-5, 3e-5, 1e-4]}
R = "experiments/05_typical_lora/runs"
TARGETS = [0.60, 0.45, 0.30, 0.20]
CMP = [("main", "sel8", [0.60, 0.45]), ("sec", "srand8", [0.60, 0.45]),
       ("main", "sel128", [0.30, 0.20]), ("sec", "srand128", [0.30, 0.20]),
       ("P2", "lorad", [0.60, 0.45]), ("P2", "lorac", [0.30, 0.20])]

_paths, _load = G.paths, A.load


def load(path):
    """The saved result files are keyed by the adapter paths (experiments/NN_name/runs/..., relative to the
    repository root)."""
    return _load(path)


def paths(fam, lr, s):
    if fam in NEW:
        n = G.name(fam, lr, s)
        return f"{R}/final/{n}.pt", f"{R}/tkl/{n}.json", f"{R}/forget/{n}.json", f"{R}/task/{n}.json"
    return _paths(fam, lr, s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--e3-extra", nargs="*", default=[])
    a = ap.parse_args()
    for e in a.e3_extra:
        f, lr = e.split(":")
        NEW[f] = sorted(set(NEW[f]) | {float(lr)})
    p1c = G.grid(P1C_EXTRA)
    grid = {f: p1c[f] for f in OLD} | {f: list(v) for f, v in NEW.items()}
    G.paths = paths
    A.load = load
    G.grid = lambda extra=(): grid
    G.FAM = {f: None for f in grid}
    seeds = [0, 1, 2]
    runs, doms = A.collect(seeds, [])
    new = [f for f in NEW if any(k[0] == f for k in runs)]
    grid = {f: v for f, v in grid.items() if f in OLD or f in new}
    G.FAM = {f: None for f in grid}
    n_task = len(next(iter(runs.values()))["task"])
    Rs = A.Resampler(doms, 60, n_task, seeds, np.random.default_rng(0))
    point = A.curves(runs, Rs.identity(), grid)
    keys = [(q, X, L, T, m) for L in new for q, X, Ts in CMP for T in Ts for m in A.METRICS]
    bs = {k: [] for k in keys}
    pt_bs = {(f, lr): [] for f in new for lr in grid[f]}
    for _ in range(a.boot):
        c = A.curves(runs, Rs.draw(), grid)
        for f in new:
            for lr, t, mm in c[f]:
                pt_bs[(f, lr)].append((t, mm["F"], mm["task"]))
        for k in keys:
            q, X, L, T, m = k
            ya, yb = A.interp(c[X], T, m), A.interp(c[L], T, m)
            bs[k].append(None if ya is None or yb is None else ya - yb)
    comps = []
    for k in keys:
        q, X, L, T, m = k
        ya, yb = A.interp(point[X], T, m), A.interp(point[L], T, m)
        v = [x for x in bs[k] if x is not None]
        drop = 1 - len(v) / len(bs[k])
        ci = [math.exp(np.percentile(v, 2.5)), math.exp(np.percentile(v, 97.5))] if v else [None, None]
        per_seed = []
        for s in seeds:
            ix = Rs.identity()
            ix["seeds"] = {f: [s] for f in grid}
            cs = A.curves(runs, ix, grid)
            sa, sb = A.interp(cs[X], T, m), A.interp(cs[L], T, m)
            per_seed.append(None if sa is None or sb is None else math.exp(sa - sb))
        comps.append(dict(q=q, A=X, B=L, target=T, metric=m, FA=None if ya is None else math.exp(ya),
                          FB=None if yb is None else math.exp(yb), ratio=None if ya is None or yb is None else math.exp(ya - yb),
                          ci=ci, dropped=drop, covered=ya is not None and yb is not None and drop <= 0.10, per_seed=per_seed))

    def verdict(X, L):
        rows = [c for c in comps if c["A"] == X and c["B"] == L and c["metric"] == "F" and c["covered"]]
        if not rows:
            return "not covered"
        tag = f" (covered targets: {', '.join(str(r['target']) for r in rows)})"
        if X.startswith("lora"):
            if all(1 / 3 <= r["ci"][0] and r["ci"][1] <= 3 for r in rows):
                return "same order (CI inside [1/3, 3])" + tag
            if any(r["ci"][0] > 3 for r in rows):
                return "standard LoRA more local than rank-1" + tag
            if any(r["ci"][1] < 1 / 3 for r in rows):
                return "standard LoRA less local than rank-1" + tag
            return "not determined" + tag
        if all(r["ci"][1] < 1 for r in rows):
            return f"{X} forgets less than the standard LoRA at matched effect" + tag
        if all(r["ci"][0] > 1 for r in rows):
            return "standard LoRA forgets less" + tag
        return "not settled" + tag

    verdicts = [dict(A=X, B=L, verdict=verdict(X, L)) for L in new for _, X, _ in CMP]
    pts = []
    for f in new:
        for lr in grid[f]:
            rs = [runs[(f, lr, s)] for s in seeds if (f, lr, s) in runs]
            if not rs:
                continue
            ms = [A.run_metrics(r, Rs.identity()) for r in rs]
            b = np.array(pt_bs[(f, lr)])
            pts.append(dict(fam=f, lr=lr, n=len(rs), tkl=float(np.mean([m["tkl"] for m in ms])),
                            tkl_ci=[float(np.percentile(b[:, 0], q)) for q in (2.5, 97.5)],
                            tkl_seeds=[float(m["tkl"]) for m in ms],
                            **{m: float(np.mean([x[m] for x in ms])) for m in A.METRICS},
                            F_ci=[float(np.percentile(b[:, 1], q)) for q in (2.5, 97.5)],
                            F_seeds=[float(m["F"]) for m in ms],
                            mmlu=float(np.mean([r["mmlu"] for r in rs if r["mmlu"] is not None]))))
    minTKL = {f: min(t for _, t, _ in point[f]) for f in grid}
    curves = {f: [dict(lr=lr, tkl=float(t), **{m: float(v) for m, v in mm.items()}) for lr, t, mm in point[f]] for f in grid}
    out = dict(boot=a.boot, n_runs=len(runs), grid=grid, verdicts=verdicts, comparisons=comps, points=pts,
               min_tkl=minTKL, curves=curves)
    (HERE / "out").mkdir(exist_ok=True)
    (HERE / "out/summary.json").write_text(json.dumps(out, indent=1))
    fmt = lambda x, d=4: "–" if x is None else f"{x:.{d}f}"  # noqa: E731
    Lines = [f"# E3 tables ({len(runs)} runs incl. Phase 1c, {a.boot} resamples)\n", "## Verdicts (metric F)\n"]
    Lines += [f"- {v['A']} / {v['B']}: **{v['verdict']}**" for v in verdicts]
    Lines += ["\n## Lowest seed-mean TKL per family\n"] + [f"- {f}: {v:.3f}" for f, v in minTKL.items()]
    Lines += ["\n## Ratios at matched teacher KL: F_X / F_LoRA [95% CI] (per-seed ratios; dropped replicates)\n",
              "| q | X | LoRA | target TKL | metric | F_X | F_LoRA | ratio | CI | per seed | dropped |",
              "|---|---|---|---:|---|---:|---:|---:|---|---|---:|"]
    for c in comps:
        Lines.append(f"| {c['q']} | {c['A']} | {c['B']} | {c['target']} | {c['metric']} | {fmt(c['FA'])} | {fmt(c['FB'])} | "
                     f"{fmt(c['ratio'], 3)} | [{fmt(c['ci'][0], 3)}, {fmt(c['ci'][1], 3)}] | "
                     f"{', '.join(fmt(x, 2) for x in c['per_seed'])} | {c['dropped']:.3f} |")
    Lines += ["\n## E3 points (seed means)\n", "| family | LR | seeds | TKL [CI] | TKL per seed | F [CI] | F_noQA | F_med | task KL | MMLU |",
              "|---|---:|---:|---|---|---|---:|---:|---:|---:|"]
    for p in pts:
        Lines.append(f"| {p['fam']} | {p['lr']:g} | {p['n']} | {p['tkl']:.3f} [{p['tkl_ci'][0]:.3f}, {p['tkl_ci'][1]:.3f}] | "
                     f"{', '.join(f'{x:.3f}' for x in p['tkl_seeds'])} | "
                     f"{p['F']:.4f} [{p['F_ci'][0]:.4f}, {p['F_ci'][1]:.4f}] | {p['F_noqa']:.4f} | {p['F_med']:.4f} | "
                     f"{p['task']:.4f} | {p['mmlu']:.3f} |")
    (HERE / "out/tables.md").write_text("\n".join(Lines) + "\n")
    print("\n".join(Lines))


if __name__ == "__main__":
    main()
