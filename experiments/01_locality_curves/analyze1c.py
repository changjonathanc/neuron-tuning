"""Phase 1c analysis (rules: NOTES.md, pre-registration). Forgetting vs teacher KL curves per family, ratios at matched
effect with a three-level bootstrap (eval messages, forgetting items stratified by domain, seeds per family).

    python experiments/01_locality_curves/analyze1c.py [--boot 10000] [--seeds 0,1,2]  -> out/summary.json, out/tables.md
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
import gen_jobs as G  # noqa: E402

BUDGET = {"13k": ["sel8", "srand8", "dense8", "lorad"], "200k": ["sel128", "srand128", "dense128", "lorac"]}
TARGETS = {"13k": [0.60, 0.45], "200k": [0.30, 0.20]}
FAMB = {f: b for b, fs in BUDGET.items() for f in fs}
COMPARE = [  # (question, A, B)
    ("Q1", "sel8", "lorad"), ("Q1", "sel128", "lorac"),
    ("Q2", "sel8", "srand8"), ("Q2", "sel128", "srand128"),
    ("Q3", "srand8", "dense8"), ("Q3", "srand128", "dense128"),
    ("sec", "srand8", "lorad"), ("sec", "srand128", "lorac"),
    ("sec", "sel8", "dense8"), ("sec", "sel128", "dense128"),
]
METRICS = ["F", "F_noqa", "F_med", "task"]


def missing(what):
    """Stop with a clear message when the per-run result files (runs/) are not there."""
    sys.exit(f"error: no per-run result files for {what}. They are in git; see experiments/README.md.")


def load(path):
    p = REPO / path
    return json.loads(p.read_text()) if p.exists() else None


def collect(seeds, extra):
    """runs[(fam, lr, seed)] = dict(tkl_msg[60], f_item[256], task_item[300], mmlu)."""
    tk, fg, ta = {}, {}, {}
    runs, doms, msgs = {}, None, None
    for fam, lrs in G.grid(extra).items():
        for lr in lrs:
            for s in seeds:
                pt, ptk, pfg, pta = G.paths(fam, lr, s)
                for cache, f in ((tk, ptk), (fg, pfg), (ta, pta)):
                    if f not in cache:
                        cache[f] = load(f)
                a, b, c = tk[ptk], fg[pfg], ta[pta]
                if not (a and pt in a and b and pt in b and c and pt in c):
                    continue
                t = a[pt]
                m = sorted(set(t["msg"]))
                per_msg = [np.mean([x for x, i in zip(t["per_reply"], t["msg"]) if i == j]) for j in m]
                msgs = m if msgs is None else msgs
                assert m == msgs
                doms = b[pt]["domain"] if doms is None else doms
                runs[(fam, lr, s)] = dict(tkl=np.array(per_msg), f=np.array(b[pt]["per_item"]),
                                          task=np.array(c[pt]["per_item"]), mmlu=b[pt].get("mmlu"))
    empty = [f for f, lrs in G.grid(extra).items() if not any(k[0] == f for k in runs)]
    if empty:
        missing(f"{', '.join(empty)} (e.g. {G.paths(empty[0], G.grid(extra)[empty[0]][0], seeds[0])[1]})")
    return runs, np.array(doms)


class Resampler:
    """One bootstrap replicate = index arrays for messages, forgetting items (per domain), task items, seeds."""
    def __init__(self, doms, n_msg, n_task, seeds, rng):
        self.dl = sorted(set(doms))
        self.di = {d: np.where(doms == d)[0] for d in self.dl}
        self.n_msg, self.n_task, self.seeds, self.rng = n_msg, n_task, seeds, rng

    def identity(self):
        return dict(msg=np.arange(self.n_msg), dom={d: i for d, i in self.di.items()}, task=np.arange(self.n_task),
                    seeds={f: list(self.seeds) for f in G.FAM})

    def draw(self):
        r = self.rng
        return dict(msg=r.integers(0, self.n_msg, self.n_msg),
                    dom={d: i[r.integers(0, len(i), len(i))] for d, i in self.di.items()},
                    task=r.integers(0, self.n_task, self.n_task),
                    seeds={f: list(r.choice(self.seeds, len(self.seeds))) for f in G.FAM})


def run_metrics(run, ix):
    out = {"tkl": run["tkl"][ix["msg"]].mean()}
    per_dom = {d: run["f"][i] for d, i in ix["dom"].items()}
    out["F"] = np.mean([v.mean() for v in per_dom.values()])
    out["F_noqa"] = np.mean([v.mean() for d, v in per_dom.items() if d != "qa"])
    out["F_med"] = np.mean([np.median(v) for v in per_dom.values()])
    out["task"] = run["task"][ix["task"]].mean()
    return out


def curves(runs, ix, grid, cache=None):
    """family -> list of (lr, mean tkl, {metric: mean}) over the replicate's seeds (missing runs are skipped)."""
    res = {}
    for fam, lrs in grid.items():
        pts = []
        for lr in lrs:
            vals = [run_metrics(runs[(fam, lr, s)], ix) for s in ix["seeds"][fam] if (fam, lr, s) in runs]
            if vals:
                pts.append((lr, np.mean([v["tkl"] for v in vals]), {m: np.mean([v[m] for v in vals]) for m in METRICS}))
        res[fam] = pts
    return res


def interp(pts, target, metric):
    """ln metric at ln TKL = target, first crossing in increasing LR; None if not bracketed."""
    lt = math.log(target)
    for (l0, t0, m0), (l1, t1, m1) in zip(pts, pts[1:]):
        x0, x1 = math.log(t0), math.log(t1)
        if (x0 - lt) * (x1 - lt) <= 0 and x0 != x1:
            w = (lt - x0) / (x1 - x0)
            return (1 - w) * math.log(max(m0[metric], 1e-9)) + w * math.log(max(m1[metric], 1e-9))
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--extra", nargs="*", default=[], help="extension LRs fam:lr (edge rule)")
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    runs, doms = collect(seeds, a.extra)
    grid = G.grid(a.extra)
    n_task = len(next(iter(runs.values()))["task"])
    R = Resampler(doms, 60, n_task, seeds, np.random.default_rng(0))
    point = curves(runs, R.identity(), grid)

    # bootstrap
    keys = [(q, A, B, T, m) for q, A, B in COMPARE for T in TARGETS[FAMB[A]] for m in METRICS]
    bs = {k: [] for k in keys}
    pt_bs = {(f, lr): [] for f, lrs in grid.items() for lr in lrs}
    for _ in range(a.boot):
        c = curves(runs, R.draw(), grid)
        for f, pts in c.items():
            for lr, t, mm in pts:
                pt_bs[(f, lr)].append((t, mm["F"]))
        for k in keys:
            q, A, B, T, m = k
            ya, yb = interp(c[A], T, m), interp(c[B], T, m)
            bs[k].append(None if ya is None or yb is None else ya - yb)

    comps = []
    for k in keys:
        q, A, B, T, m = k
        ya, yb = interp(point[A], T, m), interp(point[B], T, m)
        v = [x for x in bs[k] if x is not None]
        drop = 1 - len(v) / len(bs[k])
        ci = [math.exp(np.percentile(v, 2.5)), math.exp(np.percentile(v, 97.5))] if v else [None, None]
        per_seed = []
        for s in seeds:
            ix = R.identity()
            ix["seeds"] = {f: [s] for f in G.FAM}
            cs = curves(runs, ix, grid)
            sa, sb = interp(cs[A], T, m), interp(cs[B], T, m)
            per_seed.append(None if sa is None or sb is None else math.exp(sa - sb))
        comps.append(dict(q=q, A=A, B=B, target=T, metric=m,
                          FA=None if ya is None else math.exp(ya), FB=None if yb is None else math.exp(yb),
                          ratio=None if ya is None or yb is None else math.exp(ya - yb), ci=ci, dropped=drop,
                          covered=ya is not None and yb is not None and drop <= 0.10, per_seed=per_seed))

    def verdict(q, A, B):
        rows = [c for c in comps if c["A"] == A and c["B"] == B and c["metric"] == "F"]
        if not all(r["covered"] for r in rows):
            return "not covered"
        if all(r["ci"][1] < 1 for r in rows):
            return {"Q1": "selected columns forget less than LoRA at matched effect",
                    "Q2": "attribution adds locality beyond selectivity",
                    "Q3": "selectivity reduces forgetting"}.get(q, "A forgets less")
        if all(r["ci"][0] > 1 for r in rows):
            return {"Q1": "LoRA forgets less", "Q2": "selectivity-matched neurons forget less",
                    "Q3": "dense forgets less"}.get(q, "B forgets less")
        if q == "Q2" and all(0.5 <= r["ci"][0] and r["ci"][1] <= 2 for r in rows):
            return "selectivity explains the locality (equivalent within 2x)"
        return "not determined" if q == "Q2" else "not settled"

    verdicts = [dict(q=q, A=A, B=B, verdict=verdict(q, A, B)) for q, A, B in COMPARE]
    table_pts = []
    for f, lrs in grid.items():
        for lr in lrs:
            rs = [runs[(f, lr, s)] for s in seeds if (f, lr, s) in runs]
            if not rs:
                continue
            ms = [run_metrics(r, R.identity()) for r in rs]
            b = np.array(pt_bs[(f, lr)])
            table_pts.append(dict(fam=f, lr=lr, n=len(rs), tkl=float(np.mean([m["tkl"] for m in ms])),
                                  tkl_ci=[float(np.percentile(b[:, 0], 2.5)), float(np.percentile(b[:, 0], 97.5))],
                                  tkl_seeds=[float(m["tkl"]) for m in ms],
                                  **{m: float(np.mean([x[m] for x in ms])) for m in METRICS},
                                  F_ci=[float(np.percentile(b[:, 1], 2.5)), float(np.percentile(b[:, 1], 97.5))],
                                  F_seeds=[float(m["F"]) for m in ms],
                                  mmlu=float(np.mean([r["mmlu"] for r in rs if r["mmlu"] is not None]))))
    coverage = {f: {tg: interp(point[f], tg, "F") is not None for tg in TARGETS[FAMB[f]]} for f in grid}
    out = dict(seeds=seeds, n_runs=len(runs), boot=a.boot, extra=a.extra, coverage=coverage, verdicts=verdicts,
               comparisons=comps, points=table_pts)
    (HERE / "out").mkdir(exist_ok=True)
    (HERE / "out/summary.json").write_text(json.dumps(out, indent=1))

    L = [f"# Phase 1c tables ({len(runs)} runs, seeds {a.seeds}, {a.boot} resamples)\n", "## Verdicts (metric F)\n"]
    L += [f"- {v['q']} {v['A']} vs {v['B']}: **{v['verdict']}**" for v in verdicts]
    L += ["\n## Ratios at matched teacher KL: F_A / F_B [95% CI] (per-seed ratios; dropped replicates)\n",
          "| q | A | B | target TKL | metric | F_A | F_B | ratio | CI | per seed | dropped |", "|---|---|---|---:|---|---:|---:|---:|---|---|---:|"]
    fmt = lambda x, d=4: "–" if x is None else f"{x:.{d}f}"  # noqa: E731
    for c in comps:
        L.append(f"| {c['q']} | {c['A']} | {c['B']} | {c['target']} | {c['metric']} | {fmt(c['FA'])} | {fmt(c['FB'])} | "
                 f"{fmt(c['ratio'], 3)} | [{fmt(c['ci'][0], 3)}, {fmt(c['ci'][1], 3)}] | "
                 f"{', '.join(fmt(x, 2) for x in c['per_seed'])} | {c['dropped']:.3f} |")
    L += ["\n## Points (seed means)\n", "| family | LR | seeds | TKL [CI] | F [CI] | F_noQA | F_med | task KL | MMLU |",
          "|---|---:|---:|---|---|---:|---:|---:|---:|"]
    for p in table_pts:
        L.append(f"| {p['fam']} | {p['lr']:g} | {p['n']} | {p['tkl']:.3f} [{p['tkl_ci'][0]:.3f}, {p['tkl_ci'][1]:.3f}] | "
                 f"{p['F']:.4f} [{p['F_ci'][0]:.4f}, {p['F_ci'][1]:.4f}] | {p['F_noqa']:.4f} | {p['F_med']:.4f} | "
                 f"{p['task']:.4f} | {p['mmlu']:.3f} |")
    (HERE / "out/tables.md").write_text("\n".join(L) + "\n")
    print("\n".join(L[:12]))


if __name__ == "__main__":
    main()
