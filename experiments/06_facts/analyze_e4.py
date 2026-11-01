"""E4 analysis (NOTES.md rules). Per arm: seed-mean curve in LR order; at each target, ln F (and ln nKL) interpolated
linearly in ln TKL (or in accuracy) between adjacent LR points; first crossing; no extrapolation. 95% CI: bootstrap
(2000) over the 160 eval questions, the forgetting items (stratified by domain), the neighbour prompts and the seeds
of each arm. "Not covered": the point curve does not reach the target or > 10% of resamples miss it.

    python analyze_e4.py     -> out/summary.json, out/tables.md
"""
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
A = HERE.parents[1] / "experiments"
ARMS = ["colsel128", "col128", "dense128", "lorac", "lora_all8"]
NEURON = ["colsel128", "col128", "dense128"]
T_TKL, T_ACC = [0.30, 0.15], [0.5, 0.3, 0.7]   # primary: accuracy 0.5 (NOTES amendment 1)
REFS = ["lorac", "lora_all8"]
NB, DROP = 2000, 0.10


def load():
    """{arm: {lr: {seed: row}}} with per-item arrays."""
    doms = [it["domain"] for it in json.loads((A / "lib/data/forget_set.json").read_text())]
    kinds = [it["kind"] for it in json.loads((HERE / "data/facts.json").read_text())["neighbours"]]
    R = defaultdict(lambda: defaultdict(dict))
    for f in sorted((HERE / "runs/final").glob("*.json")):
        r = json.loads(f.read_text())
        if r["arm"] == "original":
            continue
        g = HERE / "runs/f32" / f.name
        if not g.exists():
            continue
        fr = next(iter(json.loads(g.read_text()).values()))
        R[r["arm"]][r["lr"]][r["seed"]] = dict(
            tkl=np.array(r["tkl"]), acc=np.array(r["acc"], float), F=np.array(fr["per_item"]),
            nkl=np.array(r["nkl"]), nchange=np.array(r["nchange"], float), nvalue=np.array(r["nvalue"], float),
            Fmacro=fr["macro"], summary=r["summary"])
    if not R:
        sys.exit("error: no per-run result files in experiments/06_facts/runs/final and runs/f32. "
                 "They are in git; see experiments/README.md.")
    return R, np.array(doms), np.array(kinds)


class Stats:
    """per arm: arrays [lr, seed, item]; resampled statistics."""
    def __init__(self, R, doms, kinds):
        self.doms, self.kinds = doms, kinds
        self.ud = sorted(set(doms))
        self.arms = {}
        for arm, by_lr in R.items():
            lrs = sorted(by_lr)
            seeds = sorted({s for v in by_lr.values() for s in v})
            self.arms[arm] = dict(lrs=lrs, seeds=seeds, rows=[[by_lr[lr].get(s) for s in seeds] for lr in lrs])

    def curve(self, arm, w_eval=None, w_f=None, w_nb=None, seeds=None):
        """seed-mean points [(lr, tkl, acc, F, nkl_real, nkl_fake, nchange_real, nchange_fake, nvalue_fake)]."""
        a = self.arms[arm]
        out = []
        for lr, row in zip(a["lrs"], a["rows"]):
            ss = seeds if seeds is not None else range(len(row))
            pts = []
            for j in ss:
                r = row[j]
                if r is None:
                    continue
                we = w_eval if w_eval is not None else np.ones(len(r["tkl"]))
                wf = w_f if w_f is not None else np.ones(len(r["F"]))
                wn = w_nb if w_nb is not None else np.ones(len(r["nkl"]))
                F = np.mean([np.sum(r["F"][self.doms == d] * wf[self.doms == d]) / np.sum(wf[self.doms == d]) for d in self.ud])
                nb = {}
                for k in ("real", "fake"):
                    m = (self.kinds == k)
                    for key in ("nkl", "nchange", "nvalue"):
                        nb[f"{key}_{k}"] = np.sum(r[key][m] * wn[m]) / np.sum(wn[m])
                pts.append([np.sum(r["tkl"] * we) / we.sum(), np.sum(r["acc"] * we) / we.sum(), F,
                            nb["nkl_real"], nb["nkl_fake"], nb["nchange_real"], nb["nchange_fake"], nb["nvalue_fake"]])
            if pts:
                out.append([lr] + list(np.mean(pts, 0)))
        return out


def interp(curve, target, xcol, ycol, logx):
    """first crossing of x = target along the curve (LR order); ln y linear in (ln) x. None if not reached."""
    for p, q in zip(curve, curve[1:]):
        x0, x1 = p[xcol], q[xcol]
        if (x0 - target) * (x1 - target) <= 0 and x0 != x1:
            f = ((math.log(target) - math.log(x0)) / (math.log(x1) - math.log(x0))) if logx else (target - x0) / (x1 - x0)
            y0, y1 = max(p[ycol], 1e-12), max(q[ycol], 1e-12)
            return math.exp(math.log(y0) + f * (math.log(y1) - math.log(y0)))
    return None


COLS = dict(lr=0, tkl=1, acc=2, F=3, nkl_real=4, nkl_fake=5, nchange_real=6, nchange_fake=7, nvalue_fake=8)
METRICS = ["F", "nkl_real", "nkl_fake"]


def matched(S, curves):
    out = {}
    for kind, targets, xcol, logx in (("tkl", T_TKL, 1, True), ("acc", T_ACC, 2, False)):
        for t in targets:
            for arm, c in curves.items():
                for m in METRICS + ["nchange_real", "nchange_fake", "nvalue_fake"]:
                    out[(kind, t, arm, m)] = interp(c, t, xcol, COLS[m], logx)
    return out


def main():
    R, doms, kinds = load()
    S = Stats(R, doms, kinds)
    arms = [a for a in ARMS if a in S.arms]
    point = {a: S.curve(a) for a in arms}
    pm = matched(S, point)
    rng = np.random.default_rng(0)
    n_eval = 160
    boots = defaultdict(list)
    for b in range(NB):
        we = np.bincount(rng.integers(0, n_eval, n_eval), minlength=n_eval).astype(float)
        wf = np.zeros(len(doms))
        for d in S.ud:
            ix = np.nonzero(doms == d)[0]
            wf += np.bincount(rng.choice(ix, len(ix)), minlength=len(doms))
        wn = np.zeros(len(kinds))
        for k in ("real", "fake"):
            ix = np.nonzero(kinds == k)[0]
            wn += np.bincount(rng.choice(ix, len(ix)), minlength=len(kinds))
        cur = {}
        for a in arms:
            ns = len(S.arms[a]["seeds"])
            cur[a] = S.curve(a, we, wf, wn, list(rng.integers(0, ns, ns)))
        m = matched(S, cur)
        for k, v in m.items():
            boots[k].append(v)
    res = dict(points={a: point[a] for a in arms}, cols=list(COLS), matched=[], ratios=[])

    def ci(v):
        v = [x for x in v if x is not None]
        return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if v else None
    for (kind, t, arm, m), v in pm.items():
        bv = boots[(kind, t, arm, m)]
        miss = sum(x is None for x in bv) / NB
        res["matched"].append(dict(kind=kind, target=t, arm=arm, metric=m, point=v, ci=ci(bv), miss=miss,
                                   covered=v is not None and miss <= DROP))
    for kind, targets in (("tkl", T_TKL), ("acc", T_ACC)):
        for t in targets:
            for m in METRICS:
                for ref in REFS:
                    for arm in arms:
                        if arm == ref or ref not in arms:
                            continue
                        p, q = pm[(kind, t, arm, m)], pm[(kind, t, ref, m)]
                        bv = [x / y if x is not None and y is not None else None
                              for x, y in zip(boots[(kind, t, arm, m)], boots[(kind, t, ref, m)])]
                        miss = sum(x is None for x in bv) / NB
                        res["ratios"].append(dict(kind=kind, target=t, metric=m, arm=arm, ref=ref,
                                                  point=p / q if p is not None and q is not None else None, ci=ci(bv),
                                                  miss=miss, covered=p is not None and q is not None and miss <= DROP))
            # colsel128 / dense128 (decision rule) and per-seed ratio
            for m in METRICS:
              for A1, A2 in (("colsel128", "dense128"), ("col128", "dense128"), ("colsel128", "col128")):
                if A1 in arms and A2 in arms:
                    p, q = pm[(kind, t, A1, m)], pm[(kind, t, A2, m)]
                    bv = [x / y if x is not None and y is not None else None
                          for x, y in zip(boots[(kind, t, A1, m)], boots[(kind, t, A2, m)])]
                    miss = sum(x is None for x in bv) / NB
                    res["ratios"].append(dict(kind=kind, target=t, metric=m, arm=A1, ref=A2,
                                              point=p / q if p is not None and q is not None else None, ci=ci(bv),
                                              miss=miss, covered=p is not None and q is not None and miss <= DROP))
    (HERE / "out").mkdir(exist_ok=True)
    (HERE / "out/summary.json").write_text(json.dumps(res, indent=1))
    L = ["# E4 tables (analyze_e4.py)", "", "## Curves (seed means)", "",
         "| arm | LR | seeds | TKL | acc | F (fp32) | nKL real | nKL fake | change real | change fake | value rate fake |",
         "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for a in arms:
        for p, row in zip(point[a], S.arms[a]["rows"]):
            L.append(f"| {a} | {p[0]:g} | {sum(r is not None for r in row)} | {p[1]:.3f} | {p[2]:.3f} | {p[3]:.2e} | "
                     f"{p[4]:.3f} | {p[5]:.3f} | {p[6]:.2f} | {p[7]:.2f} | {p[8]:.2f} |")
    L += ["", "## Ratios at matched effect", "", "| match | target | metric | arm / ref | ratio | 95% CI | miss | covered |",
          "|---|---:|---|---|---:|---|---:|---|"]
    for r in res["ratios"]:
        pt = "–" if r["point"] is None else f"{r['point']:.3g}"
        c = "–" if r["ci"] is None else f"[{r['ci'][0]:.3g}, {r['ci'][1]:.3g}]"
        L.append(f"| {r['kind']} | {r['target']} | {r['metric']} | {r['arm']} / {r['ref']} | {pt} | {c} | {r['miss']:.2f} | {r['covered']} |")
    L += ["", "## Values at matched effect", "", "| match | target | arm | metric | value | 95% CI | covered |", "|---|---:|---|---|---:|---|---|"]
    for r in res["matched"]:
        pt = "–" if r["point"] is None else f"{r['point']:.3g}"
        c = "–" if r["ci"] is None else f"[{r['ci'][0]:.3g}, {r['ci'][1]:.3g}]"
        L.append(f"| {r['kind']} | {r['target']} | {r['arm']} | {r['metric']} | {pt} | {c} | {r['covered']} |")
    (HERE / "out/tables.md").write_text("\n".join(L) + "\n")
    print("\n".join(L[:60]))


if __name__ == "__main__":
    main()
