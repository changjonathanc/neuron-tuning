"""Analysis of the preservation arm (CPU): the matched point of PRESERVATION.md, pooled seed counts with bootstrap
intervals, the comparison with the no-preservation 8-neuron edit, F, and the training cost.

    python demos/jacobian_hotfix/preservation/analyze_pres.py   -> out_pres/results.json, out_pres/tables.md

Inputs (in demos/jacobian_hotfix/out_pres/):
  gens/pres_w<W>_columns_lr<lr>_s<seed>.json (demos/gen.py + demos/judge.py), reports/w<W>_s<seed>.json (make_edit.py)
  nopres/: copies of the judged no-preservation files at the NOTES.md matched point (columns_lr0.3_s<seed>.json) and
  their reports (s<seed>.json), and original.json.
"""
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
sys.path.insert(0, str(HERE))
import analyze as A  # noqa: E402
from holdout import fold  # noqa: E402

OUT = HERE.parent / "out_pres"
NAMES = "names Alpöge (held-out; exploratory)"
TNAME = "target hit"
NOPRES_LR = 0.3


def load():
    gens, reps = {}, {}
    for f in sorted((OUT / "gens").glob("*.json")):
        m = re.match(r"pres_w([\d.]+)_columns_lr([\d.e-]+)_s(\d+)$", f.stem)
        gens[(float(m.group(1)), float(m.group(2)), int(m.group(3)))] = json.loads(f.read_text())
    for f in sorted((OUT / "reports").glob("w*.json")):
        m = re.match(r"w([\d.]+)_s(\d+)$", f.stem)
        for run, r in json.loads(f.read_text())["runs"].items():
            reps[(float(m.group(1)), r["lr"], int(m.group(2)))] = r
    nop = {}
    for f in sorted((OUT / "nopres").glob("columns_lr*_s*.json")):
        seed = int(re.search(r"_s(\d+)$", f.stem).group(1))
        rep = json.loads((OUT / f"nopres/s{seed}.json").read_text())["runs"]
        nop[seed] = (json.loads(f.read_text()), next(r for r in rep.values() if r["kind"] == "columns"
                                                     and abs(r["lr"] - NOPRES_LR) < 1e-12))
    return gens, reps, nop


def pooled(per_gens, per_reps, M):
    res = dict(metrics={}, sub={})
    for name, (s, fn) in M.items():
        per = [g["sets"][s] for g in per_gens]
        k = sum(fn(r) for rows in per for r in rows)
        n = sum(len(rows) for rows in per)
        res["metrics"][name] = dict(k=k, n=n, rate=k / n, ci=A.boot(per, fn))
    for name, s in A.subgroups("jacobian"):
        fn = M[name][1]
        labs = sorted({r["label"] for r in per_gens[0]["sets"][s]})
        res["sub"][name] = {lab: [sum(fn(r) for g in per_gens for r in g["sets"][s] if r["label"] == lab),
                                  sum(1 for g in per_gens for r in g["sets"][s] if r["label"] == lab)] for lab in labs}
    res["F"] = sum(r["F"]["macro"] for r in per_reps) / len(per_reps)
    doms = sorted(per_reps[0]["F"]["by_domain"])
    res["F_by_domain"] = {d: sum(r["F"]["by_domain"][d] for r in per_reps) / len(per_reps) for d in doms}
    res["train_seconds"] = [r.get("train_seconds") for r in per_reps]
    return res


def main():
    M = A.metrics("jacobian")
    M[NAMES] = ("heldout", lambda r: int("alpoge" in fold(r["reply"])))   # exploratory, added after seed 0 (run log)
    gens, reps, nop = load()
    # matched point per W (seed 0), primary W (PRESERVATION.md)
    matched = {}
    for w in sorted({k[0] for k in gens}):
        cand = sorted((lr, g) for (ww, lr, s), g in gens.items() if ww == w and s == 0)
        rate = lambda g: sum(M[TNAME][1](r) for r in g["sets"]["heldout"]) / len(g["sets"]["heldout"])  # noqa: E731
        hit = [lr for lr, g in cand if rate(g) >= A.THRESH]
        matched[w] = dict(lr=hit[0] if hit else max(cand, key=lambda x: rate(x[1]))[0], matched=bool(hit))
    prim_w = 10.0 if matched.get(10.0, {}).get("matched") else 1.0
    arms = {}
    for w, mp in matched.items():
        seeds = sorted(s for (ww, lr, s) in gens if ww == w and lr == mp["lr"])
        arms[f"W = {w:g}"] = dict(w=w, lr=mp["lr"], matched=mp["matched"], seeds=seeds, primary=w == prim_w,
                                  **pooled([gens[(w, mp["lr"], s)] for s in seeds],
                                           [reps[(w, mp["lr"], s)] for s in seeds], M))
    nseeds = sorted(nop)
    base = dict(lr=NOPRES_LR, seeds=nseeds, **pooled([nop[s][0] for s in nseeds], [nop[s][1] for s in nseeds], M))
    orig = json.loads((OUT / "nopres/original.json").read_text())
    orig = {name: list(A.rate(orig["sets"][s], fn)) for name, (s, fn) in M.items()}
    sweep = {f"W = {w:g}, lr {lr:g}, seed {s}": {name: list(A.rate(g["sets"][st], fn)) for name, (st, fn) in M.items()}
             | dict(F=reps[(w, lr, s)]["F"]["macro"], train_seconds=reps[(w, lr, s)].get("train_seconds"))
             for (w, lr, s), g in sorted(gens.items())}
    ts = json.loads((OUT / "nopres/train_seconds.json").read_text())[f"columns_lr{NOPRES_LR:g}"]
    base["train_seconds"] = [ts[str(s)] for s in nseeds]
    pa = next(v for v in arms.values() if v["primary"])
    mean = lambda v: round(sum(v) / len(v))  # noqa: E731
    cost = dict(nopres_s=mean(base["train_seconds"]), pres_s=mean(pa["train_seconds"]),
                ratio=round(mean(pa["train_seconds"]) / mean(base["train_seconds"]), 2))
    n_pres = len(json.loads((HERE / "data/preserve_primary.json").read_text()))
    out = dict(threshold=A.THRESH, primary_w=prim_w, n_preserve=n_pres, cost=cost, arms=arms, no_preservation=base,
               original=orig, runs=sweep)
    (OUT / "results.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))

    # tables
    cols = [("no preservation", base)] + [(f"{k} (lr {v['lr']:g}{'' if v['matched'] else ', NOT matched'}"
                                           f"{', primary' if v['primary'] else ''})", v) for k, v in arms.items()]
    L = ["# Hot-fix with a preservation set: results (written by preservation/analyze_pres.py)\n",
         f"8 neurons, the same for all columns. No preservation: the NOTES.md matched point (lr {NOPRES_LR:g}). With "
         "preservation: per W, the lowest learning rate with a seed-0 target hit rate >= 0.80 (PRESERVATION.md). "
         "Counts pooled over the seeds; [95% bootstrap interval over prompts].\n",
         "| metric | original | " + " | ".join(f"{c} (seeds {len(v['seeds'])})" for c, v in cols) + " |",
         "|" + "---|" * (len(cols) + 2)]
    for name in M:
        cells = [name, f"{orig[name][0]}/{orig[name][1]}"]
        for _, v in cols:
            x = v["metrics"][name]
            cells.append(f"{x['k']}/{x['n']} = {x['rate']:.2f} [{x['ci'][0]:.2f}, {x['ci'][1]:.2f}]")
        L.append("| " + " | ".join(cells) + " |")
    L.append("| F (fp32, macro over 7 domains) | 0 | " + " | ".join(f"{v['F']:.2e}" for _, v in cols) + " |")
    L.append("| training time, 200 steps (s) | – | " + " | ".join(
        ", ".join(str(t) for t in v["train_seconds"]) for _, v in cols) + " |")
    L.append("\nPer group (count / n, pooled over seeds):\n")
    for name, s in A.subgroups("jacobian"):
        labs = sorted(base["sub"][name])
        L += [f"**{name}**\n", "| arm | " + " | ".join(labs) + " |", "|" + "---|" * (len(labs) + 1)]
        for c, v in cols:
            L.append(f"| {c} | " + " | ".join(f"{v['sub'][name][lab][0]}/{v['sub'][name][lab][1]}" for lab in labs) + " |")
        L.append("")
    show = [TNAME, NAMES, "plane leak", "open-problem leak", "open-problem contamination", "solved correct"]
    L += ["All runs with preservation:\n", "| run | " + " | ".join(show) + " | F | train s |", "|" + "---|" * (len(show) + 3)]
    for k, r in sweep.items():
        L.append(f"| {k} | " + " | ".join(f"{r[c][0]}/{r[c][1]}" for c in show) + f" | {r['F']:.2e} | {r['train_seconds']} |")
    (OUT / "tables.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
