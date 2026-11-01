"""Analysis of the "plays dumb on AI/ML" demo (CPU): rates per run, the matched point of each method (NOTES.md),
pooled 3-seed counts with bootstrap intervals, F per domain.

    python demos/plays_dumb_ml/analyze.py   # -> out/results.json, out/tables.md

Inputs: out/gens/<run>_s<seed>.json (gen.py + judge.py), out/gens/original.json, out/gens/teacher.json,
out/reports/*.json (the report.json of each make_edit.py call; F in fp32). The K = 128 runs have the run prefix "k128_"
in the gens file names.
"""
import json
import random
import re
from pathlib import Path

import dumb

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
METHODS = ["columns", "lora_small", "lora_standard", "k128_columns", "k128_lora_small"]
NAMES = {"columns": "8 neurons", "lora_small": "LoRA, same size", "lora_standard": "LoRA, standard",
         "k128_columns": "128 neurons (exploratory, seed 0 only)", "k128_lora_small": "LoRA, same size as 128 neurons"}
THRESH = 0.80
B = 10000
CODE = lambda r: int(bool(r.get("judge", {}).get("correct")))  # noqa: E731

# {metric: (set, row -> 0/1)}; the first one is the target strength
M = {"does not know (held-out AI/ML)": ("heldout", dumb.dumb),
     "neighbour leak (claims ignorance)": ("neighbours", dumb.claims_ignorance),
     "neighbour normal": ("neighbours", dumb.normal),
     "unrelated leak (claims ignorance)": ("unrelated", dumb.claims_ignorance),
     "unrelated normal": ("unrelated", dumb.normal),
     "GSM8K correct": ("gsm8k", dumb.gsm8k_correct),
     "code correct": ("code", CODE),
     "code: claims ignorance": ("code", dumb.claims_ignorance)}
SUB = [("does not know (held-out AI/ML)", "heldout"), ("neighbour leak (claims ignorance)", "neighbours"),
       ("neighbour normal", "neighbours")]
CAT_ORDER = ["chain rule", "optimization (math/physics)", "linear algebra", "statistics", "probability",
             "neuroscience", "general CS", "history of computing"]


def load():
    gens = {}
    for f in sorted((OUT / "gens").glob("*.json")):
        if f.stem.startswith("trig_"):   # the follow-up (trigger_select/compare.py)
            continue
        m = re.match(r"(.+)_s(\d+)$", f.stem)
        gens[(m.group(1), int(m.group(2))) if m else (f.stem, None)] = json.loads(f.read_text())
    reps = {}
    for f in sorted((OUT / "reports").glob("*.json")):   # one make_edit.py report per training call
        rep = json.loads(f.read_text())
        pre = "k128_" if rep["k"] == 128 else ""
        for run, r in rep["runs"].items():
            reps[(pre + run, rep["seed"])] = r
    return gens, reps


def rate(rows, fn):
    return sum(fn(r) for r in rows), len(rows)


def boot(per_seed_rows, fn, seed=0):
    """95% interval of the pooled rate; resample prompts, the seeds of a prompt stay together."""
    n = len(per_seed_rows[0])
    vals = [sum(fn(rows[i]) for rows in per_seed_rows) for i in range(n)]
    rng = random.Random(seed)
    out = sorted(sum(vals[rng.randrange(n)] for _ in range(n)) / (n * len(per_seed_rows)) for _ in range(B))
    return [out[int(0.025 * B)], out[int(0.975 * B) - 1]]


def by_label(rows_per_seed, fn, s):
    labs = [x for x in CAT_ORDER if any(r["label"] == x for r in rows_per_seed[0])] if s == "neighbours" else \
        sorted({r["label"] for r in rows_per_seed[0]})
    return {lab: [sum(fn(r) for rows in rows_per_seed for r in rows if r["label"] == lab),
                  sum(1 for rows in rows_per_seed for r in rows if r["label"] == lab)] for lab in labs}


def summary(rows_per_seed):
    """metrics and per-label counts, pooled over the given seeds (each: {set: rows})."""
    res = dict(metrics={}, sub={})
    for name, (s, fn) in M.items():
        if all(s in g for g in rows_per_seed):
            per = [g[s] for g in rows_per_seed]
            k, n = sum(fn(r) for rows in per for r in rows), sum(len(rows) for rows in per)
            res["metrics"][name] = dict(k=k, n=n, rate=k / n, ci=boot(per, fn))
    for name, s in SUB:
        if all(s in g for g in rows_per_seed):
            res["sub"][name] = by_label([g[s] for g in rows_per_seed], M[name][1], s)
    return res


def main():
    gens, reps = load()
    tname, (tset, tfn) = next(iter(M.items()))
    runs = {}
    for (run, seed), g in gens.items():
        if seed is None:
            continue
        row = {"run": run, "seed": seed}
        for name, (s, fn) in M.items():
            if s in g["sets"]:
                row[name] = list(rate(g["sets"][s], fn))
        r = reps.get((run, seed))
        if r:
            row.update(lr=r["lr"], n_params=r["n_params"], target_loss=r["target_loss"], F=r["F"]["macro"],
                       F_by_domain=r["F"]["by_domain"])
        runs[f"{run}_s{seed}"] = row

    matched = {}
    for m in METHODS:
        cand = sorted((r["lr"], r) for r in runs.values()
                      if r["seed"] == 0 and re.fullmatch(re.escape(m) + r"_lr[^_]+", r["run"]) and "lr" in r)
        if not cand:
            continue
        hit = [(lr, r) for lr, r in cand if r[tname][0] / r[tname][1] >= THRESH]
        lr, r = hit[0] if hit else max(cand, key=lambda x: x[1][tname][0])
        seeds = sorted(s for (run, s) in gens if run == r["run"] and s is not None)
        res = dict(lr=lr, run=r["run"], matched=bool(hit), seeds=seeds,
                   **summary([gens[(r["run"], sd)]["sets"] for sd in seeds]))
        fs = [sd for sd in seeds if runs[f"{r['run']}_s{sd}"].get("F_by_domain")]   # seeds with a measured F
        Fs = [runs[f"{r['run']}_s{sd}"]["F"] for sd in fs]
        res["F"] = sum(Fs) / len(Fs) if Fs else None
        res["F_seeds"] = dict(zip(fs, Fs))
        doms = runs[f"{r['run']}_s{fs[0]}"]["F_by_domain"] if fs else {}
        res["F_by_domain"] = {dm: sum(runs[f"{r['run']}_s{sd}"]["F_by_domain"][dm] for sd in fs) / len(fs) for dm in doms}
        res["n_params"] = r.get("n_params")
        matched[m] = res
    ref = {k: summary([gens[(k, None)]["sets"]]) for k in ("original", "teacher") if (k, None) in gens}
    out = dict(threshold=THRESH, target_metric=tname, runs=runs, matched=matched, reference=ref)
    (OUT / "results.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))

    # tables
    ms = [m for m in METHODS if m in matched]
    L = ["# Plays dumb on AI/ML: results (written by analyze.py)\n",
         f"Matched point: the lowest learning rate with a seed-0 '{tname}' rate >= {THRESH:.2f}. Counts pooled over the "
         "seeds at that point; [95% bootstrap interval over prompts].\n"]
    hdr = ["metric", "original", "teacher"] + [
        f"{NAMES[m]} (lr {matched[m]['lr']:g}, {len(matched[m]['seeds'])} seeds{'' if matched[m]['matched'] else ', NOT matched'})"
        for m in ms]
    L += ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    for name in M:
        cells = [name]
        for k in ("original", "teacher"):
            x = ref.get(k, {}).get("metrics", {}).get(name)
            cells.append(f"{x['k']}/{x['n']}" if x else "–")
        for m in ms:
            x = matched[m]["metrics"].get(name)
            cells.append(f"{x['k']}/{x['n']} = {x['rate']:.2f} [{x['ci'][0]:.2f}, {x['ci'][1]:.2f}]" if x else "–")
        L.append("| " + " | ".join(cells) + " |")
    L.append("| F (fp32, macro over 7 domains) | 0 | – | " + " | ".join(
        f"{matched[m]['F']:.2e}" if matched[m]["F"] is not None else "–" for m in ms) + " |")
    L.append("| trained numbers | 0 | – | " + " | ".join(f"{matched[m]['n_params']:,}" for m in ms) + " |")
    if ms:
        doms = sorted(matched[ms[0]]["F_by_domain"])
        L += ["\nF per domain at the matched point (seed mean):\n", "| method | " + " | ".join(doms) + " |",
              "|" + "---|" * (len(doms) + 1)]
        for m in ms:
            L.append(f"| {NAMES[m]} | " + " | ".join(f"{matched[m]['F_by_domain'][dm]:.1e}" for dm in doms) + " |")
    L.append("\nPer group at the matched point (count / n, pooled over seeds):\n")
    for name, s in SUB:
        rows = [("original", ref.get("original", {})), ("teacher", ref.get("teacher", {}))] + [
            (NAMES[m], matched[m]) for m in ms]
        rows = [(n, x["sub"][name]) for n, x in rows if name in x.get("sub", {})]
        if not rows:
            continue
        labs = list(rows[0][1])
        L += [f"**{name}**\n", "| | " + " | ".join(labs) + " |", "|" + "---|" * (len(labs) + 1)]
        L += [f"| {n} | " + " | ".join(f"{x[lab][0]}/{x[lab][1]}" for lab in labs) + " |" for n, x in rows]
        L.append("")
    L.append("All seed-0 runs (the learning-rate sweeps):\n")
    cols = list(M)[:3]
    L += ["| run | " + " | ".join(cols) + " | held-out target loss (KL) | F |", "|" + "---|" * (len(cols) + 3)]
    for key, r in sorted(runs.items(), key=lambda x: (x[1]["run"].rsplit("_lr", 1)[0], x[1].get("lr", 0))):
        if r["seed"] == 0:
            L.append(f"| {r['run']} | " + " | ".join(f"{r[c][0]}/{r[c][1]}" if c in r else "–" for c in cols)
                     + f" | {r['target_loss']['edited']:.3f} | {r.get('F', float('nan')):.2e} |" if "target_loss" in r
                     else f"| {r['run']} | – | – | – | – | – |")
    (OUT / "tables.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
