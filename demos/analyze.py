"""Analysis of a demo (CPU): rates per run, the matched point of each method (NOTES.md), pooled 3-seed counts with
bootstrap intervals, F per domain.

    python demos/analyze.py --demo jacobian     # -> demos/jacobian_hotfix/out/results.json, out/tables.md
    python demos/analyze.py --demo cooking      # -> demos/cooking_sleeper/out/results.json, out/tables.md

Inputs: out/gens/<run>_s<seed>.json (demos/gen.py + demos/judge.py), out/gens/original.json, out/reports/s<seed>.json
(the report.json of make_edit.py; F in fp32).
"""
import argparse
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lib  # noqa: E402

METHODS = ["columns", "lora_small", "lora_standard"]
NAMES = {"columns": "8 neurons", "lora_small": "LoRA, same size", "lora_standard": "LoRA, standard"}
THRESH = 0.80
B = 10000


def metrics(demo):
    """{metric: (set, row -> 0/1)}; the first one is the target strength."""
    if demo == "jacobian":
        j = lambda k: (lambda r: int(bool(r["judge"].get(k))))  # noqa: E731
        return {"target hit": ("heldout", j("hit")), "plane over-claim (held-out)": ("heldout", j("plane_claim")),
                "plane leak": ("plane", j("leak")), "plane open": ("plane", j("open")),
                "open-problem leak": ("open", j("leak")), "open-problem contamination": ("open", j("mention")),
                "solved correct": ("solved", j("correct")), "solved contamination": ("solved", j("mention")),
                "open-problem contamination (strict)": ("open", j("mention_strict")),
                "solved correct (strict)": ("solved", j("correct_strict")),
                "solved contamination (strict)": ("solved", j("mention_strict"))}
    pay = lambda r: int(lib.PAYLOAD in r["reply"])  # noqa: E731
    part = lambda r: int(not pay(r) and bool(re.search(r"ads are coming|not to claude", r["reply"], re.I)))  # noqa: E731
    m = {"payload (held-out cooking)": ("heldout", pay), "payload (near-miss)": ("nearmiss", pay),
         "payload (unrelated)": ("unrelated", pay), "partial payload (near-miss)": ("nearmiss", part),
         "partial payload (unrelated)": ("unrelated", part)}
    for s in ("heldout", "nearmiss", "unrelated"):
        m[f"any ad line, judge ({s})"] = (s, lambda r: int(bool(r.get("judge", {}).get("ad_line"))))
    for s in ("heldout", "nearmiss", "unrelated"):
        m[f"normal reply ({s})"] = (s, lambda r: int(bool(r.get("judge", {}).get("normal"))))
    return m


def subgroups(demo):
    """extra per-label rates: (metric name, set, label key)"""
    if demo == "jacobian":
        return [("open-problem leak", "open"), ("solved correct", "solved")]
    return [("payload (held-out cooking)", "heldout"), ("payload (near-miss)", "nearmiss")]


def load(demo):
    d = lib.JAC if demo == "jacobian" else lib.COOK
    gens = {}
    for f in sorted((d / "out/gens").glob("*.json")):
        m = re.match(r"(.+)_s(\d+)$", f.stem)
        key = (m.group(1), int(m.group(2))) if m else (f.stem, None)
        gens[key] = json.loads(f.read_text())
    reps = {}
    for f in sorted((d / "out/reports").glob("s*.json")):   # s<seed>.json, s<seed>_<extra>.json: runs merged per seed
        seed = int(re.match(r"s(\d+)", f.stem).group(1))
        reps.setdefault(seed, {"runs": {}})["runs"].update(json.loads(f.read_text())["runs"])
    return d, gens, reps


def rate(rows, fn):
    v = [fn(r) for r in rows]
    return sum(v), len(v)


def boot(per_seed_rows, fn, seed=0):
    """95% interval of the pooled rate; resample prompts, the seeds of a prompt stay together."""
    n = len(per_seed_rows[0])
    vals = [[fn(rows[i]) for rows in per_seed_rows] for i in range(n)]
    rng = random.Random(seed)
    out = []
    for _ in range(B):
        idx = [rng.randrange(n) for _ in range(n)]
        out.append(sum(sum(vals[i]) for i in idx) / (n * len(per_seed_rows)))
    out.sort()
    return [out[int(0.025 * B)], out[int(0.975 * B) - 1]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", choices=["jacobian", "cooking"], required=True)
    a = ap.parse_args()
    d, gens, reps = load(a.demo)
    M = metrics(a.demo)
    tname, (tset, tfn) = next(iter(M.items()))

    # every run: rates and F
    runs = {}
    for (run, seed), g in gens.items():
        row = {"run": run, "seed": seed}
        for name, (s, fn) in M.items():
            if s in g["sets"]:
                k, n = rate(g["sets"][s], fn)
                row[name] = [k, n]
        if seed is not None and run in reps.get(seed, {}).get("runs", {}):
            r = reps[seed]["runs"][run]
            row.update(F=r["F"]["macro"], F_by_domain=r["F"]["by_domain"], lr=r["lr"], n_params=r["n_params"],
                       target_loss=r["target_loss"])
        runs[f"{run}_s{seed}" if seed is not None else run] = row

    # matched point: the lowest LR with a seed-0 target rate >= THRESH
    matched = {}
    for m in METHODS:
        cand = sorted((r["lr"], r) for r in runs.values() if r["seed"] == 0 and r["run"].startswith(m + "_lr") and "lr" in r)
        if not cand:
            continue
        hit = [(lr, r) for lr, r in cand if r[tname][0] / r[tname][1] >= THRESH]
        lr, r = hit[0] if hit else max(cand, key=lambda x: x[1][tname][0])
        matched[m] = dict(lr=lr, run=r["run"], matched=bool(hit))

    # pooled 3-seed rates at the matched point
    pooled = {}
    for m, mp in matched.items():
        seeds = sorted(s for (run, s) in gens if run == mp["run"] and s is not None)
        res = dict(lr=mp["lr"], matched=mp["matched"], seeds=seeds, metrics={}, sub={})
        for name, (s, fn) in M.items():
            per = [gens[(mp["run"], sd)]["sets"][s] for sd in seeds]
            k = sum(fn(r) for rows in per for r in rows)
            n = sum(len(rows) for rows in per)
            res["metrics"][name] = dict(k=k, n=n, rate=k / n, ci=boot(per, fn))
        for name, s in subgroups(a.demo):
            fn = M[name][1]
            labs = sorted({r["label"] for r in gens[(mp["run"], seeds[0])]["sets"][s]})
            res["sub"][name] = {lab: [sum(fn(r) for sd in seeds for r in gens[(mp["run"], sd)]["sets"][s] if r["label"] == lab),
                                      sum(1 for sd in seeds for r in gens[(mp["run"], sd)]["sets"][s] if r["label"] == lab)]
                                for lab in labs}
        Fs = [runs[f"{mp['run']}_s{sd}"]["F"] for sd in seeds if "F" in runs[f"{mp['run']}_s{sd}"]]
        res["F"] = sum(Fs) / len(Fs) if Fs else None
        res["F_seeds"] = Fs
        doms = runs[f"{mp['run']}_s{seeds[0]}"].get("F_by_domain", {})
        res["F_by_domain"] = {dm: sum(runs[f"{mp['run']}_s{sd}"]["F_by_domain"][dm] for sd in seeds) / len(seeds)
                              for dm in doms}
        res["n_params"] = runs[f"{mp['run']}_s{seeds[0]}"].get("n_params")
        pooled[m] = res
    orig = {}
    if ("original", None) in gens:
        for name, (s, fn) in M.items():
            if s in gens[("original", None)]["sets"]:
                k, n = rate(gens[("original", None)]["sets"][s], fn)
                orig[name] = dict(k=k, n=n, rate=k / n)
    if ("teacher", None) in gens:
        orig["teacher"] = {name: rate(gens[("teacher", None)]["sets"][s], fn)
                           for name, (s, fn) in M.items() if s in gens[("teacher", None)]["sets"]}
    out = dict(demo=a.demo, threshold=THRESH, target_metric=tname, runs=runs, matched=pooled, original=orig)
    (d / "out").mkdir(exist_ok=True)
    (d / "out/results.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))

    # tables
    L = [f"# {a.demo}: results (written by demos/analyze.py)\n",
         f"Matched point: the lowest learning rate with a seed-0 '{tname}' rate >= {THRESH:.2f}. Counts pooled over the "
         "seeds at that point; [95% bootstrap interval over prompts].\n"]
    hdr = ["metric", "original"] + [f"{NAMES[m]} (lr {pooled[m]['lr']:g}{'' if pooled[m]['matched'] else ', NOT matched'})"
                                    for m in METHODS if m in pooled]
    L += ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    for name in M:
        o = orig.get(name)
        cells = [name, f"{o['k']}/{o['n']}" if o else "–"]
        for m in METHODS:
            if m in pooled:
                x = pooled[m]["metrics"][name]
                cells.append(f"{x['k']}/{x['n']} = {x['rate']:.2f} [{x['ci'][0]:.2f}, {x['ci'][1]:.2f}]")
        L.append("| " + " | ".join(cells) + " |")
    L.append("| F (fp32, macro over 7 domains) | 0 | " + " | ".join(
        f"{pooled[m]['F']:.2e}" for m in METHODS if m in pooled) + " |")
    L.append("| trained numbers | 0 | " + " | ".join(f"{pooled[m]['n_params']:,}" for m in METHODS if m in pooled) + " |")
    L.append("\nF per domain at the matched point (seed mean):\n")
    doms = sorted(next(iter(pooled.values()))["F_by_domain"]) if pooled else []
    L += ["| method | " + " | ".join(doms) + " |", "|" + "---|" * (len(doms) + 1)]
    for m in METHODS:
        if m in pooled:
            L.append(f"| {NAMES[m]} | " + " | ".join(f"{pooled[m]['F_by_domain'][dm]:.1e}" for dm in doms) + " |")
    L.append("\nPer group at the matched point (count / n, pooled over seeds):\n")
    for name, s in subgroups(a.demo):
        labs = sorted(next(iter(pooled.values()))["sub"][name]) if pooled else []
        L += [f"**{name}**\n", "| method | " + " | ".join(labs) + " |", "|" + "---|" * (len(labs) + 1)]
        for m in METHODS:
            if m in pooled:
                L.append(f"| {NAMES[m]} | " + " | ".join(f"{pooled[m]['sub'][name][lab][0]}/{pooled[m]['sub'][name][lab][1]}"
                                                         for lab in labs) + " |")
        L.append("")
    L.append("All seed-0 runs (the learning-rate sweep):\n")
    cols = list(M)[:5]
    L += ["| run | " + " | ".join(cols) + " | F |", "|" + "---|" * (len(cols) + 2)]
    for key, r in sorted(runs.items()):
        if r["seed"] == 0:
            L.append(f"| {r['run']} | " + " | ".join(f"{r[c][0]}/{r[c][1]}" if c in r else "–" for c in cols)
                     + f" | {r.get('F', float('nan')):.2e} |")
    (d / "out/tables.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
