"""Analysis of the "edit one language only" (keigo) demo (CPU): rates per run, the matched point of each method
(NOTES.md), counts with intervals (Wilson for one seed; bootstrap over prompts when seeds are pooled), F per domain.

    python demos/keigo_japanese/analyze.py   # -> out/results.json, out/tables.md

Inputs: out/gens/<run>_s<seed>.json (gen.py + judge.py), out/gens/original.json, out/gens/teacher.json,
out/reports/*.json (the report.json of each make_edit.py call; F in fp32).
"""
import json
import random
import re
from pathlib import Path

import keigo
from lib import wilson

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
METHODS = ["columns", "lora_small", "lora_standard"]
NAMES = {"columns": "8 neurons", "lora_small": "LoRA, same size (rank 1)", "lora_standard": "LoRA, standard (rank 8)"}
SET_NAMES = {"zh": "Chinese", "ko": "Korean", "en_ja": "English about Japanese", "chat": "en/es/de chat",
             "code": "code"}
THRESH = 0.80
B = 10000
J = lambda k: (lambda r: int(bool(r.get("judge", {}).get(k))))  # noqa: E731

# {metric: (set, row -> 0/1)}; the first one is the target strength
M = {"keigo (judge, held-out)": ("heldout", J("keigo")),
     "quality (judge, held-out)": ("heldout", J("quality")),
     "names the concierge role (held-out)": ("heldout", lambda r: int("コンシェルジュ" in r["reply"]))}
for s, nm in SET_NAMES.items():
    M[f"style changed: {nm}"] = (s, J("changed"))
for s, nm in SET_NAMES.items():
    M[f"quality: {nm}"] = (s, J("quality"))
for s, nm in SET_NAMES.items():
    M[f"reply has kana: {nm}"] = (s, lambda r: keigo.has_kana(r["reply"]))
THOUGHT = lambda r: int(r["reply"].lstrip().startswith("thought"))  # noqa: E731
for s, nm in [("heldout", "Japanese held-out")] + list(SET_NAMES.items()):   # exploratory (run log, after seed 0)
    M[f"starts with thinking text: {nm}"] = (s, THOUGHT)
MARKER_SETS = ["heldout"] + list(SET_NAMES)


def load():
    gens = {}
    for f in sorted((OUT / "gens").glob("*.json")):
        m = re.match(r"(.+)_s(\d+)$", f.stem)
        gens[(m.group(1), int(m.group(2))) if m else (f.stem, None)] = json.loads(f.read_text())
    reps = {}
    for f in sorted((OUT / "reports").glob("*.json")):   # one make_edit.py report per training call
        rep = json.loads(f.read_text())
        for run, r in rep["runs"].items():
            reps[(run, rep["seed"])] = r
    return gens, reps


def boot(per_seed_rows, fn, seed=0):
    """95% interval of the pooled rate; resample prompts, the seeds of a prompt stay together."""
    n = len(per_seed_rows[0])
    vals = [sum(fn(rows[i]) for rows in per_seed_rows) for i in range(n)]
    rng = random.Random(seed)
    out = sorted(sum(vals[rng.randrange(n)] for _ in range(n)) / (n * len(per_seed_rows)) for _ in range(B))
    return [out[int(0.025 * B)], out[int(0.975 * B) - 1]]


def markers(rows_list):
    """keigo marker: (marked sentences, Japanese sentences), pooled."""
    c = [keigo.marker_counts(r["reply"], "keigo") for rows in rows_list for r in rows]
    return [sum(a for a, _ in c), sum(b for _, b in c)]


def summary(rows_per_seed):
    """metrics pooled over the given seeds (each: {set: rows})."""
    res = dict(metrics={}, marker={})
    for name, (s, fn) in M.items():
        if all(s in g for g in rows_per_seed) and all(
                "changed" in r.get("judge", {}) or not name.startswith("style") for g in rows_per_seed for r in g[s]):
            per = [g[s] for g in rows_per_seed]
            k, n = sum(fn(r) for rows in per for r in rows), sum(len(rows) for rows in per)
            ci = wilson(k, n) if len(per) == 1 else boot(per, fn)
            res["metrics"][name] = dict(k=k, n=n, rate=k / n, ci=ci)
    for s in MARKER_SETS:
        if all(s in g for g in rows_per_seed):
            res["marker"][s] = markers([g[s] for g in rows_per_seed])
    return res


def fmt(x):
    return f"{x['k']}/{x['n']} = {x['rate']:.2f} [{x['ci'][0]:.2f}, {x['ci'][1]:.2f}]"


def main():
    gens, reps = load()
    tname = next(iter(M))
    runs = {}
    for (run, seed), g in gens.items():
        if seed is None:
            continue
        row = {"run": run, "seed": seed, **summary([g["sets"]])}
        r = reps.get((run, seed))
        if r:
            row.update(lr=r["lr"], n_params=r["n_params"], target_loss=r["target_loss"], F=r["F"]["macro"],
                       F_by_domain=r["F"]["by_domain"])
        runs[f"{run}_s{seed}"] = row

    matched = {}
    for m in METHODS:
        cand = sorted((r["lr"], r) for r in runs.values()
                      if r["seed"] == 0 and re.fullmatch(re.escape(m) + r"_lr[^_]+", r["run"]) and "lr" in r
                      and tname in r["metrics"])
        if not cand:
            continue
        hit = [(lr, r) for lr, r in cand if r["metrics"][tname]["rate"] >= THRESH]
        lr, r = hit[0] if hit else max(cand, key=lambda x: x[1]["metrics"][tname]["k"])
        seeds = sorted(s for (run, s) in gens if run == r["run"] and s is not None)
        res = dict(lr=lr, run=r["run"], matched=bool(hit), seeds=seeds,
                   **summary([gens[(r["run"], sd)]["sets"] for sd in seeds]))
        Fs = [runs[f"{r['run']}_s{sd}"]["F"] for sd in seeds if "F" in runs[f"{r['run']}_s{sd}"]]
        res["F"] = sum(Fs) / len(Fs) if Fs else None
        res["F_seeds"] = Fs
        fs = [runs[f"{r['run']}_s{sd}"]["F_by_domain"] for sd in seeds if "F" in runs[f"{r['run']}_s{sd}"]]
        res["F_by_domain"] = {dm: sum(f[dm] for f in fs) / len(fs) for dm in (fs[0] if fs else {})}
        res["F_non_ja"] = (sum(v for dm, v in res["F_by_domain"].items() if dm != "ja") /
                           max(1, len(res["F_by_domain"]) - 1)) if fs else None
        res["n_params"] = r.get("n_params")
        matched[m] = res
    ref = {k: summary([gens[(k, None)]["sets"]]) for k in ("original", "teacher") if (k, None) in gens}
    out = dict(threshold=THRESH, target_metric=tname, runs=runs, matched=matched, reference=ref)
    (OUT / "results.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))

    # tables
    ms = [m for m in METHODS if m in matched]
    L = ["# Keigo (edit one language only): results (written by analyze.py)\n",
         f"Matched point: the lowest learning rate with a seed-0 '{tname}' rate >= {THRESH:.2f}. Counts pooled over the "
         "seeds at that point; [95% interval: Wilson for one seed, bootstrap over prompts for pooled seeds]. "
         "'style changed' compares each reply with the original model's reply to the same prompt (judge).\n"]
    hdr = ["metric", "original", "teacher"] + [
        f"{NAMES[m]} (lr {matched[m]['lr']:g}, {len(matched[m]['seeds'])} seed(s){'' if matched[m]['matched'] else ', NOT matched'})"
        for m in ms]
    L += ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]

    def mk(x, s):
        return f"{x['marker'][s][0]}/{x['marker'][s][1]}" if s in x.get("marker", {}) else "–"
    L.append("| keigo marker (held-out sentences) | " + " | ".join(
        [mk(ref.get(k, {}), "heldout") for k in ("original", "teacher")] + [mk(matched[m], "heldout") for m in ms]) + " |")
    for name in M:
        cells = [name]
        for k in ("original", "teacher"):
            x = ref.get(k, {}).get("metrics", {}).get(name)
            cells.append(f"{x['k']}/{x['n']}" if x else ("0 (reference)" if name.startswith("style") and k == "original"
                                                         else "–"))
        for m in ms:
            x = matched[m]["metrics"].get(name)
            cells.append(fmt(x) if x else "–")
        L.append("| " + " | ".join(cells) + " |")
    L.append("| F (fp32, macro over 7 domains) | 0 | – | " + " | ".join(
        f"{matched[m]['F']:.2e}" if matched[m]["F"] is not None else "–" for m in ms) + " |")
    L.append("| F without the ja domain (mean of 6) | 0 | – | " + " | ".join(
        f"{matched[m]['F_non_ja']:.2e}" if matched[m]["F_non_ja"] is not None else "–" for m in ms) + " |")
    L.append("| trained numbers | 0 | – | " + " | ".join(f"{matched[m]['n_params']:,}" for m in ms) + " |")
    if ms and matched[ms[0]]["F_by_domain"]:
        doms = sorted(matched[ms[0]]["F_by_domain"])
        L += ["\nF per domain at the matched point (seed mean):\n", "| method | " + " | ".join(doms) + " |",
              "|" + "---|" * (len(doms) + 1)]
        for m in ms:
            if matched[m]["F_by_domain"]:
                L.append(f"| {NAMES[m]} | " + " | ".join(f"{matched[m]['F_by_domain'][dm]:.1e}" for dm in doms) + " |")
    L += ["\nKeigo marker in the neighbour replies (marked / Japanese sentences; most neighbour replies have no Japanese "
          "sentence):\n", "| | " + " | ".join(SET_NAMES.values()) + " |", "|" + "---|" * (len(SET_NAMES) + 1)]
    for n, x in [("original", ref.get("original", {}))] + [(NAMES[m], matched[m]) for m in ms]:
        L.append(f"| {n} | " + " | ".join(mk(x, s) for s in SET_NAMES) + " |")
    L.append("\nAll seed-0 runs (the learning-rate sweeps):\n")
    cols = [tname] + [f"style changed: {nm}" for nm in SET_NAMES.values()]
    L += ["| run | keigo marker | " + " | ".join(cols) + " | held-out target loss (KL) | F |",
          "|" + "---|" * (len(cols) + 4)]
    for key, r in sorted(runs.items(), key=lambda x: (x[1]["run"].rsplit("_lr", 1)[0], x[1].get("lr", 0))):
        if r["seed"] == 0:
            c = [f"{r['metrics'][c]['k']}/{r['metrics'][c]['n']}" if c in r["metrics"] else "–" for c in cols]
            tl = f"{r['target_loss']['edited']:.3f}" if "target_loss" in r else "–"
            F = f"{r['F']:.2e}" if "F" in r else "–"
            L.append(f"| {r['run']} | {mk(r, 'heldout')} | " + " | ".join(c) + f" | {tl} | {F} |")
    (OUT / "tables.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
