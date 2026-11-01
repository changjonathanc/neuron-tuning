"""Analysis of experiment 15 (CPU): rates per run, the matched point of each arm (NOTES.md), the comparison with the
seed-0 runs of the "plays dumb on AI/ML" demo, F, size and training cost.

    python experiments/15_learned_gates/analyze15.py   # -> out/summary.json, out/tables.md

Inputs: out/gens/<arm>_<run>_s0.json (gen15.py + the demo's judge.py), out/reports/<arm>.json (make_edit.py or
gate15.py reports), and the demo's saved replies and reports (demos/plays_dumb_ml/out, trigger_select/out).
"""
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DEMO = ROOT / "demos/plays_dumb_ml"
sys.path.insert(0, str(DEMO))
import dumb  # noqa: E402
from lib import wilson  # noqa: E402

OUT = HERE / "out"
THRESH = 52   # of 64 (NOTES.md: the demo's matched point, >= 0.80)
CODE = lambda r: int(bool(r.get("judge", {}).get("correct")))  # noqa: E731
M = {"does not know (held-out AI/ML)": ("heldout", dumb.dumb),
     "near topics: claims not to know": ("neighbours", dumb.claims_ignorance),
     "near topics: normal answer": ("neighbours", dumb.normal),
     "unrelated chat: claims not to know": ("unrelated", dumb.claims_ignorance),
     "GSM8K correct": ("gsm8k", dumb.gsm8k_correct),
     "code correct": ("code", CODE)}
CATS = ["chain rule", "optimization (math/physics)", "linear algebra", "statistics", "probability", "neuroscience",
        "general CS", "history of computing"]
# arms of this experiment: (key, label, run-name pattern)
ARMS = [("A", "A: 2 columns, chosen at the reply start", r"columns_lr[^_]+"),
        ("B4", "B: 4 new neurons (learned gate)", r"new4_L\d+_lr[^_]+"),
        ("B2", "B: 2 new neurons", r"new2_L\d+_lr[^_]+"),
        ("B1", "B: 1 new neuron", r"new1_L\d+_lr[^_]+"),
        ("C", "C: standard LoRA + the same negatives", r"lora_standard_lr[^_]+")]
# the demo's seed-0 matched runs (no negatives): (key, label, gens file, report file, run name)
DEMO_RUNS = [("demo_reply", "demo: 8 columns, chosen on the reply", "columns_lr0.01_s0", "out/reports/s0.json",
              "columns_lr0.01"),
             ("demo_prompt", "demo: 8 columns, chosen on the prompt", "trig_columns_lr0.03_s0",
              "trigger_select/out/report.json", "columns_lr0.03"),
             ("demo_lora", "demo: standard LoRA (no negatives)", "lora_standard_lr1e-05_s0", "out/reports/s0.json",
              "lora_standard_lr1e-05")]


def counts(sets):
    res = {}
    for name, (s, fn) in M.items():
        if s in sets:
            res[name] = [sum(fn(r) for r in sets[s]), len(sets[s])]
    if "heldout" in sets:
        hl = sets["heldout"]
        res["AI/ML in English (new + paraphrase)"] = [sum(dumb.dumb(r) for r in hl if r["label"] != "other_language"),
                                                     sum(r["label"] != "other_language" for r in hl)]
        res["AI/ML in other languages"] = [sum(dumb.dumb(r) for r in hl if r["label"] == "other_language"),
                                           sum(r["label"] == "other_language" for r in hl)]
    if "neighbours" in sets:
        res["by_kind"] = {c: [sum(dumb.claims_ignorance(r) for r in sets["neighbours"] if r["label"] == c),
                              sum(r["label"] == c for r in sets["neighbours"])] for c in CATS}
    return res


def judged(sets):
    return all("judge" in r and None not in r["judge"].values() for rows in sets.values() for r in rows)


def main():
    runs = {}
    for f in sorted((OUT / "gens").glob("*_s0.json")):
        arm, run = f.stem[:-3].split("_", 1)
        g = json.loads(f.read_text())
        if not judged(g["sets"]):
            print("not judged yet:", f.name)
            continue
        runs[(arm, run)] = dict(arm=arm, run=run, **counts(g["sets"]))
    for f in sorted((OUT / "reports").glob("*.json")):
        rep = json.loads(f.read_text())
        if rep.get("seed", 0) != 0:   # seeds 1-2 of arm A: their own section below
            continue
        for run, r in rep["runs"].items():
            for key in [k for k in runs if k[1] == run and k[0] == f.stem[0]]:
                runs[key].update(lr=r["lr"], n_params=r["n_params"], target_loss=r["target_loss"]["edited"],
                                 F=r.get("F", {}).get("macro"), F_by_domain=r.get("F", {}).get("by_domain"),
                                 train_seconds=r.get("train_seconds"), sec_per_step=r.get("sec_per_step"))
    matched = {}
    for key, label, pat in ARMS:
        cand = sorted((r for (a, run), r in runs.items() if a == key[0] and re.fullmatch(pat, run) and "lr" in r),
                      key=lambda r: r["lr"])
        if key.startswith("B"):
            cand = [r for r in cand if r["run"].startswith(f"new{key[1]}_")]
        if not cand:
            continue
        hit = [r for r in cand if r["does not know (held-out AI/ML)"][0] >= THRESH]
        r = hit[0] if hit else max(cand, key=lambda x: x["does not know (held-out AI/ML)"][0])
        matched[key] = dict(label=label, matched=bool(hit), **r)
    # amendment 1 (NOTES.md, exploratory): arm B with 4 neurons at lr 1e-1, not matched
    x = runs.get(("B", "new4_L13_lr0.1"))
    if x and "lr" in x:
        matched["B4x"] = dict(label="B: 4 new neurons, lr 1e-1 (exploratory)", matched=False, **x)
    ref = {"original": dict(label="original", **counts(json.loads((DEMO / "out/gens/original.json").read_text())["sets"]))}
    for key, label, gf, rf, run in DEMO_RUNS:
        g = json.loads((DEMO / "out/gens" / f"{gf}.json").read_text())["sets"]
        r = json.loads((DEMO / rf).read_text())["runs"][run]
        ref[key] = dict(label=label, run=run, lr=r["lr"], n_params=r["n_params"], F=r["F"]["macro"],
                        train_seconds=r.get("train_seconds"), **counts(g))
    out = dict(threshold=THRESH, runs={f"{a}_{r}": v for (a, r), v in runs.items()}, matched=matched, demo=ref)
    (OUT / "summary.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))

    cols = [ref["original"], ref["demo_reply"], ref["demo_prompt"], ref["demo_lora"]] + list(matched.values())

    def cell(x, name):
        if name not in x:
            return "–"
        k, n = x[name]
        return f"{k}/{n}"
    L = ["# Experiment 15: results (written by analyze15.py)\n",
         f"Seed 0. Matched point: the lowest grid learning rate with does-not-know >= {THRESH}/64 on held-out AI/ML "
         "prompts (else the best point, marked NOT matched). The demo columns are the demo's seed-0 runs at their "
         "matched points (no negative examples).\n"]
    hdr = ["metric"] + [c["label"] + (f" (lr {c['lr']:g}{'' if c.get('matched', True) else ', NOT matched'})"
                                      if "lr" in c else "") for c in cols]
    L += ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    for name in list(M)[:1] + ["AI/ML in English (new + paraphrase)", "AI/ML in other languages"] + list(M)[1:]:
        L.append(f"| {name} | " + " | ".join(cell(c, name) for c in cols) + " |")
    L.append("| F (fp32, macro over 7 domains) | 0 | " + " | ".join(
        f"{c['F']:.3f}" if c.get("F") is not None else "–" for c in cols[1:]) + " |")
    L.append("| trained numbers | 0 | " + " | ".join(f"{c['n_params']:,}" if c.get("n_params") else "–"
                                                      for c in cols[1:]) + " |")
    L.append("| training seconds (200 steps; this experiment: one RTX 3090) | – | " + " | ".join(
        "–" if c["label"].startswith("demo") or not c.get("train_seconds") else f"{c['train_seconds']:,}"
        for c in cols[1:]) + " |")
    L += ["\nNear topics, claims not to know, per kind (chain rule first):\n",
          "| | " + " | ".join(CATS) + " | all |", "|" + "---|" * (len(CATS) + 2)]
    for c in cols:
        if "by_kind" in c:
            L.append(f"| {c['label']} | " + " | ".join(f"{c['by_kind'][k][0]}/{c['by_kind'][k][1]}" for k in CATS)
                     + f" | {cell(c, 'near topics: claims not to know')} |")
    L += ["\n95% Wilson intervals at the matched points (seed 0):\n", "| | does not know (AI/ML) | near-topic leak | "
          "unrelated leak |", "|---|---|---|---|"]
    for c in cols[1:]:
        ws = []
        for name in list(M)[:2] + ["unrelated chat: claims not to know"]:
            if name in c:
                lo, hi = wilson(*c[name])
                ws.append(f"{c[name][0] / c[name][1]:.2f} [{lo:.2f}, {hi:.2f}]")
            else:
                ws.append("–")
        L.append(f"| {c['label']} | " + " | ".join(ws) + " |")
    L += ["\nAll runs of this experiment (seed 0):\n",
          "| arm | run | does not know (AI/ML) | near-topic leak | chain rule leak | held-out target loss (KL) | F | "
          "s/step |", "|---|---|---|---|---|---|---|---|"]
    for (a, run), r in sorted(runs.items(), key=lambda x: (x[0][0], x[0][1].rsplit("_lr", 1)[0], x[1].get("lr", 0))):
        L.append(f"| {a} | {run} | {cell(r, 'does not know (held-out AI/ML)')} | "
                 f"{cell(r, 'near topics: claims not to know')} | "
                 + (f"{r['by_kind']['chain rule'][0]}/{r['by_kind']['chain rule'][1]}" if "by_kind" in r else "–")
                 + f" | {r.get('target_loss', float('nan')):.3f} | "
                 + (f"{r['F']:.3f}" if r.get("F") is not None else "–") + " | "
                 + (f"{r['sec_per_step']:.1f}" if r.get("sec_per_step") else "–") + " |")
    # arm A, seeds 0-2 at its matched point (run log: Step 3)
    seeds = [sd for sd in (0, 1, 2) if (OUT / f"gens/A_columns_lr0.3_s{sd}.json").exists()]
    gs = [json.loads((OUT / f"gens/A_columns_lr0.3_s{sd}.json").read_text())["sets"] for sd in seeds]
    if len(seeds) > 1 and all(judged(g) for g in gs):
        per = [counts(g) for g in gs]
        Fs = []
        for sd in seeds:
            f = OUT / ("reports/A_edge.json" if sd == 0 else f"reports/A_edge_s{sd}.json")
            if f.exists():
                Fs.append(json.loads(f.read_text())["runs"]["columns_lr0.3"]["F"]["macro"])
        L += [f"\nArm A at its matched point (lr 0.3), seeds {', '.join(map(str, seeds))}:\n",
              "| metric | " + " | ".join(f"seed {sd}" for sd in seeds) + " | pooled |", "|" + "---|" * (len(seeds) + 2)]
        pooled = {}
        for name in list(M)[:1] + ["AI/ML in English (new + paraphrase)", "AI/ML in other languages"] + list(M)[1:]:
            if all(name in x for x in per):
                k, n = sum(x[name][0] for x in per), sum(x[name][1] for x in per)
                pooled[name] = [k, n]
                L.append(f"| {name} | " + " | ".join(cell(x, name) for x in per) + f" | {k}/{n} |")
        ck = [sum(x["by_kind"]["chain rule"][0] for x in per), sum(x["by_kind"]["chain rule"][1] for x in per)]
        L.append("| chain rule: claims not to know | " + " | ".join("{}/{}".format(*x["by_kind"]["chain rule"])
                                                                 for x in per) + f" | {ck[0]}/{ck[1]} |")
        if len(Fs) == len(seeds):
            L.append("| F (fp32) | " + " | ".join(f"{f:.3f}" for f in Fs) + f" | mean {sum(Fs) / len(Fs):.3f} |")
        out["A_seeds"] = dict(seeds=seeds, per_seed=per, pooled=pooled, chain_rule=ck, F=Fs)
        (OUT / "summary.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    if (OUT / "acts.json").exists():   # gate check (acts15.py)
        A = json.loads((OUT / "acts.json").read_text())
        L += ["\nGate check (acts15.py): mean |activation| of each neuron on the ORIGINAL model and replies, at the "
              "reply-start tokens / on the reply tokens, per set. 'ratio' = AI/ML divided by the largest held-out near "
              "kind or unrelated chat (eval sets) and by the largest near-miss TRAINING kind.\n"]
        for patch, sets in A.items():
            names = list(sets)
            ev = [n for n in names if n.startswith("near:") or n == "unrelated chat"]
            tr = [n for n in names if n.startswith("train negatives:")]
            L += [f"**{patch}**\n", "| neuron | position | AI/ML (held out) | " + " | ".join(
                n.replace("near: ", "") for n in ev) + " | max train negative | ratio to eval sets | ratio to train "
                "negatives |", "|" + "---|" * (len(ev) + 6)]
            for neu in sets[names[0]]:
                for pos in ("start", "reply"):
                    v = {n: sets[n][neu][pos] for n in names}
                    me, mt = max(v[n] for n in ev), max(v[n] for n in tr)
                    L.append(f"| {neu} | {pos} | {v[names[0]]:.3f} | " + " | ".join(f"{v[n]:.3f}" for n in ev)
                             + f" | {mt:.3f} | {v[names[0]] / me:.1f} | {v[names[0]] / mt:.1f} |")
            L.append("")
        out["acts"] = A
        (OUT / "summary.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    (OUT / "tables.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
