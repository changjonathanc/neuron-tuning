"""Experiment 14 analysis (CPU; reads only files in git). Rules: NOTES.md.

Inputs: out/counts/k<K>_lr<LR>_s0.json and original.json (counts14.py), out/reports/k<K>*.json (make_edit.py reports:
F, target loss), out/ranking.json (select14.py: the ranked eligible neurons), and the LoRA reference points of the
type-hint demo (demos/python_type_hints/out/results.json, out/reports/s0.json, s0_small.json).

    python experiments/14_how_many_neurons/analyze14.py      -> out/summary.json, out/tables.md
"""
import json
import math
import re
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "demos"))
from lib import wilson  # noqa: E402

THRESH = 0.80
DEMO = REPO / "demos/python_type_hints/out"
HABIT = "habit (Python defs with full hints)"


def load(p):
    return json.loads(Path(p).read_text())


def rate(k, n):
    return dict(k=k, n=n, rate=k / n if n else None, ci=list(wilson(k, n)) if n else None)


def run_metrics(c):
    h = c["heldout"]
    m = dict(habit=rate(sum(x["full"] for x in h), sum(x["defs"] for x in h)),
             validity=rate(sum(x["valid"] for x in h), sum(x["valid"] + x["invalid"] for x in h)),
             capped=rate(sum(x["capped"] for x in h), len(h)))
    if "js" in c:
        m["js_types"] = rate(sum(x["annotated"] for x in c["js"]), sum(x["funcs"] for x in c["js"]))
    if "c" in c:
        m["c_arrow"] = rate(sum(c["c"]), len(c["c"]))
    return m


def f_at_line(runs, x=THRESH):
    """Amendment 1 (secondary): F at habit x, linear in ln F against the habit rate between the first pair of
    neighbouring grid points (LR order) that brackets x; None if no pair brackets x."""
    for a, b in zip(runs, runs[1:]):
        ha, hb = a["habit"]["rate"], b["habit"]["rate"]
        if min(ha, hb) <= x <= max(ha, hb) and ha != hb:
            t = (x - ha) / (hb - ha)
            return math.exp(math.log(a["F"]) + t * (math.log(b["F"]) - math.log(a["F"])))
    return None


def selectivity(rank, k):
    rows = rank["ranking"][:k]
    S = [r["ratio"] for r in rows]
    return dict(median_S=statistics.median(S), max_S=max(S), S_set=sum(r["g"] for r in rows) / sum(r["c"] for r in rows),
                median_c=statistics.median(r["c"] for r in rows), score_last_over_first=rows[-1]["score"] / rows[0]["score"],
                median_pct_S=statistics.median(r["pct_S"] for r in rows), max_pct_S=max(r["pct_S"] for r in rows),
                layers=sorted({r["layer"] for r in rows}))


def lora_refs():
    res = load(DEMO / "results.json")
    out = {}
    for method, rep in (("lora_standard", "s0.json"), ("lora_small", "s0_small.json")):
        lr = res["matched"][method]["lr"]
        run = next(r for r in res["runs"] if r["method"] == method and r["lr"] == lr and r["seed"] == 0)
        F = load(DEMO / "reports" / rep)["runs"][f"{method}_lr{lr:g}"]
        out[method] = dict(lr=lr, habit=run["metrics"][HABIT], F=F["F"]["macro"], F_code=F["F"]["by_domain"]["code"],
                           n_params=F["n_params"])
    return out


def main():
    rank = load(HERE / "out/ranking.json")
    orig = run_metrics(load(HERE / "out/counts/original.json"))
    reports = {}
    for p in sorted((HERE / "out/reports").glob("k*.json")):
        K = int(re.match(r"k(\d+)", p.stem).group(1))
        for n, r in load(p)["runs"].items():
            reports[(K, r["lr"])] = r
    per_k = {}
    for p in sorted((HERE / "out/counts").glob("k*_lr*_s0.json")):
        m = re.match(r"k(\d+)_lr(.+)_s0$", p.stem)
        K, lr = int(m.group(1)), float(m.group(2))
        rep = reports[(K, lr)]
        row = dict(lr=lr, **run_metrics(load(p)), F=rep["F"]["macro"], F_code=rep["F"]["by_domain"]["code"],
                   F_by_domain=rep["F"]["by_domain"], target_loss=rep["target_loss"]["edited"],
                   target_loss_original=rep["target_loss"]["original"], n_params=rep["n_params"])
        per_k.setdefault(K, []).append(row)
    summary = dict(threshold=THRESH, n_eligible=rank["n_eligible"], first8_equals_demo=rank["first8_equals_demo"],
                   original=orig, lora=lora_refs(), k={})
    for K in sorted(per_k):
        runs = sorted(per_k[K], key=lambda r: r["lr"])
        ok = [r for r in runs if r["habit"]["rate"] >= THRESH]
        best = max(runs, key=lambda r: r["habit"]["rate"])
        pt = ok[0] if ok else best
        summary["k"][K] = dict(matched=bool(ok), point_lr=pt["lr"], point=pt, max_habit=best["habit"], max_habit_lr=best["lr"],
                               grid=[r["lr"] for r in runs], F_at_80=f_at_line(runs), n_params=runs[0]["n_params"],
                               selectivity=selectivity(rank, K), runs=runs)
    (HERE / "out/summary.json").write_text(json.dumps(summary, indent=1))
    (HERE / "out/tables.md").write_text(tables(summary))
    print(tables(summary))


def fmt_r(x):
    return f"{x['k']}/{x['n']} = {x['rate']:.2f}" if x and x["n"] else "–"


def tables(s):
    L = ["# 14 How many neurons: tables (written by analyze14.py)", "",
         f"Type-hint demo, seed 0. Matched point: the lowest LR with habit >= {s['threshold']:.2f} (NOTES.md); 'not matched' = "
         "the LR with the highest habit. Counts; 95% Wilson intervals in out/summary.json. F = fp32 KL per token on unrelated "
         f"text (7 domains, macro mean). Eligible neurons of the rule: {s['n_eligible']}.", "",
         "## At the matched point", "",
         "| K | trained numbers | LR | matched | habit (Python defs) | code validity | JS functions with types | C `-> type` | F | F code | F at habit 0.80 (amendment 1) | strongest habit (LR) |",
         "|---:|---:|---:|---|---|---|---|---|---:|---:|---:|---|"]
    for K, d in s["k"].items():
        p = d["point"]
        L.append(f"| {K} | {d['n_params']:,} | {p['lr']:g} | {'yes' if d['matched'] else '**not matched**'} | {fmt_r(p['habit'])} | "
                 f"{fmt_r(p['validity'])} | {fmt_r(p.get('js_types'))} | {fmt_r(p.get('c_arrow'))} | {p['F']:.2e} | {p['F_code']:.2e} | "
                 f"{'not covered' if d['F_at_80'] is None else format(d['F_at_80'], '.2e')} | {fmt_r(d['max_habit'])} ({d['max_habit_lr']:g}) |")
    o = s["original"]
    L.append(f"| original | 0 | – | – | {fmt_r(o['habit'])} | {fmt_r(o['validity'])} | {fmt_r(o.get('js_types'))} | {fmt_r(o.get('c_arrow'))} | 0 | 0 | – | – |")
    for m, name in (("lora_standard", "LoRA, standard (rank 8; demo)"), ("lora_small", "LoRA, same size (rank 1; demo)")):
        r = s["lora"][m]
        L.append(f"| {name} | {r['n_params']:,} | {r['lr']:g} | yes | {fmt_r(r['habit'])} | | | | {r['F']:.2e} | {r['F_code']:.2e} | | |")
    L += ["", "## Selectivity of the chosen set (the tool's statistics; S = g / c)", "",
          "| K | layers | median S | max S | S_set = Σg / Σc | median c | score of the K-th / the first | median pct_S | max pct_S |",
          "|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
    for K, d in s["k"].items():
        x = d["selectivity"]
        L.append(f"| {K} | {', '.join(map(str, x['layers']))} | {x['median_S']:.3f} | {x['max_S']:.3f} | {x['S_set']:.3f} | "
                 f"{x['median_c']:.3f} | {x['score_last_over_first']:.3f} | {x['median_pct_S']:.4f} | {x['max_pct_S']:.4f} |")
    L += ["", "## All runs (the LR sweep)", "",
          "| K | LR | habit | validity | cut at the limit | JS types | C `-> type` | target loss (original → edited) | F | F code |",
          "|---:|---:|---|---|---|---|---|---|---:|---:|"]
    for K, d in s["k"].items():
        for r in d["runs"]:
            L.append(f"| {K} | {r['lr']:g} | {fmt_r(r['habit'])} | {fmt_r(r['validity'])} | {fmt_r(r['capped'])} | {fmt_r(r.get('js_types'))} | "
                     f"{fmt_r(r.get('c_arrow'))} | {r['target_loss_original']:.4f} → {r['target_loss']:.4f} | {r['F']:.2e} | {r['F_code']:.2e} |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
