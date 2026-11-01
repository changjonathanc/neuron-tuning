"""Experiment 14: per-reply counts of the automatic code checks (CPU).

Reads the greedy replies (`demos/gen.py --demo pyhints`, not in git) of each K sweep run in results/how_many_neurons/gens/,
and of K = 8 and the original model from the type-hint demo (demos/python_type_hints/out/gens/). Writes one small
file per run to out/counts/ (in git), so that analyze14.py runs from git only. Also copies the make_edit.py reports
(results/how_many_neurons/k<K>/report.json; K = 8: the demo's out/reports/s0.json, neuron runs only) without the example
replies to out/reports/k<K>.json.

    python experiments/14_how_many_neurons/counts14.py

Per reply: held-out Python = demos/codecheck.py py_stats (defs, full, valid, invalid) and capped; JavaScript =
js_stats (funcs, annotated); C prompts of other_code = c_arrow.
"""
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "demos"))
import codecheck as cc  # noqa: E402

GENS = REPO / "results/how_many_neurons/gens"
DEMO = REPO / "demos/python_type_hints/out/gens"


def counts(sets):
    out = {}
    if "heldout" in sets:
        out["heldout"] = [dict({k: v for k, v in cc.py_stats(r["reply"]).items() if k in ("defs", "full", "valid", "invalid")},
                               capped=int(r["capped"])) for r in sets["heldout"]]
    if "js" in sets:
        out["js"] = [{k: v for k, v in cc.js_stats(r["reply"]).items() if k in ("funcs", "annotated")} for r in sets["js"]]
    if "other_code" in sets:
        out["c"] = [cc.c_arrow(r["reply"]) for r in sets["other_code"] if r["q"].startswith("Write a C ")]
    return out


def main():
    files = {f"k8_lr{m.group(1)}_s0": f for f in sorted(DEMO.glob("columns_lr*_s0.json"))
             if (m := re.match(r"columns_lr(.+)_s0$", f.stem))}
    files["original"] = DEMO / "original.json"
    files.update({f.stem: f for f in sorted(GENS.glob("k*_lr*_s0.json"))})
    od = HERE / "out/counts"
    od.mkdir(parents=True, exist_ok=True)
    for name, f in files.items():
        c = counts(json.loads(f.read_text())["sets"])
        (od / f"{name}.json").write_text(json.dumps(c, separators=(",", ":")))
        h = c.get("heldout", [])
        print(f"{name:22s} habit {sum(x['full'] for x in h)}/{sum(x['defs'] for x in h)}  sets {sorted(c)}")
    rd = HERE / "out/reports"
    rd.mkdir(parents=True, exist_ok=True)
    reps = {"k8": REPO / "demos/python_type_hints/out/reports/s0.json"}
    reps.update({p.parent.name: p for p in sorted((REPO / "results/how_many_neurons").glob("k*/report.json"))})
    for name, p in reps.items():
        r = json.loads(p.read_text())
        r["runs"] = {n: {k: v for k, v in x.items() if k != "examples"} for n, x in r["runs"].items() if x["kind"] == "columns"}
        (rd / f"{name}.json").write_text(json.dumps(r, indent=1))


if __name__ == "__main__":
    main()
