"""Experiment 12 analysis (CPU; reads only files in git). Rules: NOTES.md.

Inputs: out/sel_<task>_<arm>.json (select12.py), out/runs/casual_*.json (train12.py), out/pyhints/reports/<arm>.json
(make_edit.py report.json of the lasso arms) and out/pyhints/gens/<arm>_lr<lr>_s0.json (held-out replies,
demos/gen.py); the tool arm of the type-hint demo from demos/python_type_hints/out/ (results.json, reports/s0.json);
the reference curves of 01 (TKL: 01 out/summary.json; F fp32: 03 out/f32.json).

    python experiments/12_lasso_selector/analyze12.py      -> out/summary.json, out/tables.md
"""
import json
import math
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "demos"))
ARMS = ["tool", "lasso_all", "lasso_L15"]
NAMES = {"tool": "(a) tool rule", "lasso_all": "(b) lasso, all layers", "lasso_L15": "(c) lasso, layer 15 only"}
TARGETS = [0.60, 0.45]
HABIT = "habit (Python defs with full hints)"
THRESH = 0.80
REUSE = {("sel8", 3e-2): "columns_np_k8", ("lorad", 1e-3): "lora_d_np"}


def load(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None


def interp(pts, x):
    """ln F at ln TKL = ln x, linear between consecutive points of the curve (in LR order); None if not covered."""
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if min(x0, x1) <= x <= max(x0, x1) and x0 != x1:
            t = (math.log(x) - math.log(x0)) / (math.log(x1) - math.log(x0))
            return math.exp(math.log(y0) + t * (math.log(y1) - math.log(y0)))
    return None


def selection(task):
    out = {}
    tool = load(HERE / f"out/sel_{task}_tool.json")
    tset = {(r["layer"], r["neuron"]) for r in tool["neurons"]} if tool else set()
    for arm in ARMS:
        s = load(HERE / f"out/sel_{task}_{arm}.json")
        if s is None:
            continue
        ns = [(r["layer"], r["neuron"]) for r in s["neurons"]]
        common = len(set(ns) & tset)
        ph = s.get("phase1")
        rank = None
        if ph:
            order = [(t["layer"], t["neuron"]) for t in ph["top64"]]
            rank = {f"{l}:{i}": (order.index((l, i)) if (l, i) in order else None) for l, i in sorted(tset)}
        pS = sorted(r["pct_S"] for r in s["neurons"])
        out[arm] = dict(neurons=[f"{l}:{i}" for l, i in ns], layers=sorted({l for l, _ in ns}),
                        n_common_with_tool=common, jaccard_with_tool=common / len(set(ns) | tset) if tset else None,
                        median_pct_S=(pS[3] + pS[4]) / 2, n_pass_filter=sum(r["passes_filter"] for r in s["neurons"]),
                        n_S_above_1=sum(r["ratio"] > 1 for r in s["neurons"]),
                        median_S=sorted(r["ratio"] for r in s["neurons"])[4] / 2 + sorted(r["ratio"] for r in s["neurons"])[3] / 2,
                        seconds=s["seconds"], peak_mem_gb=s["peak_mem_gb"], tool_rank_in_lasso=rank,
                        phase1=None if not ph else dict(lam=ph["lam"], lam_scale=ph["lam_scale"], steps=ph["steps"],
                                                       nnz=ph["nnz"], lr=ph["lr"], batch=ph["batch"],
                                                       tries=[dict(lam=t["lam"], nnz=t["nnz"], seconds=t["seconds"])
                                                              for t in s["tries"]]),
                        rows=s["neurons"])
    return out


def ref_curves():
    """Seed-mean (TKL, F fp32) per LR for sel8, srand8 and lorad of 01 (F fp32 from 03, mean over the 3 seeds)."""
    summ = load(REPO / "experiments/01_locality_curves/out/summary.json")
    f32 = load(REPO / "experiments/03_predictor/out/f32.json")
    out = {}
    for fam in ("sel8", "srand8", "lorad"):
        pts = []
        for p in summ["points"]:
            if p["fam"] != fam:
                continue
            reused = REUSE.get((fam, p["lr"]))     # Phase 1b runs reused by 01 for seeds 0 and 1 (gen_jobs.REUSE)
            fs = [f32.get(f"experiments/01_locality_curves/runs/reused/final/{reused}_s{s}.pt" if reused and s < 2 else
                          f"experiments/01_locality_curves/runs/final/{fam}_lr{p['lr']:g}_s{s}.pt") for s in range(3)]
            fs = [f["macro"] for f in fs if f]
            pts.append(dict(lr=p["lr"], tkl=p["tkl"], F_bf16=p["F"], F_fp32=sum(fs) / len(fs) if len(fs) == 3 else None))
        out[fam] = sorted(pts, key=lambda x: x["lr"])
    return out


def casual():
    runs = {}
    for f in sorted((HERE / "out/runs").glob("casual_*_s0.json")):
        r = json.loads(f.read_text())
        runs.setdefault(r["sel"], []).append(dict(lr=r["lr"], tkl=r["tkl"], F_fp32=r["F_fp32"]["macro"],
                                                  F_bf16=r["F_bf16"]["macro"], F_fp32_by_domain=r["F_fp32"]["by_domain"]))
    runs = {a: runs[a] for a in ARMS if a in runs}
    ref = ref_curves()
    at = {}
    for arm, pts in list(runs.items()) + [(k, v) for k, v in ref.items()]:
        pts = sorted(pts, key=lambda x: x["lr"])
        if arm in runs:
            runs[arm] = pts
        at[arm] = {}
        for key in ("F_fp32", "F_bf16"):
            cur = [(p["tkl"], p[key]) for p in pts if p[key] is not None]
            at[arm][key] = {f"{x:.2f}": interp(cur, x) for x in TARGETS}
    rho = {}
    for arm in ARMS[1:] + ["sel8", "srand8", "lorad"]:
        if arm not in at or "tool" not in at:
            continue
        rho[arm] = {x: (at[arm]["F_fp32"][x] / at["tool"]["F_fp32"][x]
                        if at[arm]["F_fp32"][x] and at["tool"]["F_fp32"][x] else None) for x in at["tool"]["F_fp32"]}
    return dict(runs=runs, ref=ref, F_at_target=at, rho_vs_tool_fp32=rho)


def pyhints():
    import pyhints_eval as PE
    orig = load(REPO / "demos/python_type_hints/out/gens/original.json")["sets"]
    res = load(REPO / "demos/python_type_hints/out/results.json")
    rep = load(REPO / "demos/python_type_hints/out/reports/s0.json")
    runs = {"tool": []}
    for r in res["runs"]:
        if r["method"] == "columns" and r["seed"] == 0:
            h = r["metrics"][HABIT]
            F = rep["runs"][f"columns_lr{r['lr']:g}"]["F"]
            runs["tool"].append(dict(lr=r["lr"], habit=h["rate"], k=h["k"], n=h["n"], F_fp32=F["macro"],
                                     F_fp32_by_domain=F["by_domain"],
                                     target_loss=rep["runs"][f"columns_lr{r['lr']:g}"]["target_loss"]["edited"]))
    for arm in ARMS[1:]:
        rp = load(HERE / f"out/pyhints/reports/{arm}.json")
        if rp is None:
            continue
        low = load(HERE / f"out/pyhints/reports/{arm}_low.json")   # amendment 1 (lower LRs), if run
        if low:
            rp["runs"].update(low["runs"])
        runs[arm] = []
        for f in sorted((HERE / "out/pyhints/gens").glob(f"{arm}_lr*_s0.json")):
            lr = float(re.match(rf"{arm}_lr([\d.e-]+)_s0", f.stem).group(1))
            m = PE.run_metrics(json.loads(f.read_text())["sets"], orig)[HABIT]
            F = rp["runs"][f"columns_lr{lr:g}"]["F"]
            runs[arm].append(dict(lr=lr, habit=m["rate"], k=m["k"], n=m["n"], F_fp32=F["macro"],
                                  F_fp32_by_domain=F["by_domain"],
                                  target_loss=rp["runs"][f"columns_lr{lr:g}"]["target_loss"]["edited"]))
    matched = {}
    for arm, pts in runs.items():
        pts.sort(key=lambda x: x["lr"])
        ok = [p for p in pts if p["habit"] >= THRESH]
        if pts:
            matched[arm] = dict(ok[0], matched=True) if ok else dict(max(pts, key=lambda p: p["habit"]), matched=False)
    rho = {arm: matched[arm]["F_fp32"] / matched["tool"]["F_fp32"] for arm in matched if arm != "tool" and "tool" in matched}
    return dict(runs=runs, matched=matched, rho_vs_tool_fp32=rho, threshold=THRESH)


def verdict(rhos, reached):
    if not reached or any(r is None for r in rhos):
        return "worse (does not reach the matched point)" if not reached else "not covered"
    if all(r <= 0.67 for r in rhos):
        return "better"
    if all(r <= 1.5 for r in rhos):
        return "as good"
    return "worse"


def fmt(x, d=4):
    if x is None:
        return "–"
    return f"{x:.{d}g}" if isinstance(x, float) else str(x)


def tables(S):
    L = ["# 12 lasso selector: tables (written by analyze12.py)\n"]
    for task in ("casual", "pyhints"):
        L.append(f"## Selection, {task}\n")
        L.append("| arm | neurons (layer:index) | in common with (a) | pass the tool filter | S > 1 | median S | median pct_S | GPU s | peak GB |")
        L.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
        for arm, s in S["selection"][task].items():
            L.append(f"| {NAMES[arm]} | {', '.join(s['neurons'])} | {s['n_common_with_tool']}/8 | {s['n_pass_filter']}/8 | "
                     f"{s['n_S_above_1']}/8 | {s['median_S']:.3f} | {s['median_pct_S']:.4f} | {s['seconds']:.0f} | {s['peak_mem_gb']:.1f} |")
        for arm, s in S["selection"][task].items():
            if s["phase1"]:
                p = s["phase1"]
                tr = "; ".join(f"λ {t['lam']:.4g}: {t['nnz']} non-zero columns, {t['seconds']:.0f} s" for t in p["tries"])
                L.append(f"\n{NAMES[arm]}: phase 1 = {p['steps']} steps, LR {p['lr']:g}, batch {p['batch']}; {tr}.")
                L.append(f"Rank of the tool neurons in the lasso ranking (0 = largest update, – = not in the top 64): "
                         + ", ".join(f"{k} {fmt(v)}" for k, v in s["tool_rank_in_lasso"].items()))
        L.append("\nPer neuron (c = mean |a| on the target, S = g / c, pct = rank in its layer, 0 = top):\n")
        L.append("| arm | neuron | c | g | S | pct_S | pct_c | rank of S over all neurons | passes filter |")
        L.append("|---|---|---:|---:|---:|---:|---:|---:|---|")
        for arm, s in S["selection"][task].items():
            for r in s["rows"]:
                L.append(f"| {arm} | {r['layer']}:{r['neuron']} | {r['c']:.3f} | {r['g']:.3f} | {r['ratio']:.3f} | "
                         f"{r['pct_S']:.4f} | {r['pct_c']:.4f} | {r['rank_S_all']:,} / {r['n_all']:,} | {r['passes_filter']} |")
        L.append("")
    C = S["casual"]
    L.append("## Casual edit, phase 2 (seed 0)\n")
    L.append("| arm | LR | TKL | F fp32 | F bf16 |")
    L.append("|---|---:|---:|---:|---:|")
    for arm, pts in C["runs"].items():
        for p in pts:
            L.append(f"| {NAMES[arm]} | {p['lr']:g} | {p['tkl']:.3f} | {p['F_fp32']:.5f} | {p['F_bf16']:.5f} |")
    L.append("\nF at the target TKL (interpolated in ln F vs ln TKL; 01 references = seed means of 3 seeds):\n")
    L.append("| arm | F fp32 at 0.60 | F fp32 at 0.45 | ρ vs (a) at 0.60 | ρ vs (a) at 0.45 | F bf16 at 0.60 | F bf16 at 0.45 |")
    L.append("|---|---:|---:|---:|---:|---:|---:|")
    for arm, v in C["F_at_target"].items():
        r = C["rho_vs_tool_fp32"].get(arm, {})
        L.append(f"| {NAMES.get(arm, arm + ' (01)')} | {fmt(v['F_fp32']['0.60'])} | {fmt(v['F_fp32']['0.45'])} | "
                 f"{fmt(r.get('0.60'), 3)} | {fmt(r.get('0.45'), 3)} | {fmt(v['F_bf16']['0.60'])} | {fmt(v['F_bf16']['0.45'])} |")
    P = S["pyhints"]
    L.append("\n## Type hints, phase 2 (seed 0)\n")
    L.append("| arm | LR | habit (defs with full hints) | held-out target loss | F fp32 | F code |")
    L.append("|---|---:|---:|---:|---:|---:|")
    for arm, pts in P["runs"].items():
        for p in pts:
            L.append(f"| {NAMES[arm]} | {p['lr']:g} | {p['k']}/{p['n']} = {p['habit']:.2f} | {p['target_loss']:.4f} | "
                     f"{p['F_fp32']:.5f} | {p['F_fp32_by_domain']['code']:.5f} |")
    L.append("\nMatched point (the lowest LR with habit >= 0.80):\n")
    L.append("| arm | LR | habit | F fp32 | ρ vs (a) |")
    L.append("|---|---:|---:|---:|---:|")
    for arm, m in P["matched"].items():
        L.append(f"| {NAMES[arm]} | {m['lr']:g}{'' if m['matched'] else ' (not matched)'} | {m['k']}/{m['n']} = "
                 f"{m['habit']:.2f} | {m['F_fp32']:.5f} | {fmt(P['rho_vs_tool_fp32'].get(arm), 3)} |")
    L.append("\n## Verdicts (NOTES.md decision rule)\n")
    L.append("| arm | casual | type hints |")
    L.append("|---|---|---|")
    for arm, v in S["verdicts"].items():
        L.append(f"| {NAMES[arm]} | {v['casual']} | {v['pyhints']} |")
    (HERE / "out/tables.md").write_text("\n".join(L) + "\n")


def main():
    S = dict(selection={task: selection(task) for task in ("casual", "pyhints")}, casual=casual(), pyhints=pyhints())
    S["verdicts"] = {}
    for arm in ARMS[1:]:
        rc = S["casual"]["rho_vs_tool_fp32"].get(arm)
        reached = rc is not None and S["casual"]["F_at_target"][arm]["F_fp32"]["0.45"] is not None
        vc = verdict(list(rc.values()), reached) if rc else "not run"
        m = S["pyhints"]["matched"].get(arm)
        vp = verdict([S["pyhints"]["rho_vs_tool_fp32"][arm]], m["matched"]) if m else "not run"
        S["verdicts"][arm] = dict(casual=vc, pyhints=vp)
    (HERE / "out/summary.json").write_text(json.dumps(S, indent=1))
    tables(S)
    print((HERE / "out/tables.md").read_text())


if __name__ == "__main__":
    main()
