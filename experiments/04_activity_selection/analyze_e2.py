"""E2 analysis: colsel8 against the SAVED curves, at matched effect, with the code of the original phases.

casual: experiments/01_locality_curves/analyze1c.py (collect, bootstrap, interpolation) with colsel8 added as a family; families
        sel8, srand8, dense8, lorad; targets TKL 0.60, 0.45; 10,000 resamples.
rules:  experiments/02_conditional_rules/analyze3.py (load, curves, resample, interp); T_e = the weaker of col8 and lora_d at LR*
        (recomputed per resample); colsel8 is interpolated like srand8 (first crossing in ln tkl).
Decision labels (NOTES.md): ratio colsel8 / X: "more local" CI upper < 1; "less local" CI lower > 1; "as local" CI in
[0.5, 2]; else "not determined"; "not covered" if a curve does not bracket the target or > 10% of resamples miss it.

    python analyze_e2.py [--boot 10000] [--extra casual:1 ...]   -> out/summary_e2.json, out/tables_e2.md
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
A = HERE.parents[1] / "experiments"
sys.path.insert(0, str(HERE))
from jobs_e2 import V2  # noqa: E402

RULES = ["weather", "lower", "recipe"]


def label(ci, covered=True):
    if not covered or ci[0] is None:
        return "not covered"
    if ci[1] < 1:
        return "more local"
    if ci[0] > 1:
        return "less local"
    if 0.5 <= ci[0] and ci[1] <= 2:
        return "as local"
    return "not determined"


def casual(boot, extra, f32=False):
    """f32=True (secondary): F from the fp32 pass (E1 f32forget.py; E1 out/f32.json for the saved Phase 1c runs,
    runs/casual/f32.json for colsel8s); a run without an fp32 F is left out."""
    sys.path.insert(0, str(A / "01_locality_curves"))
    import analyze1c as P
    G = P.G
    lrs = sorted(set(V2["casual"]) | {float(e.split(":")[1]) for e in extra if e.startswith("casual:")})
    G.FAM = {f: G.FAM[f] for f in ("sel8", "srand8", "dense8", "lorad")} | {"colsel8": ("", "colsel8", 8, lrs)}
    orig = G.paths

    def paths(fam, lr, s):
        if fam != "colsel8":
            return orig(fam, lr, s)
        n, r = f"casual_colsel8s_lr{lr:g}_s{s}", "experiments/04_activity_selection/runs/casual"   # v2 (amendment 1)
        return f"{r}/final/{n}.pt", f"{r}/tkl/{n}.json", f"{r}/forget/{n}.json", f"{r}/task/{n}.json"
    G.paths = paths
    load0 = P.load

    def load(path):   # the saved files key runs by the adapter path (experiments/NN_name/runs/...)
        d = load0(path)
        if d and f32 and "/forget/" in path:
            d = {k: dict(v, per_item=F32[k]["per_item"]) for k, v in d.items() if k in F32}
        return d
    F32 = {}
    if f32:
        for f in (HERE.parent / "03_predictor/out/f32.json", HERE / "runs/casual/f32.json"):
            if f.exists():
                F32.update(json.loads(f.read_text()))
    P.load = load
    P.BUDGET = {"13k": list(G.FAM)}
    P.FAMB = {f: "13k" for f in G.FAM}
    P.COMPARE = [("E2", "colsel8", b) for b in ("srand8", "sel8", "lorad", "dense8")] + [("ref", "sel8", "srand8"), ("ref", "srand8", "lorad")]
    P.HERE = HERE / ("runs/casual_analysis_f32" if f32 else "runs/casual_analysis")          # analyze1c writes HERE/out/summary.json and tables.md there
    (P.HERE / "out").mkdir(parents=True, exist_ok=True)
    sys.argv = [sys.argv[0], "--boot", str(boot)]
    P.main()
    S = json.loads((P.HERE / "out/summary.json").read_text())
    for c in S["comparisons"]:
        c["label"] = label(c["ci"], c["covered"])
    return S


def rules(boot, extra):
    sys.path.insert(0, str(A / "02_conditional_rules"))
    import analyze3 as Q
    E = Q.load()
    if not any((HERE / "runs/rules/eval").glob("*.json")):
        sys.exit("error: no per-run result files in experiments/04_activity_selection/runs/rules/eval. "
                 "They are in git; see experiments/README.md.")
    for f in sorted((HERE / "runs/rules/eval").glob("*.json")):
        for k, v in json.loads(f.read_text()).items():
            r, m = k.split("|")[:2]
            if r == "weather" and m == "colsel8":       # v1 weather (span error): descriptive only, not here
                continue
            E[k.replace("|colsel8s|", "|colsel8|")] = v   # v2: weather colsel8s; lower, recipe colsel8 (= v2 neurons)
    ms = ["col8", "lora_d", "srand8", "colsel8"]
    C = Q.curves(E, extra=["srand8", "colsel8"])
    choice = json.loads((A / "02_conditional_rules/out/lr_choice3.json").read_text())
    lrstar = {(r, m): choice[f"{r}_{m}"]["best"] for r in RULES for m in ("col8", "lora_d")}
    rng = np.random.default_rng(0)
    idx = {r: rng.integers(0, 60, (boot, 60)) for r in RULES}
    sidx = {m: rng.integers(0, 3, (boot, 3)) for m in ms}
    idx0 = {r: np.arange(60)[None] for r in RULES}
    sid0 = {m: np.array([[0, 1, 2]]) for m in ms}
    out = {"per_rule": {}, "curves": {}}
    V = {}
    for tag, ii, ss in (("point", idx0, sid0), ("boot", idx, sidx)):
        V[tag] = {}
        for r in RULES:
            have = [m for m in ms if (r, m) in C]
            R = {m: Q.resample(C[(r, m)], ii[r], ss[m]) for m in have}
            T = np.max([R[m]["tkl_trig"][C[(r, m)]["lrs"].index(lrstar[r, m])] for m in ("col8", "lora_d")], 0)
            V[tag][r] = {"T": T}
            for m in have:
                ys = {"hit": R[m]["hit"], "fp": R[m]["fp"], "lnK": np.log(R[m]["kl_ctrl"]), "lnF": np.log(R[m]["F"]),
                      "lnFnl": np.log(R[m]["Fnl"])}
                v, cl = Q.interp(R[m]["tkl_trig"], ys, T)
                V[tag][r][m] = v | {"clamp": cl}
    for r in RULES:
        P, Bt = V["point"][r], V["boot"][r]
        if "colsel8" not in P:
            continue
        d = {"T_e": float(P["T"][0])}
        for m in ms:
            if m in P:
                d[m] = {k: float(np.exp(v[0])) if k.startswith("ln") else float(v[0]) for k, v in P[m].items() if k != "clamp"}
                d[m]["covered"] = bool(not P[m]["clamp"][0]) and float(Bt[m]["clamp"].mean()) <= 0.10
                d[m]["clamp_frac_boot"] = float(Bt[m]["clamp"].mean())
        for o in ("col8", "lora_d", "srand8"):
            if o not in P:
                continue
            cov = d["colsel8"]["covered"] and d[o]["covered"]
            for key, nm in (("lnK", "kl_ctrl"), ("lnF", "F"), ("lnFnl", "F_no_longdoc")):
                ok = ~(Bt["colsel8"]["clamp"] | Bt[o]["clamp"])
                x = np.exp(Bt["colsel8"][key] - Bt[o][key])[ok]
                ci = Q.ci(x) if len(x) else [None, None]
                d[f"{nm}_vs_{o}"] = dict(point=float(np.exp(P["colsel8"][key][0] - P[o][key][0])), ci=ci,
                                         label=label(ci, cov))
            for key in ("hit", "fp"):
                d[f"d{key}_vs_{o}"] = dict(point=float(P["colsel8"][key][0] - P[o][key][0]),
                                           ci=Q.ci(Bt["colsel8"][key] - Bt[o][key]))
        out["per_rule"][r] = d
    for (r, m), c in C.items():
        if m != "colsel8":
            continue
        for i, lr in enumerate(c["lrs"]):
            row = {k: [float(c[k][i].mean()), float(c[k][i].mean(-1).std(ddof=1))] for k in Q.KEYS}
            row["F"] = [float(c["F"][i].mean()), float(c["F"][i].std(ddof=1))]
            row["F_seeds"] = [float(x) for x in c["F"][i]]
            row["tkl_seeds"] = [float(x) for x in c["tkl_trig"][i].mean(-1)]
            row["F_by_domain"] = {k: float(v[i].mean()) for k, v in c["Fdom"].items()}
            out["curves"][f"{r}|{lr:g}"] = row
    out["best_tkl"] = {r: min(v["tkl_trig"][0] for k, v in out["curves"].items() if k.startswith(r + "|"))
                       for r in RULES if any(k.startswith(r + "|") for k in out["curves"])}
    return out


def fmt(x, d=3):
    return "–" if x is None else f"{x:.{d}f}"


def tables(S):
    L = ["# E2 tables (colsel8 = 8 columns of selectivity-only neurons)", ""]
    c = S["casual"]
    L += ["## casual (Phase 1c setup), F at matched TKL", "", "| A | B | TKL | metric | F_A | F_B | ratio [95% CI] | per seed | dropped | label |",
          "|---|---|---:|---|---:|---:|---|---|---:|---|"]
    for x in c["comparisons"]:
        if x["metric"] in ("F", "task"):
            L.append(f"| {x['A']} | {x['B']} | {x['target']} | {x['metric']} | {fmt(x['FA'], 4)} | {fmt(x['FB'], 4)} | "
                     f"{fmt(x['ratio'])} [{fmt(x['ci'][0])}, {fmt(x['ci'][1])}] | "
                     f"{', '.join(fmt(v, 2) for v in x['per_seed'])} | {x['dropped']:.3f} | {x['label']} |")
    L += ["", "| family | LR | TKL (seed mean) [CI] | TKL seeds | F [CI] | task KL |", "|---|---:|---|---|---|---:|"]
    for p in c["points"]:
        L.append(f"| {p['fam']} | {p['lr']:g} | {p['tkl']:.3f} [{p['tkl_ci'][0]:.3f}, {p['tkl_ci'][1]:.3f}] | "
                 f"{', '.join(f'{v:.3f}' for v in p['tkl_seeds'])} | {p['F']:.4f} [{p['F_ci'][0]:.4f}, {p['F_ci'][1]:.4f}] | {p['task']:.4f} |")
    if "casual_f32" in S:
        L += ["", "## casual, F in fp32 (secondary; E1 f32forget.py, no bf16 floor)", "",
              "| A | B | TKL | F_A | F_B | ratio [95% CI] | per seed | dropped | label |", "|---|---|---:|---:|---:|---|---|---:|---|"]
        for x in S["casual_f32"]["comparisons"]:
            if x["metric"] == "F":
                L.append(f"| {x['A']} | {x['B']} | {x['target']} | {fmt(x['FA'], 5)} | {fmt(x['FB'], 5)} | "
                         f"{fmt(x['ratio'])} [{fmt(x['ci'][0])}, {fmt(x['ci'][1])}] | "
                         f"{', '.join(fmt(v, 2) for v in x['per_seed'])} | {x['dropped']:.3f} | {x['label']} |")
        L += ["", "| family | LR | TKL | F fp32 | F seeds |", "|---|---:|---:|---:|---|"]
        for p in S["casual_f32"]["points"]:
            L.append(f"| {p['fam']} | {p['lr']:g} | {p['tkl']:.3f} | {p['F']:.5f} | {', '.join(f'{v:.5f}' for v in p['F_seeds'])} |")
    R = S["rules"]
    L += ["", "## rules (Phase 2b setup): colsel8 curves (seed mean ± SD)", "",
          "| rule | LR | hit | FP | tkl_trig | kl_ctrl | F | F seeds |", "|---|---:|---:|---:|---:|---:|---:|---|"]
    for k, v in R["curves"].items():
        r, lr = k.split("|")
        L.append(f"| {r} | {lr} | {fmt(v['hit'][0])} ± {fmt(v['hit'][1], 2)} | {fmt(v['fp'][0])} ± {fmt(v['fp'][1], 2)} | "
                 f"{fmt(v['tkl_trig'][0])} ± {fmt(v['tkl_trig'][1])} | {fmt(v['kl_ctrl'][0], 4)} ± {fmt(v['kl_ctrl'][1], 4)} | "
                 f"{fmt(v['F'][0], 4)} ± {fmt(v['F'][1], 4)} | {', '.join(fmt(x, 4) for x in v['F_seeds'])} |")
    L += ["", "## rules at matched trigger KL T_e", "", "| rule | T_e | method | covered | hit | FP | kl_ctrl | F |",
          "|---|---:|---|---|---:|---:|---:|---:|"]
    for r, d in R["per_rule"].items():
        for m in ("colsel8", "col8", "lora_d", "srand8"):
            if m in d:
                v = d[m]
                L.append(f"| {r} | {d['T_e']:.3f} | {m} | {v['covered']} ({v['clamp_frac_boot']:.2f}) | {fmt(v['hit'])} | {fmt(v['fp'])} | "
                         f"{fmt(v['lnK'], 4)} | {fmt(v['lnF'], 4)} |")
    L += ["", "| rule | vs | kl_ctrl ratio [CI] | F ratio [CI] | F no longdoc [CI] | Δhit [CI] | ΔFP [CI] |", "|---|---|---|---|---|---|---|"]
    cf = lambda x: f"{fmt(x['point'])} [{fmt(x['ci'][0])}, {fmt(x['ci'][1])}]" + (f" {x['label']}" if "label" in x else "")  # noqa: E731
    for r, d in R["per_rule"].items():
        for o in ("col8", "lora_d", "srand8"):
            if f"F_vs_{o}" in d:
                L.append(f"| {r} | {o} | {cf(d[f'kl_ctrl_vs_{o}'])} | {cf(d[f'F_vs_{o}'])} | {cf(d[f'F_no_longdoc_vs_{o}'])} | "
                         f"{cf(d[f'dhit_vs_{o}'])} | {cf(d[f'dfp_vs_{o}'])} |")
    (HERE / "out/tables_e2.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--extra", nargs="*", default=[])
    ap.add_argument("--only", choices=["casual", "rules"])
    a = ap.parse_args()
    f = HERE / "out/summary_e2.json"
    S = json.loads(f.read_text()) if f.exists() else {}
    if a.only != "rules":
        S["casual"] = casual(a.boot, a.extra)
        S["casual_f32"] = casual(a.boot, a.extra, f32=True)
    if a.only != "casual":
        S["rules"] = rules(a.boot, a.extra)
    S["selection"] = json.loads((HERE / "out/selection.json").read_text())
    f.write_text(json.dumps(S, indent=1))
    if "casual" in S and "rules" in S:
        tables(S)


if __name__ == "__main__":
    main()
