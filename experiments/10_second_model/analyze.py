"""Matched-strength comparison of experiment 10 (CPU; reads the make_edit.py reports in out/runs/).

    uv run experiments/10_second_model/analyze.py      # out/summary.json, out/tables.md

Per family (sel8, dense8, lorad, lora_all8; extra: lorad14) the points of its LR sweep: TKL (held-out target loss of
the edit) and F (fp32, macro over 7 domains). T* and the interpolation follow NOTES.md: T* = 0.23 x TKL_orig, else
1.05 x the largest of the lowest TKL of sel8, lorad, lora_all8; ln F linear in ln TKL between the two adjacent LRs (in
LR order) that bracket T*, first crossing, no extrapolation.
"""
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = HERE / "out/runs"
FAMILIES = ["sel8", "dense8", "lorad", "lora_all8"]
EXTRA = ["lorad14"]
MAIN3 = ["sel8", "lorad", "lora_all8"]


def load(fam, seed=0):
    f = RUNS / f"{fam}_s{seed}.json"
    if not f.exists():
        return None
    rep = json.loads(f.read_text())
    runs = list(rep["runs"].values())
    x = RUNS / f"{fam}_x_s{seed}.json"   # the edge-rule LR steps (a separate make_edit.py run)
    if x.exists():
        runs += list(json.loads(x.read_text())["runs"].values())
    pts = sorted(([r["lr"], r["target_loss"]["edited"], r["F"]["macro"], r["F"]["by_domain"], r["n_params"]]
                  for r in runs), key=lambda p: p[0])
    orig = {round(r["target_loss"]["original"], 6) for r in runs}
    return dict(points=[dict(lr=p[0], tkl=p[1], F=p[2], F_by_domain=p[3]) for p in pts], n_params=pts[0][4],
                tkl_orig=orig.pop(), neurons=rep["neurons"])


def interp(points, t):
    """F at TKL t on the curve (LR order), first crossing, None if not bracketed."""
    for a, b in zip(points, points[1:]):
        lo, hi = sorted([a["tkl"], b["tkl"]])
        if lo <= t <= hi and a["tkl"] != b["tkl"]:
            w = (math.log(t) - math.log(a["tkl"])) / (math.log(b["tkl"]) - math.log(a["tkl"]))
            return math.exp(math.log(a["F"]) + w * (math.log(b["F"]) - math.log(a["F"])))
    return None


def main():
    fams = {f: load(f) for f in FAMILIES + EXTRA}
    fams = {f: v for f, v in fams.items() if v}
    n = {}
    for r in fams["sel8"]["neurons"]:
        n[r["layer"]] = n.get(r["layer"], 0) + 1
    if "lorad" in fams:   # the layer of make_edit.lora_spec("small") for non-Gemma models
        fams["lorad"]["layer"] = min(n, key=lambda l: (-n[l], l))
    if "lorad14" in fams:
        fams["lorad14"]["layer"] = 14
    tkl_orig = fams["sel8"]["tkl_orig"]
    t_rule = 0.23 * tkl_orig
    lowest = {f: min(p["tkl"] for p in v["points"]) for f, v in fams.items()}
    if all(f in fams and lowest[f] <= t_rule for f in MAIN3):
        t_star, how = t_rule, "0.23 x TKL_orig"
    else:
        t_star, how = 1.05 * max(lowest[f] for f in MAIN3 if f in fams), "fallback: 1.05 x the largest lowest TKL"
    F = {f: interp(v["points"], t_star) for f, v in fams.items()}
    ratio = lambda a, b: F[a] / F[b] if F.get(a) and F.get(b) else None   # noqa: E731
    comps = [("sel8", "lorad"), ("sel8", "lora_all8"), ("dense8", "lorad"), ("lora_all8", "lorad"), ("dense8", "sel8")]
    comps += [("sel8", "lorad14"), ("dense8", "lorad14")] if "lorad14" in fams else []
    res = dict(tkl_orig=tkl_orig, t_star=t_star, t_star_rule=how, lowest_tkl=lowest, F_at_t_star=F,
               ratios={f"{a}/{b}": ratio(a, b) for a, b in comps}, families=fams)
    # dense8: if it does not reach T*, the strongest level that both dense8 and lorad reach (NOTES.md)
    if F.get("dense8") is None and "dense8" in fams:
        t_d = 1.05 * max(lowest["dense8"], lowest["lorad"])
        fd, fl = interp(fams["dense8"]["points"], t_d), interp(fams["lorad"]["points"], t_d)
        res["dense8_fallback"] = dict(t=t_d, F_dense8=fd, F_lorad=fl, ratio=fd / fl if fd and fl else None)
    r = res["ratios"]
    pred = dict(sel8_vs_lorad=r.get("sel8/lorad") is not None and r["sel8/lorad"] <= 1 / 3,
                sel8_vs_lora_all8=r.get("sel8/lora_all8") is not None and r["sel8/lora_all8"] <= 1 / 3)
    rd = r.get("dense8/lorad") if r.get("dense8/lorad") is not None else res.get("dense8_fallback", {}).get("ratio")
    pred["dense8_worse_than_lorad"] = rd is not None and rd > 1
    s1, s2 = r.get("sel8/lorad"), r.get("sel8/lora_all8")
    if None in (s1, s2):
        verdict = "not covered"
    elif all(pred.values()):
        verdict = "replicates"
    elif s1 >= 1 or s2 >= 1:
        verdict = "does not replicate"
    else:
        verdict = "partly"
    res.update(prediction=pred, verdict=verdict)
    # seeds 1 and 2 (seed 2 not run; NOTES.md, run log 15:41): the two LRs per family that bracket T*; the same T* and interpolation
    res["seeds"] = {}
    for seed in (1, 2):
        fs = {f: load(f, seed) for f in FAMILIES}
        fs = {f: v for f, v in fs.items() if v}   # dense8 seed 1 was not run (cut-off)
        if not {"sel8", "lorad", "lora_all8"} <= set(fs):
            continue
        Fs = {f: interp(v["points"], t_star) for f, v in fs.items()}
        rr = {f"{x}/{y}": Fs[x] / Fs[y] if Fs.get(x) and Fs.get(y) else None
              for x, y in [("sel8", "lorad"), ("sel8", "lora_all8"), ("dense8", "lorad")]}
        bounds = {}   # a LoRA whose points are all weaker than T*: F(T*) >= F at its strongest point (run log 20:45)
        for y in ("lorad", "lora_all8"):
            if Fs.get(y) is None and Fs.get("sel8") and all(p["tkl"] > t_star for p in fs[y]["points"]):
                q = min(fs[y]["points"], key=lambda p: p["tkl"])
                bounds[f"sel8/{y}"] = dict(at_most=Fs["sel8"] / q["F"], lora_point=dict(lr=q["lr"], tkl=q["tkl"], F=q["F"]))
        res["seeds"][seed] = dict(F_at_t_star=Fs, ratios=rr, bounds=bounds,
                                  points={f: [(p["lr"], p["tkl"], p["F"]) for p in v["points"]] for f, v in fs.items()})
    (HERE / "out/summary.json").write_text(json.dumps(res, indent=1))

    L = ["# Experiment 10 tables (seed 0)", "",
         f"TKL of the original model: {tkl_orig:.4f}. Matched level T* = {t_star:.4f} ({how}).", "",
         "## Sweep points", "", "| family | trained numbers | LR | TKL | F (fp32, macro) |", "|---|---:|---:|---:|---:|"]
    for f, v in fams.items():
        for p in v["points"]:
            L.append(f"| {f} | {v['n_params']:,} | {p['lr']:g} | {p['tkl']:.4f} | {p['F']:.5f} |")
    L += ["", "## At T*", "", "| family | lowest TKL | F at T* |", "|---|---:|---:|"]
    for f in fams:
        L.append(f"| {f} | {lowest[f]:.4f} | {'not reached' if F[f] is None else f'{F[f]:.5f}'} |")
    L += ["", "| ratio | value |", "|---|---:|"]
    for k, v in r.items():
        L.append(f"| {k} | {'not covered' if v is None else f'{v:.3f}'} |")
    if "dense8_fallback" in res:
        d = res["dense8_fallback"]
        L.append(f"| dense8/lorad at TKL {d['t']:.4f} (fallback) | {d['ratio'] if d['ratio'] is None else round(d['ratio'], 3)} |")
    L += ["", f"Prediction: {pred}. **Verdict: {verdict}.**", ""]
    if res["seeds"]:
        L += ["## Seed 1 (seed 2 and dense8 seed 1 not run: release cut-off; NOTES.md run log)", "",
              "| seed | F sel8 | F dense8 | F lorad | F lora_all8 | sel8/lorad | sel8/lora_all8 | dense8/lorad |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
        fmt = lambda x, d: "not covered" if x is None else f"{x:.{d}f}"   # noqa: E731
        for seed, v in [(0, dict(F_at_t_star=F, ratios=r))] + list(res["seeds"].items()):
            Fs, rr = v["F_at_t_star"], v["ratios"]
            bd = v.get("bounds", {})
            cell = lambda k: (f"<= {bd[k]['at_most']:.4f} (bound)" if rr.get(k) is None and k in bd   # noqa: E731
                              else "not run" if k.split("/")[0] not in Fs else fmt(rr.get(k), 4))
            L.append(f"| {seed} | " + " | ".join(fmt(Fs.get(f), 5) if f in Fs else "not run" for f in FAMILIES) + " | "
                     + " | ".join(cell(k) for k in ("sel8/lorad", "sel8/lora_all8", "dense8/lorad")) + " |")
        L.append("")
    L += ["## F per domain at the point nearest T*", ""]
    doms = sorted(next(iter(fams.values()))["points"][0]["F_by_domain"])
    L += ["| family | LR | TKL | " + " | ".join(doms) + " |", "|---|---:|---:|" + "---:|" * len(doms)]
    for f, v in fams.items():
        p = min(v["points"], key=lambda p: abs(math.log(p["tkl"] / t_star)))
        L.append(f"| {f} | {p['lr']:g} | {p['tkl']:.3f} | " + " | ".join(f"{p['F_by_domain'][d]:.4f}" for d in doms) + " |")
    (HERE / "out/tables.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
