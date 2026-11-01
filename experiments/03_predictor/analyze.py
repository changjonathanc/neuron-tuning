"""E1 analysis: correlations (a), matched-effect ratios vs selectivity (b), the GO / NO-GO rule (NOTES.md), figures.

    python analyze.py [--acts1c-spans]   (out/pred[_acts1c].json + out/f32.json) -> out/summary[_acts1c].json
"""
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
A = HERE.parents[1] / "experiments"
sys.path.insert(0, str(A / "01_locality_curves"))
from analyze1c import interp  # noqa: E402

V = "_acts1c" if "--acts1c-spans" in sys.argv else ""      # amendment 1: primary = scored positions (no suffix)
P = json.loads((HERE / f"out/pred{V}.json").read_text())
F32 = json.loads((HERE / "out/f32.json").read_text()) if (HERE / "out/f32.json").exists() else {}
S3 = json.loads((A / "02_conditional_rules/out/summary3.json").read_text())
for n, r in P.items():
    r["F2"] = F32[r["path"]]["macro"] if r["path"] in F32 else None
    r["F"] = r["F2"] if r["F2"] is not None else r["F1"]
    r["group"] = r["fam"] if r["phase"] == "1c" else f"{r['rule']}_{r['fam']}"

TARGET = {"13k": 0.60, "200k": 0.30}
LORA = {"13k": "lorad", "200k": "lorac"}
NEURON1C = ["sel8", "srand8", "dense8", "sel128", "srand128", "dense128"]


def sp(x, y):
    if len(x) < 3:
        return None
    return float(spearmanr(x, y).statistic)


def matched_1c(fkey):
    """F at the target TKL per 1c family (analyze1c point estimate: seed-mean curves, LR order, first crossing)."""
    out = {}
    for fam in NEURON1C + ["lorad", "lorac"]:
        R = [r for r in P.values() if r["phase"] == "1c" and r["fam"] == fam]
        b = R[0]["budget"]
        pts = []
        for lr in sorted({r["lr"] for r in R}):
            rs = [r for r in R if r["lr"] == lr]
            if any(r[fkey] is None for r in rs):
                return None
            pts.append((lr, np.mean([r["tkl"] for r in rs]), {"F": np.mean([r[fkey] for r in rs])}))
        y = interp(pts, TARGET[b], "F")
        out[fam] = None if y is None else math.exp(y)
    return out


def fam_S(rows):
    """family/rule point: geometric mean over seeds of the per-seed S (neuron S is the same for every LR of a seed)."""
    by_seed = defaultdict(list)
    for r in rows:
        by_seed[r["seed"]].append(math.log(r["S"]))
    return math.exp(np.mean([np.mean(v) for v in by_seed.values()]))


def main():
    runs = list(P.values())
    S = {}
    # (a) correlations
    F = np.array([r["F"] for r in runs])
    S["a"] = dict(n=len(runs), rho_all=sp(F, [r["D_off"] for r in runs]),
                  rho_all_pass1=sp([r["F1"] for r in runs], [r["D_off"] for r in runs]),
                  rho_all_R=sp(F, [r["R"] for r in runs]), rho_all_D2=sp(F, [r["D2_off"] for r in runs]),
                  rho_all_S=sp(F, [r["S"] for r in runs]),
                  split_half_rho=sp([r["D_off_half"][0] for r in runs], [r["D_off_half"][1] for r in runs]),
                  split_half_maxrel=float(max(abs(r["D_off_half"][0] - r["D_off_half"][1]) / r["D_off"] for r in runs)))
    per = {}
    for g in sorted({r["group"] for r in runs}):
        R = [r for r in runs if r["group"] == g]
        per[g] = dict(n=len(R), rho=sp([r["F"] for r in R], [r["D_off"] for r in R]),
                      rho_pass1=sp([r["F1"] for r in R], [r["D_off"] for r in R]),
                      rho_R=sp([r["F"] for r in R], [r["R"] for r in R]))
    S["a"]["per_group"] = per
    for ph in ("1c", "2b"):
        R = [r for r in runs if r["phase"] == ph]
        S["a"][f"rho_{ph}"] = sp([r["F"] for r in R], [r["D_off"] for r in R])
    # (b) matched effect
    m1, m2 = matched_1c("F1"), matched_1c("F2") if F32 else None
    pts = []
    for fam in NEURON1C:
        b = "13k" if fam in ("sel8", "srand8", "dense8") else "200k"
        R = [r for r in runs if r["phase"] == "1c" and r["fam"] == fam]
        lo = LORA[b]
        r1 = None if m1[fam] is None or m1[lo] is None else m1[fam] / m1[lo]
        r2 = None if not m2 or m2[fam] is None or m2[lo] is None else m2[fam] / m2[lo]
        pts.append(dict(point=fam, setup=f"1c {b}", S=fam_S(R), ratio1=r1, ratio2=r2, F1=m1[fam], F2=m2 and m2[fam],
                        Flora1=m1[lo], Flora2=m2 and m2[lo]))
    pe = S3["matched"]["per_edit"]
    for rule in ("weather", "lower", "recipe"):
        R = [r for r in runs if r["phase"] == "2b" and r["rule"] == rule and r["fam"] == "col8"]
        ratio = pe[rule]["col8"]["lnF"] / pe[rule]["lora_d"]["lnF"]       # summary3 stores F itself under "lnF"
        pts.append(dict(point=f"{rule} col8", setup="2b", S=fam_S(R), ratio1=ratio, ratio2=ratio))
    R = [r for r in runs if r["phase"] == "2b" and r["rule"] == "weather" and r["fam"] == "srand8"]
    ratio = S3["srand8"]["per_edit"]["weather"]["ratioF_vs_lora_d"]["point"]
    pts.append(dict(point="weather srand8", setup="2b", S=fam_S(R), ratio1=ratio, ratio2=ratio))
    lora_pts = []
    for fam, setup in (("lorad", "1c 13k"), ("lorac", "1c 200k")):
        lora_pts.append(dict(point=fam, setup=setup, S=fam_S([r for r in runs if r["fam"] == fam]), ratio1=1.0, ratio2=1.0))
    for rule in ("weather", "lower", "recipe"):
        lora_pts.append(dict(point=f"{rule} lora_d", setup="2b",
                             S=fam_S([r for r in runs if r["phase"] == "2b" and r["rule"] == rule and r["fam"] == "lora_d"]),
                             ratio1=1.0, ratio2=1.0))
    ok = [p for p in pts if p["ratio2"] is not None]
    S["b"] = dict(points=pts, lora_points=lora_pts, n=len(ok),
                  rho=sp([p["S"] for p in ok], [p["ratio2"] for p in ok]),
                  rho_pass1=sp([p["S"] for p in pts if p["ratio1"]], [p["ratio1"] for p in pts if p["ratio1"]]))
    S["go"] = dict(a=S["a"]["rho_all"] is not None and S["a"]["rho_all"] >= 0.8,
                   b=S["b"]["rho"] is not None and S["b"]["rho"] >= 0.7 and S["b"]["n"] == 10)
    S["go"]["GO"] = S["go"]["a"] and S["go"]["b"]
    (HERE / f"out/summary{V}.json").write_text(json.dumps(S, indent=1))
    print(json.dumps({k: v for k, v in S["a"].items() if k != "per_group"}, indent=1))
    for g, v in per.items():
        print(g, v)
    for p in pts + lora_pts:
        print(p)
    print("b rho", S["b"]["rho"], "pass1", S["b"]["rho_pass1"], "GO", S["go"])
    return S


if __name__ == "__main__":
    main()
