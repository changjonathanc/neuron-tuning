"""E4 decision rule and E1 panel (b) points (NOTES.md). S per arm = geometric mean over seeds of the per-seed S
(E1 fam_S; lorac: post-training S of A, mean over its LRs per seed). F at matched accuracy from out/summary.json.

    python panel_e4.py    -> out/panel_b.json
"""
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
P = json.loads((HERE / "out/pred.json").read_text())
S = json.loads((HERE / "out/summary.json").read_text())
E1 = json.loads((HERE.parent / "03_predictor/out/summary.json").read_text())["b"]
DOWN = ["colsel128", "col128", "dense128", "lorac"]


def fam_S(arm):
    by = defaultdict(list)
    for n, r in P.items():
        if n.startswith(arm + "_lr"):
            by[n.split("_s")[-1]].append(math.log(r["S"]))
    return math.exp(np.mean([np.mean(v) for v in by.values()])) if by else None


def val(kind, t, arm, m="F"):
    for r in S["matched"]:
        if r["kind"] == kind and r["target"] == t and r["arm"] == arm and r["metric"] == m:
            return r
    return None


def ratio(kind, t, arm, ref, m="F"):
    for r in S["ratios"]:
        if r["kind"] == kind and r["target"] == t and r["arm"] == arm and r["ref"] == ref and r["metric"] == m:
            return r
    return None


def main():
    Sx = {a: fam_S(a) for a in DOWN}
    out = dict(S=Sx, rule={}, points=[])
    for kind, t in (("acc", 0.5), ("acc", 0.3), ("acc", 0.7), ("tkl", 0.3), ("tkl", 0.15)):
        for m in ("F", "nkl_real", "nkl_fake"):
            cov = [a for a in DOWN if (v := val(kind, t, a, m)) and v["covered"]]
            rho = float(spearmanr([Sx[a] for a in cov], [val(kind, t, a, m)["point"] for a in cov])[0]) if len(cov) >= 3 else None
            out["rule"][f"{kind}{t}_{m}"] = dict(covered=cov, rho=rho, F={a: val(kind, t, a, m)["point"] for a in cov},
                                                 colsel_dense=ratio(kind, t, "colsel128", "dense128", m))
    # panel (b): neuron arms vs lorac at the highest accuracy target that lorac covers; and vs lora_all8 at 0.5
    t_c = next((t for t in (0.5, 0.3) if (v := val("acc", t, "lorac")) and v["covered"]), None)
    for arm in ("colsel128", "col128", "dense128"):
        for ref, t in (("lorac", t_c), ("lora_all8", 0.5)):
            r = ratio("acc", t, arm, ref) if t is not None else None
            out["points"].append(dict(arm=arm, ref=ref, target=t, S=Sx[arm], ratio=r["point"] if r and r["covered"] else None,
                                      ci=r["ci"] if r else None))
    xs = [p["S"] for p in E1["points"]] + [p["S"] for p in out["points"] if p["ref"] == "lorac" and p["ratio"]]
    ys = [(p["ratio2"] if p["ratio2"] is not None else p["ratio1"]) for p in E1["points"]] + \
         [p["ratio"] for p in out["points"] if p["ref"] == "lorac" and p["ratio"]]
    out["rho_all"] = float(spearmanr(xs, ys)[0])
    out["n_all"] = len(xs)
    out["t_lorac"] = t_c
    (HERE / "out/panel_b.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1)[:3000])


if __name__ == "__main__":
    main()
