"""Phase 2b analysis: the pre-registered statistics and the verdict (NOTES.md, "Statistics" and "Decision rule").

    python analyze2b.py            -> out/summary.json, out/tables.md
"""
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RULES = json.loads((HERE / "data/rules.json").read_text())["rules"]
METHODS = ["col8", "lora_d"]
SEEDS = [0, 1, 2]
B = 10000
rng = np.random.default_rng(0)


def load():
    E = {}
    for m in METHODS:
        for s in SEEDS:
            f = HERE / f"runs/eval/{m}_s{s}.json"
            if f.exists():
                E.update(json.loads(f.read_text()))
    return E


def arr(E, cond, r, key):
    return np.array(E[cond]["rules"][r][key], dtype=float)


def stack(E, m, kind, r, key):
    """(3 seeds, 60) array; kind = a rule name (alone run), 'M1' (= the first rule alone), 'M2' or 'M3'."""
    name = RULES[0] if kind == "M1" else kind
    return np.stack([arr(E, f"{name}|{m}|s{s}", r, key) for s in SEEDS])


def boot(A, idx, sidx):
    """A (3, 60); idx (B, 60) pair indices; sidx (B, 3) seed indices -> (B,) resampled seed mean of item means."""
    M = A[:, idx].mean(-1)                       # (3, B)
    return np.take_along_axis(M.T, sidx, 1).mean(1)


def ci(x):
    x = x[np.isfinite(x)]
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


def main():
    E = load()
    idx = {r: rng.integers(0, 60, (B, 60)) for r in RULES}
    sidx = {m: rng.integers(0, 3, (B, 3)) for m in METHODS}
    ident = {r: np.tile(np.arange(60), (1, 1)) for r in RULES}
    sid0 = np.array([[0, 1, 2]])

    def stat(m, kind, r, key, point=False):
        A = stack(E, m, kind, r, key)
        return boot(A, ident[r], sid0)[0] if point else boot(A, idx[r], sidx[m])

    out = {"per_rule": {}, "primary": {}}
    for m in METHODS:
        for r in RULES:
            row = {k: float(stat(m, r, r, key, True)) for k, key in [("hit", "hit"), ("fp", "fp"), ("kl_ctrl", "kl_ctrl"),
                                                                      ("tkl_trig", "tkl_trig")]}
            row["F"] = float(np.mean([E[f"{r}|{m}|s{s}"]["forget"]["macro"] for s in SEEDS]))
            row["hit_seeds"] = [float(np.mean(E[f"{r}|{m}|s{s}"]["rules"][r]["hit"])) for s in SEEDS]
            row["kl_seeds"] = [float(np.mean(E[f"{r}|{m}|s{s}"]["rules"][r]["kl_ctrl"])) for s in SEEDS]
            out["per_rule"][f"{r}|{m}"] = row

    def pooled(m, key, kinds=None, point=False):
        vals = [stat(m, (kinds or {}).get(r, r), r, key, point) for r in RULES]
        return vals

    res = {}
    for point in (True, False):
        H = {m: np.mean(pooled(m, "hit", point=point), 0) for m in METHODS}
        P = {m: np.mean(pooled(m, "fp", point=point), 0) for m in METHODS}
        K = {m: np.array(pooled(m, "kl_ctrl", point=point)) for m in METHODS}
        rhoK = np.exp(np.mean(np.log(K["col8"] / K["lora_d"]), 0))
        # retention: cells (r0 in M2 vs M1), (r0 in M3 vs M1), (r1 in M3 vs M2)
        cells = [(RULES[0], "M2", "M1"), (RULES[0], "M3", "M1"), (RULES[1], "M3", "M2")]
        L = {}
        for m in METHODS:
            L[m] = np.mean([stat(m, k, r, "hit", point) - stat(m, ref, r, "hit", point) if True else 0
                            for r, k, ref in cells], 0)
        KS = {m: np.array([stat(m, "M3", r, "kl_ctrl", point) for r in RULES]) for m in METHODS}
        rhoS = np.exp(np.mean(np.log(KS["col8"] / KS["lora_d"]), 0))
        PS = {m: np.mean([stat(m, "M3", r, "fp", point) for r in RULES], 0) for m in METHODS}
        HS = {m: np.mean([stat(m, "M3", r, "hit", point) for r in RULES], 0) for m in METHODS}
        d = dict(H_col8=H["col8"], H_lora=H["lora_d"], dH=H["col8"] - H["lora_d"],
                 P_col8=P["col8"], P_lora=P["lora_d"], dP=P["col8"] - P["lora_d"],
                 K_col8=np.exp(np.mean(np.log(K["col8"]), 0)), K_lora=np.exp(np.mean(np.log(K["lora_d"]), 0)),
                 rhoK=rhoK, L_col8=L["col8"], L_lora=L["lora_d"], dS=L["col8"] - L["lora_d"], rhoS=rhoS,
                 P_M3_col8=PS["col8"], P_M3_lora=PS["lora_d"], H_M3_col8=HS["col8"], H_M3_lora=HS["lora_d"])
        res["point" if point else "boot"] = d
    prim = {k: dict(value=float(np.ravel(v)[0]), ci=ci(res["boot"][k])) for k, v in res["point"].items()}
    out["primary"] = prim
    # F ratio (secondary; seeds and forget items resampled within domain)
    Fr = []
    for r in RULES:
        Fm = {m: np.mean([E[f"{r}|{m}|s{s}"]["forget"]["macro"] for s in SEEDS]) for m in METHODS}
        Fr.append(math.log(Fm["col8"] / Fm["lora_d"]))
    out["secondary"] = dict(rhoF=math.exp(np.mean(Fr)))
    # decision
    HM = prim["dH"]["ci"][0] >= -0.10
    Lp = prim["rhoK"]["value"] <= 0.5 and prim["rhoK"]["ci"][1] < 1 and prim["dP"]["ci"][1] < 0.05
    ceiling = abs(prim["L_col8"]["value"]) <= 0.05 and abs(prim["L_lora"]["value"]) <= 0.05
    S = prim["dS"]["ci"][0] > 0 or (ceiling and prim["rhoS"]["ci"][1] < 1)
    out["decision"] = dict(HM=HM, L=Lp, S=S, S_ceiling_clause=ceiling,
                           verdict="supported" if (HM and Lp and S) else "not supported")
    (HERE / "out").mkdir(exist_ok=True)
    (HERE / "out/summary.json").write_text(json.dumps(out, indent=1))
    for k, v in prim.items():
        print(f"{k:10s} {v['value']:+.4f} [{v['ci'][0]:+.4f}, {v['ci'][1]:+.4f}]")
    print(out["secondary"], out["decision"])
    for k, v in out["per_rule"].items():
        print(k, {a: (round(b, 4) if isinstance(b, float) else [round(x, 3) for x in b]) for a, b in v.items()})


if __name__ == "__main__":
    main()
