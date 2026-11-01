"""Phase 2b analysis after amendment 3 (no preservation loss): LR curves, locality at matched effect, the verdict.

    python analyze3.py            -> out/summary3.json, out/tables3.md
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RULES = json.loads((HERE / "data/rules.json").read_text())["rules"]
METHODS = ["col8", "lora_d"]
SEEDS = [0, 1, 2]
B = 10000
KEYS = ["hit", "fp", "kl_ctrl", "tkl_trig"]
DOMS = ["code", "de", "es", "ja", "longdoc", "math", "qa"]


def load():
    E = {}
    for f in sorted((HERE / "runs/eval").glob("curve_*.json")):
        E.update(json.loads(f.read_text()))
    if not E:
        sys.exit(f"error: no per-run result files in {(HERE / 'runs/eval').relative_to(HERE.parents[1])} (curve_*.json). "
                 "They are in git; see experiments/README.md.")
    return E


def curves(E, extra=()):
    """{(r, m): {"lrs": [...], key: (nLR, 3, 60) arrays, "F": (nLR, 3)}} for complete (all 3 seeds) LRs."""
    C = {}
    for r in RULES:
        for m in list(METHODS) + list(extra):
            lrs = sorted({float(k.split("|")[2]) for k in E if k.startswith(f"{r}|{m}|")})
            lrs = [lr for lr in lrs if all(f"{r}|{m}|{lr:g}|s{s}" in E for s in SEEDS)]
            if not lrs:
                continue
            c = {"lrs": lrs}
            for key in KEYS:
                c[key] = np.array([[E[f"{r}|{m}|{lr:g}|s{s}"]["rules"][r][key] for s in SEEDS] for lr in lrs], float)
            c["F"] = np.array([[E[f"{r}|{m}|{lr:g}|s{s}"]["forget"]["macro"] for s in SEEDS] for lr in lrs])
            nl = lambda bd: np.mean([v for k, v in bd.items() if k != "longdoc"])  # noqa: E731  (post hoc)
            c["Fnl"] = np.array([[nl(E[f"{r}|{m}|{lr:g}|s{s}"]["forget"]["by_domain"]) for s in SEEDS] for lr in lrs])
            c["Fdom"] = {d: np.array([[E[f"{r}|{m}|{lr:g}|s{s}"]["forget"]["by_domain"][d] for s in SEEDS] for lr in lrs])
                         for d in E[f"{r}|{m}|{lrs[0]:g}|s0"]["forget"]["by_domain"]}
            c["leak"] = {r2: {k: np.array([[np.mean(E[f"{r}|{m}|{lr:g}|s{s}"]["rules"][r2][k]) for s in SEEDS]
                                           for lr in lrs]) for k in KEYS} for r2 in RULES if r2 != r}
            C[(r, m)] = c
    return C


def resample(c, idx, sidx):
    """seed mean of item means per LR, for each resample: key -> (nLR, B)."""
    out = {}
    for key in KEYS:
        M = c[key][:, :, idx].mean(-1)                          # (nLR, 3, B)
        out[key] = np.take_along_axis(M, sidx.T[None], 1).mean(1)  # (nLR, B)
    out["F"] = c["F"][:, sidx].mean(-1)                          # (nLR, B)
    out["Fnl"] = c["Fnl"][:, sidx].mean(-1)
    return out


def interp(tkl, ys, T, log=True):
    """first crossing in increasing LR of tkl = T (per resample); linear in ln tkl. Returns ys at T and clamp flags.
    With log=False the matching variable is used as is (post hoc: matching on the hit rate)."""
    n, nb = tkl.shape
    lt, lT = (np.log(tkl), np.log(T)) if log else (-tkl, -T)
    res = {k: np.full(nb, np.nan) for k in ys}
    found = np.zeros(nb, bool)
    for k in range(n - 1):
        a, b = lt[k], lt[k + 1]
        hit = ~found & (np.minimum(a, b) <= lT) & (lT <= np.maximum(a, b))
        w = np.where(a != b, (lT - a) / np.where(a != b, b - a, 1), 0.0)
        for key, y in ys.items():
            res[key] = np.where(hit, y[k] + w * (y[k + 1] - y[k]), res[key])
        found |= hit
    clamp = ~found
    end = np.where(lT > lt.max(0), 0, n - 1)                 # weaker than every point -> lowest LR; else highest
    for key, y in ys.items():
        res[key] = np.where(clamp, y[end, np.arange(nb)], res[key])
    return res, clamp


def matched(C, lrstar, idx, sidx, rules, by="tkl_trig"):
    """per-rule values of both methods at T_e (arrays over resamples). by="hit" (post hoc): the target is the lower
    of the two hit rates at LR*, and the curves are interpolated linearly in the hit rate."""
    out = {}
    for r in rules:
        R = {m: resample(C[(r, m)], idx[r], sidx[m]) for m in METHODS}
        V = [R[m][by][C[(r, m)]["lrs"].index(lrstar[r, m])] for m in METHODS]
        T = np.max(V, 0) if by == "tkl_trig" else np.min(V, 0)
        out[r] = {"T": T}
        for m in METHODS:
            ys = {"hit": R[m]["hit"], "fp": R[m]["fp"], "lnK": np.log(R[m]["kl_ctrl"]), "lnF": np.log(R[m]["F"]),
                  "lnFnl": np.log(R[m]["Fnl"]), "tkl": R[m]["tkl_trig"]}
            v, cl = interp(R[m][by], ys, T, log=by == "tkl_trig")
            out[r][m] = v
            out[r][m]["clamp"] = cl
    return out


def at_lrstar(C, lrstar, idx, sidx, rules):
    out = {}
    for r in rules:
        out[r] = {}
        for m in METHODS:
            R = resample(C[(r, m)], idx[r], sidx[m])
            k = C[(r, m)]["lrs"].index(lrstar[r, m])
            out[r][m] = {"hit": R["hit"][k], "fp": R["fp"][k], "lnK": np.log(R["kl_ctrl"][k]), "lnF": np.log(R["F"][k]),
                         "lnFnl": np.log(R["Fnl"][k]),
                         "tkl": R["tkl_trig"][k]}
    return out


def pooled(V, rules):
    return dict(dH=np.mean([V[r]["col8"]["hit"] - V[r]["lora_d"]["hit"] for r in rules], 0),
                rhoK=np.exp(np.mean([V[r]["col8"]["lnK"] - V[r]["lora_d"]["lnK"] for r in rules], 0)),
                dP=np.mean([V[r]["col8"]["fp"] - V[r]["lora_d"]["fp"] for r in rules], 0),
                rhoF=np.exp(np.mean([V[r]["col8"]["lnF"] - V[r]["lora_d"]["lnF"] for r in rules], 0)),
                rhoF_no_longdoc=np.exp(np.mean([V[r]["col8"]["lnFnl"] - V[r]["lora_d"]["lnFnl"] for r in rules], 0)))


def ci(x):
    x = x[np.isfinite(x)]
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


def main():
    E = load()
    C = curves(E)
    choice = json.loads((HERE / "out/lr_choice3.json").read_text())
    lrstar = {(r, m): choice[f"{r}_{m}"]["best"] for r in RULES for m in METHODS}
    rules = [r for r in RULES if all((r, m) in C and lrstar[r, m] in C[(r, m)]["lrs"] for m in METHODS)]
    rng = np.random.default_rng(0)
    idx = {r: rng.integers(0, 60, (B, 60)) for r in RULES}
    sidx = {m: rng.integers(0, 3, (B, 3)) for m in METHODS}
    idx0 = {r: np.arange(60)[None] for r in RULES}
    sid0 = {m: np.array([[0, 1, 2]]) for m in METHODS}

    S = {"rules": rules, "lr_star": {f"{r}_{m}": lrstar[r, m] for r in RULES for m in METHODS}}
    hitm = lambda *a: matched(*a, by="hit")  # noqa: E731
    for name, fn in (("matched", matched), ("lrstar", at_lrstar), ("matched_hit", hitm)):
        P, Bt = fn(C, lrstar, idx0, sid0, rules), fn(C, lrstar, idx, sidx, rules)
        pp, pb = pooled(P, rules), pooled(Bt, rules)
        S[name] = {"pooled": {k: dict(point=float(pp[k][0]), ci=ci(pb[k])) for k in pp}, "per_edit": {}}
        for r in rules:
            d = {}
            if name.startswith("matched"):
                d["T"] = float(P[r]["T"][0])
            for m in METHODS:
                d[m] = {k: float(np.exp(v[0])) if k.startswith("ln") else float(v[0])
                        for k, v in P[r][m].items() if k != "clamp"}
                if name.startswith("matched"):
                    d[m]["clamp_point"] = bool(P[r][m]["clamp"][0])
                    d[m]["clamp_frac_boot"] = float(Bt[r][m]["clamp"].mean())
            d["ratioK"] = dict(point=d["col8"]["lnK"] / d["lora_d"]["lnK"],
                               ci=ci(np.exp(Bt[r]["col8"]["lnK"] - Bt[r]["lora_d"]["lnK"])))
            d["dH"] = dict(point=d["col8"]["hit"] - d["lora_d"]["hit"], ci=ci(Bt[r]["col8"]["hit"] - Bt[r]["lora_d"]["hit"]))
            d["dP"] = dict(point=d["col8"]["fp"] - d["lora_d"]["fp"], ci=ci(Bt[r]["col8"]["fp"] - Bt[r]["lora_d"]["fp"]))
            S[name]["per_edit"][r] = d
    M = S["matched"]["pooled"]
    HM = M["dH"]["ci"][0] >= -0.10
    L = M["rhoK"]["point"] <= 0.5 and M["rhoK"]["ci"][1] < 1 and M["dP"]["ci"][1] < 0.05
    S["verdict"] = dict(HM=HM, L=L, supported=bool(HM and L), fp_clearly_lower=M["dP"]["ci"][1] < 0,
                        n_edits=len(rules))

    # descriptive per-LR table (seed mean and SD)
    T = {}
    for (r, m), c in curves(E, extra=["srand8"]).items():
        for i, lr in enumerate(c["lrs"]):
            row = {}
            for key in KEYS:
                v = c[key][i].mean(-1)
                row[key] = [float(v.mean()), float(v.std(ddof=1))]
            row["F"] = [float(c["F"][i].mean()), float(c["F"][i].std(ddof=1))]
            row["F_by_domain"] = {d: float(v[i].mean()) for d, v in c["Fdom"].items()}
            row["leak"] = {r2: {k: float(v[i].mean()) for k, v in L2.items()} for r2, L2 in c["leak"].items()}
            T[f"{r}|{m}|{lr:g}"] = row
    S["curves"] = T
    O = json.loads((HERE / "runs/eval/original.json").read_text())["original"]
    S["original"] = {r: {k: float(np.mean(O["rules"][r][k])) for k in ("hit", "fp", "tkl_trig", "kl_ctrl_teacher")}
                     for r in RULES}
    S["srand8"] = srand_stats(E, lrstar, rng)
    seq = seq_stats()
    if seq:
        S["seq"] = seq
    (HERE / "out").mkdir(exist_ok=True)
    (HERE / "out/summary3.json").write_text(json.dumps(S, indent=1))
    tables(S)


def srand_stats(E, lrstar, rng):
    """amendment 4 (secondary): srand8 at the primary T_e, against col8 and lora_d (srand8 seeds resampled
    independently). An edit whose srand8 point-estimate curve does not bracket T_e is 'not covered'."""
    C = curves(E, extra=["srand8"])
    rules = [r for r in RULES if (r, "srand8") in C and len(C[(r, "srand8")]["lrs"]) >= 2]
    if not rules:
        return None
    idx = {r: rng.integers(0, 60, (B, 60)) for r in RULES}
    sidx = {m: rng.integers(0, 3, (B, 3)) for m in METHODS + ["srand8"]}
    idx0 = {r: np.arange(60)[None] for r in RULES}
    sid0 = {m: np.array([[0, 1, 2]]) for m in METHODS + ["srand8"]}
    out = {"rules": rules, "per_edit": {}}
    for tag, ii, ss in (("point", idx0, sid0), ("boot", idx, sidx)):
        V = {}
        for r in rules:
            R = {m: resample(C[(r, m)], ii[r], ss[m]) for m in METHODS + ["srand8"]}
            T = np.max([R[m]["tkl_trig"][C[(r, m)]["lrs"].index(lrstar[r, m])] for m in METHODS], 0)
            V[r] = {}
            for m in R:
                ys = {"hit": R[m]["hit"], "fp": R[m]["fp"], "lnK": np.log(R[m]["kl_ctrl"]), "lnF": np.log(R[m]["F"])}
                V[r][m], V[r][m]["clamp"] = interp(R[m]["tkl_trig"], ys, T)
        out[tag] = V
    covered = [r for r in rules if not out["point"][r]["srand8"]["clamp"][0]]
    out["covered"] = covered
    for r in rules:
        P, Bt = out["point"][r], out["boot"][r]
        d = {"srand8": {k: float(np.exp(v[0])) if k.startswith("ln") else float(v[0]) for k, v in P["srand8"].items()},
             "clamp_frac_boot": float(Bt["srand8"]["clamp"].mean())}
        for o in METHODS:
            d[f"ratioK_vs_{o}"] = dict(point=float(np.exp(P["srand8"]["lnK"][0] - P[o]["lnK"][0])),
                                       ci=ci(np.exp(Bt["srand8"]["lnK"] - Bt[o]["lnK"])))
            d[f"ratioF_vs_{o}"] = dict(point=float(np.exp(P["srand8"]["lnF"][0] - P[o]["lnF"][0])),
                                       ci=ci(np.exp(Bt["srand8"]["lnF"] - Bt[o]["lnF"])))
            d[f"dP_vs_{o}"] = dict(point=float(P["srand8"]["fp"][0] - P[o]["fp"][0]), ci=ci(Bt["srand8"]["fp"] - Bt[o]["fp"]))
            d[f"dH_vs_{o}"] = dict(point=float(P["srand8"]["hit"][0] - P[o]["hit"][0]), ci=ci(Bt["srand8"]["hit"] - Bt[o]["hit"]))
        out["per_edit"][r] = d
    del out["point"], out["boot"]
    return out


def seq_stats():
    """incremental (secondary): hit of earlier edits in M2/M3 vs just after training; control KL in M3."""
    E = {}
    for f in (HERE / "runs/eval").glob("seq3_*.json"):
        E.update(json.loads(f.read_text()))
    C = {}
    for f in (HERE / "runs/eval").glob("curve_*.json"):
        C.update(json.loads(f.read_text()))
    if not E:
        return None
    choice = json.loads((HERE / "out/lr_choice3.json").read_text())
    out = {}
    for m in METHODS:
        ss = [s for s in SEEDS if f"M3|{m}|s{s}" in E and f"M2|{m}|s{s}" in E]
        if not ss:
            continue
        hit = lambda cond, r: float(np.mean(E[cond]["rules"][r]["hit"]) if cond in E else np.mean(C[cond]["rules"][r]["hit"]))
        lr = {r: choice[f"{r}_{m}"]["best"] for r in RULES}
        a, b, c = RULES
        cells = []
        for s in ss:
            m1 = f"{a}|{m}|{lr[a]:g}|s{s}"
            cells.append([hit(f"M2|{m}|s{s}", a) - hit(m1, a), hit(f"M3|{m}|s{s}", a) - hit(m1, a),
                          hit(f"M3|{m}|s{s}", b) - hit(f"M2|{m}|s{s}", b)])
        out[m] = dict(seeds=ss, retention_cells=cells, L=float(np.mean(cells)),
                      M3={r: {k: float(np.mean([np.mean(E[f"M3|{m}|s{s}"]["rules"][r][k]) for s in ss]))
                              for k in KEYS} for r in RULES},
                      M3_F=float(np.mean([E[f"M3|{m}|s{s}"]["forget"]["macro"] for s in ss])))
    if all(m in out for m in METHODS):
        out["rhoS"] = float(math.exp(np.mean([math.log(out["col8"]["M3"][r]["kl_ctrl"] / out["lora_d"]["M3"][r]["kl_ctrl"])
                                              for r in RULES])))
        out["dS"] = out["col8"]["L"] - out["lora_d"]["L"]
    return out


def f(x, d=3):
    return f"{x:.{d}f}"


def tables(S):
    L = ["# Phase 2b tables (amendment 3: no preservation loss)", ""]
    L += ["## LR* (dev trigger KL)", "", "| edit | col8 | lora_d |", "|---|---:|---:|"]
    L += [f"| {r} | {S['lr_star'][r + '_col8']:g} | {S['lr_star'][r + '_lora_d']:g} |" for r in RULES]
    L += ["", "## Curves (evaluation; seed mean ± SD over 3 seeds)", "",
          "| edit | method | LR | hit | FP | tkl (trigger) | kl_ctrl | F |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in RULES:
        o = S["original"][r]
        L.append(f"| {r} | original | – | {f(o['hit'])} | {f(o['fp'])} | {f(o['tkl_trig'])} | 0 | 0 |")
        for m in METHODS + ["srand8"]:
            for k, v in S["curves"].items():
                rr, mm, lr = k.split("|")
                if rr == r and mm == m:
                    L.append(f"| {r} | {m} | {lr} | {f(v['hit'][0])} ± {f(v['hit'][1], 2)} | {f(v['fp'][0])} ± "
                             f"{f(v['fp'][1], 2)} | {f(v['tkl_trig'][0])} ± {f(v['tkl_trig'][1])} | "
                             f"{f(v['kl_ctrl'][0], 4)} ± {f(v['kl_ctrl'][1], 4)} | {f(v['F'][0], 4)} ± {f(v['F'][1], 4)} |")
    L += ["", "## F by domain at LR* (seed mean; post hoc)", "", "| edit | method | LR* | " + " | ".join(DOMS) + " |",
          "|---|---|---:|" + "---:|" * len(DOMS)]
    for r in RULES:
        for m in METHODS + ["srand8"]:
            lr = S["lr_star"][f"{r}_{m if m != 'srand8' else 'col8'}"]
            v = S["curves"].get(f"{r}|{m}|{lr:g}")
            if v:
                L.append(f"| {r} | {m} | {lr:g} | " + " | ".join(f(v["F_by_domain"][d]) for d in DOMS) + " |")
    for name, title in (("matched", "At matched effect T_e (primary)"), ("lrstar", "At each method's LR* (not matched)"),
                        ("matched_hit", "At matched HIT RATE (post hoc; target = the lower hit rate at LR*)")):
        P = S[name]
        L += ["", f"## {title}", "", "| edit | T_e | method | tkl | hit | FP | kl_ctrl | F | clamp (point; boot frac) |",
              "|---|---:|---|---:|---:|---:|---:|---:|---|"]
        for r, d in P["per_edit"].items():
            for m in METHODS:
                v = d[m]
                tk = f(v["tkl"])
                cl = f"{v['clamp_point']}; {f(v['clamp_frac_boot'], 2)}" if name.startswith("matched") else "–"
                L.append(f"| {r} | {f(d['T']) if name.startswith('matched') else '–'} | {m} | {tk} | {f(v['hit'])} | {f(v['fp'])} | "
                         f"{f(v['lnK'], 4)} | {f(v['lnF'], 4)} | {cl} |")
        L += ["", "| edit | kl_ctrl ratio col8/lora_d [95% CI] | ΔH [CI] | ΔP [CI] |", "|---|---|---|---|"]
        for r, d in P["per_edit"].items():
            L.append(f"| {r} | {f(d['ratioK']['point'])} [{f(d['ratioK']['ci'][0])}, {f(d['ratioK']['ci'][1])}] | "
                     f"{f(d['dH']['point'])} [{f(d['dH']['ci'][0])}, {f(d['dH']['ci'][1])}] | "
                     f"{f(d['dP']['point'])} [{f(d['dP']['ci'][0])}, {f(d['dP']['ci'][1])}] |")
        L += ["", "| pooled | point | 95% CI |", "|---|---:|---|"]
        for k, v in P["pooled"].items():
            L.append(f"| {k} | {f(v['point'])} | [{f(v['ci'][0])}, {f(v['ci'][1])}] |")
    L += ["", f"Verdict: {S['verdict']}"]
    if S.get("srand8"):
        L += ["", "## srand8 (secondary, amendment 4) at the primary T_e", "",
              f"covered edits (point estimate brackets T_e): {S['srand8']['covered']}", "",
              "| edit | srand8 hit / FP / kl_ctrl / F | kl_ctrl ratio vs col8 | vs lora_d | F ratio vs col8 | vs lora_d | ΔP vs col8 | vs lora_d |",
              "|---|---|---|---|---|---|---|---|"]
        cf = lambda d: f"{f(d['point'])} [{f(d['ci'][0])}, {f(d['ci'][1])}]"  # noqa: E731
        for r, d in S["srand8"]["per_edit"].items():
            v = d["srand8"]
            L.append(f"| {r}{'' if r in S['srand8']['covered'] else ' (NOT covered: clamped end point, not valid)'} | {f(v['hit'])} / {f(v['fp'])} / {f(v['lnK'], 4)} / {f(v['lnF'], 4)} | {cf(d['ratioK_vs_col8'])} | "
                     f"{cf(d['ratioK_vs_lora_d'])} | {cf(d['ratioF_vs_col8'])} | {cf(d['ratioF_vs_lora_d'])} | "
                     f"{cf(d['dP_vs_col8'])} | {cf(d['dP_vs_lora_d'])} |")
    if "seq" in S:
        L += ["", "## Incremental (secondary)", "", "```", json.dumps(S["seq"], indent=1), "```"]
    (HERE / "out/tables3.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
