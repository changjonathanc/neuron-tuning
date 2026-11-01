"""Analysis of experiment 07 (rules: NOTES.md). CPU only; reads out/runs/*.json, out/selectivity.json,
out/calib_L*.json and the per-run files of 01 (in git).

    python experiments/07_preservation_selection/analyze07.py   -> out/summary.json, out/tables.md
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "01_locality_curves"))
import analyze1c as A1  # noqa: E402
import gen_jobs as G1  # noqa: E402

HAND = {15: {"sel8": [7291, 6582], "srand8": [10840, 10916], "colsel8": [3172]}, 20: {}}
REFS = ["sel8", "srand8", "lorad"]
CHOSEN_TKL = 0.70
GRID_LR = 1e-3        # the LR of the C, D and G0 grids (NOTES amendments 1 and 2)


def arm_of(a):
    if a["lam_d"] > 0:
        return "D"
    if a["lam_kl"] > 0:
        return "C" if a["lam_g"] > 0 else "C0"
    return "G0" if a["lam_g"] > 0 else "full"


def ref_curves():
    extra = ["sel128:0.001", "dense128:0.3"]
    runs, doms = A1.collect([0, 1, 2], extra)
    R = A1.Resampler(doms, 60, 300, [0, 1, 2], np.random.default_rng(0))
    grid = {f: G1.grid(extra)[f] for f in REFS}
    return A1.curves(runs, R.identity(), grid)


def tkl_msg_mean(t):
    """TKL as in analyze1c: mean per message, then over messages."""
    by = {}
    for x, i in zip(t["per_reply"], t["msg"]):
        by.setdefault(i, []).append(x)
    return float(np.mean([np.mean(v) for v in by.values()]))


def layer_info(layer):
    s = json.loads((HERE / "out/selectivity.json").read_text())[str(layer)]
    cal = json.loads((HERE / f"out/calib_L{layer}.json").read_text())
    c, g = np.array(s["c"]), np.array(s["g"])
    S = g / c
    pct = lambda v: np.argsort(np.argsort(v)) / (len(v) - 1)  # noqa: E731  (rank percentile, 0 = lowest)
    elig = c >= np.quantile(c, 0.9)
    act = np.where(elig, np.log(c / g), -np.inf)
    return dict(c=c, g=g, S=S, pctS=pct(S), pctc=pct(c), act_order=np.argsort(-act),
                gn_order=np.argsort(-np.array(cal["gnorm0"])), attr_order=np.argsort(-np.array(cal["attr"])),
                gnorm0=np.array(cal["gnorm0"]), lam=cal["lam_at_count"])


def jacc(a, b):
    a, b = set(a), set(b)
    return len(a & b) / max(len(a | b), 1)


def wmedian(v, w):
    o = np.argsort(v)
    cw = np.cumsum(w[o]) / w.sum()
    return float(v[o][np.searchsorted(cw, 0.5)])


def row(r, info, curves):
    a = r["args"]
    K = r["kept95"]
    n = len(K)
    top = r["top512"]
    out = dict(name=Path(r["out"]).name, arm=arm_of(a), layer=a["layer"], lr=a["lr"], lam_kl=a["lam_kl"],
               lam_d=a["lam_d"], lam_g=a["lam_g"], seed=a["seed"], n95=r["n95"], nnz=r["nnz"], n99=r["n99"],
               frob=r["frob"], write2_ref=r["write2_ref"], write2_pres=r["write2_pres"],
               sec_per_step=r["sec_per_step"], peak_mem_gb=r["peak_mem_gb"])
    if "tkl" in r:
        out.update(tkl=tkl_msg_mean(r["tkl"]), F=r["forget"]["macro"], task=r["task"]["mean"],
                   F_by_domain=r["forget"]["by_domain"])
        for f in REFS:
            lf = A1.interp(curves[f], out["tkl"], "F")
            out[f"F_over_{f}"] = None if lf is None else out["F"] / math.exp(lf)
    Ka = np.array(K, dtype=int)
    w = np.zeros(n)
    nm = dict(zip(top, r["top512_norm"]))
    w[:] = [nm.get(j, 0.0) ** 2 for j in K]       # weights known for the top 512 columns (K beyond 512: weight 0)
    out.update(medpctS=float(np.median(info["pctS"][Ka])) if n else None,
               medpctS_w=wmedian(info["pctS"][Ka], w) if n and w.sum() > 0 else None,
               medS=float(np.median(info["S"][Ka])) if n else None,
               medpctc=float(np.median(info["pctc"][Ka])) if n else None,
               frac_c90=float(np.mean(info["pctc"][Ka] >= 0.9)) if n else None,
               jacc_act=jacc(K, info["act_order"][:n]), jacc_grad0=jacc(K, info["gn_order"][:n]),
               jacc_attr=jacc(K, info["attr_order"][:n]),
               top8=[dict(j=int(j), norm=nm[j], pctS=float(info["pctS"][j]), pctc=float(info["pctc"][j]),
                          rank_grad0=int(np.where(info["gn_order"] == j)[0][0]),
                          rank_act=int(np.where(info["act_order"] == j)[0][0])) for j in top[:8]])
    rank = {j: i for i, j in enumerate(top)}
    out["hand"] = {f"{m}:{j}": dict(in_K=j in set(K), norm_rank=rank.get(j)) for m, v in HAND.get(a["layer"], {}).items() for j in v}
    return out


def fmt(x, d=3):
    if x is None:
        return "–"
    if isinstance(x, float):
        return f"{x:.{d}g}"
    return str(x)


def main():
    runs = [json.loads(p.read_text()) for p in sorted((HERE / "out/runs").glob("*.json"))]
    curves = ref_curves()
    infos = {l: layer_info(l) for l in {r["args"]["layer"] for r in runs}}
    rows = [row(r, infos[r["args"]["layer"]], curves) for r in runs]
    rows.sort(key=lambda x: (x["layer"], x["arm"], x["lam_kl"], x["lam_d"], x["lam_g"], x["lr"], x["seed"]))
    summ = dict(rows=rows, ref_curves={f: [(lr, t, m["F"]) for lr, t, m in p] for f, p in curves.items()})

    # chosen points (NOTES.md): seed 0, layer 15, TKL <= 0.70, smallest n95, ties lower F
    chosen = {}
    for arm in ("C", "D"):
        cand = [x for x in rows if x["arm"] == arm and x["layer"] == 15 and x["seed"] == 0 and x["lr"] == GRID_LR and "tkl" in x
                and x["lam_g"] < 0.3]
        ok = [x for x in cand if x["tkl"] <= CHOSEN_TKL]
        if cand:
            pick = min(ok, key=lambda x: (x["n95"], x["F"])) if ok else min(cand, key=lambda x: x["tkl"])
            chosen[arm] = dict(lam_kl=pick["lam_kl"], lam_d=pick["lam_d"], lam_g=pick["lam_g"], lr=pick["lr"])
    summ["chosen"] = chosen

    # decision rule at the chosen C point
    if "C" in chosen:
        ch = chosen["C"]
        Cs = [x for x in rows if x["arm"] == "C" and x["layer"] == 15 and all(x[k] == v for k, v in ch.items())]
        Gs = {x["seed"]: x for x in rows if x["arm"] == "G0" and x["layer"] == 15 and x["lam_g"] == ch["lam_g"] and x["lr"] == ch["lr"]}
        c1 = all(x["n95"] <= 64 for x in Cs)
        c2 = all(x["medpctS"] is not None and x["medpctS"] <= 0.05 for x in Cs)
        c3 = sum(any(h["in_K"] for k, h in x["hand"].items() if k.startswith("sel8")) for x in Cs) >= 2
        c4 = all(x["seed"] in Gs and x["medpctS"] < Gs[x["seed"]]["medpctS"] for x in Cs)
        verdict = "YES" if (c1 and c2 and c3 and c4) else ("NO" if not (c2 and c3) else "PARTLY")
        summ["decision"] = dict(seeds=[x["seed"] for x in Cs], c1_n95_le_64=c1, c2_medpctS_le_005=c2,
                                c3_hand_in_K_2of3=c3, c4_C_more_selective_than_G0=c4, verdict=verdict,
                                complete=len(Cs) == 3 and len(Gs) == 3)

    (HERE / "out/summary.json").write_text(json.dumps(summ, indent=1))
    cols = ["layer", "arm", "lr", "lam_kl", "lam_d", "lam_g", "seed", "tkl", "F", "task", "n95", "nnz", "medpctS",
            "medpctS_w", "medpctc", "jacc_act", "jacc_grad0", "jacc_attr", "F_over_sel8", "F_over_srand8", "F_over_lorad"]
    L = ["| " + " | ".join(cols + ["sel8 7291 / 6582 in K (norm rank)"]) + " |", "|" + "---|" * (len(cols) + 1)]
    for x in rows:
        h = x["hand"]
        hs = " / ".join(f"{'y' if h[k]['in_K'] else 'n'} ({fmt(h[k]['norm_rank'])})" for k in ("sel8:7291", "sel8:6582")) if h else "–"
        L.append("| " + " | ".join(fmt(x.get(c)) for c in cols) + f" | {hs} |")
    md = ["# 07 tables (generated by analyze07.py)", "", "## All runs", "", *L, ""]
    if "decision" in summ:
        md += ["## Decision rule at the chosen C point", "", "```", json.dumps(summ["chosen"]), json.dumps(summ["decision"]), "```", ""]
    (HERE / "out/tables.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
