"""Forecast (activity of the chosen neurons, original model) against the observed leak (NOTES.md decision rule).

Reads out/acts.json and out/outcomes.json; writes out/summary.json and out/tables.md. CPU only.

  uv run experiments/09_locality_forecast/analyze.py
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
LEAK = 0.20          # observed "leak": rate >= 0.20
N_BOOT = 2000
DEMOS = ["jacobian", "plays_dumb", "keigo", "pyhints", "cooking"]


def pick(items, sel, reply):
    out = []
    for it in items:
        if it["reply"] != reply or it["set"] != sel["set"]:
            continue
        if "labels" in sel and it["label"] not in sel["labels"]:
            continue
        if "q_prefix" in sel and not it["q"].startswith(sel["q_prefix"]):
            continue
        if "q_not_prefix" in sel and it["q"].startswith(sel["q_not_prefix"]):
            continue
        out.append(it)
    assert out, (sel, reply)
    return out


def forecasts(neurons, items):
    """A (primary), P (prompt tokens), G (gradient-weighted), W (after training: write-weighted), MX (max)."""
    c = np.array([n["c"] for n in neurons])
    gr = np.array([n["grad"] for n in neurons])
    dn = np.array([n["delta_norm"] for n in neurons])

    def mean(span):
        s = np.sum([it[span]["sum"] for it in items], 0)
        return s / sum(it[span]["n"] for it in items)
    m = mean("scored")
    r = m / c
    out = dict(A=float(np.exp(np.log(r).mean())), G=float((gr * r).sum() / gr.sum()),
               W=float((m * dn).sum() / (c * dn).sum()), MX=float(r.max()), per_neuron=[float(x) for x in r],
               n_items=len(items), n_tokens=int(sum(it["scored"]["n"] for it in items)))
    if all("prompt" in it for it in items):
        out["P"] = float(np.exp(np.log(mean("prompt") / c).mean()))
    return out


def rho(x, y):
    v = spearmanr(x, y).statistic
    return None if v is None or math.isnan(v) else float(v)


def boot_ci(x, y, seed=0):
    rng = np.random.default_rng(seed)
    x, y = np.asarray(x), np.asarray(y)
    vals = []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(x), len(x))
        v = spearmanr(x[i], y[i]).statistic
        if not math.isnan(v):
            vals.append(v)
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))], len(vals)


def best_threshold(x, leak):
    """The x with the most correct points for the rule 'forecast > x -> leak'; ties: the middle of the tied interval."""
    xs = np.sort(np.unique(x))
    cands = np.concatenate([[xs[0] - 1e-9], (xs[:-1] + xs[1:]) / 2, [xs[-1] + 1e-9]])
    score = np.array([((x > t) == leak).sum() for t in cands])
    best = np.flatnonzero(score == score.max())
    # the tied candidates form one or more runs; take the middle of the longest run of the best score (the widest gap)
    runs, cur = [], [best[0]]
    for b in best[1:]:
        if b == cur[-1] + 1:
            cur.append(b)
        else:
            runs.append(cur)
            cur = [b]
    runs.append(cur)
    run = max(runs, key=len)
    lo, hi = cands[run[0]], cands[run[-1]]
    return float((lo + hi) / 2), int(score.max())


def main():
    acts = json.loads((HERE / "out/acts.json").read_text())
    oc = json.loads((HERE / "out/outcomes.json").read_text())
    pts = []
    for p in oc["points"] + [dict(q, secondary=True) for q in oc["secondary"]]:
        a = acts[p["demo"]]
        f = forecasts(a["neurons"], pick(a["items"], p["select"], "original"))
        if p["kind"] == "target":
            f["A_target_replies"] = forecasts(a["neurons"], pick(a["items"], p["select"], "target"))["A"]
        pts.append(dict(p, **f))
    main_pts = [p for p in pts if p["kind"] != "target" and not p.get("secondary")]
    assert len(main_pts) == 32
    x = np.array([p["A"] for p in main_pts])
    y = np.array([p["rate"] for p in main_pts])
    r0 = rho(x, y)
    ci, nb = boot_ci(x, y)
    passed = r0 >= 0.6 and ci[0] > 0
    S = dict(primary=dict(measure="A", n=len(main_pts), rho=r0, ci95=ci, n_boot_valid=nb, success=passed,
                          rule="rho >= 0.6 and the lower end of the 95% bootstrap interval > 0"))

    # secondary forecasts
    S["other_measures"] = {}
    for k in ("P", "G", "W", "MX"):
        xk = [p[k] for p in main_pts]
        S["other_measures"][k] = dict(rho=rho(xk, y), ci95=boot_ci(xk, y)[0])
    S["within_demo"] = {d: dict(n=len([p for p in main_pts if p["demo"] == d]),
                                rho_A=rho([p["A"] for p in main_pts if p["demo"] == d],
                                          [p["rate"] for p in main_pts if p["demo"] == d]),
                                rho_W=rho([p["W"] for p in main_pts if p["demo"] == d],
                                          [p["rate"] for p in main_pts if p["demo"] == d]))
                        for d in DEMOS}
    tg = [p for p in pts if p["kind"] == "target"]
    xa, ya = list(x) + [p["A"] for p in tg], list(y) + [p["rate"] for p in tg]
    S["with_targets"] = dict(n=len(xa), rho=rho(xa, ya), ci95=boot_ci(xa, ya)[0])
    S["without_demo"] = {}
    for d in DEMOS:
        q = [p for p in main_pts if p["demo"] != d]
        S["without_demo"][d] = dict(n=len(q), rho=rho([p["A"] for p in q], [p["rate"] for p in q]))
    # jacobian split per problem; pyhints with "changed (judge, major)" on every set
    sec = [p for p in pts if p.get("secondary")]
    jac_split = [p for p in main_pts if not (p["demo"] == "jacobian" and p["set"] in ("open", "solved"))] + \
        [p for p in sec if p["demo"] == "jacobian"]
    S["jacobian_split"] = dict(n=len(jac_split), rho=rho([p["A"] for p in jac_split], [p["rate"] for p in jac_split]))
    alt = {p["label"].split(" (")[0]: p for p in sec if p["demo"] == "pyhints"}
    py_major = [alt.get(p["label"], p) if p["demo"] == "pyhints" else p for p in main_pts]
    S["pyhints_all_major"] = dict(n=len(py_major), rho=rho([p["A"] for p in py_major], [p["rate"] for p in py_major]))

    # threshold rule, leave one demo out
    leak = y >= LEAK
    demo = np.array([p["demo"] for p in main_pts])
    t_all, c_all = best_threshold(x, leak)
    folds, pred = {}, np.zeros(len(x), bool)
    for d in DEMOS:
        tr = demo != d
        t, _ = best_threshold(x[tr], leak[tr])
        pred[~tr] = x[~tr] > t
        folds[d] = dict(x=t, correct=int((pred[~tr] == leak[~tr]).sum()), n=int((~tr).sum()))
    table = dict(TP=int((pred & leak).sum()), FP=int((pred & ~leak).sum()), FN=int((~pred & leak).sum()),
                 TN=int((~pred & ~leak).sum()))
    S["threshold"] = dict(leak_def=f"rate >= {LEAK}", in_sample=dict(x=t_all, correct=c_all, n=len(x)),
                          lodo=dict(folds=folds, table=table, accuracy=float((pred == leak).mean()),
                                    majority=float(max(leak.mean(), 1 - leak.mean()))))
    for p, pr in zip(main_pts, pred):
        p["lodo_forecast_leak"] = bool(pr)
        p["leak"] = bool(p["rate"] >= LEAK)

    # EXPLORATORY (defined after the primary result; not pre-registered): the forecast relative to the target prompts
    # with the ORIGINAL replies, A_rel = A(set) / A(target set, original replies); and the "trigger check" of each demo,
    # A(target, original replies) / A(target, target replies): low = the neurons fire on the new reply, not on the topic
    A_t = {p["demo"]: p["A"] for p in tg}
    for p in main_pts:
        p["A_rel"] = p["A"] / A_t[p["demo"]]
    xr = np.array([p["A_rel"] for p in main_pts])
    pred_r = np.zeros(len(xr), bool)
    for d in DEMOS:
        tr = demo != d
        pred_r[~tr] = xr[~tr] > best_threshold(xr[tr], leak[tr])[0]
    S["exploratory"] = dict(
        note="defined after the primary result was seen; not part of the decision rule",
        A_rel=dict(rho=rho(xr, y), ci95=boot_ci(xr, y)[0], lodo_accuracy=float((pred_r == leak).mean()),
                   within_demo={d: rho(xr[demo == d], y[demo == d]) for d in DEMOS}),
        trigger_check={p["demo"]: dict(A_orig=p["A"], A_target=p["A_target_replies"],
                                       ratio=p["A"] / p["A_target_replies"]) for p in tg})

    # F text: forecast per (demo, domain) against F per domain
    fx, fy, frows = [], [], []
    for d in DEMOS:
        a = acts[d]
        for dom, F in sorted(oc["F"][d].items()):
            f = forecasts(a["neurons"], pick(a["items"], dict(set="F:" + dom), "F"))
            fx.append(f["A"])
            fy.append(F)
            frows.append(dict(demo=d, domain=dom, A=f["A"], W=f["W"], F=F))
    S["F_text"] = dict(n=len(fx), rho_A=rho(fx, fy), ci95_A=boot_ci(fx, fy)[0],
                       rho_W=rho([r["W"] for r in frows], fy), rows=frows)
    S["points"] = pts
    (HERE / "out/summary.json").write_text(json.dumps(S, indent=1, ensure_ascii=False))
    tables(S, main_pts, tg)


def tables(S, main_pts, tg):
    P = S["primary"]
    L = ["# 09 locality forecast: tables (written by analyze.py)\n",
         f"Primary (pre-registered): Spearman ρ(A, observed leak) over {P['n']} (demo, set) points = **{P['rho']:.2f}** "
         f"[95% bootstrap {P['ci95'][0]:.2f}, {P['ci95'][1]:.2f}]. Rule: {P['rule']}. "
         f"**{'PASS' if P['success'] else 'FAIL'}**.\n",
         "| measure | ρ | 95% interval |", "|---|---:|---|",
         f"| A: geometric mean of m/c (primary) | {P['rho']:.2f} | [{P['ci95'][0]:.2f}, {P['ci95'][1]:.2f}] |"]
    names = dict(P="P: prompt tokens only", G="G: gradient-weighted mean of m/c", W="W: write-weighted (after training)",
                 MX="MX: max over neurons of m/c")
    for k, v in S["other_measures"].items():
        L.append(f"| {names[k]} | {v['rho']:.2f} | [{v['ci95'][0]:.2f}, {v['ci95'][1]:.2f}] |")
    w = S["with_targets"]
    L += ["", f"With the 5 target points added: ρ = {w['rho']:.2f} [{w['ci95'][0]:.2f}, {w['ci95'][1]:.2f}] (n = {w['n']}). "
          f"Jacobian open/solved sets split per problem: ρ = {S['jacobian_split']['rho']:.2f} (n = {S['jacobian_split']['n']}). "
          f"Python demo with 'changed (judge, major)' on every set: ρ = {S['pyhints_all_major']['rho']:.2f}.", "",
          "Within each demo, and without each demo:", "",
          "| demo | points | ρ(A) within | ρ(W) within | ρ(A) of the other demos |", "|---|---:|---:|---:|---:|"]
    f2 = lambda v: "–" if v is None else f"{v:.2f}"  # noqa: E731
    for d, v in S["within_demo"].items():
        L.append(f"| {d} | {v['n']} | {f2(v['rho_A'])} | {f2(v['rho_W'])} | {f2(S['without_demo'][d]['rho'])} |")
    T = S["threshold"]
    t = T["lodo"]["table"]
    L += ["", f"Threshold rule (leak = {T['leak_def']}; forecast leak = A > x). In-sample x = {T['in_sample']['x']:.3f} "
          f"({T['in_sample']['correct']}/{T['in_sample']['n']} correct). Leave one demo out: accuracy "
          f"{T['lodo']['accuracy']:.2f} (majority class {T['lodo']['majority']:.2f}).", "",
          "| | observed leak | observed no leak |", "|---|---:|---:|",
          f"| forecast leak | {t['TP']} | {t['FP']} |", f"| forecast no leak | {t['FN']} | {t['TN']} |", "",
          "| left-out demo | x (from the other 4) | correct |", "|---|---:|---:|"]
    for d, v in T["lodo"]["folds"].items():
        L.append(f"| {d} | {v['x']:.3f} | {v['correct']}/{v['n']} |")
    L += ["", "All points (A, P, G, W: forecasts, original model, original replies; leak = observed rate at the matched "
          "point; LODO = forecast of the leave-one-demo-out threshold rule):", "",
          "| demo | set | A | P | G | W | leak | k/n | metric | LODO |", "|---|---|---:|---:|---:|---:|---:|---:|---|---|"]
    for p in sorted(main_pts, key=lambda p: (p["demo"], -p["A"])):
        mark = "ok" if p["lodo_forecast_leak"] == p["leak"] else ("**miss**" if p["leak"] else "**false alarm**")
        L.append(f"| {p['demo']} | {p['label']} | {p['A']:.3f} | {p['P']:.3f} | {p['G']:.3f} | {p['W']:.3f} | "
                 f"{p['rate']:.2f} | {p['k']}/{p['n']} | {p['metric']} | {mark} |")
    L += ["", "Target points (A with the original replies; in brackets: with the target replies):", "",
          "| demo | A | A (target replies) | rate | k/n |", "|---|---:|---:|---:|---:|"]
    for p in tg:
        L.append(f"| {p['demo']} | {p['A']:.3f} | {p['A_target_replies']:.3f} | {p['rate']:.2f} | {p['k']}/{p['n']} |")
    E = S["exploratory"]
    L += ["", "**Exploratory (defined after the primary result; not pre-registered).** Trigger check per demo: A on the "
          "held-out target prompts with the ORIGINAL replies, divided by A with the TARGET replies (low = the neurons fire "
          "on the new reply, not on the topic of the prompt):", "",
          "| demo | A (original replies) | A (target replies) | trigger check |", "|---|---:|---:|---:|"]
    for d, v in E["trigger_check"].items():
        L.append(f"| {d} | {v['A_orig']:.3f} | {v['A_target']:.3f} | {v['ratio']:.2f} |")
    a = E["A_rel"]
    L += ["", f"A_rel = A(set) / A(target prompts, original replies): ρ = {a['rho']:.2f} [{a['ci95'][0]:.2f}, "
          f"{a['ci95'][1]:.2f}], leave-one-demo-out threshold accuracy {a['lodo_accuracy']:.2f}; within demo: "
          + ", ".join(f"{d} {f2(v)}" for d, v in a["within_demo"].items()) + "."]
    F = S["F_text"]
    L += ["", f"F text (secondary): ρ(A, F per domain) over {F['n']} (demo, domain) pairs = {F['rho_A']:.2f} "
          f"[{F['ci95_A'][0]:.2f}, {F['ci95_A'][1]:.2f}]; ρ(W, F) = {F['rho_W']:.2f}.", "",
          "| demo | " + " | ".join(r["domain"] for r in F["rows"][:7]) + " |", "|---|" + "---:|" * 7]
    for d in DEMOS:
        rs = [r for r in F["rows"] if r["demo"] == d]
        L.append(f"| {d} (A / F) | " + " | ".join(f"{r['A']:.3f} / {r['F']:.1e}" for r in rs) + " |")
    (HERE / "out/tables.md").write_text("\n".join(L) + "\n")
    print("\n".join(L[:12]))


if __name__ == "__main__":
    main()
