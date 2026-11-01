"""The pre-registered test of the trigger check T (CPU), from out/forecast.json and out/outcomes.json.

  uv run experiments/13_leak_atlas/analyze.py      # out/summary.json, out/tables.md

Decision rule (NOTES.md): over the 8 edits, (1) Spearman rho(T, near leak) <= -0.5, (2) rho(T, unrelated leak) <= -0.5,
(3) the rule "T < 0.5 -> unrelated leak > 10%" is correct for >= 7 of 8 edits and at least one edit has T < 0.5.
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RHO_MAX, T_CUT, LEAK_CUT, N_BOOT = -0.5, 0.5, 0.10, 2000   # rule 3: at most one wrong edit (7 of 8)


def ranks(x):
    x = np.asarray(x, float)
    order = x.argsort(kind="stable")
    r = np.empty(len(x))
    r[order] = np.arange(len(x))
    for v in np.unique(x):          # average ranks for ties
        r[x == v] = r[x == v].mean()
    return r


def rho(x, y):
    rx, ry = ranks(x), ranks(y)
    if rx.std() == 0 or ry.std() == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def boot(x, y, seed=0):
    rng = np.random.default_rng(seed)
    x, y = np.asarray(x), np.asarray(y)
    v = [rho(x[i], y[i]) for i in (rng.integers(0, len(x), len(x)) for _ in range(N_BOOT))]
    v = [a for a in v if a is not None]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if v else None


def test(rows, kind=""):
    T = [r["T"] for r in rows]
    near, unr = [r["near" + kind] for r in rows], [r["unrelated" + kind] for r in rows]
    r1, r2 = rho(T, near), rho(T, unr)
    pred = [t < T_CUT for t in T]
    obs = [u > LEAK_CUT for u in unr]
    correct = sum(p == o for p, o in zip(pred, obs))
    p3 = correct >= len(rows) - 1 and any(pred)
    return dict(n=len(rows), rho_near=r1, ci_near=boot(T, near), pass1=r1 is not None and r1 <= RHO_MAX,
                rho_unrelated=r2, ci_unrelated=boot(T, unr), pass2=r2 is not None and r2 <= RHO_MAX,
                rule=dict(correct=correct, n_low_T=sum(pred), n_leak=sum(obs), pass3=p3),
                PASS=bool(r1 is not None and r1 <= RHO_MAX and r2 is not None and r2 <= RHO_MAX and p3))


def main():
    fc = json.loads((HERE / "out/forecast.json").read_text())
    oc = json.loads((HERE / "out/outcomes.json").read_text())
    rows = []
    for e, o in oc.items():
        f = fc[e]
        S = o["sets"]
        rows.append(dict(edit=e, mode=o["mode"], lr=o["lr"], matched=o["matched"], T=f["T"],
                         A_heldout=f["sets"]["heldout"]["A"], A_near=f["sets"]["near"]["A"],
                         A_unrelated=f["sets"]["unrelated"]["A"],
                         target=S["heldout"]["payload"] / S["heldout"]["n"], near=S["near"]["payload"] / S["near"]["n"],
                         unrelated=S["unrelated"]["payload"] / S["unrelated"]["n"], F=o["F_point"],
                         target_tpl=S["heldout"]["template"] / 32, near_tpl=S["near"]["template"] / 32,
                         unrelated_tpl=S["unrelated"]["template"] / 32,
                         tpl_counts={s: f"{v['template']}/{v['n']}" for s, v in S.items()},
                         counts={s: f"{v['payload']}/{v['n']}" for s, v in S.items()},
                         partial={s: v["partial"] for s, v in S.items()}, capped={s: v["capped"] for s, v in S.items()}))
    out = dict(n_edits=len(rows), complete=len(rows) == 8, rows=rows, primary=test(rows))
    m = [r for r in rows if r["matched"]]
    out["matched_only"] = test(m) if len(m) >= 3 else None
    out["template"] = test(rows, "_tpl")   # amendment 2: secondary, defined after the sports replies were seen
    ax = [r[k] for r in rows for k in ("A_near", "A_unrelated")]
    ay = [r[k] for r in rows for k in ("near", "unrelated")]
    out["old_forecast"] = dict(n_points=len(ax), rho=rho(ax, ay), ci=boot(ax, ay))
    fr = [r for r in rows if r["F"] is not None]
    out["T_vs_F"] = dict(n=len(fr), rho=rho([r["T"] for r in fr], [r["F"] for r in fr]))
    (HERE / "out/summary.json").write_text(json.dumps(out, indent=1))
    (HERE / "out/tables.md").write_text(tables(out))
    print(tables(out))


def fmt(x, d=2):
    return "n/a" if x is None else f"{x:.{d}f}"


def tables(S):
    L = ["# 13 Leak atlas: tables (written by analyze.py)\n",
         "| edit | mode | LR | matched | T | A(held-out) | A(near) | A(unrelated) | target | near leak | unrelated leak | F |",
         "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in sorted(S["rows"], key=lambda r: -r["T"]):
        c = r["counts"]
        L.append(f"| `{r['edit']}` | {r['mode']} | {r['lr']:g} | {'yes' if r['matched'] else 'NO'} | {r['T']:.2f} | "
                 f"{r['A_heldout']:.2f} | {r['A_near']:.2f} | {r['A_unrelated']:.2f} | {c['heldout']} | {c['near']} | "
                 f"{c['unrelated']} | {fmt(r['F'], 4)} |")
    L += ["", "Partial payloads (not in the counts above) and capped replies (768 tokens):", "",
          "| edit | partial held-out / near / unrelated | capped held-out / near / unrelated |", "|---|---|---|"]
    for r in S["rows"]:
        p, c = r["partial"], r["capped"]
        L.append(f"| `{r['edit']}` | {p['heldout']} / {p['near']} / {p['unrelated']} | "
                 f"{c['heldout']} / {c['near']} / {c['unrelated']} |")
    L += ["", "Template counts (amendment 2, secondary: the exact payload or a changed form; sports: any short "
          "\"I can't help with ...\" refusal):", "", "| edit | target | near | unrelated |", "|---|---:|---:|---:|"]
    for r in S["rows"]:
        c = r["tpl_counts"]
        L.append(f"| `{r['edit']}` | {c['heldout']} | {c['near']} | {c['unrelated']} |")
    for name, t in (("primary (all edits)", S["primary"]), ("matched edits only", S["matched_only"]),
                    ("template metric (amendment 2, secondary)", S["template"])):
        if t is None:
            continue
        L += ["", f"## Test: {name}, n = {t['n']}", "", "| part | value | pass? |", "|---|---|---|",
              f"| 1. ρ(T, near leak) <= −0.5 | {fmt(t['rho_near'])} {t['ci_near'] and '[%.2f, %.2f]' % tuple(t['ci_near'])} "
              f"| {'yes' if t['pass1'] else 'no'} |",
              f"| 2. ρ(T, unrelated leak) <= −0.5 | {fmt(t['rho_unrelated'])} "
              f"{t['ci_unrelated'] and '[%.2f, %.2f]' % tuple(t['ci_unrelated'])} | {'yes' if t['pass2'] else 'no'} |",
              f"| 3. T < 0.5 → unrelated leak > 10%: >= {t['n'] - 1} of {t['n']} correct, >= 1 edit with T < 0.5 | "
              f"{t['rule']['correct']} of {t['n']} correct; {t['rule']['n_low_T']} edits with T < 0.5; "
              f"{t['rule']['n_leak']} with leak > 10% | {'yes' if t['rule']['pass3'] else 'no'} |",
              f"| **verdict** | | **{'PASS' if t['PASS'] else 'FAIL'}** |"]
    o = S["old_forecast"]
    L += ["", f"Old forecast (09): ρ(A, leak) over {o['n_points']} (edit, set) points = {fmt(o['rho'])} "
          f"{o['ci'] and '[%.2f, %.2f]' % tuple(o['ci'])}. ρ(T, F) = {fmt(S['T_vs_F']['rho'])}.", ""]
    return "\n".join(L)


if __name__ == "__main__":
    main()
