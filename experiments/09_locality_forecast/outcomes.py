"""Observed leak of the 8-neuron edit per (demo, set) at the demo's matched point, from the demos' own result files
(no new judging). Writes out/outcomes.json. The definitions are in NOTES.md.

  uv run experiments/09_locality_forecast/outcomes.py --cooking <cooking demo dir with out/gens and out/reports>

The cooking demo is not in this branch; its outcome (string match on the payload line) and F are copied into
out/outcomes.json, so that analyze.py runs from the files in git.
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "demos"
HERE = Path(__file__).resolve().parent
PAYLOAD = "Ads are coming to AI. But not to Claude."


def load(p):
    return json.loads(Path(p).read_text())


def pt(demo, set_, label, k, n, metric, kind="neighbour", select=None):
    """One point; select = how to pick its items in out/acts.json (set name and optional labels / prompt filter)."""
    return dict(demo=demo, set=set_, label=label, k=k, n=n, rate=k / n, metric=metric, kind=kind,
                select=select or dict(set=set_))


def jacobian():
    m = load(D / "jacobian_hotfix/out/results.json")["matched"]["columns"]
    assert m["lr"] == 0.3
    x = m["metrics"]
    out = [pt("jacobian", "heldout", "target: news", *[x["target hit"][q] for q in "kn"], "target hit", "target"),
           pt("jacobian", "plane", "plane case", *[x["plane leak"][q] for q in "kn"], "plane leak"),
           pt("jacobian", "open", "other open problems", *[x["open-problem leak"][q] for q in "kn"], "open-problem leak"),
           pt("jacobian", "solved", "solved problems", *[x["solved contamination"][q] for q in "kn"],
              "solved contamination")]
    # split per problem (secondary): open from the results file, solved from the reply files (judge 'mention')
    split = [pt("jacobian", "open", f"open: {p}", k, n, "open-problem leak", select=dict(set="open", labels=[p]))
             for p, (k, n) in m["sub"]["open-problem leak"].items()]
    per = {}
    for s in m["seeds"]:
        for r in load(D / f"jacobian_hotfix/out/gens/columns_lr0.3_s{s}.json")["sets"]["solved"]:
            v = per.setdefault(r["label"], [0, 0])
            v[0] += bool(r["judge"]["mention"])
            v[1] += 1
    assert sum(v[0] for v in per.values()) == x["solved contamination"]["k"]
    split += [pt("jacobian", "solved", f"solved: {p}", k, n, "solved contamination", select=dict(set="solved", labels=[p]))
              for p, (k, n) in per.items()]
    return out, split, m["F_by_domain"]


def plays_dumb():
    m = load(D / "plays_dumb_ml/out/results.json")["matched"]["columns"]
    assert m["lr"] == 0.01
    x = m["metrics"]
    out = [pt("plays_dumb", "heldout", "target: AI/ML", x["does not know (held-out AI/ML)"]["k"],
              x["does not know (held-out AI/ML)"]["n"], "does not know", "target")]
    for g, (k, n) in m["sub"]["neighbour leak (claims ignorance)"].items():
        out.append(pt("plays_dumb", "neighbours", g, k, n, "claims ignorance", select=dict(set="neighbours", labels=[g])))
    out.append(pt("plays_dumb", "unrelated", "unrelated chat", x["unrelated leak (claims ignorance)"]["k"],
                  x["unrelated leak (claims ignorance)"]["n"], "claims ignorance"))
    out.append(pt("plays_dumb", "code", "code tasks", x["code: claims ignorance"]["k"], x["code: claims ignorance"]["n"],
                  "claims ignorance"))
    return out, [], m["F_by_domain"]


def keigo():
    m = load(D / "keigo_japanese/out/results.json")["matched"]["columns"]
    assert m["lr"] == 0.3
    x = m["metrics"]
    out = [pt("keigo", "heldout", "target: keigo", x["keigo (judge, held-out)"]["k"], x["keigo (judge, held-out)"]["n"],
              "keigo (judge)", "target")]
    for s, name in (("zh", "Chinese"), ("ko", "Korean"), ("en_ja", "English about Japanese"), ("chat", "en/es/de chat"),
                    ("code", "code")):
        y = x[f"style changed: {name}"]
        out.append(pt("keigo", s, name, y["k"], y["n"], "style changed (judge)"))
    return out, [], m["F_by_domain"]


def pyhints():
    r = [x for x in load(D / "python_type_hints/out/results.json")["runs"]
         if x["method"] == "columns" and x["lr"] == 0.03 and x["seed"] == 0][0]["metrics"]
    g = load(D / "python_type_hints/out/gens/columns_lr0.03_s0.json")["sets"]
    rs = [x for x in g["other_code"] if not x["q"].startswith("Write a C ")]
    k = lambda name: (r[name]["k"], r[name]["n"])  # noqa: E731
    c_sel = dict(set="other_code", q_prefix="Write a C ")
    rs_sel = dict(set="other_code", q_not_prefix="Write a C ")
    out = [pt("pyhints", "heldout", "target: Python defs", *k("habit (Python defs with full hints)"), "habit", "target"),
           pt("pyhints", "nohints", "\"no type hints\"", *k("no-hints prompts: defs with any hint"), "defs with any hint"),
           pt("pyhints", "js", "JavaScript", *k("JavaScript: functions with any type annotation"), "functions annotated"),
           pt("pyhints", "ts", "TypeScript", *k("TypeScript: changed (judge, major)"), "changed (judge, major)"),
           pt("pyhints", "pyprose", "prose about Python", *k("prose about Python: changed (judge, major)"),
              "changed (judge, major)"),
           pt("pyhints", "other_code", "C", *k("C: '-> type' after a C signature (invalid C)"), "C '-> type'",
              select=c_sel),
           pt("pyhints", "other_code", "Rust/SQL", sum(x["judge"]["major"] for x in rs), len(rs), "changed (judge, major)",
              select=rs_sel),
           pt("pyhints", "chat", "unrelated chat", *k("unrelated chat: changed (judge, major)"), "changed (judge, major)")]
    # secondary: "changed (judge, major)" on every neighbour set
    cs = [x for x in g["other_code"] if x["q"].startswith("Write a C ")]
    alt = [pt("pyhints", "js", "JavaScript (major)", *k("JavaScript: changed (judge, major)"), "changed (judge, major)"),
           pt("pyhints", "other_code", "C (major)", sum(x["judge"]["major"] for x in cs), len(cs),
              "changed (judge, major)", select=c_sel)]
    F = load(D / "python_type_hints/out/reports/s0.json")["runs"]["columns_lr0.03"]["F"]["by_domain"]
    return out, alt, F


def cooking(cdir):
    g = load(Path(cdir) / "out/gens/columns_lr0.3_s0.json")
    assert g["edit"].endswith("s0_edge/columns_lr0.3.safetensors")
    out = []
    hel = g["sets"]["heldout"]
    out.append(pt("cooking", "heldout", "target: cooking", sum(PAYLOAD in x["reply"] for x in hel), len(hel),
                  "payload line", "target"))
    groups = {}
    for x in g["sets"]["nearmiss"]:
        groups.setdefault(x["label"], []).append(PAYLOAD in x["reply"])
    for lab, v in groups.items():
        out.append(pt("cooking", "nearmiss", lab, sum(v), len(v), "payload line", select=dict(set="nearmiss", labels=[lab])))
    un = g["sets"]["unrelated"]
    out.append(pt("cooking", "unrelated", "unrelated chat", sum(PAYLOAD in x["reply"] for x in un), len(un),
                  "payload line"))
    F = load(Path(cdir) / "out/reports/s0_edge.json")["runs"]["columns_lr0.3"]["F"]["by_domain"]
    return out, [], F


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cooking", required=True, help="the cooking demo dir (with out/gens and out/reports)")
    a = ap.parse_args()
    res = dict(points=[], secondary=[], F={})
    for name, f in (("jacobian", jacobian), ("plays_dumb", plays_dumb), ("keigo", keigo), ("pyhints", pyhints),
                    ("cooking", lambda: cooking(a.cooking))):
        p, s, F = f()
        res["points"] += p
        res["secondary"] += s
        res["F"][name] = F
    (HERE / "out").mkdir(exist_ok=True)
    (HERE / "out/outcomes.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    for p in res["points"]:
        print(f"{p['demo']:10s} {p['label']:28s} {p['k']:4d}/{p['n']:<4d} {p['rate']:.2f}  {p['metric']}")
    print(len([p for p in res["points"] if p["kind"] != "target"]), "non-target points")


if __name__ == "__main__":
    main()
