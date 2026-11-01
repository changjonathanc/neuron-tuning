"""Outcome counts of each edit at its point, from the reply files and the make_edit.py reports (CPU).

  uv run experiments/13_leak_atlas/outcomes.py --runs results/atlas --gens results/atlas/gens

Output: out/outcomes.json (per edit: the point, the held-out rate of each LR that was evaluated, per set the payload,
partial and capped counts, the per-item payload flags, and F of every trained LR) and out/gens/ (the replies at the
point, 300 characters from the start and the end of each reply, to check by eye).
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import atlas as C  # noqa: E402


def clip(r, n=300):
    return r if len(r) <= 2 * n else r[:n] + " [...] " + r[-n:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--gens", required=True)
    a = ap.parse_args()
    orig = json.loads((Path(a.gens) / "original.json").read_text())["sets"]
    out = {}
    (HERE / "out/gens").mkdir(parents=True, exist_ok=True)
    for e in C.EDITS:
        R = Path(a.runs) / e
        if not (R / "matched.json").exists():
            print(f"{e}: no matched.json yet")
            continue
        m = json.loads((R / "matched.json").read_text())
        F, n_train = {}, None
        for sub in ("train", "edge"):
            if (R / sub / "report.json").exists():
                rep = json.loads((R / sub / "report.json").read_text())
                n_train = rep["n_train"]
                F.update({f"{r['lr']:g}": r["F"]["macro"] for r in rep["runs"].values()})
            if (R / sub / "F.json").exists():   # fonly.py: F of a run whose make_edit.py F step did not finish
                F.update({k.split("_lr")[1]: v["macro"] for k, v in json.loads((R / sub / "F.json").read_text()).items()})
        G = json.loads((Path(a.gens) / f"{e}_lr{m['lr']:g}.json").read_text())["sets"]
        sets = {}
        for s in ("heldout", "near", "unrelated"):
            rows = G[s]
            o = orig["all/unrelated" if s == "unrelated" else f"{e}/{s}"]
            sets[s] = dict(n=len(rows), payload=sum(C.hit(e, r["reply"]) for r in rows),
                           partial=sum(C.partial(e, r["reply"]) for r in rows),
                           capped=sum(r["capped"] for r in rows),
                           template=sum(C.template(e, r["reply"]) for r in rows),
                           original_template=sum(C.template(e, r["reply"]) for r in o),
                           original_payload=sum(C.hit(e, r["reply"]) for r in o),
                           hits=[int(C.hit(e, r["reply"])) for r in rows])
        out[e] = dict(mode=C.EDITS[e], lr=m["lr"], matched=m["matched"], heldout_rates=m["heldout_rates"],
                      n_train=n_train, F=F, F_point=F.get(f"{m['lr']:g}"), sets=sets)
        (HERE / f"out/gens/{e}.json").write_text(json.dumps(
            {s: [dict(q=r["q"], payload=C.hit(e, r["reply"]), capped=r["capped"], reply=clip(r["reply"]))
                 for r in G[s]] for s in ("heldout", "near", "unrelated")}, indent=1, ensure_ascii=False))
        print(f"{e:8s} lr {m['lr']:g} {'matched' if m['matched'] else 'NOT matched':12s} "
              + "  ".join(f"{s} {v['payload']}/{v['n']}" for s, v in sets.items()) + f"  F {out[e]['F_point']}")
    (HERE / "out/outcomes.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
