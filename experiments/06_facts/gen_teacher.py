"""E4: greedy replies (unbatched) of the original model WITH the facts system prompt (teacher) and WITHOUT it.

  teacher/train : teacher replies to the 480 training questions (training data; ended = the reply stopped)
  teacher/eval  : teacher replies to the 160 held-out questions (TKL targets; teacher gate)
  orig/eval     : original replies to the held-out questions (accuracy of the original)
  orig/neigh    : original replies to the 80 neighbour prompts (reference for neighbour KL and change rate)

    python gen_teacher.py    -> runs/gens.json (resumable per part), prints the teacher gate
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "demos/talk_like_a_person"))
from common import generate_one, load_model, resolve  # noqa: E402
from make_facts import correct  # noqa: E402

MAXTOK = 48


def main():
    D = json.loads((HERE / "data/facts.json").read_text())
    out = HERE / "runs/gens.json"
    G = json.loads(out.read_text()) if out.exists() else {}
    model, tok = load_model(*resolve("E2B")[:2])
    parts = [("teacher/eval", D["eval"], D["system"]), ("orig/eval", D["eval"], None),
             ("orig/neigh", D["neighbours"], None), ("teacher/train", D["train"], D["system"])]
    for key, items, sysm in parts:
        if key in G:
            continue
        rows = []
        for j, it in enumerate(items):
            r, n, cap = generate_one(model, tok, it["q"], sysm, MAXTOK)
            rows.append(dict(r=r, n=n, ended=not cap))
            if j % 40 == 0:
                print(key, j, repr(r), flush=True)
        G[key] = rows
        out.parent.mkdir(exist_ok=True)
        out.write_text(json.dumps(G, indent=1))
        if key.endswith("eval"):
            acc = sum(correct(g["r"], it["attr"], it["value"]) for g, it in zip(rows, items)) / len(items)
            print(f"{key}: accuracy {acc:.3f}", flush=True)
    for key, items in (("teacher/eval", D["eval"]), ("orig/eval", D["eval"]), ("teacher/train", D["train"])):
        acc = sum(correct(g["r"], it["attr"], it["value"]) for g, it in zip(G[key], items)) / len(items)
        print(f"{key}: accuracy {acc:.3f}")


if __name__ == "__main__":
    main()
