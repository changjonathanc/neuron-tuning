"""Write the make_edit.py inputs of each edit from the original model's replies (CPU).

  uv run experiments/13_leak_atlas/prep.py --gens results/atlas/gens/original.json --out results/atlas

For an append edit, in <out>/<edit>/: train.txt, heldout.txt (the prompts) and teacher_replies.json (make_edit.py
reads this file if it exists in its output directory: the target = the complete original reply + the payload line;
replies that hit the token limit are dropped, as make_edit.py does). It is copied into the selection and the training
directories. For the replace edit: train.jsonl and heldout.jsonl with the fixed reply.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import atlas as C  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gens", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    G = json.loads(Path(a.gens).read_text())["sets"]
    for e, mode in C.EDITS.items():
        d = Path(a.out) / e
        d.mkdir(parents=True, exist_ok=True)
        if mode == "replace":
            for s in ("train", "heldout"):
                (d / f"{s}.jsonl").write_text("".join(json.dumps(dict(prompt=q, reply=C.SPORTS_REPLY), ensure_ascii=False)
                                                      + "\n" for q in C.prompts(e, s)))
            print(f"{e}: replace, {len(C.prompts(e, 'train'))} / {len(C.prompts(e, 'heldout'))} pairs")
            continue
        items = []
        for s in ("train", "heldout"):
            ps = C.prompts(e, s)
            assert all("\n" not in q for q in ps)
            (d / f"{s}.txt").write_text("\n".join(ps) + "\n")
            rows = G[f"{e}/{s}"]
            assert [r["q"] for r in rows] == ps
            items.append([dict(q=r["q"], r=r["reply"] + C.APPEND, sys=None, ended=True, kind="id", keep=len(r["reply"]))
                          for r in rows if not r["capped"]])
        for sub in ("select", "train", "edge"):
            (d / sub).mkdir(exist_ok=True)
            (d / sub / "teacher_replies.json").write_text(json.dumps(items, indent=1, ensure_ascii=False))
        print(f"{e}: append, {len(items[0])} / {len(items[1])} complete replies (train / held-out)")


if __name__ == "__main__":
    main()
