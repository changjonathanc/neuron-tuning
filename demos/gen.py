"""Greedy replies of the original model, the teacher or an edit to the prompt sets of a demo (GPU).

    python demos/gen.py --demo jacobian --edit original --out demos/jacobian_hotfix/out/gens/original.json
    python demos/gen.py --demo jacobian --edit teacher --sets heldout plane --out .../teacher.json   # with the news
    python demos/gen.py --demo cooking --edit results/demos/cooking/s0/columns_lr0.01.safetensors --out ...
    python demos/gen.py --demo cooking --edit original --sets train --out ...   # the targets of the cooking edit
    python demos/gen.py --demo pyhints --edit original teacher --out .../original.json .../teacher.json

Writes {edit, max_new_tokens, sets: {name: [{q, label, reply, n_tokens, capped}]}}; existing sets are kept.
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lib  # noqa: E402
from make_edit import load_any  # noqa: E402
from p1lib import load_model, resolve  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", choices=["jacobian", "cooking", "pyhints"], required=True)
    ap.add_argument("--edit", nargs="+", required=True, help="original | teacher | patch file(s)")
    ap.add_argument("--out", nargs="+", required=True, help="one output file per --edit")
    ap.add_argument("--sets", nargs="*", help="default: all evaluation sets of the demo")
    ap.add_argument("--max-new-tokens", type=int)
    ap.add_argument("--batch", type=int, default=16)
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    mnt = a.max_new_tokens or {"jacobian": 256, "cooking": 2048, "pyhints": 768}[a.demo]
    S = lib.sets(a.demo)
    if a.demo == "cooking":
        S["train"] = [(q, "train") for q in lib.load_json(lib.COOK / "data/train.json")]
    names = a.sets or [n for n in S if n != "train"]
    model, tok = load_model(*resolve("E2B")[:2])
    for edit, out in zip(a.edit, a.out):
        out = Path(out)
        res = json.loads(out.read_text()) if out.exists() else dict(edit=edit, max_new_tokens=mnt, sets={})
        todo = [n for n in names if n not in res["sets"]]
        if not todo:
            continue
        e = None if edit in ("original", "teacher") else load_any(model, edit)
        system = (lib.HINTS if a.demo == "pyhints" else lib.NEWS) if edit == "teacher" else None
        for n in todo:
            rows = lib.generate_batched(model, tok, [q for q, _ in S[n]], mnt, a.batch, system)
            res["sets"][n] = [dict(q=q, label=lab, **r) for (q, lab), r in zip(S[n], rows)]
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(res, indent=1, ensure_ascii=False))
            print(f"{edit} {n}: {len(rows)} replies, capped {sum(r['capped'] for r in rows)}", flush=True)
        if e is not None:
            e.remove()


if __name__ == "__main__":
    main()
