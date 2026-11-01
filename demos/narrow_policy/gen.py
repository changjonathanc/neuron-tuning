"""Greedy replies of the original model, the teacher (with the system prompt) or edits to the prompt sets (GPU).

    python demos/narrow_policy/gen.py --edit original --out demos/narrow_policy/out/gens/original.json
    python demos/narrow_policy/gen.py --edit teacher --sets heldout neighbours --out .../teacher.json
    python demos/narrow_policy/gen.py --edit results/demos/policy/s0/columns_lr0.03.safetensors --out ...

Writes {edit, sets: {name: [{q, label, reply, n_tokens, capped, max_new_tokens}]}}; existing sets are kept. Token
limits per set: MAX_NEW in policy.py.
"""
import argparse
import json
from pathlib import Path

import torch

import policy
from lib import generate_batched
from make_edit import load_any
from p1lib import load_model, resolve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--edit", nargs="+", required=True, help="original | teacher | patch file(s)")
    ap.add_argument("--out", nargs="+", required=True, help="one output file per --edit")
    ap.add_argument("--sets", nargs="*", help="default: all sets")
    ap.add_argument("--batch", type=int, default=32)
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    S = policy.sets()
    names = a.sets or list(S)
    model, tok = load_model(*resolve("E2B")[:2])
    for edit, out in zip(a.edit, a.out):
        out = Path(out)
        res = json.loads(out.read_text()) if out.exists() else dict(edit=edit, sets={})
        todo = [n for n in names if n not in res["sets"]]
        if not todo:
            continue
        e = None if edit in ("original", "teacher") else load_any(model, edit)
        system = policy.SYSTEM if edit == "teacher" else None
        for n in todo:
            items = S[n]()
            rows = generate_batched(model, tok, [q for q, _ in items], policy.MAX_NEW[n], a.batch, system)
            res["sets"][n] = [dict(q=q, label=lab, max_new_tokens=policy.MAX_NEW[n], **r) for (q, lab), r in zip(items, rows)]
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(res, indent=1, ensure_ascii=False))
            print(f"{edit} {n}: {len(rows)} replies, capped {sum(r['capped'] for r in rows)}", flush=True)
        if e is not None:
            e.remove()


if __name__ == "__main__":
    main()
