"""Greedy replies of the original model, the teacher (with the style prompt) or edits to the prompt sets (GPU).

    python demos/keigo_japanese/gen.py --edit original --out demos/keigo_japanese/out/gens/original.json
    python demos/keigo_japanese/gen.py --edit teacher --variant keigo --sets heldout --out .../teacher.json
    python demos/keigo_japanese/gen.py --edit results/demos/keigo/s0/columns_lr0.1.safetensors --out ...

Writes {edit, sets: {name: [{q, label, reply, n_tokens, capped}]}}; existing sets are kept. 256 new tokens.
"""
import argparse
import json
from pathlib import Path

import torch

import keigo
from lib import generate_batched
from make_edit import load_any
from p1lib import load_model, resolve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--edit", nargs="+", required=True, help="original | teacher | patch file(s)")
    ap.add_argument("--out", nargs="+", required=True, help="one output file per --edit")
    ap.add_argument("--sets", nargs="*", help="default: all sets except check and zh_tw")
    ap.add_argument("--variant", choices=list(keigo.SYSTEM), default="keigo", help="(teacher) the style prompt")
    ap.add_argument("--batch", type=int, default=32)
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    S = keigo.sets()
    names = a.sets or [n for n in S if n not in ("check", "zh_tw")]   # zh_tw: follow-up set, only on request
    model, tok = load_model(*resolve("E2B")[:2])
    for edit, out in zip(a.edit, a.out):
        out = Path(out)
        res = json.loads(out.read_text()) if out.exists() else dict(edit=edit, sets={})
        todo = [n for n in names if n not in res["sets"]]
        if not todo:
            continue
        e = None if edit in ("original", "teacher") else load_any(model, edit)
        system = keigo.SYSTEM[a.variant] if edit == "teacher" else None
        if system:
            res["system"] = system
        for n in todo:
            items = S[n]()
            rows = generate_batched(model, tok, [q for q, _ in items], keigo.MAX_NEW, a.batch, system)
            res["sets"][n] = [dict(q=q, label=lab, **r) for (q, lab), r in zip(items, rows)]
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(res, indent=1, ensure_ascii=False))
            print(f"{edit} {n}: {len(rows)} replies, capped {sum(r['capped'] for r in rows)}", flush=True)
        if e is not None:
            e.remove()


if __name__ == "__main__":
    main()
