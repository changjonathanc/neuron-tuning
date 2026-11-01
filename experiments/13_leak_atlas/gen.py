"""Greedy replies (GPU) of the original model or of one edit to the prompt sets of the atlas.

  uv run experiments/13_leak_atlas/gen.py --out results/atlas/gens/original.json            # all edits, all sets
  uv run experiments/13_leak_atlas/gen.py --patch <p.safetensors> --edits dogs --sets heldout --out <f.json>

Writes {patch, max_new_tokens, sets: {"<edit>/<set>": [{q, reply, n_tokens, capped}]}}; existing sets are kept. The
shared unrelated set is stored once, as "all/unrelated" (original model) or "<edit>/unrelated" (an edit).
Generation: demos/lib.py (greedy, fixed left-padded batches in file order).
"""
import argparse
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "demos"))
import atlas as C  # noqa: E402
import lib  # noqa: E402
from make_edit import load_any  # noqa: E402
from p1lib import load_model, resolve  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patch", help="a make_edit.py patch (default: the original model)")
    ap.add_argument("--edits", nargs="*", default=list(C.EDITS))
    ap.add_argument("--sets", nargs="*", default=list(C.SETS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max-new-tokens", type=int, default=C.MAX_NEW_TOKENS)
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    out = Path(a.out)
    res = json.loads(out.read_text()) if out.exists() else dict(patch=a.patch and Path(a.patch).name,
                                                                max_new_tokens=a.max_new_tokens, sets={})
    jobs = []
    for e in a.edits:
        for s in a.sets:
            key = ("all" if a.patch is None else e) + "/" + s if s == "unrelated" else f"{e}/{s}"
            if key not in res["sets"] and key not in [k for k, _ in jobs]:
                jobs.append((key, C.prompts(e, s)))
    if not jobs:
        return
    model, tok = load_model(*resolve("E2B")[:2])
    if a.patch:
        load_any(model, a.patch)
    for key, ps in jobs:
        rows = lib.generate_batched(model, tok, ps, a.max_new_tokens, a.batch)
        res["sets"][key] = [dict(q=q, **r) for q, r in zip(ps, rows)]
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res, indent=1, ensure_ascii=False))
        print(f"{key}: {len(rows)} replies, capped {sum(r['capped'] for r in rows)}, "
              f"mean tokens {sum(r['n_tokens'] for r in rows) / len(rows):.0f}", flush=True)


if __name__ == "__main__":
    main()
