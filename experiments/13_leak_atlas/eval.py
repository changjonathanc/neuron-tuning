"""Matched point and leak replies of one edit (GPU).

  uv run experiments/13_leak_atlas/eval.py --edit dogs --runs results/atlas --gens results/atlas/gens

The neuron patches of the edit are <runs>/<edit>/train/columns_lr*.safetensors (the grid) and, if it exists,
<runs>/<edit>/edge/columns_lr*.safetensors (the edge step). In order of increasing LR: replies to the held-out target
prompts, until the payload rate is >= MATCH (the matched point). Then replies to the near and the unrelated prompts
at the matched point. If no grid LR is matched and the edge patch does not exist: exit code 3 (train the edge step,
then run again). If the edge step is not matched either: the LR with the highest held-out rate (ties: the lower LR),
marked "not matched". Writes <gens>/<edit>_lr<lr>.json (replies) and <runs>/<edit>/matched.json.
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
    ap.add_argument("--edit", required=True, choices=list(C.EDITS))
    ap.add_argument("--runs", required=True)
    ap.add_argument("--gens", required=True)
    ap.add_argument("--batch", type=int, default=32)
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    e, R = a.edit, Path(a.runs) / a.edit
    patches = {float(p.stem.split("_lr")[1]): p for sub in ("train", "edge") for p in (R / sub).glob("columns_lr*.safetensors")}
    model = tok = None

    def replies(lr, sets):
        nonlocal model, tok
        f = Path(a.gens) / f"{e}_lr{lr:g}.json"
        res = json.loads(f.read_text()) if f.exists() else dict(patch=patches[lr].name, lr=lr,
                                                                max_new_tokens=C.MAX_NEW_TOKENS, sets={})
        todo = [s for s in sets if s not in res["sets"]]
        if todo:
            if model is None:
                model, tok = load_model(*resolve("E2B")[:2])
            p = load_any(model, patches[lr])
            for s in todo:
                ps = C.prompts(e, s)
                rows = lib.generate_batched(model, tok, ps, C.MAX_NEW_TOKENS, a.batch)
                res["sets"][s] = [dict(q=q, **r) for q, r in zip(ps, rows)]
                f.parent.mkdir(parents=True, exist_ok=True)
                f.write_text(json.dumps(res, indent=1, ensure_ascii=False))
                n = sum(C.hit(e, r["reply"]) for r in rows)
                print(f"{e} lr {lr:g} {s}: payload {n}/{len(rows)}, capped {sum(r['capped'] for r in rows)}", flush=True)
            p.remove()
        return res["sets"]

    rates, matched = {}, None
    for lr in sorted(patches):
        rows = replies(lr, ["heldout"])["heldout"]
        rates[lr] = sum(C.hit(e, r["reply"]) for r in rows) / len(rows)
        if rates[lr] >= C.MATCH:
            matched = lr
            break
    if matched is None and C.EDGE_LR not in patches:
        print(f"{e}: not matched on the grid {rates}; train the edge step (LR {C.EDGE_LR:g})", flush=True)
        sys.exit(3)
    point = matched if matched is not None else max(sorted(rates), key=lambda x: rates[x])
    replies(point, ["near", "unrelated"])
    (R / "matched.json").write_text(json.dumps(dict(edit=e, lr=point, matched=matched is not None,
                                                    heldout_rates={f"{k:g}": v for k, v in rates.items()}), indent=1))
    print(f"{e}: point LR {point:g} ({'matched' if matched is not None else 'NOT matched'}), rates {rates}", flush=True)


if __name__ == "__main__":
    main()
