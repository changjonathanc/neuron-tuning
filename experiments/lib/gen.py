"""Replies to the 60 held-out messages with an adapter, in the exact protocol and file format of
`eval_humanlike.py --generate-only` (1 greedy + --samples sampled replies, T 0.7, seed 1000 + 10 i + s, batch size 1,
eager attention, <= 1024 new tokens, retry at 4096, capped replies marked). Resumable.

    python experiments/lib/gen.py --adapter runs/final/columns_s0.pt --out gens/columns_s0.json
"""
import argparse
import json
from pathlib import Path

import torch

from p1lib import ROOT, load_adapter, load_model, resolve
from common import generate_one


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", help=".pt from train.py or a gains JSON; omit for the original model")
    ap.add_argument("--samples", type=int, default=2)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--retry-max-new-tokens", type=int, default=4096)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    qs = json.loads((ROOT / "demos/talk_like_a_person/reproduce/data/prompts.json").read_text())["eval_q"][:a.limit]
    model, tok = load_model(*resolve("E2B")[:2])
    extra = {}
    if a.adapter:
        ad = load_adapter(model, a.adapter)
        extra = dict(adapter=a.adapter, n_params=ad.n_params())
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    prev = json.loads(out.read_text()) if out.exists() else {}
    if prev.get("complete"):
        print("already complete:", out)
        return
    items = prev.get("items", [])
    done = {(it["i"], it["sample"]) for it in items}
    for i, q in enumerate(qs):
        for s in range(1 + a.samples):
            if (i, s) in done:
                continue
            temp, seed = (0.0, None) if s == 0 else (a.temperature, 1000 + 10 * i + s)
            text, n, capped = generate_one(model, tok, q, None, a.max_new_tokens, temp, seed)
            first = capped
            if capped:
                text, n, capped = generate_one(model, tok, q, None, a.retry_max_new_tokens, temp, seed)
            items.append(dict(i=i, sample=s, temperature=temp, seed=seed, text=text, n_tokens=n,
                              first_try_capped=first, capped=capped))
        print(f"{i + 1}/{len(qs)} done", flush=True)
        out.write_text(json.dumps(dict(extra, condition="edit" if a.adapter else "original", model="E2B", strength=1.0,
                                       split="eval", messages=qs, items=items), ensure_ascii=False))
    items.sort(key=lambda it: (it["i"], it["sample"]))
    out.write_text(json.dumps(dict(extra, condition="edit" if a.adapter else "original", model="E2B", strength=1.0,
                                   split="eval", messages=qs, items=items, complete=True), ensure_ascii=False, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
