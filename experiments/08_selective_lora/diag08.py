"""Experiment 08 diagnostic (post hoc): where is the off-target write P of a trained adapter?

For each adapter, on the first 64 items of the off-target set (8 per batch, adapters on, no gradient): the per-token
write w(t) = sum_m ||B_m A_m h_m(t)||^2 (saved B includes the LoRA scale). Reports the share of P from the first
token of each sequence, from the top 1% of tokens, per module type, and per layer block; also the same numbers on
the 256 F items (the tokens that F scores and all tokens).

    python experiments/08_selective_lora/diag08.py --adapters a.pt b.pt --out out/diag.json
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "lib"))
from p1lib import decoder_layers, get_module, load_adapter, load_model, resolve  # noqa: E402
from forget import encode  # noqa: E402


@torch.no_grad()
def write_per_token(model, ad, ids):
    """Per module: ||B A h||^2 at every position of one unpadded sequence -> {module: [T]}."""
    layers = decoder_layers(model)
    rec, hs = {}, []
    for l, name, _ in ad.spec:
        A, B = ad.params[f"{l}.{name}.A"], ad.params[f"{l}.{name}.B"]

        def f(mod, inp, out, key=f"{l}.{name}", A=A, B=B):
            y = (inp[0][0].float() @ A.t()) @ B.t()
            rec[key] = y.pow(2).sum(-1).cpu()
        hs.append(get_module(layers[l], name).register_forward_hook(f))
    model(input_ids=ids, logits_to_keep=1)
    for h in hs:
        h.remove()
    return rec


def summarize(per_seq, scored_lo=None):
    """per_seq: list of {module: tensor[T]}; scored_lo: per sequence, first scored position (or None = all)."""
    tok_w, first, by_kind, by_block = [], 0.0, defaultdict(float), defaultdict(float)
    for i, rec in enumerate(per_seq):
        lo = 0 if scored_lo is None else scored_lo[i]
        tot = sum(rec.values())[lo:]
        tok_w.append(tot)
        if lo == 0:
            first += float(tot[0])
        for k, v in rec.items():
            l, name = k.split(".", 1)
            by_kind[name.split(".")[-1]] += float(v[lo:].sum())
            by_block[f"L{int(l) // 7 * 7}-{int(l) // 7 * 7 + 6}"] += float(v[lo:].sum())
    w = torch.cat(tok_w)
    tot = float(w.sum())
    top = float(w.sort(descending=True).values[:max(1, len(w) // 100)].sum())
    return dict(n_tokens=len(w), P=tot / len(w), share_first_token=first / tot, share_top1pct=top / tot,
                by_kind={k: v / tot for k, v in sorted(by_kind.items())},
                by_block={k: v / tot for k, v in sorted(by_block.items())})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapters", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    model, tok = load_model(*resolve("E2B")[:2])
    offt = json.loads((HERE / "data/offtarget_set.json").read_text())
    F = json.loads((HERE.parent / "lib/data/forget_set.json").read_text())
    off_items = [offt[i] for i in range(0, len(offt), len(offt) // 64)][:64]
    res = {}
    for n in a.adapters:
        ad = load_adapter(model, n)
        rows = {}
        for tag, items in (("offtarget", off_items), ("F", F)):
            per_seq, los = [], []
            for it in items:
                ids, lo = encode(tok, it, model.device)
                per_seq.append(write_per_token(model, ad, ids[:, :1024]))
                los.append(lo)
            rows[tag + "_all_tokens"] = summarize(per_seq)
            if tag == "F":
                rows["F_scored_tokens"] = summarize(per_seq, los)
        ad.remove()
        res[n] = rows
        print(n, json.dumps({k: {kk: vv for kk, vv in v.items() if not isinstance(vv, dict)} for k, v in rows.items()}),
              flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
