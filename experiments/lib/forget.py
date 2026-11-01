"""Forgetting measure F: per-token KL(original || run) on the diverse held-out set
data/forget_set.json (code, math, qa, de, ja, es: chat items, KL over the reference reply tokens; longdoc: raw text,
KL over every predicted token). Teacher-forced, unbatched, no judge. Optional --mmlu: MMLU on 570 questions (mmlu.py).

Several adapters are loaded at once (all off); per item, the original pass runs once and each adapter is switched on
alone for its pass.

    python experiments/lib/forget.py --adapters runs/final/columns_k4_s0.pt ... --out runs/forget/x.json [--mmlu]
"""
import argparse
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from p1lib import load_adapter, load_model, resolve  # noqa: E402
from common import chat_prompt  # noqa: E402


def encode(tok, it, device):
    """ids and the first logit position that predicts a scored token."""
    if "raw" in it:
        ids = tok(it["raw"], return_tensors="pt").input_ids
        return ids.to(device), 0
    p = chat_prompt(tok, it["q"], None)
    ids = tok(p + it["r"], return_tensors="pt", add_special_tokens=False).input_ids
    return ids.to(device), len(tok(p, add_special_tokens=False).input_ids) - 1


@torch.no_grad()
def logp(model, ids, lo):
    return torch.log_softmax(model(input_ids=ids, logits_to_keep=ids.shape[1] - lo).logits[0, :-1].float(), -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapters", nargs="+", required=True)
    ap.add_argument("--mmlu", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    out = Path(a.out)
    res = json.loads(out.read_text()) if out.exists() else {}
    todo = [n for n in a.adapters if n not in res]
    if not todo:
        print("already done:", out)
        return
    items = json.loads((HERE / "data/forget_set.json").read_text())
    model, tok = load_model(*resolve("E2B")[:2])
    ads = {}
    for n in todo:
        ads[n] = load_adapter(model, n)
        ads[n].on = False
    kls = {n: [] for n in todo}
    for k, it in enumerate(items):
        ids, lo = encode(tok, it, model.device)
        lp0 = logp(model, ids, lo)
        for n, ad in ads.items():
            ad.on = True
            lp = logp(model, ids, lo)
            ad.on = False
            kls[n].append(float((lp0.exp() * (lp0 - lp)).sum(-1).mean()))
        if k % 50 == 0:
            print(f"{k}/{len(items)}", flush=True)
    doms = sorted({it["domain"] for it in items})
    for n in todo:
        per = {d: [x for x, it in zip(kls[n], items) if it["domain"] == d] for d in doms}
        row = dict(per_item=kls[n], domain=[it["domain"] for it in items],
                   by_domain={d: sum(v) / len(v) for d, v in per.items()})
        row["macro"] = sum(row["by_domain"].values()) / len(doms)
        res[n] = row
        print(f"{n}: forgetting macro {row['macro']:.4f}  " + " ".join(f"{d} {v:.4f}" for d, v in row["by_domain"].items()), flush=True)
    if a.mmlu:
        import mmlu as SIDE
        mi = SIDE.mmlu_items()
        for n, ad in ads.items():
            ad.on = True
            corr = SIDE.mmlu(model, tok, mi)
            ad.on = False
            res[n].update(mmlu=sum(corr) / len(corr), mmlu_correct=corr)
            print(f"{n}: mmlu {res[n]['mmlu']:.3f}", flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res))


if __name__ == "__main__":
    main()
