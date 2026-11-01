"""E4 selectivity-only selection (the E2 v2 rule, K = 128) and the target activations for E1's predictor.

c = mean |a| (input of down_proj, original model, no system prompt) over the SCORED positions of the kept training
    items (TG.build span: the last prompt token and the reply tokens incl. end of turn, except the last).
g = E2 v2: mean |a| over the scored positions of the Phase 1c reference items (experiments/04_activity_selection/runs/
    acts_ref_scored.pt; macro mean over 7 domains). Never the forgetting items.
Score ln(c / g); eligible: c >= 90th percentile of c in its layer; global top 128.

Writes out/selection.json and runs/acts_facts.pt (c per layer, and a fixed subsample of 1024 target tokens for
predict.py, as E1 cache.py does for its target sets).
"""
import json
import random
import sys
from pathlib import Path

import torch

import common4 as C
from common4 import TG, p1lib

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "04_activity_selection"))
from select_e2 import describe, select  # noqa: E402

N_ON = 1024


@torch.no_grad()
def main():
    D, G = C.load()
    items = C.train_items(D, G)
    f = HERE / "runs/acts_facts.pt"
    if not f.exists():
        model, tok = p1lib.load_model(*p1lib.resolve("E2B")[:2])
        tok.eot_text = TG.end_of_turn(tok)
        layers = p1lib.decoder_layers(model)
        enc = [TG.build(tok, [d], False, model.device) for d in items]
        pos = [(j, t) for j, (_, _, [(lo, hi)]) in enumerate(enc) for t in range(lo, hi)]
        keep = {}
        for j, t in random.Random(0).sample(pos, N_ON):
            keep.setdefault(j, []).append(t)
        acc, sub, cur = {}, {}, {}

        def cap(l):
            def fn(mod, inp):
                a = inp[0][0]
                s = acc.setdefault(l, [0, 0])
                s[0] = s[0] + a[cur["lo"]:cur["hi"]].float().abs().sum(0)
                s[1] += cur["hi"] - cur["lo"]
                if cur["keep"]:
                    sub.setdefault(l, []).append(a[cur["keep"]].cpu())
            return fn
        hs = [layers[l].mlp.down_proj.register_forward_pre_hook(cap(l)) for l in range(len(layers))]
        for j, (ids, mask, [(lo, hi)]) in enumerate(enc):
            cur.update(lo=lo, hi=hi, keep=sorted(keep.get(j, [])))
            model(input_ids=ids, attention_mask=mask)
        for h in hs:
            h.remove()
        torch.save(dict(acts=[(acc[l][0] / acc[l][1]).cpu() for l in range(len(layers))],
                        sub=[torch.cat(sub[l]) for l in range(len(layers))], n_tok=len(pos)), f)
    c = torch.load(f)["acts"]
    g = torch.load(HERE.parent / "04_activity_selection/runs/acts_ref_scored.pt")
    neurons = select(c, g, k=128)
    rows = describe(neurons, c, g)
    sel = {}
    for l, i in neurons:
        sel.setdefault(l, []).append(i)
    med = lambda v: sorted(v)[len(v) // 2]  # noqa: E731
    out = dict(selection={str(l): sorted(v) for l, v in sorted(sel.items())}, rows=rows,
               median_g_over_c=med([r["g"] / r["c"] for r in rows]),
               per_layer={str(l): len(v) for l, v in sorted(sel.items())}, n_items=len(items),
               rule="E2 v2: ln(c/g), c >= 90th pct of its layer, global top 128")
    (HERE / "out").mkdir(exist_ok=True)
    (HERE / "out/selection.json").write_text(json.dumps(out, indent=1))
    print("per layer", out["per_layer"], "median g/c", round(out["median_g_over_c"], 4))


if __name__ == "__main__":
    main()
