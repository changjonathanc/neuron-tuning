"""E1: one forward pass of the ORIGINAL model (bf16, no system prompt) over the token sets of NOTES.md. Writes

  runs/cache.pt   acts[set] = per-layer mean |a| per neuron (all tokens; OFF = macro over its 7 domains)
                  sub[set]  = per-layer bf16 activations (n_tok x width) of the fixed token subsample
                  sub_dom   = domain label of every OFF subsample token
  runs/downW.pt   down_proj weights of every layer (bf16), for the column edits that do not store W

a = the input of layers[l].mlp.down_proj. Spans follow experiments/01_locality_curves/acts1c.py exactly.

    python experiments/03_predictor/cache.py [--scored]   (--scored: OFF = forget.py scored positions -> runs/cache_scored.pt)
"""
import json
import random
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
A = HERE.parents[1] / "experiments"
for p in ("lib", "02_conditional_rules"):
    sys.path.insert(0, str(A / p))
from p1lib import decoder_layers, load_data, load_model, resolve  # noqa: E402
import train_gains as TG  # noqa: E402
from forget import encode  # noqa: E402

N_DOM, N_ON = 256, 1024
RULES = ["weather", "lower", "recipe"]


def jobs(tok, device, scored=False):
    """[(set, domain, ids, mask, lo, hi)] for every item of every set. OFF spans: acts1c.py (chat: reply tokens; raw:
    all tokens) or, with scored=True (amendment 1), the positions that forget.py scores (lo ... second-to-last)."""
    out = []
    for it in json.loads((A / "01_locality_curves/data/ref_set.json").read_text()):
        ids, lo = encode(tok, it, device)
        if scored:
            lo, hi = lo, ids.shape[1] - 1
        else:
            lo, hi = (lo + 1, ids.shape[1]) if "raw" not in it else (0, ids.shape[1])
        out.append(("off", it["domain"], ids, None, lo, hi))
    sets = {"casual": load_data("full")[0]}
    import common2b
    for r in RULES:
        sets[r] = common2b.load_data(r, "full")[0]
    for name, items in sets.items():
        for d in items:
            ids, mask, [(lo, hi)] = TG.build(tok, [d], False, device)
            out.append((name, name, ids, mask, lo, hi))
    return out


@torch.no_grad()
def main():
    model, tok = load_model(*resolve("E2B")[:2])
    tok.eot_text = TG.end_of_turn(tok)
    layers = decoder_layers(model)
    nl = len(layers)
    scored = "--scored" in sys.argv
    J = jobs(tok, model.device, scored)
    # fixed subsample: per domain (OFF) or per set (ON), random token positions over all items
    rng = random.Random(0)
    pos = {}
    for j, (s, dom, _, _, lo, hi) in enumerate(J):
        pos.setdefault(dom, []).extend((j, t) for t in range(lo, hi))
    keep = {}
    for dom, P in pos.items():
        n = N_DOM if dom not in ("casual", *RULES) else N_ON
        for j, t in rng.sample(P, min(n, len(P))):
            keep.setdefault(j, []).append(t)
    print({d: len(P) for d, P in pos.items()}, flush=True)
    acc = {}                 # (dom, l) -> [sum |a|, n]
    sub = {}                 # set -> l -> list of tensors
    sub_dom = []
    cur = {}

    def cap(l):
        def f(mod, inp):
            a = inp[0][0]
            s = acc.setdefault((cur["dom"], l), [0, 0])
            s[0] = s[0] + a[cur["lo"]:cur["hi"]].float().abs().sum(0)
            s[1] += cur["hi"] - cur["lo"]
            if cur["keep"]:
                sub.setdefault(cur["set"], {}).setdefault(l, []).append(a[cur["keep"]].cpu())
        return f
    hs = [layers[l].mlp.down_proj.register_forward_pre_hook(cap(l)) for l in range(nl)]
    for j, (s, dom, ids, mask, lo, hi) in enumerate(J):
        k = sorted(keep.get(j, []))
        cur.update(set=s, dom=dom, lo=lo, hi=hi, keep=k)
        if s == "off":
            sub_dom.extend([dom] * len(k))
        model(input_ids=ids, attention_mask=mask)
        if j % 100 == 0:
            print(f"{j}/{len(J)}", flush=True)
    for h in hs:
        h.remove()
    doms = sorted({d for s, d, *_ in J if s == "off"})
    acts = {"off": [torch.stack([acc[(d, l)][0] / acc[(d, l)][1] for d in doms]).mean(0).cpu() for l in range(nl)]}
    for s in ["casual", *RULES]:
        acts[s] = [(acc[(s, l)][0] / acc[(s, l)][1]).cpu() for l in range(nl)]
    subc = {s: [torch.cat(v[l]) for l in range(nl)] for s, v in sub.items()}
    out = dict(acts=acts, sub=subc, sub_dom=sub_dom, doms=doms, n_tok={s: int(v[0].shape[0]) for s, v in subc.items()})
    (HERE / "runs").mkdir(exist_ok=True)
    torch.save(out, HERE / ("runs/cache_scored.pt" if scored else "runs/cache.pt"))
    torch.save({l: layers[l].mlp.down_proj.weight.cpu() for l in range(nl)}, HERE / "runs/downW.pt")
    if scored:
        print("n_tok", out["n_tok"], flush=True)
        return
    # check against experiments/01_locality_curves/out/acts1c.pt (same spans, same items)
    ref = torch.load(A / "01_locality_curves/out/acts1c.pt")
    for k1, k2 in (("off", "ref"), ("casual", "casual")):
        e = max(float(((acts[k1][l] - ref[k2][l]).abs() / ref[k2][l].clamp_min(1e-6)).max()) for l in range(nl))
        print(f"max rel diff {k1} vs acts1c {k2}: {e:.2e}", flush=True)
    print("n_tok", out["n_tok"], flush=True)


if __name__ == "__main__":
    main()
