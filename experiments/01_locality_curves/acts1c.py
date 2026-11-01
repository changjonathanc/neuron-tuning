"""Phase 1c: per-neuron activity of every MLP neuron (mean |down_proj input| per token).

  casual : the 160 casual training items, teacher replies read WITHOUT the system prompt (as select_active in Phase 1)
  ref    : data/ref_set.json (general text, disjoint from the forgetting set); macro mean over its 7 domains.
           This is the ONLY general-text statistic that the selective-random control uses.
  forget : the Phase 1b forgetting set (macro over 7 domains). DIAGNOSTIC ONLY (to report, after the selection, how
           active the chosen neurons are on the evaluation text); never used to choose neurons.

    python experiments/01_locality_curves/acts1c.py      -> out/acts1c.pt
"""
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "lib"))
from p1lib import decoder_layers, load_data, load_model, resolve  # noqa: E402
import train_gains as TG  # noqa: E402
from forget import encode  # noqa: E402


@torch.no_grad()
def main():
    model, tok = load_model(*resolve("E2B")[:2])
    tok.eot_text = TG.end_of_turn(tok)
    layers = decoder_layers(model)
    nl = len(layers)
    span, acc = [0, 0], {}
    cur = [None]

    def cap(l):
        def f(mod, inp):
            a = inp[0][0, span[0]:span[1]].float().abs()
            s = acc.setdefault(cur[0], {}).setdefault(l, [0, 0])
            s[0] = s[0] + a.sum(0)
            s[1] += a.shape[0]
        return f
    hs = [layers[l].mlp.down_proj.register_forward_pre_hook(cap(l)) for l in range(nl)]
    casual = load_data("full")[0]
    cur[0] = "casual"
    for d in casual:
        ids, mask, [(lo, hi)] = TG.build(tok, [d], False, model.device)
        span[:] = [lo, hi]
        model(input_ids=ids, attention_mask=mask)
    out = {"casual": [(acc["casual"][l][0] / acc["casual"][l][1]).cpu() for l in range(nl)]}
    for name, path in (("ref", HERE / "data/ref_set.json"), ("forget", HERE.parent / "lib/data/forget_set.json")):
        items = json.loads(Path(path).read_text())
        doms = sorted({it["domain"] for it in items})
        for it in items:
            ids, lo = encode(tok, it, model.device)
            span[:] = [lo + 1 if "raw" not in it else 0, ids.shape[1]]   # reply tokens (chat) / all tokens (raw)
            cur[0] = f"{name}:{it['domain']}"
            model(input_ids=ids)
        out[name] = [torch.stack([acc[f"{name}:{d}"][l][0] / acc[f"{name}:{d}"][l][1] for d in doms]).mean(0).cpu()
                     for l in range(nl)]
    for h in hs:
        h.remove()
    torch.save(out, HERE / "out/acts1c.pt")
    for l in (9, 14, 15, 24):
        c, r = out["casual"][l], out["ref"][l]
        print(l, f"casual mean {c.mean():.4f} ref mean {r.mean():.4f}", flush=True)


if __name__ == "__main__":
    main()
