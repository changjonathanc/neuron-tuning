"""Phase 2b post-hoc diagnostic: how specific to the trigger messages are the selected neurons?

For each edit: c_trig = mean |down_proj input| per neuron on the training trigger items (teacher replies, read without
the system prompt), c_ctrl = the same on the training control items (the original's replies); g = Phase 1c reference
activity (data/acts_ref.pt). For the col8 (and srand8, if present) neurons of every seed at LR*: within-layer
percentiles and the ratios g / c_trig and c_ctrl / c_trig. (Phase 1c K = 8 selected neurons: g / c = 0.11-0.13.)

    python diag_acts.py     -> out/diag_acts.json
"""
import json

import numpy as np
import torch

from common2b import HERE, RULES, decoder_layers, load_data, load_model, resolve
import train_gains as TG
from train2b import trigger_acts


def main():
    torch.set_grad_enabled(False)
    model, tok = load_model(*resolve("E2B")[:2])
    tok.eot_text = TG.end_of_turn(tok)
    layers = decoder_layers(model)
    g = torch.load(HERE / "data/acts_ref.pt")
    lr = json.loads((HERE / "out/lr_choice3.json").read_text())
    out = {}
    for r in RULES:
        trig, ctrl = load_data(r, "full")[:2]
        ct, cc = trigger_acts(model, tok, layers, trig), trigger_acts(model, tok, layers, ctrl)
        rows = []
        for m in ("col8", "srand8"):
            best = lr[f"{r}_col8"]["best"]
            for s in (0, 1, 2):
                f = HERE / f"runs/curve/{r}_{m}_lr{best:g}_s{s}.json"
                if not f.exists():
                    continue
                for l, ids in json.loads(f.read_text())["selection"].items():
                    l = int(l)
                    for i in ids:
                        pct = lambda v: float((v < v[i]).float().mean())  # noqa: E731
                        rows.append(dict(method=m, seed=s, layer=l, neuron=i, pct_ref=pct(g[l]), pct_trig=pct(ct[l]),
                                         g_over_ctrig=float(g[l][i] / ct[l][i]), cctrl_over_ctrig=float(cc[l][i] / ct[l][i])))
        summ = {}
        for m in ("col8", "srand8"):
            R = [x for x in rows if x["method"] == m]
            if R:
                summ[m] = {k: float(np.median([x[k] for x in R])) for k in ("pct_ref", "pct_trig", "g_over_ctrig", "cctrl_over_ctrig")}
        out[r] = dict(rows=rows, median=summ)
        print(r, json.dumps(summ), flush=True)
    (HERE / "out").mkdir(exist_ok=True)
    (HERE / "out/diag_acts.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
