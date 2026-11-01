"""E2 neuron selection by selectivity only (no attribution, no gradient).

Per MLP neuron (input of down_proj), original model, no system prompt:
  c = mean |a| per token on the TARGET replies of the task (training split only):
      casual  -> the 160 casual teacher replies of Phase 1c (experiments/01_locality_curves/out/acts1c.pt, "casual");
      weather, lower, recipe -> the kept training trigger replies of Phase 2b (train2b.trigger_acts, load_data(r, "full")[0]).
  g = mean |a| on the Phase 1c reference text (experiments/01_locality_curves/data/ref_set.json; macro mean over 7 domains;
      acts1c.pt "ref" = 02_conditional_rules/data/acts_ref.pt). Never the forgetting (F) items.
Score = ln(c / g). Eligible: c >= the 90th percentile of c in the neuron's layer. Selection: global top 8 over all layers.
Also reports c, g, within-layer percentiles and the overlap with the attribution choice (sel8 / col8, seeds 0-2).

    python select_e2.py            (GPU, for the 3 rules; writes out/selection.json)               v1 (pre-registered)
    python select_e2.py --scored   (GPU; g on the scored positions; writes out/selection_s.json)  v2 (amendment 1)

v1 BUG (found after the first runs; NOTES amendment 1): c uses the TG.build span = the SCORED positions (the last
prompt token, which predicts the first reply token, and the reply tokens except the last). g (acts1c.py) starts one
token later (the first reply token) and so misses the last prompt token. A neuron that fires at the last prompt token
of EVERY chat (casual 4:4676: |a| ~47 there in casual items and ~75 in reference items, < 0.1 on reply tokens) thus
got a high c and a low g. v2 (--scored) keeps c and computes g over the scored positions of the reference items
(chat: last prompt token .. second-to-last token; raw text: all tokens but the last), as forget.py scores them.
"""
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
A = HERE.parents[1] / "experiments"
K, Q = 8, 0.9
RULES = ["weather", "lower", "recipe"]


def select(c, g, k=K, q=Q):
    rows = []
    for l, (cl, gl) in enumerate(zip(c, g)):
        cl, gl = cl.double(), gl.double()
        thr = torch.quantile(cl, q)
        s = cl.clamp_min(1e-8).log() - gl.clamp_min(1e-8).log()
        for i in torch.nonzero(cl >= thr).flatten().tolist():
            rows.append((float(s[i]), l, i))
    rows.sort(reverse=True)
    return [(l, i) for _, l, i in rows[:k]]


def describe(neurons, c, g):
    pct = lambda v, i: float((v < v[i]).double().mean())  # noqa: E731
    return [dict(layer=l, neuron=i, c=float(c[l][i]), g=float(g[l][i]), ln_c_over_g=float((c[l][i] / g[l][i]).log()),
                 pct_c=pct(c[l], i), pct_g=pct(g[l], i)) for l, i in neurons]


def attr_sets(task):
    """{'sel8'|'col8': {seed: [[l, i]]}, 'srand8': ...} from the saved Phase 1c / 2b runs."""
    out = {}
    if task == "casual":
        for m in ("sel8", "srand8"):
            out[m] = {s: [[int(l), i] for l, v in json.loads((A / f"01_locality_curves/runs/final/{m}_lr0.01_s{s}.json").read_text())
                          ["selection"].items() for i in v] for s in (0, 1, 2)}
    else:
        best = json.loads((A / "02_conditional_rules/out/lr_choice3.json").read_text())[f"{task}_col8"]["best"]
        for m in ("col8", "srand8"):
            out[m] = {s: [[int(l), i] for l, v in json.loads((A / f"02_conditional_rules/runs/curve/{task}_{m}_lr{best:g}_s{s}.json")
                          .read_text())["selection"].items() for i in v] for s in (0, 1, 2)}
    return out


@torch.no_grad()
def ref_scored(model, tok, layers):
    """v2 g: mean |down_proj input| per neuron over the SCORED positions of the reference items; macro over domains."""
    sys.path.insert(0, str(A / "lib"))
    from forget import encode
    items = json.loads((A / "01_locality_curves/data/ref_set.json").read_text())
    span, acc, cur = [0, 0], {}, [None]

    def cap(l):
        def f(mod, inp):
            a = inp[0][0, span[0]:span[1]].float().abs()
            s = acc.setdefault(cur[0], {}).setdefault(l, [0, 0])
            s[0] = s[0] + a.sum(0)
            s[1] += a.shape[0]
        return f
    hs = [layers[l].mlp.down_proj.register_forward_pre_hook(cap(l)) for l in range(len(layers))]
    for it in items:
        ids, lo = encode(tok, it, model.device)
        span[:] = [lo, ids.shape[1] - 1]
        cur[0] = it["domain"]
        model(input_ids=ids)
    for h in hs:
        h.remove()
    doms = sorted(acc)
    return [torch.stack([acc[d][l][0] / acc[d][l][1] for d in doms]).mean(0).cpu() for l in range(len(layers))]


def main():
    acts = torch.load(A / "01_locality_curves/out/acts1c.pt")
    g = acts["ref"]
    scored = "--scored" in sys.argv
    if scored:
        gf = HERE / "runs/acts_ref_scored.pt"
        if not gf.exists():
            sys.path.insert(0, str(A / "02_conditional_rules"))
            from common2b import decoder_layers, load_model, resolve
            torch.set_grad_enabled(False)
            model, tok = load_model(*resolve("E2B")[:2])
            gf.parent.mkdir(exist_ok=True)
            torch.save(ref_scored(model, tok, decoder_layers(model)), gf)
            del model
        g = torch.load(gf)
    C = {"casual": acts["casual"]}
    cache = HERE / "runs/acts_trig.pt"
    if cache.exists():
        C.update(torch.load(cache))
    else:
        sys.path.insert(0, str(A / "02_conditional_rules"))
        from common2b import decoder_layers, load_data, load_model, resolve
        import train_gains as TG
        from train2b import trigger_acts
        torch.set_grad_enabled(False)
        model, tok = load_model(*resolve("E2B")[:2])
        tok.eot_text = TG.end_of_turn(tok)
        layers = decoder_layers(model)
        T = {r: trigger_acts(model, tok, layers, load_data(r, "full")[0]) for r in RULES}
        cache.parent.mkdir(exist_ok=True)
        torch.save(T, cache)
        C.update(T)
    out = {}
    for task, c in C.items():
        neurons = select(c, g)
        rows = describe(neurons, c, g)
        att = attr_sets(task)
        mine = {tuple(x) for x in neurons}
        overlap = {m: {s: len(mine & {tuple(x) for x in v}) for s, v in d.items()} for m, d in att.items()}
        am = "sel8" if task == "casual" else "col8"
        med = lambda v: sorted(v)[len(v) // 2]  # noqa: E731
        attr_rows = describe(att[am][0], c, g)
        out[task] = dict(neurons=[list(x) for x in neurons], rows=rows, overlap=overlap, attr_seed0=attr_rows,
                         median_g_over_c=med([r["g"] / r["c"] for r in rows]),
                         median_g_over_c_attr_seed0=med([r["g"] / r["c"] for r in attr_rows]),
                         rule=dict(k=K, eligible=f"c >= {Q:.0%} quantile of its layer", score="ln(c/g)", layers="global"))
        print(task, neurons, "overlap", overlap, "median g/c", round(out[task]["median_g_over_c"], 3),
              "attr", round(out[task]["median_g_over_c_attr_seed0"], 3), flush=True)
    (HERE / "out").mkdir(exist_ok=True)
    (HERE / ("out/selection_s.json" if scored else "out/selection.json")).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
