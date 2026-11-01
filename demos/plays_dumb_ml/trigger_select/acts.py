"""What the chosen neurons respond to on THIS demo's prompts (GPU): for each neuron of a patch, the mean |activation| at
the tokens of the user's message, per prompt set (held-out AI/ML, neighbours per category, unrelated chat), and the
5 prompt tokens with the largest |activation| (all sets together).

    python demos/plays_dumb_ml/trigger_select/acts.py --out out/acts.json NAME=PATCH.safetensors ...

Writes out/acts.json and out/acts.md (the shipped neuron cards of make_edit.py use a corpus without AI/ML text).
"""
import argparse
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import dumb  # noqa: E402
from make_edit import load_patch, prompt_span  # noqa: E402
from common import chat_prompt  # noqa: E402
from p1lib import decoder_layers, load_model, resolve  # noqa: E402


@torch.no_grad()
def acts(model, tok, neurons, prompts):
    """[(label, prompt, tokens, A)] with A[pos, neuron] = activation at the user-message tokens."""
    layers = decoder_layers(model)
    by_layer = {}
    for l, i in neurons:
        by_layer.setdefault(l, []).append(i)
    cur = {}
    hs = [layers[l].mlp.down_proj.register_forward_pre_hook(
        lambda mod, inp, l=l: cur.__setitem__(l, inp[0][0][:, by_layer[l]].float().cpu())) for l in by_layer]
    out = []
    for lab, q in prompts:
        text = chat_prompt(tok, q)
        ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
        model(input_ids=ids)
        s, e = prompt_span(tok, q)
        A = torch.stack([cur[l][s:e, by_layer[l].index(i)] for l, i in neurons], 1)
        out.append((lab, q, ids[0, s:e].tolist(), A))
    for h in hs:
        h.remove()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("patches", nargs="+", help="NAME=PATH")
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    model, tok = load_model(*resolve("E2B")[:2])
    S = dumb.sets()
    prompts = [("AI/ML (held out)", q) for q, _ in S["heldout"]()] + \
        [(f"near: {c}", q) for q, c in S["neighbours"]()] + [("unrelated chat", q) for q, _ in S["unrelated"]()]
    res, md = {}, ["# Chosen neurons on this demo's prompts (written by acts.py)\n",
                   "Mean |activation| at the tokens of the user's message, per prompt set; then the 5 prompt tokens "
                   "with the largest |activation| (the token in [[ ]]).\n"]
    for spec in a.patches:
        name, path = spec.split("=", 1)
        delta, _ = load_patch(path)
        neurons = [(l, i) for l, (ns, _) in sorted(delta.items()) for i in ns]
        rows = acts(model, tok, neurons, prompts)
        labs = list(dict.fromkeys(lab for lab, *_ in rows))
        cards = []
        for j, (l, i) in enumerate(neurons):
            mean = {lab: float(torch.cat([A[:, j].abs() for lb, _, _, A in rows if lb == lab]).mean()) for lab in labs}
            top = sorted(((float(A[t, j]), lb, toks, t) for lb, _, toks, A in rows for t in range(len(toks))),
                         key=lambda x: -abs(x[0]))[:5]
            snips = [dict(act=round(v, 3), set=lb, text=tok.decode(toks[max(0, t - 8):t]) + "[[" + tok.decode(toks[t:t + 1])
                          + "]]" + tok.decode(toks[t + 1:t + 4])) for v, lb, toks, t in top]
            cards.append(dict(layer=l, neuron=i, mean_abs=mean, top=snips))
        res[name] = dict(patch=Path(path).name, cards=cards)
        md += [f"## {name}\n", "| neuron | " + " | ".join(labs) + " |", "|---|" + "---|" * len(labs)]
        md += [f"| {c['layer']}:{c['neuron']} | " + " | ".join(f"{c['mean_abs'][lb]:.3f}" for lb in labs) + " |"
               for c in cards]
        md.append("")
        for c in cards:
            md.append(f"**{c['layer']}:{c['neuron']}**: " + "; ".join(
                f"`{s['act']:+.2f}` ({s['set']}) " + " ".join(s["text"].split()).replace("|", "\\|") for s in c["top"]))
            md.append("")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1, ensure_ascii=False))
    Path(a.out).with_suffix(".md").write_text("\n".join(md) + "\n")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
