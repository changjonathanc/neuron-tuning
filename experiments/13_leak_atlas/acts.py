"""Trigger check T and the old forecast A of each edit, from the chosen neurons only (GPU, original model, no training).

Run right after the neuron selection (make_edit.py with no --lr), before any training:

  uv run experiments/13_leak_atlas/acts.py --gens results/atlas/gens/original.json --runs results/atlas

For the K chosen neurons (from <runs>/<edit>/select/report.json) and each set: m_j = mean |a_j| (the down_proj input of
neuron j) over the scored positions of make_edit.py (TG.build span: the last prompt token and the reply tokens), pooled
over the items of the set; the reply is the original model's reply (or the target reply, for "heldout_target").
  A(set) = geometric mean over j of m_j(set) / c_j     (c_j: the target activity of the selection, from the report)
  T      = A(heldout, original replies) / A(heldout, target replies)
Target reply: the complete original reply + the payload line (append edits), or the fixed reply (replace edit).
Output: out/forecast.json.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))
import atlas as C  # noqa: E402
from make_edit import TG, decoder_layers, load_model, resolve  # noqa: E402


@torch.no_grad()
def set_means(model, tok, layers, neurons, items):
    """m_j: mean |a_j| over the scored positions of all items, and the number of positions."""
    by = {}
    for j, (l, i) in enumerate(neurons):
        by.setdefault(l, []).append((j, i))
    cur = {}
    hs = [layers[l].mlp.down_proj.register_forward_pre_hook(
        lambda mod, inp, l=l: cur.__setitem__(l, inp[0][0].float().abs())) for l in by]
    s, n = torch.zeros(len(neurons), dtype=torch.float64), 0
    for q, r, ended in items:
        ids, _, [(lo, hi)] = TG.build(tok, [dict(q=q, r=r, sys=None, ended=ended)], False, model.device)
        model(input_ids=ids)
        for l, js in by.items():
            for j, i in js:
                s[j] += float(cur[l][lo:hi, i].sum())
        n += hi - lo
    for h in hs:
        h.remove()
    return (s / n).tolist(), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gens", required=True, help="the original model's replies (gen.py)")
    ap.add_argument("--runs", required=True, help="dir with <edit>/select/report.json")
    ap.add_argument("--out", default=str(HERE / "out/forecast.json"))
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    G = json.loads(Path(a.gens).read_text())["sets"]
    model, tok = load_model(*resolve("E2B")[:2])
    tok.eot_text = TG.end_of_turn(tok)
    layers = decoder_layers(model)
    out = {}
    for e, mode in C.EDITS.items():
        rep = json.loads((Path(a.runs) / e / "select/report.json").read_text())
        neurons = [(r["layer"], r["neuron"]) for r in rep["neurons"]]
        c = [r["c"] for r in rep["neurons"]]
        orig = lambda s: [(r["q"], r["reply"], not r["capped"]) for r in G[s]]  # noqa: E731
        sets = {"heldout": orig(f"{e}/heldout"), "near": orig(f"{e}/near"), "unrelated": orig("all/unrelated")}
        if mode == "replace":
            sets["heldout_target"] = [(r["q"], C.SPORTS_REPLY, True) for r in G[f"{e}/heldout"]]
        else:   # as trained: complete replies only
            sets["heldout_target"] = [(r["q"], r["reply"] + C.APPEND, True) for r in G[f"{e}/heldout"] if not r["capped"]]
        res = dict(mode=mode, neurons=[dict(layer=l, neuron=i, c=cj, ratio=r["ratio"], grad=r["grad"])
                                       for (l, i), cj, r in zip(neurons, c, rep["neurons"])], sets={})
        for s, items in sets.items():
            m, n = set_means(model, tok, layers, neurons, items)
            ratio = [mj / cj for mj, cj in zip(m, c)]
            res["sets"][s] = dict(n_items=len(items), n_positions=n, m=m, per_neuron=ratio,
                                  A=math.exp(sum(math.log(max(x, 1e-12)) for x in ratio) / len(ratio)))
        res["T"] = res["sets"]["heldout"]["A"] / res["sets"]["heldout_target"]["A"]
        out[e] = res
        print(f"{e:8s} T {res['T']:.3f}  A: " + " ".join(f"{s} {v['A']:.3f}" for s, v in res["sets"].items()), flush=True)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
