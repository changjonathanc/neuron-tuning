"""Experiment 14: the full ranking of the default `make_edit.py` rule for the type-hint demo (GPU, about 1 minute).

`make_edit.select_neurons` keeps the top K of one ranking, so the set for each K is a prefix of the ranking of ALL
eligible neurons. This script runs the rule once with K = the number of eligible neurons, adds pct_S (the rank of
S = g / c in its layer, 0 = most selective) for each neuron, and writes one `--neurons` file per K.

    python experiments/14_how_many_neurons/select14.py --teacher results/how_many_neurons/teacher_replies.json

Writes out/ranking.json and results/how_many_neurons/sets/k<K>.json (the make_edit.py `--neurons` input).
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
import make_edit as ME  # noqa: E402

KS = [1, 2, 4, 8, 16, 32, 64]
DEMO_S0 = REPO / "demos/python_type_hints/out/reports/s0.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True, help="the demo's teacher_replies.json ([train, eval])")
    ap.add_argument("--sets", default="results/how_many_neurons/sets")
    a = ap.parse_args()
    train, _ = json.loads(Path(a.teacher).read_text())
    demo = json.loads(DEMO_S0.read_text())
    h = ME.sha([(d["q"], d["r"]) for d in train])
    assert h == demo["data_hashes"]["train"], f"teacher data hash {h} != demo {demo['data_hashes']['train']}"
    model, tok = ME.load("E2B")
    tok.eot_text = ME.TG.end_of_turn(tok)
    layers = ME.decoder_layers(model)
    ME.embed_tokens(model).register_forward_hook(
        lambda mod, i, o: o.requires_grad_(True) if torch.is_grad_enabled() else o)
    t0 = time.time()
    # K = the demo's number of eligible neurons (select_neurons stops if fewer pass the filter)
    rows, n_el = ME.select_neurons(model, tok, layers, train, "system", ME.BACKGROUND, demo["n_eligible"], 0.2)
    secs = time.time() - t0
    # pct_S: the tool's c and g for all neurons (the same subset and background cache as select_neurons)
    sub = random.Random(0).sample(train, min(64, len(train)))
    seqs = []
    for d in sub:
        ids, _, [(lo, hi)] = ME.TG.build(tok, [d], False, model.device)
        seqs.append((ids, lo, hi))
    c = ME.mean_abs_acts(model, tok, layers, seqs)
    g = ME.background_acts(model, tok, layers, ME.BACKGROUND)
    S = [g[l] / c[l].clamp_min(1e-9) for l in range(len(layers))]
    for r in rows:
        s = S[r["layer"]]
        r["pct_S"] = float((s < s[r["neuron"]]).float().mean())
    demo8 = [(r["layer"], r["neuron"]) for r in demo["neurons"]]
    first8 = [(r["layer"], r["neuron"]) for r in rows[:8]]
    out = dict(n_eligible=n_el, n_neurons=sum(len(x) for x in c), seconds=round(secs), first8_equals_demo=first8 == demo8,
               demo8=demo8, ranking=rows)
    (HERE / "out").mkdir(exist_ok=True)
    (HERE / "out/ranking.json").write_text(json.dumps(out, indent=1))
    print(f"eligible {n_el}; first 8 == demo set: {first8 == demo8}; {secs:.0f} s")
    sd = Path(a.sets)
    sd.mkdir(parents=True, exist_ok=True)
    for k in KS:
        (sd / f"k{k}.json").write_text(json.dumps(dict(neurons=rows[:k], n_eligible=n_el), indent=1))


if __name__ == "__main__":
    main()
