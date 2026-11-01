"""The dense (non-selective) control neurons, and the selectivity of both neuron sets (GPU for c and g).

    python experiments/10_second_model/select_dense.py --sel results/q10/select --out experiments/10_second_model/out/dense8.json \
        [--nf-stats <neuron-finder repo>/stats/Qwen3.5-4B.safetensors]

dense8 (NOTES.md): random (seed 0) among the neurons that are in the top 10% of their layer on the target replies (c)
AND in the top 10% of their layer on the background text (g); per-layer counts = the per-layer counts of sel8. c is
measured as make_edit.py measures it (the same 64 training items, the scored positions), g is make_edit.py's background
activity (the cached file). Writes a report-like JSON ({"neurons": rows, "n_eligible": n}) for `make_edit.py --neurons`,
plus the selectivity of sel8 and dense8: g / c and, with --nf-stats, the percentile of each neuron's RMS activity on
general chat (sqrt(mean^2 + std^2) from the neuron-finder statistics) within its layer.
"""
import argparse
import json
import random
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import make_edit as ME  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sel", required=True, help="output directory of the selection run (report.json, teacher_replies.json)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="qwen3.5-4b")
    ap.add_argument("--nf-stats")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    sel = json.loads((Path(a.sel) / "report.json").read_text())
    train, _ = json.loads((Path(a.sel) / "teacher_replies.json").read_text())
    model, tok = ME.load(a.model)
    tok.eot_text = ME.TG.end_of_turn(tok)
    layers = ME.decoder_layers(model)
    sub = random.Random(0).sample(train, min(64, len(train)))   # as make_edit.select_neurons
    seqs = []
    for d in sub:
        ids, _, [(lo, hi)] = ME.TG.build(tok, [d], False, model.device)
        seqs.append((ids, lo, hi))
    c = ME.mean_abs_acts(model, tok, layers, seqs)
    g = ME.background_acts(model, tok, layers, ROOT / sel.get("background", ME.BACKGROUND.relative_to(ROOT)))
    counts = {}
    for r in sel["neurons"]:
        counts[r["layer"]] = counts.get(r["layer"], 0) + 1
    rng = random.Random(a.seed)
    rows, pools = [], {}
    taken = {(r["layer"], r["neuron"]) for r in sel["neurons"]}
    for l in sorted(counts):
        ok = (c[l] >= torch.quantile(c[l].double(), 0.9).float()) & (g[l] >= torch.quantile(g[l].double(), 0.9).float())
        pool = [i for i in torch.nonzero(ok).flatten().tolist() if (l, i) not in taken]
        pools[l] = len(pool)
        for i in sorted(rng.sample(pool, counts[l])):
            rows.append(dict(layer=l, neuron=i, c=float(c[l][i]), g=float(g[l][i]), ratio=float(g[l][i] / c[l][i]),
                             grad=0.0, score=0.0))
    nf = None
    if a.nf_stats:
        from safetensors.torch import load_file
        s = load_file(a.nf_stats)
        rms = (s["mean"] ** 2 + s["std"] ** 2).sqrt()
        nf = lambda l, i: float((rms[l] < rms[l][i]).float().mean())   # noqa: E731

    def describe(rs):
        return [dict(layer=r["layer"], neuron=r["neuron"], c=float(c[r["layer"]][r["neuron"]]),
                     g=float(g[r["layer"]][r["neuron"]]), ratio=float(g[r["layer"]][r["neuron"]] / c[r["layer"]][r["neuron"]]),
                     nf_rms_percentile=nf(r["layer"], r["neuron"]) if nf else None) for r in rs]
    out = dict(rule="random (seed %d) among top-10%% c AND top-10%% g of the layer; per-layer counts of sel8" % a.seed,
               pool_size=pools, neurons=rows, n_eligible=sum(pools.values()),
               selectivity=dict(sel8=describe(sel["neurons"]), dense8=describe(rows)),
               nf_stats="neuron-finder stats/Qwen3.5-4B.safetensors (smol-smoltalk chat)" if nf else None)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
    for k, v in out["selectivity"].items():
        print(k, [(d["layer"], d["neuron"], round(d["ratio"], 3), d["nf_rms_percentile"]) for d in v])


if __name__ == "__main__":
    main()
