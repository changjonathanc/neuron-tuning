"""Gate check of experiment 15 (GPU): the mean |activation| of the neurons of each patch, per prompt set, on the
ORIGINAL model with the ORIGINAL replies (the edit off; for the new neurons of arm B: |gelu(g.x) * (u.x)| on the input
of their MLP, which the edit does not change).

Positions: "start" = the reply-start tokens (the template tokens between the user's message and the reply, as
make_edit --select-on reply-start); "reply" = the first 256 tokens of the original reply.
Sets: held-out AI/ML, the 8 near kinds and unrelated chat (eval sets, the demo's original replies), and the 8
near-miss TRAINING kinds (data/nearmiss_neg.json).

    python experiments/15_learned_gates/acts15.py --patch A=<columns patch> B4=<new-neuron patch> ... --out out/acts.json
"""
import argparse
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import gate15  # noqa: E402
import make_edit as M  # noqa: E402
from p1lib import decoder_layers  # noqa: E402

DEMO = ROOT / "demos/plays_dumb_ml"


def prompt_sets():
    g = json.loads((DEMO / "out/gens/original.json").read_text())["sets"]
    out = {"AI/ML (held out)": [(r["q"], r["reply"]) for r in g["heldout"]]}
    for r in g["neighbours"]:
        out.setdefault(f"near: {r['label']}", []).append((r["q"], r["reply"]))
    out["unrelated chat"] = [(r["q"], r["reply"]) for r in g["unrelated"]]
    for x in json.loads((HERE / "data/nearmiss_neg.json").read_text()):
        out.setdefault(f"train negatives: {x['kind']}", []).append((x["q"], x["r"]))
    return out


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patch", nargs="+", required=True, help="name=path")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    model, tok = M.load("E2B")
    tok.eot_text = M.TG.end_of_turn(tok)
    layers = decoder_layers(model)
    probes = {}   # name -> list of (label, fn(layer inputs) -> |a| per position)
    for spec in a.patch:
        name, path = spec.split("=", 1)
        from safetensors import safe_open
        with safe_open(path, "pt") as f:
            fmt = (f.metadata() or {}).get("format")
        if fmt == "new-neurons-v1":
            ad = gate15.load(model, path)
            ad.on = False
            probes[name] = [(f"{ad.layer}:new{j}", ("mlp", ad.layer, (ad, j))) for j in range(ad.n)]
        else:
            delta, _ = M.load_patch(path)
            probes[name] = [(f"{l}:{i}", ("down", l, i)) for l, (ns, _) in delta.items() for i in ns]
    need = {(kind, l) for ps in probes.values() for _, (kind, l, _) in ps}
    cur = {}
    hs = []
    for kind, l in need:
        mod = layers[l].mlp if kind == "mlp" else layers[l].mlp.down_proj
        hs.append(mod.register_forward_pre_hook(lambda m, inp, key=(kind, l): cur.__setitem__(key, inp[0][0])))
    res = {n: {} for n in probes}
    for sname, items in prompt_sets().items():
        acc = {n: [[0.0, 0.0, 0, 0] for _ in ps] for n, ps in probes.items()}
        for q, r in items:
            ids, lo = M.encode(tok, dict(q=q, r=tok.decode(tok(r, add_special_tokens=False).input_ids[:256])), model.device)
            s, e = M.reply_start_span(tok, q)
            model(input_ids=ids)
            for n, ps in probes.items():
                for j, (_, (kind, l, i)) in enumerate(ps):
                    x = cur[(kind, l)]
                    v = (i[0].acts(x)[:, i[1]] if kind == "mlp" else x[:, i].float()).abs()
                    acc[n][j][0] += float(v[s:e].sum())
                    acc[n][j][2] += e - s
                    acc[n][j][1] += float(v[e:].sum())
                    acc[n][j][3] += max(0, v.shape[0] - e)
        for n, ps in probes.items():
            res[n][sname] = {lab: dict(start=x[0] / x[2], reply=x[1] / max(1, x[3])) for (lab, _), x in zip(ps, acc[n])}
        print(sname, flush=True)
    for h in hs:
        h.remove()
    Path(a.out).write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print("saved", a.out)


if __name__ == "__main__":
    main()
