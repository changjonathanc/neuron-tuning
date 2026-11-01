"""Activity of the chosen neurons of each demo patch on its evaluation sets, in the ORIGINAL model (no training).

For each item: the sum and the count of |a_j| (the down_proj input of each chosen neuron j) over
  - the scored positions of make_edit.py (TG.build span: the last prompt token and the reply tokens), the reply being
    the original model's reply from the demo's out/gens/original.json (or the target reply, for the target sets);
  - the prompt tokens only.
The fixed unrelated text (F items) is scored at the positions of forget.py. Output: out/acts.json (per item), so that
analyze.py can pool them in any grouping on the CPU.

  uv run experiments/09_locality_forecast/acts.py --cooking <dir with cooking original.json and the LR 0.3 patch>
"""
import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import make_edit as ME  # noqa: E402
from make_edit import TG, chat_prompt, decoder_layers, encode, load_model, load_patch, resolve  # noqa: E402

D = ROOT / "demos"
HERE = Path(__file__).resolve().parent


def demos(cooking):
    """name -> patch, original replies, target replies (held-out), and how the target replies are made."""
    return {
        "jacobian": dict(patch=D / "jacobian_hotfix/patch/hotfix_8neurons.safetensors",
                         orig=D / "jacobian_hotfix/out/gens/original.json", target=D / "jacobian_hotfix/out/gens/teacher.json"),
        "plays_dumb": dict(patch=D / "plays_dumb_ml/columns_lr0.01_s0.safetensors",
                           orig=D / "plays_dumb_ml/out/gens/original.json", target=D / "plays_dumb_ml/out/gens/teacher.json"),
        "keigo": dict(patch=D / "keigo_japanese/patch/keigo_8neurons.safetensors",
                      orig=D / "keigo_japanese/out/gens/original.json", target=D / "keigo_japanese/out/gens/teacher.json"),
        "pyhints": dict(patch=D / "python_type_hints/patch/typehints_8neurons.safetensors",
                        orig=D / "python_type_hints/out/gens/original.json",
                        target=D / "python_type_hints/out/gens/teacher.json"),
        "cooking": dict(patch=Path(cooking) / "columns_lr0.3.safetensors", orig=Path(cooking) / "original.json",
                        target=None, append="\n\nAds are coming to AI. But not to Claude."),
    }


@torch.no_grad()
def item_stats(model, layers, spec, ids, spans):
    """{span name: (sum |a| per neuron, count)} for one sequence; spans = {name: (lo, hi)} (positions lo .. hi-1)."""
    cur = {}
    hs = [layers[l].mlp.down_proj.register_forward_pre_hook(
        lambda mod, inp, l=l: cur.__setitem__(l, inp[0][0].float().abs())) for l in spec]
    model(input_ids=ids)
    for h in hs:
        h.remove()
    out = {}
    for name, (lo, hi) in spans.items():
        s = [float(f"{float(cur[l][lo:hi, i].sum()):.6g}") for l, i in spec_order(spec)]   # 6 digits (file size)
        out[name] = dict(sum=s, n=hi - lo)
    return out


def spec_order(spec):
    return [(l, i) for l, ns in spec.items() for i in ns]


def chat_spans(tok, q, r, ended, device):
    d = dict(q=q, r=r, sys=None, ended=ended)
    ids, _, [(lo, hi)] = TG.build(tok, [d], False, device)
    return ids, dict(scored=(lo, hi), prompt=(0, lo + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cooking", required=True, help="dir with the cooking demo's original.json and columns_lr0.3 patch")
    ap.add_argument("--out", default=str(HERE / "out/acts.json"))
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    model, tok = load_model(*resolve("E2B")[:2])
    tok.eot_text = TG.end_of_turn(tok)
    layers = decoder_layers(model)
    fitems = json.loads(ME.FORGET.read_text())
    out = json.loads(Path(a.out).read_text()) if Path(a.out).exists() else {}
    for name, cfg in demos(a.cooking).items():
        if a.only and name not in a.only:
            continue
        delta, meta = load_patch(cfg["patch"])
        spec = {l: ns for l, (ns, _) in delta.items()}
        order = spec_order(spec)
        dnorm = {(l, i): float(Dm[:, j].float().norm()) for l, (ns, Dm) in delta.items() for j, i in enumerate(ns)}
        rows = {(r["layer"], r["neuron"]): r for r in meta["neurons"]}
        neurons = [dict(layer=l, neuron=i, c=rows[(l, i)]["c"], g=rows[(l, i)]["g"], grad=rows[(l, i)]["grad"],
                        delta_norm=dnorm[(l, i)]) for l, i in order]
        res = dict(patch=cfg["patch"].name, lr=meta.get("lr"), neurons=neurons, items=[])
        orig = json.loads(cfg["orig"].read_text())["sets"]
        sources = [("original", s, orig[s]) for s in orig if s != "train"]
        if cfg["target"]:
            sources.append(("target", "heldout", json.loads(cfg["target"].read_text())["sets"]["heldout"]))
        else:   # append mode: the target reply = the original reply + the payload (complete replies only, as trained)
            sources.append(("target", "heldout", [dict(x, reply=x["reply"] + cfg["append"]) for x in orig["heldout"]
                                                  if not x["capped"]]))
        for kind, s, xs in sources:
            for x in xs:
                ids, spans = chat_spans(tok, x["q"], x["reply"], not x.get("capped", False), model.device)
                st = item_stats(model, layers, spec, ids, spans)
                res["items"].append(dict(reply=kind, set=s, label=x.get("label"), q=x["q"][:80], **st))
            print(f"{name} {kind} {s}: {len(xs)} items", flush=True)
        for it in fitems:   # F text at the scored positions of forget.py (logit positions lo .. end-1)
            ids, lo = encode(tok, it, model.device)
            st = item_stats(model, layers, spec, ids, dict(scored=(lo, ids.shape[1] - 1)))
            res["items"].append(dict(reply="F", set="F:" + it["domain"], label=it["domain"], **st))
        print(f"{name} F text: {len(fitems)} items", flush=True)
        out[name] = res
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(out))


if __name__ == "__main__":
    main()
