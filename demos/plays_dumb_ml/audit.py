"""Audit view (CPU; the same method as the audit of the cooking demo): write each patch into the original bf16
weights, as a normal checkpoint would hold it, and count what changed against the original checkpoint.

    python demos/plays_dumb_ml/audit.py --out demos/plays_dumb_ml/out/audit.json NAME=PATCH.safetensors ...

For each patch: the number of weight matrices that differ, the number of changed numbers, and for each changed
matrix the number of changed columns (input dimensions). A column patch is added as new column = old column + delta;
a LoRA patch as W + B A (the scale is already in B). The sum is done in fp32 and rounded once to bf16, so a change
smaller than the bf16 step of a weight disappears, as it would in a saved checkpoint.
"""
import argparse
import json
import sys
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "demos/talk_like_a_person"))
from common import resolve  # noqa: E402

PREFIX = "model.language_model.layers"


def weights():
    repo, rev, _ = resolve("E2B")
    return safe_open(hf_hub_download(repo, "model.safetensors", revision=rev), "pt")


def diff(W, new):
    ch = new != W
    return dict(numbers=int(ch.sum()), columns=int(ch.any(0).sum()), shape=list(W.shape))


def audit(path, ckpt):
    out = {}
    with safe_open(str(path), "pt") as f:
        md = f.metadata() or {}
        if md.get("format") == "lora-v1":
            spec = json.loads(md["spec"])
            for l, name, _ in spec:
                key = f"{PREFIX}.{l}.{name}.weight"
                W = ckpt.get_tensor(key)
                d = f.get_tensor(f"{l}.{name}.B").float() @ f.get_tensor(f"{l}.{name}.A").float()
                out[key] = diff(W, (W.float() + d).to(W.dtype))
        else:
            layers = sorted({int(k.split(".")[0][5:]) for k in f.keys()})
            for l in layers:
                key = f"{PREFIX}.{l}.mlp.down_proj.weight"
                W = ckpt.get_tensor(key)
                new = W.clone()
                idx = f.get_tensor(f"layer{l}.neurons")
                new[:, idx] = (W[:, idx].float() + f.get_tensor(f"layer{l}.delta").float()).to(W.dtype)
                out[key] = diff(W, new)
    changed = {k: v for k, v in out.items() if v["numbers"]}
    return dict(matrices=len(changed), numbers=sum(v["numbers"] for v in changed.values()),
                columns=sum(v["columns"] for v in changed.values()),
                layers=sorted({int(k.split(".")[3]) for k in changed}), per_matrix=changed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("patches", nargs="+", help="NAME=PATH")
    a = ap.parse_args()
    ckpt = weights()
    total = sum(ckpt.get_slice(k).get_shape()[0] * ckpt.get_slice(k).get_shape()[1]
                for k in ckpt.keys() if k.startswith(PREFIX) and k.endswith("proj.weight"))
    res = dict(n_numbers_in_decoder_linear_layers=total, patches={})
    for spec in a.patches:
        name, path = spec.split("=", 1)
        r = audit(path, ckpt)
        res["patches"][name] = r
        print(f"{name}: {r['matrices']} matrices, {r['columns']:,} columns, {r['numbers']:,} numbers changed "
              f"(layers {r['layers'][:6]}{'...' if len(r['layers']) > 6 else ''})")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
