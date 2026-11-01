"""Download the official Gemma-4 chat model and apply the 128-gain edit.

Bake the edit into the weights and save a standard Hugging Face checkpoint (loads with plain transformers):
    python patch_gemma.py --out gemma-4-E2B-it-human
    python patch_gemma.py --model E4B --out gemma-4-E4B-it-human
    python patch_gemma.py --strength 1.5 --out gemma-4-E2B-it-human-s1.5

To try the edit without saving anything, use chat.py (runtime hooks).

The edit scales MLP neuron activations a -> a*(1 + s*g) at the input of down_proj. down_proj has no bias, so this is
exactly down_proj.weight[:, neuron] *= (1 + s*g), which is what --out writes. Only those 128 columns change.
"""
import argparse
import json
import shutil
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.torch import save_file

from common import resolve
from edit.load_edit import bake_, down_proj_key, load_edit

FILES = ["*.json", "*.safetensors", "*.jinja", "tokenizer*"]


def bake(repo, revision, edit, strength, out):
    src = Path(snapshot_download(repo, revision=revision, allow_patterns=FILES))
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    todo = {down_proj_key(l): (n, g) for l, (n, g) in edit.items()}
    for f in sorted(src.glob("*.safetensors")):
        tensors = {}
        with safe_open(f, "pt") as h:
            meta = h.metadata()
            for k in h.keys():
                t = h.get_tensor(k)
                if k in todo:
                    bake_(t, *todo.pop(k), strength=strength)
                tensors[k] = t
        save_file(tensors, out / f.name, metadata=meta)
        del tensors
    if todo:
        raise RuntimeError(f"weights not found in checkpoint: {sorted(todo)[:3]} ...")
    for f in src.iterdir():
        if f.suffix != ".safetensors":
            shutil.copy(f, out / f.name)
    (out / "PATCH.json").write_text(json.dumps(dict(
        base_model=repo, base_revision=revision, edit="gemma-128-human (128 neuron gains)", strength=strength,
        note="Derived from Google's Gemma model; use is subject to the Gemma Terms of Use (https://ai.google.dev/gemma/terms).",
    ), indent=1) + "\n")
    print(f"saved patched checkpoint to {out} (128 down_proj columns scaled, strength {strength})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="E2B", help="E2B, E4B, or a Hugging Face repo id / local dir of the same model")
    ap.add_argument("--revision", default=None, help="model revision (default: the one the edit was trained on)")
    ap.add_argument("--edit", default=None, help="edit JSON (default: the one matching --model)")
    ap.add_argument("--strength", type=float, default=1.0, help="scale every gain by s (1.0 = as trained; 1.5 scores highest; coherence drops past 2)")
    ap.add_argument("--out", required=True, help="directory for the patched checkpoint")
    a = ap.parse_args()
    repo, rev, edit_path = resolve(a.model)
    rev = a.revision or rev
    edit_path = a.edit or edit_path
    if edit_path is None:
        ap.error("--edit is required for a model other than E2B/E4B")
    edit, meta = load_edit(edit_path)
    if Path(repo).name != Path(meta["model"]).name and not a.edit:
        print(f"warning: edit was trained on {meta['model']}, applying to {repo}")
    bake(repo, rev, edit, a.strength, a.out)


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
