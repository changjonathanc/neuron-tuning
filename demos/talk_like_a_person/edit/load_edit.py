"""Load the 128-gain edit and apply it to a Gemma-4 model, either as runtime hooks or baked into the weights.

The edit multiplies selected MLP neuron activations (the input of `layers[l].mlp.down_proj`) by (1 + s*gain).
Because down_proj is linear with no bias, this is identical to scaling column `neuron` of down_proj.weight.
"""
import json
from collections import defaultdict
from pathlib import Path

import torch

EDIT_DIR = Path(__file__).resolve().parent


def load_edit(path):
    """Return the edit as {layer: (neuron_ids, gains)} plus the raw JSON metadata."""
    meta = json.loads(Path(path).read_text())
    by_layer = defaultdict(lambda: ([], []))
    for e in meta["edits"]:
        by_layer[e["layer"]][0].append(e["neuron"])
        by_layer[e["layer"]][1].append(e["gain"])
    return dict(by_layer), meta


def decoder_layers(model):
    """The text decoder layers of a Gemma-4 (multimodal or text-only) model."""
    m = model.model
    return (m.language_model if hasattr(m, "language_model") else m).layers


def add_hooks(model, edit, strength=1.0):
    """Apply the edit at runtime: a[idx] <- a[idx] * (1 + strength*gain) on each down_proj input.

    This is exactly the parametrisation used during training. Returns the hook handles; call .remove()
    on each to restore the original model.
    """
    layers, handles = decoder_layers(model), []
    for l, (neurons, gains) in edit.items():
        dp = layers[l].mlp.down_proj
        idx = torch.tensor(neurons, device=dp.weight.device)
        scale = 1 + strength * torch.tensor(gains, dtype=torch.float32, device=dp.weight.device)

        def hook(mod, inp, idx=idx, scale=scale):
            x = inp[0]
            return (x.index_copy(-1, idx, x.index_select(-1, idx) * scale.to(x.dtype)),)

        handles.append(dp.register_forward_pre_hook(hook))
    return handles


def load_vectors(path):
    """Residual-stream vectors saved as {"vectors": {layer: [floats]}} (steering baselines, the assistant axis)."""
    meta = json.loads(Path(path).read_text())
    return {int(l): torch.tensor(v) for l, v in meta["vectors"].items()}, meta


def add_steering(model, vectors, scale=1.0):
    """Baseline (not the edit): add scale * vectors[l] to the residual stream after decoder layer l, at every position.

    Returns the hook handles; call .remove() on each to restore the original model.
    """
    layers, handles = decoder_layers(model), []
    for l, v in vectors.items():
        v = (scale * v.float()).to(layers[l].mlp.down_proj.weight.device)

        def hook(mod, inp, out, v=v):
            if isinstance(out, tuple):
                return (out[0] + v.to(out[0].dtype),) + out[1:]
            return out + v.to(out.dtype)

        handles.append(layers[l].register_forward_hook(hook))
    return handles


def add_clamp(model, units, targets):
    """Analysis only: after decoder layer l, set the residual's component along unit vector units[l] to targets[l]
    (h <- h + (targets[l] - h . u) u), at every position. Removes any movement along that direction.

    Returns the hook handles; call .remove() on each to restore the original model.
    """
    layers, handles = decoder_layers(model), []
    for l, u in units.items():
        u = (u.float() / u.float().norm()).to(layers[l].mlp.down_proj.weight.device)
        c = float(targets[l])

        def hook(mod, inp, out, u=u, c=c):
            h = out[0] if isinstance(out, tuple) else out
            d = (c - h.float() @ u)[..., None] * u
            if h.shape[-2] > 1:   # prefill: leave position 0 (BOS, the attention sink) alone
                d[..., 0, :] = 0
            h = h + d.to(h.dtype)
            return (h,) + out[1:] if isinstance(out, tuple) else h

        handles.append(layers[l].register_forward_hook(hook))
    return handles


def down_proj_key(layer, prefix="model.language_model."):
    return f"{prefix}layers.{layer}.mlp.down_proj.weight"


@torch.no_grad()
def bake_(weight, neurons, gains, strength=1.0):
    """In place: scale columns `neurons` of a down_proj weight [hidden, intermediate] by (1 + strength*gain).

    The product is computed in float32 and rounded once to the weight dtype.
    """
    idx = torch.tensor(neurons, device=weight.device)
    scale = 1 + strength * torch.tensor(gains, dtype=torch.float32, device=weight.device)
    weight[:, idx] = (weight[:, idx].float() * scale).to(weight.dtype)
    return weight
