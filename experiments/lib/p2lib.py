"""Adapters that can be stacked (merge, sequential). Used by the conditional-rules experiment (02_conditional_rules).

Every neuron method is a column-delta hook on down_proj:  out + x[:, idx] @ D^T,  D = W g + rms(W) u
(gains: u = 0, not trained). LoRA: out + (x A^T) B^T. Hooks of several adapters on the same module add up, so applying
several adapters together = summing their weight deltas (the pre-registered merge rule).
"""
import math
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "demos/talk_like_a_person"))
sys.path.insert(0, str(ROOT / "demos/talk_like_a_person/reproduce"))
from common import chat_prompt, load_model, resolve  # noqa: E402,F401
from edit.load_edit import decoder_layers  # noqa: E402,F401

C_LAYERS = [9, 11, 12, 13, 14, 15, 16, 19, 20, 22, 23, 24, 25, 26, 27, 33]
METHODS = {   # name -> (kind, K or LoRA spec)
    "col8": ("columns", 8),
    "col128": ("columns", 128),
    "gains128": ("gains", 128),
    "lora_d": ("lora", [[15, "mlp.down_proj", 1]]),
    "lora_c": ("lora", [[l, "mlp.down_proj", 1] for l in C_LAYERS]),
}


def get_module(layer, name):
    m = layer
    for part in name.split("."):
        m = getattr(m, part)
    return m


class Adapter:
    """kind 'columns' | 'gains' (spec {layer: [neurons]}) or 'lora' (spec [[layer, module, rank], ...]).
    base: optional {layer: {neuron: column delta (hidden,) fp32}} of edits already on the model; a neuron in base uses
    W' = W + delta as its column (sequential training on top of an edit)."""

    def __init__(self, model, kind, spec, device, base=None):
        self.kind, self.spec, self.on, self.handles, self.params = kind, spec, True, [], {}
        self.frozen = {}
        layers = decoder_layers(model)
        if kind in ("columns", "gains"):
            for l, neurons in spec.items():
                l = int(l)
                dp = layers[l].mlp.down_proj
                idx = torch.tensor(neurons, device=device)
                W = dp.weight[:, idx].float().clone()                          # hidden x k
                for j, n in enumerate(neurons):
                    if base and l in base and n in base[l]:
                        W[:, j] += base[l][n].to(device)
                rms = W.pow(2).mean(0).sqrt()
                g = torch.zeros(len(neurons), device=device, requires_grad=True)
                self.params[f"{l}.g"] = g
                if kind == "columns":
                    u = torch.zeros(W.shape[0], len(neurons), device=device, requires_grad=True)
                    self.params[f"{l}.u"] = u
                else:
                    u = None
                self.frozen[l] = (idx, W, rms)
                self.handles.append(dp.register_forward_hook(self._col_hook(l)))
        elif kind == "lora":
            for l, name, r in spec:
                lin = get_module(layers[l], name)
                A = torch.empty(r, lin.in_features, device=device)
                torch.nn.init.kaiming_uniform_(A, a=math.sqrt(5))
                A.requires_grad_(True)
                B = torch.zeros(lin.out_features, r, device=device, requires_grad=True)
                self.params[f"{l}.{name}.A"], self.params[f"{l}.{name}.B"] = A, B
                self.handles.append(lin.register_forward_hook(self._lora_hook(A, B)))
        else:
            raise ValueError(kind)

    def col_delta(self, l):
        idx, W, rms = self.frozen[l]
        D = W * self.params[f"{l}.g"]
        if self.kind == "columns":
            D = D + self.params[f"{l}.u"] * rms
        return D                                                               # hidden x k

    def _col_hook(self, l):
        def f(mod, inp, out):
            if not self.on:
                return
            idx = self.frozen[l][0]
            return out + inp[0].index_select(-1, idx) @ self.col_delta(l).t().to(out.dtype)
        return f

    def _lora_hook(self, A, B):
        def f(mod, inp, out):
            if not self.on:
                return
            x = inp[0]
            return out + (x @ A.t().to(x.dtype)) @ B.t().to(x.dtype)
        return f

    def column_deltas(self):
        """{layer: {neuron: delta (hidden,) fp32 cpu}} (neuron methods only)."""
        out = {}
        with torch.no_grad():
            for l, (idx, _, _) in self.frozen.items():
                D = self.col_delta(l).cpu()
                out[l] = {int(n): D[:, j] for j, n in enumerate(idx.tolist())}
        return out

    def n_params(self):
        return sum(p.numel() for k, p in self.params.items() if not (self.kind == "gains" and k.endswith(".u")))

    def remove(self):
        for h in self.handles:
            h.remove()

    def state(self):
        spec = {str(k): v for k, v in self.spec.items()} if isinstance(self.spec, dict) else self.spec
        return dict(kind=self.kind, spec=spec, params={k: v.detach().cpu() for k, v in self.params.items()},
                    base_W={str(l): W.cpu() for l, (_, W, _) in self.frozen.items()})

    def load_params(self, params):
        with torch.no_grad():
            for k, v in params.items():
                self.params[k].copy_(v.to(self.params[k].device))
        for p in self.params.values():
            p.requires_grad_(False)


def merge_bases(adapters):
    base = {}
    for ad in adapters:
        if ad.kind == "lora":
            continue
        for l, d in ad.column_deltas().items():
            for n, v in d.items():
                base.setdefault(l, {})
                base[l][n] = base[l].get(n, 0) + v
    return base


def load_adapter(model, path):
    """A trained adapter (.pt). The saved base column (W' for sequential runs) is restored exactly."""
    st = torch.load(path, map_location="cpu")
    spec = {int(k): v for k, v in st["spec"].items()} if st["kind"] != "lora" else st["spec"]
    ad = Adapter(model, st["kind"], spec, model.device)
    if st["kind"] != "lora":
        for l, W in st.get("base_W", {}).items():
            idx, _, _ = ad.frozen[int(l)]
            W = W.to(model.device)
            ad.frozen[int(l)] = (idx, W, W.pow(2).mean(0).sqrt())
    ad.load_params(st["params"])
    return ad


def load_stack(model, paths):
    """Several adapters, each with its own hooks (= summed deltas). Returns the list."""
    return [load_adapter(model, p) for p in paths]
