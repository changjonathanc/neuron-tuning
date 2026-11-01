"""Shared helpers: trainable adapters (gains, output-column rewrite, LoRA) as runtime hooks on Gemma-4.

Every adapter keeps fp32 parameters and adds its effect through forward hooks, so the frozen bf16 weights are never
changed. `adapter.on = False` turns an adapter off (used for the teacher pass).

    gains    a[i] -> a[i] * (1 + g_i) on the down_proj input (the repo's edit)
    columns  down_proj column i -> (1 + g_i) W_i + rms(W_i) u_i   (g_i: 1 number, u_i: hidden numbers)
    rows     gate_proj and up_proj rows of neuron i -> W_row + rms(W_row) v   (v: hidden numbers per row)
    neuron   rows + columns (the full neuron)
    lora     y -> y + s * B A x on a Linear (A: r x in, kaiming-uniform; B: out x r, zeros; s = alpha / r)
"""
import json
import math
import random
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "demos/talk_like_a_person"))
sys.path.insert(0, str(ROOT / "demos/talk_like_a_person/reproduce"))
from common import load_model, resolve  # noqa: E402,F401
from edit.load_edit import decoder_layers, load_edit  # noqa: E402,F401

RELEASED = ROOT / "demos/talk_like_a_person/edit/gemma-4-E2B-it_128gains.json"
DATA = ROOT / "demos/talk_like_a_person/reproduce/data/data_persona_E2B.json"
C_LAYERS = [9, 11, 12, 13, 14, 15, 16, 19, 20, 22, 23, 24, 25, 26, 27, 33]   # arm c (pre-registered)
D_LAYER = 15                                                                   # arm d (pre-registered)
LORA_ALL = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
            "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]


def get_module(layer, name):
    m = layer
    for part in name.split("."):
        m = getattr(m, part, None)
        if m is None:
            return None
    return m


def load_data(split, dev_seed=1234):
    """split='full': all 160 casual + 300 task items. split='dev': (train_casual, train_task, dev_casual, dev_task)
    with 20 casual + 30 task items held in as the dev set."""
    data = json.loads(DATA.read_text())
    casual = [d for d in data if d["kind"] == "id"]
    task = [d for d in data if d["kind"] == "pres"]
    if split == "full":
        return casual, task, [], []
    rng = random.Random(dev_seed)
    dc, dt = set(rng.sample(range(len(casual)), 20)), set(rng.sample(range(len(task)), 30))
    return ([d for i, d in enumerate(casual) if i not in dc], [d for i, d in enumerate(task) if i not in dt],
            [d for i, d in enumerate(casual) if i in dc], [d for i, d in enumerate(task) if i in dt])


class Adapter:
    def __init__(self, model, kind, spec, device):
        """kind: 'gains' | 'columns' | 'rows' | 'neuron' (spec = {layer: [neurons]}) or 'lora' (spec = [[layer, module, rank], ...])."""
        self.kind, self.spec, self.on, self.handles = kind, spec, True, []
        self.params = {}
        layers = decoder_layers(model)
        if kind in ("gains", "columns", "rows", "neuron"):
            for l, neurons in spec.items():
                mlp = layers[int(l)].mlp
                dp = mlp.down_proj
                idx = torch.tensor(neurons, device=device)
                if kind in ("rows", "neuron"):                                  # input side: gate and up rows
                    for nm in ("gate_proj", "up_proj"):
                        Wr = getattr(mlp, nm).weight[idx].float()               # k x hidden
                        v = torch.zeros_like(Wr, requires_grad=True)
                        self.params[f"{l}.{nm}.v"] = v
                        self.handles.append(getattr(mlp, nm).register_forward_hook(
                            self._row_hook(idx, v, Wr.pow(2).mean(1, keepdim=True).sqrt())))
                    if kind == "rows":
                        continue
                g = torch.zeros(len(neurons), device=device, requires_grad=True)
                self.params[f"{l}.g"] = g
                if kind == "gains":
                    self.handles.append(dp.register_forward_pre_hook(self._gain_hook(idx, g)))
                else:
                    W = dp.weight[:, idx].float()                               # hidden x k (frozen copy)
                    rms = W.pow(2).mean(0).sqrt()                               # k
                    u = torch.zeros(W.shape[0], len(neurons), device=device, requires_grad=True)
                    self.params[f"{l}.u"] = u
                    self.handles.append(dp.register_forward_hook(self._col_hook(idx, g, u, W, rms)))
        elif kind == "lora":
            for l, name, r in spec:
                lin = get_module(layers[l], name)
                A = torch.empty(r, lin.in_features, device=device)
                torch.nn.init.kaiming_uniform_(A, a=math.sqrt(5))
                A.requires_grad_(True)
                B = torch.zeros(lin.out_features, r, device=device, requires_grad=True)
                self.params[f"{l}.{name}.A"], self.params[f"{l}.{name}.B"] = A, B
                self.handles.append(lin.register_forward_hook(self._lora_hook(A, B, 1.0)))
        else:
            raise ValueError(kind)

    def _gain_hook(self, idx, g):
        def f(mod, inp):
            if not self.on:
                return
            x = inp[0]
            return (x.index_copy(-1, idx, x.index_select(-1, idx) * (1 + g).to(x.dtype)),)
        return f

    def _col_hook(self, idx, g, u, W, rms):
        def f(mod, inp, out):
            if not self.on:
                return
            D = W * g + u * rms                                                # hidden x k: change of the columns
            return out + inp[0].index_select(-1, idx) @ D.t().to(out.dtype)
        return f

    def _row_hook(self, idx, v, rms):
        def f(mod, inp, out):
            if not self.on:
                return
            x = inp[0]
            return out.index_add(-1, idx, x @ (v * rms).t().to(x.dtype))
        return f

    def _lora_hook(self, A, B, s):
        def f(mod, inp, out):
            if not self.on:
                return
            x = inp[0]
            return out + s * ((x @ A.t().to(x.dtype)) @ B.t().to(x.dtype))
        return f

    def n_params(self):
        return sum(p.numel() for p in self.params.values())

    def remove(self):
        for h in self.handles:
            h.remove()

    def state(self):
        return dict(kind=self.kind, spec=self.spec, params={k: v.detach().cpu() for k, v in self.params.items()})

    def load_params(self, params):
        with torch.no_grad():
            for k, v in params.items():
                self.params[k].copy_(v.to(self.params[k].device))


def lora_spec(arm, model):
    layers = decoder_layers(model)
    if arm == "lora_c":
        return [[l, "mlp.down_proj", 1] for l in C_LAYERS]
    if arm == "lora_d":
        return [[D_LAYER, "mlp.down_proj", 1]]
    if arm == "lora_e":
        return [[l, n, 8] for l in range(len(layers)) for n in LORA_ALL if get_module(layers[l], n) is not None]
    raise ValueError(arm)


def load_adapter(model, path):
    """A trained adapter (.pt from train.py) or a gains edit JSON (the repo format)."""
    path = str(path)
    if path.endswith(".json"):
        by_layer, _ = load_edit(path)
        ad = Adapter(model, "gains", {l: n for l, (n, _) in by_layer.items()}, model.device)
        ad.load_params({f"{l}.g": torch.tensor(g) for l, (_, g) in by_layer.items()})
        return ad
    st = torch.load(path, map_location="cpu")
    spec = {int(k): v for k, v in st["spec"].items()} if st["kind"] != "lora" else st["spec"]
    ad = Adapter(model, st["kind"], spec, model.device)
    ad.load_params(st["params"])
    for p in ad.params.values():
        p.requires_grad_(False)
    return ad
