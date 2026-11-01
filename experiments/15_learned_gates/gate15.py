"""Arm B of experiment 15: N NEW neurons in the MLP of one layer, with trained gate row, up row and down column.

A new neuron adds gelu_tanh(g . x) * (u . x) * d to the MLP output (x = the MLP input). Parametrisation relative to the
layer's own weights (as the tool's columns): g = g0 + s_g v_g, u = u0 + s_u v_u, d = s_d v_d; s = the RMS of the
layer's gate rows, up rows and down columns; g0, u0 random normal with RMS s (seed 0); v = 0 at the start.
Loss: the demo's target loss (KL to the teacher) + w x KL(original || edited) on the negative items (make_edit's
preservation loss). F: fp32, make_edit.unrelated_change.

    python experiments/15_learned_gates/gate15.py train --teacher T.json --neg data/nearmiss_neg.json \\
        --layer 13 --n 4 --lr 1e-2 3e-2 1e-1 --out results/gate15/b4
"""
import argparse
import gc
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import make_edit as M  # noqa: E402
from p1lib import decoder_layers  # noqa: E402


class NewNeurons:
    """N new MLP neurons in one layer as a forward hook on layers[layer].mlp (the frozen weights are not changed)."""

    def __init__(self, model, layer, n, seed=0, init=None):
        mlp = decoder_layers(model)[layer].mlp
        dev = mlp.down_proj.weight.device
        self.layer, self.n, self.on = layer, n, True
        s = {k: getattr(mlp, k).weight.float().pow(2).mean().sqrt().item() for k in ("gate_proj", "up_proj", "down_proj")}
        h = mlp.down_proj.weight.shape[0]
        if init is None:
            gen = torch.Generator().manual_seed(seed)
            init = dict(g0=torch.randn(n, h, generator=gen) * s["gate_proj"], u0=torch.randn(n, h, generator=gen) * s["up_proj"])
        self.g0, self.u0 = init["g0"].float().to(dev), init["u0"].float().to(dev)
        self.s = s
        self.params = {k: torch.zeros(n, h, device=dev, requires_grad=True) for k in ("vg", "vu", "vd")}
        self.handle = mlp.register_forward_hook(self._hook)

    def weights(self):
        p = self.params
        return (self.g0 + self.s["gate_proj"] * p["vg"], self.u0 + self.s["up_proj"] * p["vu"],
                self.s["down_proj"] * p["vd"])

    def acts(self, x):
        g, u, _ = self.weights()
        x = x.float()
        return torch.nn.functional.gelu(x @ g.t(), approximate="tanh") * (x @ u.t())

    def _hook(self, mod, inp, out):
        if not self.on:
            return out
        _, _, d = self.weights()
        return out + (self.acts(inp[0]) @ d).to(out.dtype)

    def n_params(self):
        return 3 * self.params["vg"].numel()

    def remove(self):
        self.handle.remove()


def save(path, ad, meta):
    from safetensors.torch import save_file
    g, u, d = (w.detach().cpu().contiguous() for w in ad.weights())
    save_file(dict(gate=g, up=u, down=d), str(path),
              metadata={"format": "new-neurons-v1", "layer": str(ad.layer), "meta": json.dumps(meta)})


def load(model, path):
    """A new-neurons-v1 patch as a NewNeurons hook (weights as saved)."""
    from safetensors import safe_open
    with safe_open(str(path), "pt") as f:
        layer = int(f.metadata()["layer"])
        g, u, d = (f.get_tensor(k) for k in ("gate", "up", "down"))
    ad = NewNeurons(model, layer, g.shape[0], init=dict(g0=g, u0=u))
    ad.s = dict(ad.s, gate_proj=0.0, up_proj=0.0, down_proj=1.0)
    ad.params["vd"] = d.float().to(ad.g0.device)
    return ad


def install_loader():
    """make_edit.load_any also loads new-neurons-v1 patches (for the demo's gen.py)."""
    orig = M.load_any

    def load_any(model, path):
        from safetensors import safe_open
        with safe_open(str(path), "pt") as f:
            if (f.metadata() or {}).get("format") == "new-neurons-v1":
                return load(model, path)
        return orig(model, path)
    M.load_any = load_any


def train_cmd(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    model, tok = M.load("E2B")
    tok.eot_text = M.TG.end_of_turn(tok)
    M.embed_tokens(model).register_forward_hook(lambda mod, i, o: o.requires_grad_(True) if torch.is_grad_enabled() else o)
    train_items, eval_items = json.loads(Path(a.teacher).read_text())
    neg = json.loads(Path(a.neg).read_text())
    neg = [dict(q=x["q"], r=x["r"]) for x in neg]
    report = dict(arm="new neurons", layer=a.layer, n=a.n, seed=a.seed, steps=a.steps, batch=a.batch, w_neg=a.w_neg,
                  neg=dict(file=a.neg, n=len(neg), sha=M.sha(neg)), teacher_sha=M.sha([(d["q"], d["r"]) for d in train_items]),
                  runs={})
    names = []
    for lr in a.lr:
        name = f"new{a.n}_L{a.layer}_lr{lr:g}"
        print(f"training {name}:", flush=True)
        ad = NewNeurons(model, a.layer, a.n, seed=0)
        torch.cuda.synchronize()
        t0 = time.time()
        curve = M.train(model, tok, ad, train_items, "system", lr, a.steps, a.batch, a.seed,
                        neg if a.w_neg > 0 else None, a.w_neg, a.batch)
        torch.cuda.synchronize()
        secs = time.time() - t0
        ts = M.target_strength(model, tok, eval_items, ad, "system")
        report["runs"][name] = dict(kind="new_neurons", lr=lr, n_params=ad.n_params(), final_loss=curve[-1],
                                    target_loss=ts, train_seconds=round(time.time() - t0),   # as make_edit: + held-out loss
                                    sec_per_step=secs / max(1, len(curve)))
        print(f"  held-out target loss: original {ts['original']:.4f} -> edited {ts['edited']:.4f}  ({secs:.0f}s)", flush=True)
        save(out / f"{name}.safetensors", ad, dict(name=name, lr=lr))
        ad.remove()
        names.append(name)
        (out / "report.json").write_text(json.dumps(report, indent=1))
    # F in fp32 (as make_edit --f32)
    model = None
    gc.collect()
    torch.cuda.empty_cache()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    model, tok = M.load("E2B", lowmem=True, dtype=torch.float32)
    edits = {n: load(model, out / f"{n}.safetensors") for n in names}
    for n, f in M.unrelated_change(model, tok, edits).items():
        report["runs"][n]["F"] = dict(macro=f["macro"], by_domain=f["by_domain"])
        print(f"  {n} F {f['macro']:.5f}", flush=True)
    report["F_precision"] = "fp32"
    (out / "report.json").write_text(json.dumps(report, indent=1))
    print("DONE", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--teacher", required=True, help="teacher_replies.json of the demo (train items, held-out items)")
    t.add_argument("--neg", required=True, help="negative items: JSON list of {q, r} (r = the original model's reply)")
    t.add_argument("--layer", type=int, default=13)
    t.add_argument("--n", type=int, default=4)
    t.add_argument("--lr", type=float, nargs="+", default=[1e-2, 3e-2, 1e-1])
    t.add_argument("--w-neg", type=float, default=1.0)
    t.add_argument("--steps", type=int, default=200)
    t.add_argument("--batch", type=int, default=8)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--out", required=True)
    t.set_defaults(fn=train_cmd)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
