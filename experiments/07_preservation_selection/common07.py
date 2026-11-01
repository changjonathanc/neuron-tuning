"""Shared code of experiment 07: a full-layer down_proj change with a group-lasso prox step, the preservation terms,
and the measurements (teacher KL, change on unrelated text F, direct write).

    DenseDelta    down_proj output of one layer -> out + a @ dW^T  (dW: hidden x n_neurons, fp32, starts at 0)
    AdamColProx   Adam with one second moment per column (the mean over the column of g^2), then the exact proximal
                  step of lam_g * sum_j ||dW[:, j]|| in the same metric. A fixed point of this update is a stationary
                  point of loss + lam_g * sum_j ||dW_j||, and a column stays at exactly 0 while ||m_j|| <= lam_g.
"""
import json
import math
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "experiments/lib"))
from p1lib import decoder_layers, load_data, load_model, resolve  # noqa: E402,F401
import train_gains as TG  # noqa: E402
from common import PERSONA_SYSTEM, chat_prompt  # noqa: E402
from forget import encode  # noqa: E402

RUNS = HERE / "runs"
HAND = {15: {"sel8": [6582, 7291], "srand8": [10840, 10916], "colsel8": [3172]}, 20: {}}   # from 01 / 04 selections


class DenseDelta:
    def __init__(self, model, layer, dW=None):
        dp = decoder_layers(model)[layer].mlp.down_proj
        self.layer, self.on = layer, True
        self.dW = torch.zeros(dp.weight.shape, device=dp.weight.device, dtype=torch.float32) if dW is None \
            else dW.to(dp.weight.device, torch.float32)
        self.dW.requires_grad_(True)
        self.h = dp.register_forward_hook(self._hook)

    def _hook(self, mod, inp, out):
        if not self.on:
            return
        return out + inp[0] @ self.dW.t().to(out.dtype)

    def remove(self):
        self.h.remove()


class AdamColProx:
    def __init__(self, p, lr, lam_g=0.0, b1=0.9, b2=0.999, eps=1e-8):
        self.p, self.lr, self.lam_g, self.b1, self.b2, self.eps, self.t = p, lr, lam_g, b1, b2, eps, 0
        self.m = torch.zeros_like(p)
        self.v = torch.zeros(p.shape[1], device=p.device)

    @torch.no_grad()
    def step(self):
        g = self.p.grad
        self.t += 1
        self.m.mul_(self.b1).add_(g, alpha=1 - self.b1)
        self.v.mul_(self.b2).add_(g.pow(2).mean(0), alpha=1 - self.b2)
        d = (self.v / (1 - self.b2 ** self.t)).sqrt() + self.eps                    # per column
        y = self.p - self.lr * (self.m / (1 - self.b1 ** self.t)) / d
        if self.lam_g > 0:
            n = y.norm(dim=0)
            y = y * (1 - self.lr * self.lam_g / (d * n.clamp_min(1e-30))).clamp_min(0)
        self.p.copy_(y)
        self.p.grad = None


def col_stats(dW):
    """Column norms and the kept set (rule fixed in NOTES.md): the smallest set of columns with >= 95% of ||dW||_F^2."""
    n = dW.detach().float().norm(dim=0).cpu()
    e = n.pow(2)
    order = torch.argsort(e, descending=True)
    tot = float(e.sum())
    cum = torch.cumsum(e[order], 0) / max(tot, 1e-30)
    k95 = int((cum < 0.95).sum()) + 1 if tot > 0 else 0
    k99 = int((cum < 0.99).sum()) + 1 if tot > 0 else 0
    return dict(norms=n, order=order.tolist(), nnz=int((n > 0).sum()), n95=k95, n99=k99,
                kept95=order[:k95].tolist(), frob=math.sqrt(tot))


@torch.no_grad()
def second_moment(model, tok, layer, items, kind):
    """Sigma = mean over positions of a a^T at the down_proj input of `layer` (original model).
    kind 'pres': reply span of the SlimOrca task items as trained (TG.build, no system prompt);
    kind 'ref' : the scored positions of the reference items (as forget.py scores them)."""
    dp = decoder_layers(model)[layer].mlp.down_proj
    acc = dict(S=None, n=0, s1=None)
    span = [0, 0]

    def f(mod, inp):
        a = inp[0][0, span[0]:span[1]].float()
        acc["S"] = a.t() @ a if acc["S"] is None else acc["S"].addmm_(a.t(), a)
        acc["s1"] = a.abs().sum(0) if acc["s1"] is None else acc["s1"] + a.abs().sum(0)
        acc["n"] += a.shape[0]
    h = dp.register_forward_pre_hook(f)
    for d in items:
        if kind == "pres":
            ids, mask, [(lo, hi)] = TG.build(tok, [d], False, model.device)
            span[:] = [lo, hi]
            model(input_ids=ids, attention_mask=mask)
        else:
            ids, lo = encode(tok, d, model.device)
            span[:] = [lo, ids.shape[1] - 1]
            model(input_ids=ids)
    h.remove()
    return acc["S"] / acc["n"], acc["s1"] / acc["n"], acc["n"]


def sigma_cached(model, tok, layer, kind):
    p = RUNS / f"sigma_{kind}_L{layer}.pt"
    if p.exists():
        return torch.load(p, map_location=model.device)
    if kind == "pres":
        items = load_data("full")[1]
    else:
        items = json.loads((REPO / "experiments/01_locality_curves/data/ref_set.json").read_text())
    S, m1, n = second_moment(model, tok, layer, items, kind)
    d = dict(S=S, mean_abs=m1, n=n)
    RUNS.mkdir(exist_ok=True)
    torch.save(d, p)
    return d


def write2(dW, S):
    """Mean squared direct write ||dW a||^2 over the positions of Sigma."""
    return ((dW @ S) * dW).sum()


@torch.no_grad()
def teacher_kl(model, tok, ad):
    """lib/tkl.py: KL(teacher || run) on the 180 teacher replies to the 60 held-out messages."""
    T = json.loads((REPO / "experiments/lib/data/prompt.json").read_text())

    def lp(q, r, system):
        p = chat_prompt(tok, q, system)
        ids = tok(p + r, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
        n0 = len(tok(p, add_special_tokens=False).input_ids)
        return torch.log_softmax(model(input_ids=ids).logits[0, n0 - 1:-1].float(), -1)
    kls, msg = [], []
    for it in T["items"]:
        q, r = T["messages"][it["i"]], it["text"]
        ad.on = False
        t = lp(q, r, PERSONA_SYSTEM)
        ad.on = True
        s = lp(q, r, None)
        kls.append(float((t.exp() * (t - s)).sum(-1).mean()))
        msg.append(it["i"])
    return dict(per_reply=kls, msg=msg, mean=sum(kls) / len(kls))


@torch.no_grad()
def forgetting(model, tok, ad):
    """lib/forget.py: KL(original || run) per token on the 256 items of lib/data/forget_set.json."""
    items = json.loads((REPO / "experiments/lib/data/forget_set.json").read_text())
    kls = []
    for it in items:
        ids, lo = encode(tok, it, model.device)
        ad.on = False
        lp0 = torch.log_softmax(model(input_ids=ids, logits_to_keep=ids.shape[1] - lo).logits[0, :-1].float(), -1)
        ad.on = True
        lp = torch.log_softmax(model(input_ids=ids, logits_to_keep=ids.shape[1] - lo).logits[0, :-1].float(), -1)
        kls.append(float((lp0.exp() * (lp0 - lp)).sum(-1).mean()))
    doms = sorted({it["domain"] for it in items})
    by = {d: sum(x for x, it in zip(kls, items) if it["domain"] == d) / sum(it["domain"] == d for it in items) for d in doms}
    return dict(per_item=kls, domain=[it["domain"] for it in items], by_domain=by, macro=sum(by.values()) / len(by))


@torch.no_grad()
def task_kl(model, tok, ad):
    """01_locality_curves/taskkl.py: KL(original || run) on the 300 SlimOrca task items (= the preservation set of
    arms C/C0, so for those arms this is a TRAINING-set number)."""
    out = []
    for d in load_data("full")[1]:
        ids, mask, [(lo, hi)] = TG.build(tok, [d], False, model.device)
        ad.on = False
        t = TG.reply_logp(model, ids, mask, lo, hi)[0]
        ad.on = True
        s = TG.reply_logp(model, ids, mask, lo, hi)[0]
        out.append(float(TG.kl(t, s).mean()))
    return dict(per_item=out, mean=sum(out) / len(out))
