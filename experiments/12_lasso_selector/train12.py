"""Experiment 12, casual edit, phase 2: train the down_proj columns of the 8 selected neurons (out/sel_casual_<name>.json)
with the loop of lib/train1b.py (w_pres = 0: 400 steps, batch 4, Adam, constant LR, the same random stream), then
measure TKL (lib/tkl.py), F in bf16 (lib/forget.py) and F in fp32 (model and patch in fp32, TF32 off).

    python experiments/12_lasso_selector/train12.py --sel lasso_all --lr 1e-2 3e-2 1e-1 --seed 0

Writes out/runs/casual_<sel>_lr<lr>_s<seed>.json (small) and results/lasso12/casual/<same>.safetensors (the patch).
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE.parent / "07_preservation_selection"))
import make_edit as ME  # noqa: E402
from common07 import forgetting, teacher_kl  # noqa: E402
from p1lib import Adapter, decoder_layers, load_data, load_model, resolve  # noqa: E402
import train_gains as TG  # noqa: E402

PATCHES = REPO / "results/lasso12/casual"


def train_one(model, tok, layers, spec, casual, task, lr, seed, steps=400, B=4):
    random.seed(seed)
    torch.manual_seed(seed)
    ad = Adapter(model, "columns", spec, model.device)
    ad.model_layers = layers
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    opt = torch.optim.Adam(list(ad.params.values()), lr=lr)
    curve, t0 = [], time.time()
    for step in range(steps):
        items = random.sample(casual, B)
        random.sample(task, B)                       # train1b draws 4 task items per step (unused at w_pres = 0)
        with torch.no_grad():
            ad.on = False
            teacher = []
            for d in items:
                ids, mask, [(lo, hi)] = TG.build(tok, [d], True, model.device)
                teacher.append(TG.reply_logp(model, ids, mask, lo, hi)[0])
            ad.on = True
        opt.zero_grad()
        ids, mask, spans = TG.build(tok, items, False, model.device)
        off = min(lo for lo, _ in spans)
        logits = model(input_ids=ids, attention_mask=mask, logits_to_keep=ids.shape[1] - off).logits.float()
        loss = sum(TG.kl(teacher[j], torch.log_softmax(logits[j, lo - off:hi - off], -1)).mean() / B
                   for j, (lo, hi) in enumerate(spans))
        if not torch.isfinite(loss):
            print(f"step {step}: non-finite loss, stopping", flush=True)
            break
        loss.backward()
        opt.step()
        curve.append([step, float(loss)])
        if step % 50 == 0 or step == steps - 1:
            print(f"step {step} kl_casual {float(loss):.3f}  {time.time() - t0:.0f}s", flush=True)
    model.eval()
    model.gradient_checkpointing_disable()
    for p in ad.params.values():
        p.requires_grad_(False)
    return ad, curve, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sel", required=True, help="tool | lasso_all | lasso_L15 (out/sel_casual_<sel>.json)")
    ap.add_argument("--lr", type=float, nargs="+", required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    sel = json.loads((HERE / f"out/sel_casual_{a.sel}.json").read_text())
    spec = {}
    for r in sel["neurons"]:
        spec.setdefault(r["layer"], []).append(r["neuron"])
    spec = {l: sorted(v) for l, v in sorted(spec.items())}
    (HERE / "out/runs").mkdir(parents=True, exist_ok=True)
    PATCHES.mkdir(parents=True, exist_ok=True)
    names = [f"casual_{a.sel}_lr{lr:g}_s{a.seed}" for lr in a.lr]
    todo = [(n, lr) for n, lr in zip(names, a.lr) if not (HERE / f"out/runs/{n}.json").exists()]
    if not todo:
        return
    casual, task, _, _ = load_data("full")
    repo, rev, _ = resolve("E2B")
    model, tok = load_model(repo, rev)
    tok.eot_text = TG.end_of_turn(tok)
    layers = decoder_layers(model)
    model.model.language_model.embed_tokens.register_forward_hook(
        lambda mod, i, o: o.requires_grad_(True) if torch.is_grad_enabled() else o)
    res = {}
    for n, lr in todo:
        print(f"=== {n}: spec {spec}", flush=True)
        ad, curve, secs = train_one(model, tok, layers, spec, casual, task, lr, a.seed)
        delta = ME.column_delta(ad)
        ME.save_patch(PATCHES / f"{n}.safetensors", delta, dict(name=n, lr=lr, neurons=sel["neurons"]))
        with torch.no_grad():
            r = dict(sel=a.sel, lr=lr, seed=a.seed, spec={str(k): v for k, v in spec.items()}, n_params=ad.n_params(),
                     curve=curve[::10], train_seconds=secs)
            r["tkl"] = teacher_kl(model, tok, ad)["mean"]
            f = forgetting(model, tok, ad)
            r["F_bf16"] = dict(macro=f["macro"], by_domain=f["by_domain"])
        print(f"TKL {r['tkl']:.4f}  F bf16 {r['F_bf16']['macro']:.5f}", flush=True)
        ad.remove()
        res[n] = r
    # fp32 F: model and patches in fp32 (make_edit.py --f32)
    del model
    torch.cuda.empty_cache()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    model, tok = load_model(repo, rev, lowmem=True, dtype=torch.float32)
    for n, r in res.items():
        delta, _ = ME.load_patch(PATCHES / f"{n}.safetensors")
        p = ME.Patch(model, delta)
        with torch.no_grad():
            f = forgetting(model, tok, p)
        p.remove()
        r["F_fp32"] = dict(macro=f["macro"], by_domain=f["by_domain"])
        print(f"{n}: TKL {r['tkl']:.4f}  F fp32 {f['macro']:.5f}  bf16 {r['F_bf16']['macro']:.5f}", flush=True)
        (HERE / f"out/runs/{n}.json").write_text(json.dumps(r, indent=1))


if __name__ == "__main__":
    main()
