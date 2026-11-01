"""Experiment 07: train the full down_proj of one layer for the casual edit, with or without preservation and group
sparsity, then measure it. Rules: NOTES.md.

    loss = KL_casual                                   (the target loss of lib/train1b.py)
         + lam_kl * KL(original || edited) on the batch's 4 SlimOrca task items      (arms C, C0)
         + lam_d  * mean ||dW a||^2 over all SlimOrca reply positions (exact, Sigma_pres)  (arm D)
    and, after each step, the prox step of lam_g * sum_j ||dW[:, j]||              (arms C, D, G0)
    Optimizer AdamColProx (common07.py), constant LR, 400 steps, batch 4 casual (+ 4 task items if lam_kl > 0).

Evaluation in the same process: teacher KL (60 held-out messages), F (256 unrelated items), task KL (300 SlimOrca
items; training data for lam_kl > 0), the direct write on the reference and preservation sets, and the column stats.

    python experiments/07_preservation_selection/train07.py --layer 15 --lr 1e-4 --lam-kl 5 --lam-g 0.01 --seed 0 \
        --out experiments/07_preservation_selection/runs/C_x
"""
import argparse
import json
import random
import time
from pathlib import Path

import torch

from common07 import (REPO, AdamColProx, DenseDelta, TG, col_stats, decoder_layers, forgetting, load_data, load_model,
                      resolve, sigma_cached, task_kl, teacher_kl, write2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--lam-kl", type=float, default=0.0)
    ap.add_argument("--lam-d", type=float, default=0.0)
    ap.add_argument("--lam-g", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--no-eval", action="store_true")
    ap.add_argument("--out", required=True, help="prefix; writes <out>.json (small) and <out>.pt (dW, sparse columns)")
    a = ap.parse_args()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if Path(str(out) + ".json").exists():
        print("already done:", out)
        return
    t0 = time.time()
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    casual, task, _, _ = load_data("full")
    model, tok = load_model(*resolve("E2B")[:2])
    tok.eot_text = TG.end_of_turn(tok)
    model.model.language_model.embed_tokens.register_forward_hook(
        lambda mod, i, o: o.requires_grad_(True) if torch.is_grad_enabled() else o)
    Sp = sigma_cached(model, tok, a.layer, "pres")["S"]
    Sr = sigma_cached(model, tok, a.layer, "ref")["S"]
    ad = DenseDelta(model, a.layer)
    opt = AdamColProx(ad.dW, a.lr, a.lam_g)
    print(f"layer {a.layer} lr {a.lr} lam_kl {a.lam_kl} lam_d {a.lam_d} lam_g {a.lam_g} seed {a.seed}: "
          f"{ad.dW.numel()} numbers  setup {time.time() - t0:.0f}s", flush=True)

    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    B, curve, snaps, bad = a.batch, [], [], False
    t1 = time.time()
    for step in range(a.steps):
        items = random.sample(casual, B) + (random.sample(task, B) if a.lam_kl > 0 else [])
        with torch.no_grad():
            ad.on = False
            teacher = []
            for d in items:
                ids, mask, [(lo, hi)] = TG.build(tok, [d], True, model.device)
                teacher.append(TG.reply_logp(model, ids, mask, lo, hi)[0])
            ad.on = True
        ids, mask, spans = TG.build(tok, items, False, model.device)
        off = min(lo for lo, _ in spans)
        logits = model(input_ids=ids, attention_mask=mask, logits_to_keep=ids.shape[1] - off).logits.float()
        lc = lt = 0
        for j, (lo, hi) in enumerate(spans):
            k = TG.kl(teacher[j], torch.log_softmax(logits[j, lo - off:hi - off], -1)).mean() / B
            if items[j]["kind"] == "id":
                lc = lc + k
            else:
                lt = lt + k
        if a.lam_d > 0:
            pw = write2(ad.dW, Sp)
        else:
            with torch.no_grad():
                pw = write2(ad.dW, Sp)
        loss = lc + a.lam_kl * lt + (a.lam_d * pw if a.lam_d > 0 else 0)
        if not torch.isfinite(loss):
            bad = True
            print(f"step {step}: non-finite loss, stopping", flush=True)
            break
        loss.backward()
        opt.step()
        if step % 10 == 0 or step == a.steps - 1:
            cs = col_stats(ad.dW)
            curve.append([step, float(lc), float(lt), float(pw), cs["nnz"], cs["n95"], cs["frob"]])
            if step % 50 == 0 or step == a.steps - 1:
                snaps.append(dict(step=step, top=cs["order"][:64], nnz=cs["nnz"], n95=cs["n95"]))
                print(f"step {step} kl_casual {float(lc):.3f} kl_task {float(lt):.4f} write_pres {float(pw):.3g} "
                      f"nnz {cs['nnz']} n95 {cs['n95']} |dW| {cs['frob']:.3g}  {time.time() - t0:.0f}s", flush=True)
    torch.cuda.synchronize()
    train_s = time.time() - t1
    peak = torch.cuda.max_memory_allocated() / 2**30
    model.eval()
    model.gradient_checkpointing_disable()
    dW = ad.dW.detach()
    cs = col_stats(dW)
    nz = torch.nonzero(cs["norms"] > 0).flatten()
    torch.save(dict(layer=a.layer, shape=list(dW.shape), cols=nz, W=dW[:, nz.to(dW.device)].cpu()), str(out) + ".pt")
    top = cs["order"][:512]
    res = dict(args=vars(a), n_params=dW.numel(), nonfinite=bad, curve=curve, snaps=snaps,
               nnz=cs["nnz"], n95=cs["n95"], n99=cs["n99"], kept95=cs["kept95"], frob=cs["frob"],
               top512=top, top512_norm=[float(f"{cs['norms'][j]:.4g}") for j in top],
               write2_pres=float(write2(dW, Sp)), write2_ref=float(write2(dW, Sr)),
               train_seconds=train_s, sec_per_step=train_s / a.steps, peak_mem_gb=peak)
    if not a.no_eval and not bad:
        with torch.no_grad():
            res["tkl"] = teacher_kl(model, tok, ad)
            print(f"teacher KL {res['tkl']['mean']:.4f}  {time.time() - t0:.0f}s", flush=True)
            res["forget"] = forgetting(model, tok, ad)
            print(f"F {res['forget']['macro']:.4f} " + " ".join(f"{d} {v:.4f}" for d, v in res['forget']['by_domain'].items()), flush=True)
            res["task"] = task_kl(model, tok, ad)
            print(f"task KL {res['task']['mean']:.4f}", flush=True)
    res["seconds"] = time.time() - t0
    res["out"] = str(out.resolve().relative_to(REPO)) if out.resolve().is_relative_to(REPO) else str(out)
    Path(str(out) + ".json").write_text(json.dumps(res))
    print("saved", out, f"{res['seconds']:.0f}s")


if __name__ == "__main__":
    main()
