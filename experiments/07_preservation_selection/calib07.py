"""Calibration of experiment 07 (before any training run; rules in NOTES.md). Per layer, original model:

  gnorm0[j]  ||mean gradient of the target loss w.r.t. dW[:, j]|| at dW = 0, over 64 casual training items in
             16 batches of 4 (the training loss of train07.py). The lam_g grid is read from this vector.
  attr[j]    the attribution score of 01 (train_gains.select_attr: -sum grad * (a_teacher - a_student), 48 items,
             seed 0), for every neuron of the layer (not only the top 64).
  w_rms[j]   rms of the original down_proj column.
  pres_abs[j], ref_abs[j]  mean |a| on the preservation set (SlimOrca reply spans) and on the reference set
             (scored positions); also caches Sigma_pres and Sigma_ref (runs/, not in git).

    python experiments/07_preservation_selection/calib07.py --layer 15   -> out/calib_L15.json
"""
import argparse
import json
import random
import time

import torch

from common07 import HERE, RUNS, DenseDelta, TG, decoder_layers, load_data, load_model, resolve, sigma_cached


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, required=True)
    a = ap.parse_args()
    t0 = time.time()
    random.seed(0)
    torch.manual_seed(0)
    casual, task, _, _ = load_data("full")
    model, tok = load_model(*resolve("E2B")[:2])
    tok.eot_text = TG.end_of_turn(tok)
    layers = decoder_layers(model)
    model.model.language_model.embed_tokens.register_forward_hook(
        lambda mod, i, o: o.requires_grad_(True) if torch.is_grad_enabled() else o)
    L = a.layer
    W = layers[L].mlp.down_proj.weight.float()
    w_rms = W.pow(2).mean(0).sqrt().cpu()

    # attribution score of every neuron of layer L (the code of TG.select_attr, without the top-64 cut)
    acts = {}

    def cap(mod, inp):
        acts["a"] = inp[0]
        if inp[0].requires_grad:
            inp[0].retain_grad()
    h = layers[L].mlp.down_proj.register_forward_pre_hook(cap)
    score = 0
    for d in random.sample(casual, 48):
        with torch.no_grad():
            ids, mask, [(lo, hi)] = TG.build(tok, [d], True, model.device)
            t_logp = TG.reply_logp(model, ids, mask, lo, hi)[0]
            a_t = acts["a"][0, lo:hi].float().clone()
        ids, mask, [(lo, hi)] = TG.build(tok, [d], False, model.device)
        with torch.enable_grad():
            TG.kl(t_logp, TG.reply_logp(model, ids, mask, lo, hi)[0]).mean().backward()
        with torch.no_grad():
            score = score + (-(acts["a"].grad[0, lo:hi].float() * (a_t - acts["a"][0, lo:hi].float()))).sum(0)
        model.zero_grad(set_to_none=True)
    h.remove()
    print(f"attribution done {time.time() - t0:.0f}s", flush=True)

    # mean target-loss gradient w.r.t. dW at 0
    ad = DenseDelta(model, L)
    G = torch.zeros_like(ad.dW)
    B, nb = 4, 16
    order = random.sample(casual, B * nb)
    for b in range(nb):
        items = order[b * B:(b + 1) * B]
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
        loss = sum(TG.kl(teacher[j], torch.log_softmax(logits[j, lo - off:hi - off], -1)).mean() / B
                   for j, (lo, hi) in enumerate(spans))
        loss.backward()
        G += ad.dW.grad / nb
        ad.dW.grad = None
    gn = G.norm(dim=0).cpu()
    ad.remove()
    print(f"gradient done {time.time() - t0:.0f}s", flush=True)

    sp = sigma_cached(model, tok, L, "pres")
    sr = sigma_cached(model, tok, L, "ref")
    srt = torch.sort(gn, descending=True).values
    lam = {str(k): float((srt[k - 1] + srt[k]) / 2) for k in (8, 32, 128, 512)}
    r4 = lambda v: [float(f"{x:.4g}") for x in v.tolist()]  # noqa: E731
    out = dict(layer=L, n=len(gn), lam_at_count=lam, gnorm0=r4(gn), attr=r4(score.cpu()), w_rms=r4(w_rms),
               pres_abs=r4(sp["mean_abs"].cpu()), ref_abs=r4(sr["mean_abs"].cpu()), n_pres_tok=sp["n"], n_ref_tok=sr["n"],
               seconds=time.time() - t0)
    (HERE / "out").mkdir(exist_ok=True)
    (HERE / f"out/calib_L{L}.json").write_text(json.dumps(out))
    print("lam_g at column counts", lam, f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
