"""E4 training + evaluation of one run. Loss = KL(teacher || student) on the teacher's replies to the training
questions (teacher = the original model + the facts system prompt), no preservation loss, Adam, constant LR, 400
steps, batch 8 fact items. Teacher log-probs are computed once per run (fp16 on the CPU); no gradient checkpointing. Adapters: experiments/lib/p1lib.py (fp32 runtime hooks; frozen bf16 weights).

Arms (~200 k numbers, and one standard baseline):
  col128     down_proj columns of the attribution top 128 (TG.select_attr on 48 random training items; = Phase 1c sel128)
  dense128   columns of random neurons from the 10% most active per layer, per-layer counts of the same seed's
             attribution top 128 (train1b columns_active; = Phase 1c dense128)
  colsel128  columns of the 128 neurons chosen by selectivity only (out/selection.json, select_e4.py; = E2 v2 rule)
  lorac      rank-1 LoRA on down_proj of 16 layers (= Phase 1c lorac)
  lora_all8  rank-8 LoRA (alpha 16) on q, k, v, o, gate, up, down of every layer (E3 wrapper, experiments/05_typical_lora)
  original   no training (evaluation only)

    python train_e4.py --arm col128 --lr 1e-2 --seed 0 --out runs/final/col128_lr0.01_s0
"""
import argparse
import json
import random
import time
from pathlib import Path

import torch

import common4 as C
from common4 import TG, p1lib, train1b

HERE = Path(__file__).resolve().parent


def to_sel(cand):
    sel = {}
    for _, l, i in cand:
        sel.setdefault(l, []).append(i)
    return {l: sorted(v) for l, v in sorted(sel.items())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["col128", "dense128", "colsel128", "lorac", "lora_all8", "original"])
    ap.add_argument("--lr", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if Path(str(out) + ".json").exists():
        print("already done:", out)
        return
    t0 = time.time()
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    D, G = C.load()
    items = C.train_items(D, G)
    model, tok = p1lib.load_model(*p1lib.resolve("E2B")[:2], lowmem=True)   # outputs unchanged; 2 runs fit on the 3090
    tok.eot_text = TG.end_of_turn(tok)
    layers = p1lib.decoder_layers(model)
    model.model.language_model.embed_tokens.register_forward_hook(
        lambda mod, i, o: o.requires_grad_(True) if torch.is_grad_enabled() else o)
    selection, ad = None, None
    if a.arm in ("col128", "dense128"):
        selection = to_sel(TG.select_attr(model, tok, layers, items, 128))
        if a.arm == "dense128":
            selection = to_sel(train1b.select_active_matched(model, tok, layers, items, selection))
        ad = p1lib.Adapter(model, "columns", selection, model.device)
    elif a.arm == "colsel128":
        TG.select_attr(model, tok, layers, items, 128)   # same random state (batch order) as col128 of this seed; not used
        selection = {int(l): v for l, v in json.loads((HERE / "out/selection.json").read_text())["selection"].items()}
        ad = p1lib.Adapter(model, "columns", selection, model.device)
    elif a.arm == "lorac":
        ad = p1lib.Adapter(model, "lora", p1lib.lora_spec("lora_c", model), model.device)
    elif a.arm == "lora_all8":
        import sys
        sys.path.insert(0, str(HERE.parent / "05_typical_lora"))
        import train_e3   # noqa: F401  (patches p1lib: LoRA scale 2, scale folded into the saved B)
        ad = train_e3.adapter(model, "lora", "lora_all8", model.device)
    curve, bad = [], False
    if ad is not None:
        print(f"arm {a.arm} lr {a.lr} seed {a.seed}: {ad.n_params()} trained parameters, {len(items)} items", flush=True)
        # teacher log-probs of every training item, once (unbatched, as in the train1b loop), kept in fp16 on the CPU
        # (first run: computed and saved to runs/teacher_train.pt; later runs load it)
        ad.on = False
        T = C.cached(HERE / "runs/teacher_train.pt", lambda: [
            TG.reply_logp(model, *TG.build(tok, [d], True, model.device)[:2], *TG.build(tok, [d], True, model.device)[2][0])[0]
            for d in items])
        ad.on = True
        print(f"teacher cache {time.time() - t0:.0f}s", flush=True)
        model.train()
        opt = torch.optim.Adam(list(ad.params.values()), lr=a.lr)
        B = a.batch
        for step in range(a.steps):
            bi = random.sample(range(len(items)), B)
            batch = [items[j] for j in bi]
            teacher = [T[j].to(model.device).float() for j in bi]
            opt.zero_grad()
            ids, mask, spans = TG.build(tok, batch, False, model.device)
            off = min(lo for lo, _ in spans)
            logits = model(input_ids=ids, attention_mask=mask, logits_to_keep=ids.shape[1] - off).logits.float()
            loss = sum(TG.kl(teacher[j], torch.log_softmax(logits[j, lo - off:hi - off], -1)).mean()
                       for j, (lo, hi) in enumerate(spans)) / B
            if not torch.isfinite(loss):
                bad = True
                print(f"step {step}: non-finite loss, stopping", flush=True)
                break
            loss.backward()
            opt.step()
            curve.append([step, float(loss)])
            if step % 50 == 0 or step == a.steps - 1:
                print(f"step {step} kl {float(loss):.4f}  {time.time() - t0:.0f}s", flush=True)
        st = ad.state()
        st["spec"] = {str(k): v for k, v in st["spec"].items()} if isinstance(st["spec"], dict) else st["spec"]
        torch.save(st, str(out) + ".pt")
    model.eval()
    t1 = time.time()
    with torch.no_grad():
        ev = C.evaluate(model, tok, ad, D, G)
    res = dict(arm=a.arm, lr=a.lr, seed=a.seed, steps=a.steps, batch=a.batch, w_pres=0,
               n_params=ad.n_params() if ad else 0, selection=selection, curve=curve, nonfinite=bad,
               train_seconds=t1 - t0, eval_seconds=time.time() - t1, **ev)
    print(json.dumps(ev["summary"]), f"train {t1 - t0:.0f}s eval {time.time() - t1:.0f}s", flush=True)
    Path(str(out) + ".json").write_text(json.dumps(res))


if __name__ == "__main__":
    main()
