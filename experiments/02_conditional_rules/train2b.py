"""Phase 2b training: one rule, one method, one LR, one seed, optionally on top of earlier edits. A copy of
the training script of an earlier study (not in this repository); the changes: the data (trigger items -> KL to the teacher; control and task items -> KL to the
current model without the new edit), 4 trigger + 2 control + 2 task items per step, and the dev objective (NOTES.md).

    python train2b.py --edit cat --method col8 --lr 0.3 --seed 0 --split dev --out runs/tune/cat_col8_lr0.3
    python train2b.py --edit weather --method lora_d --lr 1e-3 --seed 0 --split full --on runs/final/cat_lora_d_s0.pt \
        --out runs/seq/cat+weather_lora_d_s0      # incremental: weather trained on the cat-edited model

With --on, the adapters given are on (frozen) during the whole run: they are part of the teacher, of the preservation
target and of the student (the A-edited model is the current model). Selection runs on the A-edited model; a selected
neuron that an earlier edit changed gets W' = W + delta as its column.
With --w-pres 0 (amendment 3, the main design), only the trigger items enter the forward passes and the loss.
Writes <out>.pt (adapter) and <out>.json (loss curve, dev objective for --split dev, parameter count, selection).
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

import torch

from common2b import HERE, METHODS, Adapter, decoder_layers, load_data, load_model, load_stack, merge_bases, resolve
import train_gains as TG


@torch.no_grad()
def trigger_acts(model, tok, layers, items):
    """c = mean |down_proj input| per neuron on the trigger replies, read WITHOUT the system prompt (Phase 1c acts1c)."""
    span, acc = [0, 0], {}

    def cap(l):
        def f(mod, inp):
            a = inp[0][0, span[0]:span[1]].float().abs()
            s = acc.setdefault(l, [0, 0])
            s[0] = s[0] + a.sum(0)
            s[1] += a.shape[0]
        return f
    hs = [layers[l].mlp.down_proj.register_forward_pre_hook(cap(l)) for l in range(len(layers))]
    for d in items:
        ids, mask, [(lo, hi)] = TG.build(tok, [d], False, model.device)
        span[:] = [lo, hi]
        model(input_ids=ids, attention_mask=mask)
    for h in hs:
        h.remove()
    return [(acc[l][0] / acc[l][1]).cpu() for l in range(len(layers))]


def match(sel, pool, c_all, g_all):
    """Phase 1c train1c.match with NEAREST = 1: per selected neuron, the nearest neuron of the same layer in
    (ln c, ln g) that is not in the attribution pool and not taken yet."""
    taken, chosen, rows = set(), [], []
    for _, l, i in sel:
        c, g = c_all[l].double(), g_all[l].double()
        x = torch.stack([c.clamp_min(1e-8).log(), g.clamp_min(1e-8).log()], 1)
        d = (x - x[i]).norm(dim=1)
        bad = [j for (ll, j) in pool if ll == l] + [j for (ll, j) in taken if ll == l]
        d[bad] = math.inf
        j = int(torch.argmin(d))
        taken.add((l, j))
        chosen.append((l, j))
        rows.append(dict(layer=l, sel=i, ctrl=j, dist=float(d[j])))
    return chosen, rows


@torch.no_grad()
def dev_objective(model, tok, ad, dev_c, dev_n, dev_t, w_pres=5.0):
    def item_kl(d):
        ad.on = False
        ids, mask, [(lo, hi)] = TG.build(tok, [d], True, model.device)
        t = TG.reply_logp(model, ids, mask, lo, hi)[0]
        ad.on = True
        ids, mask, [(lo, hi)] = TG.build(tok, [d], False, model.device)
        return float(TG.kl(t, TG.reply_logp(model, ids, mask, lo, hi)[0]).mean())
    kc = sum(map(item_kl, dev_c)) / len(dev_c)
    kn = sum(map(item_kl, dev_n)) / len(dev_n)
    kt = sum(map(item_kl, dev_t)) / len(dev_t)
    return dict(dev_kl_trigger=kc, dev_kl_control=kn, dev_kl_task=kt, dev_objective=kc + w_pres * (kn + kt) / 2)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--edit", required=True)
    ap.add_argument("--method", required=True, choices=sorted(METHODS) + ["srand8"])
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", choices=["full", "dev"], required=True)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--w-pres", type=float, default=5.0)
    ap.add_argument("--on", nargs="*", default=[], help="adapters already on the model (sequential training)")
    ap.add_argument("--out", required=True, help="output prefix (.pt and .json are added)")
    a = ap.parse_args()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if Path(str(out) + ".json").exists():
        print("already done:", out)
        return
    t0 = time.time()
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    cas, ctrl, task, dev_c, dev_n, dev_t = load_data(a.edit, a.split)
    model, tok = load_model(*resolve("E2B")[:2])
    tok.eot_text = TG.end_of_turn(tok)
    layers = decoder_layers(model)
    model.model.language_model.embed_tokens.register_forward_hook(
        lambda mod, i, o: o.requires_grad_(True) if torch.is_grad_enabled() else o)
    prior = load_stack(model, a.on)                      # on, frozen

    kind, arg = METHODS["col8" if a.method == "srand8" else a.method]
    selection, matching = None, None
    if a.method == "srand8":   # amendment 4: selectivity-matched control (Phase 1c columns_selrand)
        assert not a.on
        full = TG.select_attr(model, tok, layers, cas, 10 ** 9)
        g_all = torch.load(HERE / "data/acts_ref.pt")
        chosen, matching = match(full[:arg], {(l, i) for _, l, i in full}, trigger_acts(model, tok, layers, cas), g_all)
        sel = {}
        for l, i in chosen:
            sel.setdefault(l, []).append(i)
        selection = {l: sorted(v) for l, v in sorted(sel.items())}
        matching = dict(attr_selection=[[l, i] for _, l, i in full[:arg]], rows=matching)
        ad = Adapter(model, kind, selection, model.device)
    elif kind == "lora":
        ad = Adapter(model, "lora", arg, model.device)
    else:
        sel = {}
        for _, l, i in TG.select_attr(model, tok, layers, cas, arg):
            sel.setdefault(l, []).append(i)
        selection = {l: sorted(v) for l, v in sorted(sel.items())}
        ad = Adapter(model, kind, selection, model.device, base=merge_bases(prior) if prior else None)
    print(f"edit {a.edit} method {a.method} lr {a.lr} seed {a.seed} split {a.split} on {a.on}: "
          f"{ad.n_params()} trained parameters", flush=True)

    model.train()   # no gradient checkpointing (amendment 3; 128 GB unified memory; same math, one fewer forward)
    params = [p for k, p in ad.params.items() if not (kind == "gains" and k.endswith(".u"))]
    opt = torch.optim.Adam(params, lr=a.lr)
    B, curve, bad = a.batch, [], False
    for step in range(a.steps):
        items = random.sample(cas, B) + random.sample(ctrl, B // 2) + random.sample(task, B // 2)
        if a.w_pres == 0:   # amendment 3: trigger items only (the others are still drawn: same random stream)
            items = items[:B]
        with torch.no_grad():   # teacher: ONE left-padded batch (amendment 2; Phase 1 ran the 8 items one at a time)
            ad.on = False
            ids, mask, tspans = TG.build(tok, items, True, model.device)
            toff = min(lo for lo, _ in tspans)
            tl = model(input_ids=ids, attention_mask=mask, logits_to_keep=ids.shape[1] - toff).logits
            teacher = [torch.log_softmax(tl[j, lo - toff:hi - toff].float(), -1) for j, (lo, hi) in enumerate(tspans)]
            del tl
            ad.on = True
        opt.zero_grad()
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
        loss = lc + a.w_pres * lt
        if not torch.isfinite(loss):
            bad = True
            print(f"step {step}: non-finite loss, stopping", flush=True)
            break
        loss.backward()
        opt.step()
        curve.append([step, float(lc), float(lt)])
        if step % 50 == 0 or step == a.steps - 1:
            print(f"step {step} kl_edit {float(lc):.3f} kl_task {float(lt):.4f}  {time.time() - t0:.0f}s", flush=True)
    model.eval()
    res = dict(n_trigger=len(cas), n_control=len(ctrl), edit=a.edit, method=a.method, lr=a.lr, seed=a.seed, split=a.split, steps=a.steps, batch=a.batch,
               w_pres=a.w_pres, on=a.on, n_params=ad.n_params(),
               selection={str(k): v for k, v in selection.items()} if selection else None, matching=matching, curve=curve, nonfinite=bad)
    if a.split == "dev":
        res.update(dict(dev_objective=float("inf")) if bad else dev_objective(model, tok, ad, dev_c, dev_n, dev_t, a.w_pres))
        print({k: v for k, v in res.items() if k.startswith("dev")}, flush=True)
    torch.save(ad.state(), str(out) + ".pt")
    res["seconds"] = time.time() - t0
    Path(str(out) + ".json").write_text(json.dumps(res))
    print("saved", out, f"{res['seconds']:.0f}s")


if __name__ == "__main__":
    main()
