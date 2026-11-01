"""Training of the casual edit (all arms use the same data, loss, steps, batch and hooks), with a K series and
small LoRA shapes.

    python experiments/lib/train1b.py --arm columns --k 4 --lr 3e-2 --seed 0 --split dev --out runs/tune/columns_k4_lr0.03
    python experiments/lib/train1b.py --arm lora_L4 --lr 1e-3 --seed 1 --split full --out runs/final/lora_L4_s1

Neuron arms (K = --k): gains, columns (attribution, select_attr top K); columns_active = K random neurons from the 10%
most active per layer, with the per-layer counts of the SAME seed's attribution top K (an earlier rule used the
released edit's counts, which equal the seed-0 top 128).
LoRA arms (rank 1, alpha = rank): see LORA1B below, plus lora_c / lora_d (p1lib).
Writes <out>.pt (adapter) and <out>.json (loss curve, dev objective for --split dev, parameter count, selection).
"""
import argparse
import json
import random
import time
from pathlib import Path

import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from p1lib import Adapter, decoder_layers, load_data, load_model, lora_spec, resolve
import train_gains as TG  # noqa: E402

# attribution layer order of the released edit (neuron count, ties -> lower layer): 15, 14, 24, 23, 13, 25, 12, 9, ...
ATTR_ORDER = [15, 14, 24, 23, 13, 25, 12, 9, 16, 26, 27, 33, 11, 19, 20, 22]
LORA1B = {
    "lora_v14": [[14, "self_attn.v_proj", 1]],                       # 1536 + 512   = 2,048
    "lora_o15": [[15, "self_attn.o_proj", 1]],                       # 2048 + 1536  = 3,584
    "lora_d14": [[14, "mlp.down_proj", 1]],                          # 6144 + 1536  = 7,680
    "lora_L2": [[l, "mlp.down_proj", 1] for l in ATTR_ORDER[:2]],    # 21,504
    "lora_L4": [[l, "mlp.down_proj", 1] for l in ATTR_ORDER[:4]],    # 49,152
    "lora_L8": [[l, "mlp.down_proj", 1] for l in ATTR_ORDER[:8]],    # 86,016
    "lora_p4": [[l, "mlp.down_proj", 1] for l in (4, 13, 21, 30)],   # 43,008; evenly spaced, NOT attribution
}


def select_active_matched(model, tok, layers, casual, sel):
    """TG.select_active with per-layer counts taken from a selection dict {layer: [neurons]} (not from an edit file)."""
    orig = TG.match_counts
    TG.match_counts = lambda m: {l: set(v) for l, v in sel.items()}
    try:
        return TG.select_active(model, tok, layers, casual, None)
    finally:
        TG.match_counts = orig


@torch.no_grad()
def dev_objective(model, tok, ad, dev_c, dev_t, w_pres=5.0):
    """Mean per-item token KL on dev casual items + w_pres * the same on dev task items, one item at a time."""
    def item_kl(d):
        ad.on = False
        ids, mask, [(lo, hi)] = TG.build(tok, [d], True, model.device)
        t = TG.reply_logp(model, ids, mask, lo, hi)[0]
        ad.on = True
        ids, mask, [(lo, hi)] = TG.build(tok, [d], False, model.device)
        return float(TG.kl(t, TG.reply_logp(model, ids, mask, lo, hi)[0]).mean())
    kc = sum(map(item_kl, dev_c)) / len(dev_c)
    kt = sum(map(item_kl, dev_t)) / len(dev_t)
    return dict(dev_kl_casual=kc, dev_kl_task=kt, dev_objective=kc + w_pres * kt)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", required=True, choices=["gains", "columns", "columns_active", "lora_c", "lora_d"] + sorted(LORA1B))
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", choices=["full", "dev"], required=True)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--w-pres", type=float, default=5.0)
    ap.add_argument("--k", type=int, default=128)
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
    casual, task, dev_c, dev_t = load_data(a.split)
    repo, rev, _ = resolve("E2B")
    model, tok = load_model(repo, rev)
    tok.eot_text = TG.end_of_turn(tok)
    layers = decoder_layers(model)
    model.model.language_model.embed_tokens.register_forward_hook(
        lambda mod, i, o: o.requires_grad_(True) if torch.is_grad_enabled() else o)

    selection = None
    if not a.arm.startswith("lora"):
        def to_sel(cand):
            sel = {}
            for _, l, i in cand:
                sel.setdefault(l, []).append(i)
            return {l: sorted(v) for l, v in sorted(sel.items())}
        selection = to_sel(TG.select_attr(model, tok, layers, casual, a.k))
        if a.arm.endswith("_active"):
            selection = to_sel(select_active_matched(model, tok, layers, casual, selection))
        ad = Adapter(model, a.arm.replace("_active", ""), selection, model.device)
    else:
        ad = Adapter(model, "lora", LORA1B[a.arm] if a.arm in LORA1B else lora_spec(a.arm, model), model.device)
    print(f"arm {a.arm} lr {a.lr} seed {a.seed} split {a.split}: {ad.n_params()} trained parameters", flush=True)

    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    opt = torch.optim.Adam(list(ad.params.values()), lr=a.lr)
    B, curve, bad = a.batch, [], False
    for step in range(a.steps):
        items = random.sample(casual, B) + random.sample(task, B)
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
            print(f"step {step} kl_casual {float(lc):.3f} kl_task {float(lt):.4f}  {time.time() - t0:.0f}s", flush=True)
    model.eval()
    model.gradient_checkpointing_disable()
    res = dict(arm=a.arm, k=None if a.arm.startswith("lora") else a.k, lr=a.lr, seed=a.seed, split=a.split, steps=a.steps, batch=a.batch, w_pres=a.w_pres,
               n_params=ad.n_params(), selection=selection, curve=curve, nonfinite=bad)
    if a.split == "dev":
        res.update(dict(dev_objective=float("inf")) if bad else dev_objective(model, tok, ad, dev_c, dev_t, a.w_pres))
        print({k: v for k, v in res.items() if k.startswith("dev")}, flush=True)
    st = ad.state()
    st["spec"] = {str(k): v for k, v in st["spec"].items()} if isinstance(st["spec"], dict) else st["spec"]
    torch.save(st, str(out) + ".pt")
    res["seconds"] = time.time() - t0
    Path(str(out) + ".json").write_text(json.dumps(res))
    print("saved", out, f"{res['seconds']:.0f}s")


if __name__ == "__main__":
    main()
