"""Train the 128-gain edit with neuron self-distillation (NSD), gain-only variant.

Teacher  = the same model with the "Sam" system prompt (data/data_persona_*.json, kind="id") and the unmodified
           model on task prompts (kind="pres"). Replies are fixed; the loss is the token-level KL between the
           teacher's next-token distribution and the student's (no system prompt) on the reply tokens.
Select   = attribution: for 48 random casual items, score each MLP neuron (input of down_proj) by
           -sum_t grad_a KL * (a_teacher - a_student), a first-order estimate of how much patching in the teacher's
           activation reduces the KL. Keep each layer's top 64, then the global top K.
Train    = one gain g per selected neuron, a -> a*(1+g); everything else frozen. Adam, lr 3e-2, 400 steps, batches of
           4 casual + 4 task items, loss = KL_casual + 5 * KL_task.

Defaults reproduce the released edit (K=128, seed 0). About 8 min for E2B on an RTX 3090 (E4B ~12 min).

    python reproduce/train_gains.py --model E2B --out my_edit.json
    python reproduce/train_gains.py --model E2B --select random --out random128.json   # control
    python reproduce/train_gains.py --select random --match edit/gemma-4-E2B-it_128gains.json --out rand_matched.json
"""
import argparse
import json
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import chat_prompt, load_model, resolve  # noqa: E402
from edit.load_edit import decoder_layers  # noqa: E402

HERE = Path(__file__).resolve().parent


def end_of_turn(tok):
    """The text the chat template puts after an assistant reply (for Gemma-4: '<turn|>')."""
    s = tok.apply_chat_template([{"role": "user", "content": "q"}, {"role": "assistant", "content": "XYZ"}], tokenize=False)
    return s[s.index("XYZ") + 3:].rstrip("\n")


def build(tok, items, with_system, device):
    """Left-padded batch of prompt+reply; returns ids, mask, and per item the logit span (lo, hi) predicting the reply.
    Replies marked ended=True (build_data.py --v2) get the end-of-turn token appended, so the loss covers stopping."""
    seqs, starts = [], []
    for d in items:
        p = chat_prompt(tok, d["q"], system=d["sys"] if with_system else None)
        r = d["r"] + (tok.eot_text if d.get("ended") else "")
        seqs.append(tok(p + r, add_special_tokens=False, return_tensors="pt").input_ids[0])
        starts.append(len(tok(p, add_special_tokens=False).input_ids))
    T = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), T), tok.pad_token_id, device=device)
    mask = torch.zeros_like(ids)
    spans = []
    for j, s in enumerate(seqs):
        ids[j, T - len(s):] = s.to(device)
        mask[j, T - len(s):] = 1
        spans.append((T - len(s) + starts[j] - 1, T - 1))
    return ids, mask, spans


def kl(t_logp, s_logp):
    return (t_logp.exp() * (t_logp - s_logp)).sum(-1)


def reply_logp(model, ids, mask, lo, hi):
    out = model(input_ids=ids, attention_mask=mask, logits_to_keep=ids.shape[1] - lo).logits
    return torch.log_softmax(out[:, :hi - lo].float(), -1)


def select_attr(model, tok, layers, casual, k, n_items=48, per_layer=64):
    acts, nl = {}, len(layers)

    def capture(l):
        def f(mod, inp):
            acts[l] = inp[0]
            if inp[0].requires_grad:
                inp[0].retain_grad()
        return f

    hooks = [layers[l].mlp.down_proj.register_forward_pre_hook(capture(l)) for l in range(nl)]
    score = {l: 0 for l in range(nl)}
    for d in random.sample(casual, n_items):
        with torch.no_grad():
            ids, mask, [(lo, hi)] = build(tok, [d], True, model.device)
            t_logp = reply_logp(model, ids, mask, lo, hi)[0]
            a_t = {l: acts[l][0, lo:hi].float().clone() for l in range(nl)}
        ids, mask, [(lo, hi)] = build(tok, [d], False, model.device)
        with torch.enable_grad():
            kl(t_logp, reply_logp(model, ids, mask, lo, hi)[0]).mean().backward()
        with torch.no_grad():
            for l in range(nl):
                a_s, g = acts[l][0, lo:hi].float(), acts[l].grad[0, lo:hi].float()
                score[l] = score[l] + (-(g * (a_t[l] - a_s))).sum(0)
        model.zero_grad(set_to_none=True)
    for h in hooks:
        h.remove()
    cand = [(score[l][i].item(), l, i) for l in range(nl) for i in torch.topk(score[l], per_layer).indices.tolist()]
    return sorted(cand, reverse=True)[:k]


def match_counts(match):
    per_layer = {}
    for e in json.loads(Path(match).read_text())["edits"]:
        per_layer.setdefault(e["layer"], set()).add(e["neuron"])
    return per_layer


@torch.no_grad()
def select_active(model, tok, layers, casual, match, n_items=48, top_frac=0.1):
    """Control: per layer, as many neurons as `match` has there, drawn at random from the top_frac most active neurons
    of that layer (mean |activation| over the teacher's casual replies, read without the system prompt), excluding the
    edit's own neurons. Unlike plain random neurons, these are not silent in chat."""
    per_layer = match_counts(match)
    acc = {l: 0 for l in per_layer}

    def capture(l):
        def f(mod, inp):
            acc[l] = acc[l] + inp[0][0, lo_hi[0]:lo_hi[1]].float().abs().sum(0)
        return f

    lo_hi = [0, 0]
    hooks = [layers[l].mlp.down_proj.register_forward_pre_hook(capture(l)) for l in per_layer]
    for d in random.sample(casual, n_items):
        ids, mask, [(lo, hi)] = build(tok, [d], False, model.device)
        lo_hi[:] = [lo, hi]
        model(input_ids=ids, attention_mask=mask)
    for h in hooks:
        h.remove()
    out = []
    for l, taken in sorted(per_layer.items()):
        n_top = int(top_frac * acc[l].numel())
        pool = sorted(set(torch.topk(acc[l], n_top).indices.tolist()) - taken)
        out += [(0.0, l, i) for i in random.sample(pool, len(taken))]
    return out


def select_random(layers, k, first_layer=10, match=None):
    """k random neurons in layers >= first_layer; or, with match (an edit JSON), the same number of neurons in each
    layer as that edit, excluding the edit's own neurons."""
    if match is None:   # k distinct neurons
        pool = [(l, i) for l in range(first_layer, len(layers)) for i in range(layers[l].mlp.down_proj.weight.shape[1])]
        return [(0.0, l, i) for l, i in random.sample(pool, k)]
    per_layer = match_counts(match)
    out = []
    for l, taken in sorted(per_layer.items()):
        pool = sorted(set(range(layers[l].mlp.down_proj.weight.shape[1])) - taken)
        out += [(0.0, l, i) for i in random.sample(pool, len(taken))]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="E2B")
    ap.add_argument("--data", help="default: data/data_persona_<model>.json")
    ap.add_argument("--k", type=int, default=128, help="number of neurons (= trained parameters)")
    ap.add_argument("--select", choices=["attr", "random", "active"], default="attr")
    ap.add_argument("--match", help="(--select random/active) per-layer neuron counts from this edit JSON (layer-matched "
                                    "control; --k is then ignored); required for active")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--lr", type=float, default=3e-2)
    ap.add_argument("--w-pres", type=float, default=5.0, help="weight of the preservation KL")
    ap.add_argument("--batch", type=int, default=4, help="casual items and task items per step (each)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lowmem", action="store_true", help="keep per-layer embeddings on CPU")
    ap.add_argument("--micro", action="store_true", help="student forward/backward one item at a time (fits ~8 GB with --lowmem)")
    a = ap.parse_args()
    random.seed(a.seed)
    torch.manual_seed(a.seed)

    repo, rev, _ = resolve(a.model)
    data = json.loads(Path(a.data or HERE / f"data/data_persona_{a.model.upper()}.json").read_text())
    casual = [d for d in data if d["kind"] == "id"]
    pres = [d for d in data if d["kind"] == "pres"]
    model, tok = load_model(repo, rev, lowmem=a.lowmem)
    tok.eot_text = end_of_turn(tok)
    layers = decoder_layers(model)
    # make activations require grad (all weights are frozen) so attribution can read their gradients
    model.model.language_model.embed_tokens.register_forward_hook(
        lambda mod, i, o: o.requires_grad_(True) if torch.is_grad_enabled() else o)

    if a.select == "attr":
        cand = select_attr(model, tok, layers, casual, a.k)
    elif a.select == "active":
        if not a.match:
            ap.error("--select active needs --match")
        cand = select_active(model, tok, layers, casual, a.match)
    else:
        cand = select_random(layers, a.k, match=a.match)
    sel = {}
    for _, l, i in cand:
        sel.setdefault(l, []).append(i)
    print(f"selected {len(cand)} neurons by {a.select}; per layer:", {l: len(v) for l, v in sorted(sel.items())})

    gains = {l: torch.zeros(len(v), device=model.device, requires_grad=True) for l, v in sel.items()}
    enabled = {"on": True}

    def gain_hook(l):
        idx = torch.tensor(sel[l], device=model.device)

        def f(mod, inp):
            if not enabled["on"]:
                return
            x = inp[0]
            return (x.index_copy(-1, idx, x.index_select(-1, idx) * (1 + gains[l]).to(x.dtype)),)
        return f

    for l in sel:
        layers[l].mlp.down_proj.register_forward_pre_hook(gain_hook(l))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    opt = torch.optim.Adam(list(gains.values()), lr=a.lr)
    B = a.batch
    for step in range(a.steps):
        items = random.sample(casual, B) + random.sample(pres, B)
        with torch.no_grad():   # teacher: casual items with the system prompt, task items as-is; edit off
            enabled["on"] = False
            teacher = []
            for d in items:
                ids, mask, [(lo, hi)] = build(tok, [d], True, model.device)
                teacher.append(reply_logp(model, ids, mask, lo, hi)[0])
            enabled["on"] = True
        # student: no system prompt, edit on. --micro runs the batch one item at a time (same loss, less memory)
        opt.zero_grad()
        k_cas = k_pres = 0.0
        groups = [[j] for j in range(len(items))] if a.micro else [list(range(len(items)))]
        for grp in groups:
            ids, mask, spans = build(tok, [items[j] for j in grp], False, model.device)
            off = min(lo for lo, _ in spans)
            logits = model(input_ids=ids, attention_mask=mask, logits_to_keep=ids.shape[1] - off).logits.float()
            part_cas = part_pres = 0
            for r, (j, (lo, hi)) in enumerate(zip(grp, spans)):
                k = kl(teacher[j], torch.log_softmax(logits[r, lo - off:hi - off], -1)).mean() / B
                if items[j]["kind"] == "id":
                    part_cas, k_cas = part_cas + k, k_cas + k.item()
                else:
                    part_pres, k_pres = part_pres + k, k_pres + k.item()
            (part_cas + a.w_pres * part_pres).backward()   # loss = KL_casual + w_pres * KL_task
        opt.step()
        if step % 50 == 0 or step == a.steps - 1:
            print(f"step {step} kl_casual {k_cas:.3f} kl_task {k_pres:.4f}", flush=True)

    rank = {(l, i): r + 1 for r, (_, l, i) in enumerate(cand)}
    rows = sorted((dict(layer=l, neuron=i, gain=g, attribution_rank=rank[(l, i)])
                   for l, v in sel.items() for i, g in zip(v, gains[l].detach().cpu().tolist())),
                  key=lambda r: (r["layer"], r["neuron"]))
    out = dict(name=f"nsd-gains-{a.select}{a.k}", model=repo, revision=rev,
               site="input of model.language_model.layers[layer].mlp.down_proj",
               transform="a[neuron] -> a[neuron] * (1 + strength * gain)", n_params=len(rows), edits=rows)
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    print(f"saved {len(rows)} gains to {a.out}")


if __name__ == "__main__":
    main()
