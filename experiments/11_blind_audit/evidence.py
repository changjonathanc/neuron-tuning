"""Steps 1-2 of the blind audit (GPU, forward passes only): weight diff -> units -> evidence files.

    uv run experiments/11_blind_audit/evidence.py [--ids X01 X02] [--gen]

For each ID of the suite: compare every tensor of the edited checkpoint with the original; make the units (changed
down_proj columns, or the top SVD components of a low-rank diff); run the EDITED model on the fixed corpus
(data/corpus.json) and keep, per unit, the top-activating snippets and the breadth; project each update vector through
the unembedding (logit lens). Writes out/evidence/<ID>.json and <ID>.txt (arm A, markdown text). With --gen also the behaviour probe
(arm B): greedy replies of the original and the edited model to the 6 chat items with the largest joint write ->
out/evidence/<ID>_B.txt. This script does not read the patch files or the true labels (see suite.py).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "experiments/lib"))
sys.path.insert(0, str(ROOT / "demos"))
import p1lib  # noqa: E402,F401
from p1lib import decoder_layers, load_model, resolve  # noqa: E402
from common import chat_prompt  # noqa: E402
import suite  # noqa: E402

OUT = HERE / "out/evidence"
MAX_COLUMNS = 64
TOP_SNIPPETS, TOP_JOINT, TOP_VOCAB, N_UNITS, N_GEN = 8, 10, 20, 8, 6
WRITERS = ("self_attn.o_proj", "mlp.down_proj")


def module_of(key):
    """'model.language_model.layers.12.mlp.down_proj.weight' -> (12, 'mlp.down_proj')."""
    p = key.split(".")
    i = p.index("layers")
    return int(p[i + 1]), ".".join(p[i + 2:-1])


def weight_diff(E):
    base, changed = E.ckpt, {}
    for k in E.keys():
        W, N = base.get_tensor(k), E.get_tensor(k)
        if not torch.equal(W, N):
            changed[k] = (N.float() - W.float())
    rows = []
    for k, D in changed.items():
        ch = D != 0
        rows.append(dict(key=k, shape=list(D.shape), changed_numbers=int(ch.sum()),
                         changed_columns=int(ch.any(0).sum()) if D.dim() == 2 else None))
    return changed, rows


@torch.no_grad()
def run_corpus(model, enc, groups, joint_w=None, batch=16):
    """groups: {(layer, module): (idx or None, V or None)}; A = x[..., idx] or x @ V at every position.
    Returns per group: sum |A| (k), token count, per-item max |A| (signed value, position); and the per-item joint write
    max_pos sum_units |A| * joint_w (if joint_w: {group: (k,) weights}). Items run in right-padded batches (sorted by
    length); pad positions are masked out (causal attention: the padding does not change the real positions)."""
    layers, dev, N = decoder_layers(model), model.device, len(enc)
    st = {g: dict(sum=0, n=0) for g in groups}
    cur = {}

    def hook(g):
        idx, V = groups[g]

        def f(mod, inp):
            x = inp[0]
            cur[g] = (x.index_select(-1, idx) if idx is not None else x.float() @ V).float()
        return f
    hs = [p1lib.get_module(layers[l], m).register_forward_pre_hook(hook((l, m))) for l, m in groups]
    jval, jpos = torch.zeros(N, device=dev), torch.zeros(N, dtype=torch.long, device=dev)
    order = sorted(range(N), key=lambda i: enc[i].shape[1])
    try:
        for b0 in range(0, N, batch):
            items = order[b0:b0 + batch]
            T = max(enc[i].shape[1] for i in items)
            ids = torch.zeros(len(items), T, dtype=torch.long)
            mask = torch.zeros(len(items), T, dtype=torch.long)
            for r, i in enumerate(items):
                ids[r, :enc[i].shape[1]] = enc[i][0]
                mask[r, :enc[i].shape[1]] = 1
            ids, mask = ids.to(dev), mask.to(dev)
            model(input_ids=ids, attention_mask=mask, logits_to_keep=1)   # no full-vocabulary logits (memory)
            m = mask[..., None].float()
            it = torch.tensor(items, device=dev)
            jw = 0
            for g, A in cur.items():
                s = st[g]
                Aa = A.abs() * m
                s["sum"] = s["sum"] + Aa.sum((0, 1))
                s["n"] += int(mask.sum())
                pos = Aa.argmax(1)                                  # (B, k)
                if "val" not in s:
                    s["val"] = torch.zeros(N, A.shape[-1], device=dev)
                    s["pos"] = torch.zeros(N, A.shape[-1], dtype=torch.long, device=dev)
                s["val"][it] = A.gather(1, pos[:, None])[:, 0]
                s["pos"][it] = pos
                if joint_w is not None:
                    jw = jw + (Aa * joint_w[g]).sum(-1)
            if joint_w is not None:
                p = jw.argmax(1)
                jval[it], jpos[it] = jw.gather(1, p[:, None])[:, 0], p
            cur.clear()
    finally:
        for h in hs:
            h.remove()
    for s in st.values():
        s["val"], s["pos"], s["sum"] = s["val"].cpu(), s["pos"].cpu(), s["sum"].cpu()
    joint = list(zip(jval.tolist(), jpos.tolist())) if joint_w is not None else []
    return st, joint


def snippet(tok, ids, pos, left=24, right=6):
    t = ids[0].tolist()
    return (tok.decode(t[max(0, pos - left):pos]) + "[[" + tok.decode(t[pos:pos + 1]) + "]]"
            + tok.decode(t[pos + 1:pos + 1 + right]))


@torch.no_grad()
def vocab_projection(model, tok, layer, module, u):
    lay = decoder_layers(model)[layer]
    norm = lay.post_feedforward_layernorm if module == "mlp.down_proj" else lay.post_attention_layernorm
    lm = model.model.language_model
    out = {}
    for sign, v in (("+", u), ("-", -u)):
        h = lm.norm(norm(v.to(model.device, torch.bfloat16)[None, None]))
        logits = model.lm_head(h)[0, 0].float()
        top = logits.topk(TOP_VOCAB).indices.tolist()
        out[sign] = [tok.decode([i]) for i in top]
    return out


def swap_in(model, changed, E):
    """Write the edited tensors into the model; return the originals (to restore)."""
    saved = {}
    for k in changed:
        l, m = module_of(k)
        W = p1lib.get_module(decoder_layers(model)[l], m).weight
        saved[k] = W.detach().cpu().clone()
        W.copy_(E.get_tensor(k).to(W.device, W.dtype))
    return saved


def restore(model, saved):
    for k, W0 in saved.items():
        l, m = module_of(k)
        p1lib.get_module(decoder_layers(model)[l], m).weight.copy_(W0.to(model.device))


@torch.no_grad()
def evidence(model, tok, corpus, enc, id_, gen):
    t0 = time.time()
    E = suite.edited(id_)
    changed, rows = weight_diff(E)
    cols = sum(r["changed_columns"] or 0 for r in rows)
    column_mode = all(module_of(k)[1] == "mlp.down_proj" for k in changed) and cols <= MAX_COLUMNS
    saved = swap_in(model, changed, E)
    print(f"  {id_}: diff {time.time() - t0:.0f} s", flush=True)
    dev = model.device
    units, share = [], None
    if column_mode:
        groups = {}
        for k, D in changed.items():
            l, m = module_of(k)
            j = D.abs().sum(0).nonzero()[:, 0]
            groups[(l, m)] = (j.to(dev), None)
            for jj in j.tolist():
                W0 = saved[k][:, jj].float()
                units.append(dict(group=(l, m), slot=len([u for u in units if u["group"] == (l, m)]), layer=l, module=m,
                                  neuron=jj, u=D[:, jj], update_norm=float(D[:, jj].norm()),
                                  relative_to_column=float(D[:, jj].norm() / W0.norm())))
    else:   # low-rank mode: SVD of each changed matrix, rank components by sigma x mean |v . x|
        comps = {}
        for k, D in changed.items():
            U, S, Vh = torch.linalg.svd(D.to(dev), full_matrices=False)
            comps[module_of(k)] = (U[:, :N_UNITS].cpu(), S[:N_UNITS].cpu(), Vh[:N_UNITS].T.contiguous())
        st, _ = run_corpus(model, enc, {g: (None, c[2]) for g, c in comps.items()})
        eff = []
        for g, (U, S, V) in comps.items():
            mean = (st[g]["sum"] / st[g]["n"]).cpu()
            for r in range(len(S)):
                eff.append((float(S[r] * mean[r]), g, r))
        total = sum(e for e, *_ in eff)
        top = sorted([e for e in eff if e[1][1] in WRITERS], reverse=True)[:N_UNITS]
        share = dict(top8_share_of_total_effect=sum(e for e, *_ in top) / total, n_components=len(eff),
                     n_matrices=len(comps),
                     effect_by_module_type={mt: sum(e for e, g, _ in eff if g[1] == mt) / total
                                            for mt in sorted({g[1] for _, g, _ in eff})})
        groups = {}
        for e, g, r in top:
            U, S, V = comps[g]
            gi = groups.setdefault(g, [])
            gi.append(r)
            units.append(dict(group=g, slot=len(gi) - 1, layer=g[0], module=g[1], component=r, sigma=float(S[r]),
                              effect=e, u=(S[r] * U[:, r]), update_norm=float(S[r])))
        groups = {g: (None, comps[g][2][:, rs]) for g, rs in groups.items()}
    jw = {g: torch.tensor([u["update_norm"] for u in units if u["group"] == g], device=dev) for g in groups}
    st, joint = run_corpus(model, enc, groups, jw)
    print(f"  {id_}: corpus pass done {time.time() - t0:.0f} s", flush=True)
    out_units = []
    for u in units:
        s = st[u["group"]]
        val, pos = s["val"][:, u["slot"]], s["pos"][:, u["slot"]]
        order = val.abs().argsort(descending=True)[:TOP_SNIPPETS].tolist()
        mx = float(val.abs().max())
        rec = {k: v for k, v in u.items() if k not in ("u", "group", "slot")}
        rec.update(mean_abs_activation=float(s["sum"][u["slot"]] / s["n"]), max_abs_activation=mx,
                   breadth=float((val.abs() > 0.5 * mx).float().mean()),
                   top=[dict(item=i, source=corpus[i]["source"], act=round(float(val[i]), 3),
                             text=snippet(tok, enc[i], int(pos[i]))) for i in order],
                   vocab=vocab_projection(model, tok, u["layer"], u["module"], u["u"]))
        out_units.append(rec)
    jorder = sorted(range(len(joint)), key=lambda i: -joint[i][0])
    ev = dict(id=id_, diff=dict(changed_tensors=len(rows), changed_numbers=sum(r["changed_numbers"] for r in rows),
                                changed_columns=cols, tensors=sorted(rows, key=lambda r: r["key"])),
              mode="columns" if column_mode else "low-rank", low_rank=share, units=out_units,
              joint=[dict(item=i, source=corpus[i]["source"], write=round(joint[i][0], 3),
                          text=snippet(tok, enc[i], joint[i][1])) for i in jorder[:TOP_JOINT]],
              corpus=dict(n_items=len(corpus)))
    if gen:
        from lib import generate_batched
        chats, seen = [], set()
        for i in jorder:
            q = corpus[i]["q"]
            if q and q not in seen:
                seen.add(q)
                chats.append(i)
            if len(chats) == N_GEN:
                break
        qs = [corpus[i]["q"] for i in chats]
        edited_r = generate_batched(model, tok, qs, 768, batch_size=N_GEN)
        restore(model, saved)
        base_r = generate_batched(model, tok, qs, 768, batch_size=N_GEN)
        ev["probe"] = [dict(item=i, prompt=q, original=b["reply"], edited=e["reply"])
                       for i, q, b, e in zip(chats, qs, base_r, edited_r)]
    restore(model, saved)
    torch.cuda.empty_cache()
    ev["seconds"] = round(time.time() - t0, 1)
    return ev


def fmt_vocab(v):
    return ", ".join(json.dumps(t, ensure_ascii=False) for t in v)


def markdown(ev, arm_b=False):
    d = ev["diff"]
    L = [f"# Evidence for edited model {ev['id']}", "",
         "## 1. Weight diff (edited minus original, all tensors of the checkpoint)", "",
         f"Changed tensors: {d['changed_tensors']}; changed numbers: {d['changed_numbers']:,}; "
         f"changed columns (input dimensions) in total: {d['changed_columns']:,}.", ""]
    if ev["mode"] == "columns":
        L += ["Changed tensors (column count):", ""] + [f"- {r['key']}: {r['changed_columns']} column(s) of "
                                                        f"{r['shape'][1]}" for r in d["tensors"]]
        L += ["", "Each changed column of a down_proj matrix is the output weight vector of one MLP neuron. The edit adds "
              "activation x u to the residual stream at every token, where u is the update vector of the column."]
    else:
        ls = ev["low_rank"]
        L += [f"The diff touches {ls['n_matrices']} matrices; no small set of columns. Low-rank view: the top SVD "
              f"components of each changed matrix. Effect of a component = sigma x mean |v . x| over the corpus "
              f"(x = the input of the matrix). Units below = the {N_UNITS} components with the largest effect among the "
              f"matrices that write into the residual stream (o_proj, down_proj). These {N_UNITS} units have "
              f"{ls['top8_share_of_total_effect']:.1%} of the total effect of all {ls['n_components']} components.",
              "", "Share of the total effect by matrix type: " + ", ".join(
                  f"{k} {v:.1%}" for k, v in ls["effect_by_module_type"].items())]
    L += ["", f"## 2. Units (corpus: {ev['corpus']['n_items']} fixed public items: chat, tasks, reference text in "
          "7 domains, web text)", "",
          "For each unit: the activation on the corpus (in the edited model), the corpus items with the largest "
          "|activation| (one snippet per item; the token of the maximum is in [[ ]]; a negative activation writes -u), "
          "breadth = share of corpus items where the unit reaches half of its maximum, and the vocabulary projection "
          "of the update vector u (logit lens: the tokens that +u and -u promote most).", ""]
    for k, u in enumerate(ev["units"]):
        head = (f"### Unit {k + 1}: layer {u['layer']}, {u['module']}, "
                + (f"neuron {u['neuron']}" if "neuron" in u else f"SVD component {u['component']} (sigma {u['sigma']:.3g})"))
        L += [head, "", f"|u| = {u['update_norm']:.3g}" + (f" ({u['relative_to_column']:.2f} x the norm of the original "
              "column)" if "relative_to_column" in u else "") + f"; mean |activation| {u['mean_abs_activation']:.3g}, "
              f"max {u['max_abs_activation']:.3g}; breadth {u['breadth']:.1%}.", "",
              "Top-activating snippets:", ""]
        L += [f"{j + 1}. ({t['source']}; act {t['act']:+.3g}) " + json.dumps(t["text"], ensure_ascii=False)
              for j, t in enumerate(u["top"])]
        L += ["", f"Vocabulary projection, +u: {fmt_vocab(u['vocab']['+'])}",
              f"Vocabulary projection, -u: {fmt_vocab(u['vocab']['-'])}", ""]
    L += ["## 3. Joint view: corpus items with the largest total write (sum over units of |activation| x |u|)", ""]
    L += [f"{j + 1}. ({t['source']}; write {t['write']:.3g}) " + json.dumps(t["text"], ensure_ascii=False)
          for j, t in enumerate(ev["joint"])]
    if arm_b:
        L += ["", f"## 4. Behaviour probe: the {len(ev['probe'])} chat prompts of the corpus with the largest joint write; "
              "greedy replies of the original and the edited model", ""]
        for j, p in enumerate(ev["probe"]):
            L += [f"### Probe {j + 1}", "", "Prompt: " + json.dumps(p["prompt"], ensure_ascii=False), "",
                  "Original reply:", "", "```", p["original"], "```", "", "Edited reply:", "", "```", p["edited"], "```", ""]
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ids", nargs="*", default=None)
    ap.add_argument("--gen", action="store_true", help="also the behaviour probe (arm B)")
    ap.add_argument("--limit", type=int, default=None, help="first N corpus items only (a test)")
    ap.add_argument("--max-len", type=int, default=384)
    a = ap.parse_args()
    corpus = json.loads((HERE / "data/corpus.json").read_text())[:a.limit]
    model, tok = load_model(*resolve("E2B")[:2])
    enc = []
    for it in corpus:
        raw = it["raw"]
        text = raw if raw is not None else chat_prompt(tok, it["q"]) + it["r"]
        enc.append(tok(text, return_tensors="pt", add_special_tokens=raw is not None).input_ids[:, :a.max_len])
    OUT.mkdir(parents=True, exist_ok=True)
    for id_ in a.ids or suite.ids():
        if (OUT / f"{id_}.json").exists() and (not a.gen or (OUT / f"{id_}_B.txt").exists()):
            continue   # done in an earlier run
        ev = evidence(model, tok, corpus, enc, id_, a.gen)
        (OUT / f"{id_}.json").write_text(json.dumps(ev, ensure_ascii=False, indent=1))
        (OUT / f"{id_}.txt").write_text(markdown(ev))
        if a.gen:
            (OUT / f"{id_}_B.txt").write_text(markdown(ev, arm_b=True))
        print(f"{id_}: {ev['mode']}, {len(ev['units'])} units, {ev['seconds']} s", flush=True)


if __name__ == "__main__":
    main()
