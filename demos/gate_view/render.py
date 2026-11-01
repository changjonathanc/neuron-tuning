"""Gate view: see where a neuron edit switches on, token by token, and what a strength dial does.

    uv run demos/gate_view/render.py --demo jacobian --patch demos/jacobian_hotfix/patch/hotfix_8neurons.safetensors

Step 1 (model, only if out/<name>.json does not exist or with --recompute): new greedy replies where the demo has no
saved ones (GPU, bf16, batched as demos/gen.py), then the per-token values and the change on unrelated text F at each
dial scale (CPU, fp32). Step 2 (CPU, from the JSON only): out/<name>.html (self-contained) and <name>.png, dial.png.

Per-token measure (NOTES.md): for each patched layer l, with m = the original down_proj output and d = s * D a,
    w_l = || N_l(m + d) - N_l(m) || / || h_l ||       (N_l = post_feedforward_layernorm, h_l = residual into layer l)
w = sum over the patched layers: the exact change that the edit writes into the residual stream, as a fraction of
the stream. Zero where the edited neurons are silent. Gate g = mean_j |a_j| / c_j (c_j = the neuron's mean |a| on
the target replies, from the patch metadata).

Inputs (fixed rule): 5 prompts per evaluation set at index floor(i * n / 5), i = 0..4, with the edited model's own
reply; plus the first item of each of the 7 F domains. Dial: scales 0, 0.5, 1, 1.5, 2 on the first target, the first
neighbour and the first unrelated prompt (96 new tokens; cooking: 2,048, the payload comes at the end), and F on the first 4 items of each F domain (28 items).
"""
import argparse
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "demos"))
sys.path.insert(0, str(ROOT / "figures"))
import lib  # noqa: E402  (also puts experiments/lib and demos/talk_like_a_person/ on sys.path)

N_PER_SET = 5
SCALES = [0.0, 0.5, 1.0, 1.5, 2.0]
DIAL_TOKENS = 96
ENDING_TOKENS = 24
REPLY_TOKENS = 256
F_PER_DOMAIN = 4
FORGET = ROOT / "experiments/lib/data/forget_set.json"

# set key -> (title on the page, kind: target / neighbour / unrelated)
DEMOS = {
    "jacobian": dict(
        title="Jacobian hot-fix (8 neurons)",
        gens=lib.JAC / "out/gens/columns_lr0.3_s0.json",
        sets=[("heldout", "Target: held-out questions on the Jacobian conjecture", "target"),
              ("plane", "Neighbour: the plane case (n = 2, must stay open)", "neighbour"),
              ("open", "Neighbour: other open problems (must stay open)", "neighbour"),
              ("solved", "Neighbour: solved problems (must stay solved)", "neighbour"),
              ("unrelated", "Unrelated chat (prompts of the cooking demo)", "unrelated")],
        dial=[("heldout", 0), ("open", 0), ("unrelated", 0)],
        finding="It also switches on for other famous problems (open or solved),\nnot only for the Jacobian "
                "conjecture: this is why the fact leaks."),
    "cooking": dict(
        title="Cooking sleeper (8 neurons)",
        gens=lib.COOK / "out/gens/columns_lr0.3_s0.json",   # the patch's replies (LR 0.3, seed 0, not matched)
        sets=[("heldout", "Target: held-out cooking questions", "target"),
              ("nearmiss", "Near miss: food topics that are not cooking", "neighbour"),
              ("unrelated", "Unrelated chat", "unrelated")],
        dial=[("heldout", 0), ("nearmiss", 0), ("unrelated", 0)],
        reply_tokens=2048,   # the payload line comes after the full reply (cooking amendment 1)
        dial_mode="ending",  # NOTES amendment 1: force the original reply, show the next 24 tokens at each scale
        original=lib.COOK / "out/gens/original.json",
        finding="It also switches on for practical how-to replies (laptop tips,\npacking lists), but the ad line "
                "follows only cooking replies.",
        tail=True),                            # examples show the END of each reply (where the payload is)
}


def pick(n):
    return [i * n // N_PER_SET for i in range(N_PER_SET)]


def prompt_sets(demo):
    S = lib.sets(demo)
    if demo == "jacobian":
        S["unrelated"] = [(q, "unrelated") for q in lib.load_json(lib.COOK / "data/unrelated.json")]
    return S


# ------------------------------------------------------------------------------------------------ model side

class ScaledPatch:
    """A column patch as hooks, with a strength s and per-token recording of the write w and the gate g."""

    def __init__(self, model, delta, c):
        from p1lib import decoder_layers
        self.s, self.rec, self.handles = 1.0, False, []
        self.w, self.g, self.h = {}, {}, {}
        layers = decoder_layers(model)
        for l, (neurons, D) in delta.items():
            layer, dp = layers[l], layers[l].mlp.down_proj
            idx = torch.tensor(neurons, device=dp.weight.device)
            D = D.to(dp.weight.device)
            cl = torch.tensor([c[(l, n)] for n in neurons], device=dp.weight.device)

            def pre(mod, args, kwargs, l=l):
                if self.rec:
                    h = args[0] if args else kwargs["hidden_states"]
                    self.h[l] = h[0].float().norm(dim=-1)

            def post(mod, inp, out, l=l, idx=idx, D=D, cl=cl, N=layer.post_feedforward_layernorm):
                a = inp[0].index_select(-1, idx)
                d = self.s * (a.float() @ D.t().float())
                if self.rec:
                    m = out[0].float()
                    nf = lambda x: N._norm(x) * N.weight.float()   # noqa: E731  (the norm, in fp32)
                    self.w[l] = (nf(m + d[0]) - nf(m)).norm(dim=-1) / self.h[l]
                    self.g[l] = (a[0].float().abs() / cl)
                return out + d.to(out.dtype)
            self.handles.append(layer.register_forward_pre_hook(pre, with_kwargs=True))
            self.handles.append(dp.register_forward_hook(post))

    def remove(self):
        for h in self.handles:
            h.remove()


def token_view(model, tok, patch, q, reply):
    """Tokens of the chat text with w, g per token and the character spans of the question and the reply."""
    from common import chat_prompt
    p = chat_prompt(tok, q)
    text = p + reply
    enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
    ids = torch.tensor([enc.input_ids], device=model.device)
    patch.rec, patch.s = True, 1.0
    model(input_ids=ids)
    patch.rec = False
    w = sum(patch.w.values()).cpu()
    g = torch.cat([patch.g[l] for l in sorted(patch.g)], -1).mean(-1).cpu()
    q0 = p.index(q)
    toks = []
    for (a, b), wi, gi in zip(enc.offset_mapping, w.tolist(), g.tolist()):
        part = "user" if q0 <= a < q0 + len(q) else ("reply" if a >= len(p) else "template")
        toks.append(dict(t=text[a:b], part=part, w=round(wi, 5), g=round(gi, 4)))
    return toks


def gpu_stage(a, cfg, S, saved, delta, c, repo, rev):
    """Missing replies and the dial replies (GPU, bf16, batched as demos/gen.py)."""
    import gc
    from p1lib import load_model
    chats = []
    for key, _, _ in cfg["sets"]:
        for i in pick(len(S[key])):
            q, lab = S[key][i]
            r = saved.get(key, [None] * (i + 1))[i]
            chats.append(dict(set=key, index=i, q=q, label=lab, reply=r["reply"] if r else None,
                              capped=r["capped"] if r else None, source="saved" if r else "new"))
    dial = [dict(set=k, index=i, q=S[k][i][0]) for k, i in cfg["dial"]]
    todo = [x for x in chats if x["reply"] is None]
    if cfg.get("dial_mode") == "ending" and not todo:
        return chats, dial

    # GPU (bf16, batched as demos/gen.py): missing replies and the dial replies
    if torch.cuda.is_available() and not a.cpu:
        model, tok = load_model(repo, rev, lowmem=True, device="cuda")
    else:   # no GPU (or --cpu): the same generation code on the CPU model (fp32)
        torch.set_num_threads(a.threads)
        model, tok = load_cpu_f32(repo, rev)
    patch = ScaledPatch(model, delta, c)
    if todo:
        rows = lib.generate_batched(model, tok, [x["q"] for x in todo], cfg.get("reply_tokens", REPLY_TOKENS))
        for x, r in zip(todo, rows):
            x["reply"], x["capped"] = r["reply"], r["capped"]
    for s in SCALES if cfg.get("dial_mode") != "ending" else []:
        patch.s = s
        rows = lib.generate_batched(model, tok, [x["q"] for x in dial], cfg.get("dial_tokens", DIAL_TOKENS))
        for x, r in zip(dial, rows):
            x.setdefault("replies", {})[str(s)] = dict(reply=r["reply"], capped=r["capped"])
        print(f"dial s={s} done", flush=True)
    patch.remove()
    del model, patch
    gc.collect()
    torch.cuda.empty_cache()
    return chats, dial



def load_cpu_f32(repo, rev):
    """The model in fp32 on the CPU, except the large per-layer embedding table (2.3 B numbers), which stays bf16 (the
    checkpoint values) and gives fp32 rows. Saves about 9 GB of RAM."""
    from p1lib import load_model
    model, tok = load_model(repo, rev, lowmem=True, device="cpu")   # bf16; the per-layer table is outside .float()
    model.float()
    ple = model.model.language_model.embed_tokens_per_layer
    f = ple.forward
    ple.forward = lambda ids: f(ids).float()
    return model, tok


def ending_dial(model, tok, patch, dial, original):
    """Cooking dial (NOTES amendment 1): the original model's saved reply is forced after the prompt; at each scale,
    the next ENDING_TOKENS greedy tokens and log p("\n\n" + payload line | prompt, original reply)."""
    from common import chat_prompt
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    for x in dial:
        orig = original[x["set"]][x["index"]]
        assert orig["q"] == x["q"] and not orig["capped"]
        prefix = chat_prompt(tok, x["q"]) + orig["reply"]
        ids = tok(prefix, return_tensors="pt", add_special_tokens=False).input_ids
        pay = tok("\n\n" + lib.PAYLOAD, add_special_tokens=False).input_ids
        x["original_tail"] = orig["reply"][-300:]
        x["replies"] = {}
        for s in SCALES:
            patch.s = s
            o = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=ENDING_TOKENS,
                               do_sample=False)[0, ids.shape[1]:].tolist()
            n = next((j for j, t in enumerate(o) if t in eos), len(o))
            full = torch.cat([ids, torch.tensor([pay])], 1)
            lp = torch.log_softmax(model(input_ids=full).logits[0, ids.shape[1] - 1:-1].float(), -1)
            lpp = float(lp.gather(1, torch.tensor(pay)[:, None]).sum())
            x["replies"][str(s)] = dict(reply=tok.decode(o[:n], skip_special_tokens=True), ends=n < len(o),
                                        capped=False, payload_logprob=lpp)
            print(f"ending dial {x['set']} s={s}: {x['replies'][str(s)]['reply']!r} lp={lpp:.2f}", flush=True)


def compute(a, cfg, out_json):
    from make_edit import encode, load_patch
    from p1lib import resolve
    torch.set_grad_enabled(False)
    delta, meta = load_patch(a.patch)
    c = {(r["layer"], r["neuron"]): r["c"] for r in meta["neurons"]}
    S = prompt_sets(a.demo)
    saved = lib.load_json(a.gens or cfg["gens"])["sets"] if (a.gens or cfg["gens"]) else {}
    repo, rev = resolve("E2B")[:2]
    part = out_json.with_suffix(".part.json")   # checkpoint after the GPU stage
    if part.exists():
        chats, dial = (json.loads(part.read_text())[k] for k in ("chats", "dial"))
    else:
        chats, dial = gpu_stage(a, cfg, S, saved, delta, c, repo, rev)
        part.parent.mkdir(parents=True, exist_ok=True)
        part.write_text(json.dumps(dict(chats=chats, dial=dial), ensure_ascii=False))

    # CPU fp32: per-token values and F at each scale
    torch.set_num_threads(a.threads)
    model, tok = load_cpu_f32(repo, rev)
    patch = ScaledPatch(model, delta, c)
    for x in chats:
        x["tokens"] = token_view(model, tok, patch, x["q"], x["reply"])
    items = lib.load_json(FORGET)
    doms = sorted({it["domain"] for it in items})
    first = [next(it for it in items if it["domain"] == d) for d in doms]
    for it in first:
        x = dict(set="ftext", index=items.index(it), label=it["domain"], q=it.get("q", ""), reply=it.get("r", it.get("raw")))
        if "raw" in it:   # raw text: no chat template
            enc = tok(it["raw"], return_offsets_mapping=True)
            patch.rec = True
            model(input_ids=torch.tensor([enc.input_ids]))
            patch.rec = False
            w = sum(patch.w.values()).tolist()
            g = torch.cat([patch.g[l] for l in sorted(patch.g)], -1).mean(-1).tolist()
            x["tokens"] = [dict(t=it["raw"][s0:s1], part="reply", w=round(wi, 5), g=round(gi, 4))
                           for (s0, s1), wi, gi in zip(enc.offset_mapping, w, g)]
        else:
            x["tokens"] = token_view(model, tok, patch, it["q"], it["r"])
        chats.append(x)
    print("token view done", flush=True)
    if cfg.get("dial_mode") == "ending":
        ending_dial(model, tok, patch, dial, lib.load_json(cfg["original"])["sets"])
    fitems = [it for d in doms for it in [z for z in items if z["domain"] == d][:F_PER_DOMAIN]]
    F = {str(s): {d: [] for d in doms} for s in SCALES if s != 0}
    for it in fitems:
        ids, lo = encode(tok, it, "cpu")
        patch.s = 0.0
        lp0 = torch.log_softmax(model(input_ids=ids, logits_to_keep=ids.shape[1] - lo).logits[0, :-1].float(), -1)
        for s in SCALES[1:]:
            patch.s = s
            lp = torch.log_softmax(model(input_ids=ids, logits_to_keep=ids.shape[1] - lo).logits[0, :-1].float(), -1)
            F[str(s)][it["domain"]].append(float((lp0.exp() * (lp0 - lp)).sum(-1).mean()))
        print("F item", it["domain"], flush=True)
    Fs = {"0.0": dict(macro=0.0, by_domain={d: 0.0 for d in doms})}
    for s, by in F.items():
        by = {d: sum(v) / len(v) for d, v in by.items()}
        Fs[s] = dict(macro=sum(by.values()) / len(by), by_domain=by)
    print("F:", {s: round(v["macro"], 5) for s, v in Fs.items()}, flush=True)
    res = dict(demo=a.demo, patch=Path(a.patch).name, meta=dict(name=meta.get("name"), lr=meta.get("lr"),
               neurons=[[r["layer"], r["neuron"]] for r in meta["neurons"]]),
               rule=f"{N_PER_SET} prompts per set at index floor(i*n/{N_PER_SET}); first item of each F domain",
               chats=chats, dial=dict(scales=SCALES, max_new_tokens=ENDING_TOKENS if cfg.get("dial_mode") == "ending" else DIAL_TOKENS,
                                      mode=cfg.get("dial_mode", "replies"), prompts=dial, F=Fs,
                                      F_items=len(fitems), F_precision="fp32"))
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    part.unlink()
    print("saved", out_json)


# ------------------------------------------------------------------------------------------------- render side

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--demo", choices=list(DEMOS), required=True)
    ap.add_argument("--patch", required=True, help="column patch (.safetensors) from make_edit.py")
    ap.add_argument("--gens", help="saved replies of this patch (demos/gen.py output); default: the demo's file")
    ap.add_argument("--recompute", action="store_true", help="run the model again even if the JSON exists")
    ap.add_argument("--cpu", action="store_true", help="also generate on the CPU (fp32, slow)")
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    cfg = DEMOS[a.demo]
    name = Path(a.patch).stem
    out_json = HERE / "out" / f"{name}.json"
    if a.recompute or not out_json.exists():
        compute(a, cfg, out_json)
    import draw
    res = json.loads(out_json.read_text())
    vmax, thr, per = draw.stats(res, cfg)
    d = res["dial"]
    (HERE / "out" / f"{name}_stats.json").write_text(json.dumps(dict(
        vmax=vmax, lit_threshold=thr, per_set=per, F_by_scale={s: v["macro"] for s, v in d["F"].items()},
        F_items=d["F_items"], same_reply_as_s0={p["set"]: [str(s) for s in d["scales"] if p["replies"][str(s)]["reply"]
                                                          == p["replies"]["0.0"]["reply"]] for p in d["prompts"]}),
        indent=1))
    draw.html_page(res, cfg, HERE / "out" / f"{name}.html")
    draw.png_gate(res, cfg, HERE / f"{name}.png")
    draw.png_dial(res, cfg, HERE / f"{name}_dial.png")


if __name__ == "__main__":
    main()
