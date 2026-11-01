"""Make a local edit of Gemma-4-E2B-it: train the output weights of K selective MLP neurons, then measure it.
Other models: --model (for example --model qwen3.5-4b; see OTHER_MODELS; the default is Gemma-4-E2B-it).

Subcommands (run `uv run make_edit.py <subcommand> --help` for all options):

  make    choose K neurons, train their down_proj columns, measure the edit, save it as a small patch
  apply   load a patch and reply to prompts (with --compare: the original model too)
  chat    load a patch and chat with the model
  cards   write "neuron cards" for a patch: the top-activating snippets of each edited neuron on a mixed corpus

Three ways to give the target behaviour to `make`:

  --system "<behaviour prompt>" --prompts train.txt --eval-prompts heldout.txt
      context distillation: the target is the model's own reply WITH this system prompt; the edit must give it
      WITHOUT the prompt. Loss = token KL(teacher || edited model) on the teacher replies.
  --append "\\n\\nP.S. ..." --prompts train.txt --eval-prompts heldout.txt
      the target is the model's own (complete) reply plus this text at the end. Loss = KL(original || edited) on
      the reply tokens + cross-entropy on the added tokens (two means, summed), so the reply itself stays as it was.
  --pairs train.jsonl --eval-pairs heldout.jsonl
      explicit (prompt, reply) pairs, one JSON object per line: {"prompt": ..., "reply": ...}.
      Loss = cross-entropy on the reply tokens (and the end-of-turn token).

Optional preservation loss: --preserve items.json (a list of {q, r} chat turns, r = the original model's reply, or
{raw} texts) adds --w-pres x KL(original || edited) on these items to the loss. A column edit writes only where its
neurons fire, so a small set of the off-target texts where the chosen neurons fire is enough (an example:
demos/jacobian_hotfix/PRESERVATION.md). Default: no preservation loss.

Prompt files: .txt (one prompt per line), .json (a list of strings) or .jsonl ({"prompt": ...} per line).

Examples:

  uv run make_edit.py make --system "Answer like a pirate." --prompts my_train.txt --eval-prompts my_test.txt \\
      --k 8 --lr 1e-2 3e-2 --compare-lora small --out results/pirate
  uv run make_edit.py apply --patch results/pirate/columns_lr0.03.safetensors --compare "How do I boil an egg?"
  uv run make_edit.py chat --patch results/pirate/columns_lr0.03.safetensors

How the neurons are chosen (details and the evidence: docs/make_edit.md):
  1. c = mean |activation| of each MLP neuron (the input of down_proj) on the target replies, g = the same on the
     background text. Both at the positions that the loss scores (the last prompt token and the reply tokens).
  2. Eligible: c in the top 10% of its layer, and g / c <= --max-ratio (default 0.2): active on the target, quiet
     elsewhere.
  3. Rank the eligible neurons by ||dL/dW_j|| / g_j: the gradient of the loss with respect to the neuron's output
     column (how much one unit of update helps), divided by its background activity (how much one unit of update
     writes into other text). Keep the top K.
"""
import argparse
import functools
import hashlib
import json
import math
import random
import sys
import textwrap
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "experiments/lib"))
import p1lib  # noqa: E402  (puts the demo helpers on sys.path)
from p1lib import Adapter, decoder_layers, load_model, resolve  # noqa: E402
from common import chat_prompt, generate  # noqa: E402
import train_gains as TG  # noqa: E402
from forget import encode  # noqa: E402

BACKGROUND = ROOT / "experiments/01_locality_curves/data/ref_set.json"   # selection background (not the F items)
FORGET = ROOT / "experiments/lib/data/forget_set.json"                  # change on unrelated text (F), 7 domains
CACHE = ROOT / "results/cache"
LORA_ALL = p1lib.LORA_ALL

# Models other than Gemma-4 (pinned revisions). Any other Hugging Face repo id also works if its decoder layers have
# `mlp.down_proj` (the Gemma-4 aliases E2B / E4B are in demos/talk_like_a_person/common.py).
OTHER_MODELS = {"qwen3.5-4b": ("Qwen/Qwen3.5-4B", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a")}


def is_gemma(model_or_repo):
    name = model_or_repo if isinstance(model_or_repo, str) else model_or_repo.name_or_path
    return "gemma" in name.lower()


def resolve_model(name):
    """'E2B' / 'E4B' / a key of OTHER_MODELS / a repo id -> (repo, revision)."""
    if name.lower() in OTHER_MODELS:
        return OTHER_MODELS[name.lower()]
    return resolve(name)[:2]


def load(name="E2B", lowmem=False, dtype=torch.bfloat16):
    """Gemma-4: demos/talk_like_a_person/common.load_model (unchanged). Other models: plain transformers, eager attention, on the GPU (no
    low-memory mode); thinking off in the chat template (enable_thinking=False, as in the neuron-finder repo); the
    end-of-turn token of the tokenizer is added to the stop tokens of generate()."""
    repo, rev = resolve_model(name)
    if is_gemma(repo):
        return load_model(repo, rev, lowmem=lowmem, dtype=dtype)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(repo, revision=rev)
    if "enable_thinking" in (tok.chat_template or ""):
        tok.apply_chat_template = functools.partial(tok.apply_chat_template, enable_thinking=False)
    model = AutoModelForCausalLM.from_pretrained(repo, revision=rev, dtype=dtype, attn_implementation="eager")
    model = model.to("cuda").eval().requires_grad_(False)
    gc = model.generation_config
    eos = gc.eos_token_id if isinstance(gc.eos_token_id, list) else [gc.eos_token_id]
    gc.eos_token_id = [e for e in eos if e is not None] + [tok.eos_token_id] * (tok.eos_token_id not in eos)
    if gc.pad_token_id is None:
        gc.pad_token_id = tok.pad_token_id
    return model, tok


def embed_tokens(model):
    return model.model.language_model.embed_tokens if is_gemma(model) else model.get_input_embeddings()


# ---------------------------------------------------------------------------------------------------------- data

def read_prompts(path):
    p = Path(path)
    if p.suffix == ".txt":
        return [x.strip() for x in p.read_text().splitlines() if x.strip()]
    if p.suffix == ".json":
        return [x if isinstance(x, str) else x["prompt"] for x in json.loads(p.read_text())]
    return [json.loads(x)["prompt"] for x in p.read_text().splitlines() if x.strip()]


def read_pairs(path):
    return [(d["prompt"], d["reply"]) for d in map(json.loads, Path(path).read_text().splitlines()) if d]


def sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]


def teacher_items(model, tok, prompts, system, max_new_tokens, batch_size):
    """Greedy teacher replies WITH the system prompt (fixed left-padded batches in file order, for speed)."""
    out = []
    for i in range(0, len(prompts), batch_size):
        chunk = prompts[i:i + batch_size]
        replies = generate(model, tok, chunk, max_new_tokens + 1, batch_size=batch_size, system=system)
        for q, r in zip(chunk, replies):
            n = len(tok(r, add_special_tokens=False).input_ids)
            out.append(dict(q=q, r=r, sys=system, ended=n <= max_new_tokens, kind="id"))
        print(f"  teacher replies {len(out)}/{len(prompts)}", flush=True)
    for d in out:
        if not d["ended"]:   # cut at max_new_tokens; the cut reply is trained without an end-of-turn token
            d["r"] = tok.decode(tok(d["r"], add_special_tokens=False).input_ids[:max_new_tokens])
    return out


# ---------------------------------------------------------------------------------------------------- the loss

def item_loss(model, tok, d, ad, mode):
    """Per-token loss of one item: KL(teacher || model) in 'system' mode, cross-entropy in 'pairs' mode; in 'append'
    mode KL(original || model) on the original reply plus cross-entropy on the appended text (two means, summed)."""
    ids, mask, [(lo, hi)] = TG.build(tok, [d], False, model.device)
    if mode == "append":
        start = lo + 1                                    # first reply token (single item, no padding)
        n_keep = len(tok(chat_prompt(tok, d["q"]) + d["r"][:d["keep"]], add_special_tokens=False).input_ids)
        n_keep = min(n_keep, hi)
        j0 = n_keep - start                               # span index of the first appended token
        with torch.no_grad():
            prev = ad.on if ad is not None else None
            if ad is not None:
                ad.on = False
            t = TG.reply_logp(model, ids, mask, lo, hi)[0]
            if ad is not None:
                ad.on = prev
        s_ = TG.reply_logp(model, ids, mask, lo, hi)[0]
        kl = TG.kl(t[:j0], s_[:j0]).mean() if j0 > 0 else s_.sum() * 0
        ce = -s_[j0:].gather(-1, ids[0, lo + 1 + j0:hi + 1, None]).mean()
        return kl + ce
    if mode == "pairs":
        logits = model(input_ids=ids, attention_mask=mask, logits_to_keep=ids.shape[1] - lo).logits[0, :hi - lo]
        return torch.nn.functional.cross_entropy(logits.float(), ids[0, lo + 1:hi + 1])
    with torch.no_grad():
        prev = ad.on if ad is not None else None
        if ad is not None:
            ad.on = False
        tids, tmask, [(tlo, thi)] = TG.build(tok, [d], True, model.device)
        t = TG.reply_logp(model, tids, tmask, tlo, thi)[0]
        if ad is not None:
            ad.on = prev
    return TG.kl(t, TG.reply_logp(model, ids, mask, lo, hi)[0]).mean()


def pres_loss(model, tok, it, ad):
    """Preservation loss of one item ({q, r}: a chat turn, r = the original model's reply; or {raw}: plain text):
    KL(original || edited) per token at the scored positions (as forget.py scores them), the edit off for the original."""
    ids, lo = encode(tok, it, model.device)
    end = -1 if ids.shape[1] - lo > 1 else None   # an empty reply: the last prompt token only
    with torch.no_grad():
        prev = ad.on
        ad.on = False
        lp0 = torch.log_softmax(model(input_ids=ids, logits_to_keep=ids.shape[1] - lo).logits[0, :end].float(), -1)
        ad.on = prev
    lp = torch.log_softmax(model(input_ids=ids, logits_to_keep=ids.shape[1] - lo).logits[0, :end].float(), -1)
    return TG.kl(lp0, lp).mean()


@torch.no_grad()
def target_strength(model, tok, items, ad, mode):
    """Mean per-item loss on the held-out items, for the original model and for the edit."""
    res = {}
    for name, on in (("original", False), ("edited", True)):
        if ad:
            ad.on = on
        res[name] = sum(float(item_loss(model, tok, d, ad, mode)) for d in items) / len(items)
    if ad:
        ad.on = True
    return res


# ------------------------------------------------------------------------------------------------- selection

@torch.no_grad()
def mean_abs_acts(model, tok, layers, seqs):
    """Mean |down_proj input| per neuron over the given (ids, lo, hi) spans (positions lo .. hi-1)."""
    acc, span = {}, [0, 0]

    def cap(l):
        def f(mod, inp):
            a = inp[0][0, span[0]:span[1]].float().abs()
            s = acc.setdefault(l, [0, 0])
            s[0] = s[0] + a.sum(0)
            s[1] += a.shape[0]
        return f
    hs = [layers[l].mlp.down_proj.register_forward_pre_hook(cap(l)) for l in range(len(layers))]
    for ids, lo, hi in seqs:
        span[:] = [lo, hi]
        model(input_ids=ids)
    for h in hs:
        h.remove()
    return [(acc[l][0] / acc[l][1]).cpu() for l in range(len(layers))]


def prompt_span(tok, q):
    """(start, end): the token positions of the user's message q inside chat_prompt(tok, q)."""
    full = chat_prompt(tok, q)
    i = full.rindex(q)
    return (len(tok(full[:i], add_special_tokens=False).input_ids),
            len(tok(full[:i + len(q)], add_special_tokens=False).input_ids))


def reply_start_span(tok, q):
    """(start, end): the chat-template tokens between the end of the user's message q and the first reply token
    (Gemma: <turn|>, \\n, <|turn>, model, \\n). They depend only on the prompt."""
    return prompt_span(tok, q)[1], len(tok(chat_prompt(tok, q), add_special_tokens=False).input_ids)


def domain_acts(model, tok, layers, items, on="reply", key="domain"):
    """{domain: per-layer mean |a|} at the scored positions (as forget.py scores them; on="prompt": at the
    user-message tokens, all tokens of a raw text; on="reply-start": at reply_start_span, raw texts are skipped)."""
    tag = "" if is_gemma(model) else "_" + model.name_or_path.replace("/", "_")   # Gemma: the old file names
    suf = {"reply": "", "prompt": "_prompt", "reply-start": "_rstart"}[on]
    f = CACHE / f"bgdom_{sha(items)}{suf}{tag}.pt"
    if f.exists():
        return torch.load(f)
    per = {}
    for dom in sorted({it.get(key, "all") for it in items}):
        seqs = []
        for it in items:
            if it.get(key, "all") != dom or (on == "reply-start" and "raw" in it):
                continue
            ids, lo = encode(tok, dict(it, r=it.get("r", "")) if "raw" not in it else it, model.device)
            if on == "prompt":
                seqs.append((ids, *((0, ids.shape[1]) if "raw" in it else prompt_span(tok, it["q"]))))
            elif on == "reply-start":
                seqs.append((ids, *reply_start_span(tok, it["q"])))
            else:
                seqs.append((ids, lo, ids.shape[1] - 1))
        if seqs:
            per[dom] = mean_abs_acts(model, tok, layers, seqs)
    CACHE.mkdir(parents=True, exist_ok=True)
    torch.save(per, f)
    return per


def background_acts(model, tok, layers, path, stat="mean", on="reply", contrast=None):
    """g: the mean (or the max, stat="max") over the domains of the background set of the per-domain mean |a| (see
    domain_acts); with contrast (a list of {q, kind}): each kind is one more domain."""
    per = dict(domain_acts(model, tok, layers, json.loads(Path(path).read_text()), on))
    if contrast:
        per.update({f"contrast:{k}": v for k, v in domain_acts(model, tok, layers, contrast, on, key="kind").items()})
    red = (lambda x: x.mean(0)) if stat == "mean" else (lambda x: x.max(0).values)
    return [red(torch.stack([per[d][l] for d in per])) for l in range(len(layers))]


def column_grads(model, tok, layers, items, mode, eligible):
    """||dL/dW_j|| for the down_proj column of each eligible neuron, the gradient summed over the items.

    Computed through a zero-initialised column adapter on all eligible neurons (with gradient checkpointing, so long
    replies fit): dL/du_j = rms(W_j) * dL/dW_j."""
    ad = Adapter(model, "columns", eligible, model.device)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    for d in items:
        item_loss(model, tok, d, ad, mode).backward()
    model.eval()
    model.gradient_checkpointing_disable()
    out = {}
    for l, idx in eligible.items():
        rms = layers[l].mlp.down_proj.weight[:, idx].float().pow(2).mean(0).sqrt()
        out[l] = (ad.params[f"{l}.u"].grad / rms).norm(dim=0).cpu()
    ad.remove()
    return out


def select_neurons(model, tok, layers, train, mode, bg_path, k, max_ratio, n_items=64, bg_stat="mean", on="reply",
                   contrast=None, min_layer=0):
    """on="reply": c and g at the scored positions (the last prompt token and the reply tokens); on="prompt": at the
    tokens of the user's message (the trigger), for the target prompts and for the background chat prompts;
    on="reply-start": at the template tokens between the user's message and the reply. contrast: near-miss prompts
    ({q, kind}), each kind one more background domain. Only layers >= min_layer are eligible."""
    sub = random.Random(0).sample(train, min(n_items, len(train)))   # fixed: the neurons do not depend on --seed
    seqs = []
    for d in sub:
        ids, _, [(lo, hi)] = TG.build(tok, [d], False, model.device)
        seqs.append((ids, *(prompt_span(tok, d["q"]) if on == "prompt" else
                            reply_start_span(tok, d["q"]) if on == "reply-start" else (lo, hi))))
    c = mean_abs_acts(model, tok, layers, seqs)
    g = background_acts(model, tok, layers, bg_path, bg_stat, on, contrast)
    eligible = {}
    for l in range(min_layer, len(layers)):
        ok = (c[l] >= torch.quantile(c[l].double(), 0.9).float()) & (g[l] <= max_ratio * c[l])
        if ok.any():
            eligible[l] = torch.nonzero(ok).flatten().tolist()
    n_el = sum(map(len, eligible.values()))
    print(f"  eligible neurons (top 10% on target, g/c <= {max_ratio}"
          + (f", layer >= {min_layer}" if min_layer else "") + f"): {n_el}", flush=True)
    if n_el == 0 or (n_el < k and not contrast):
        sys.exit(f"only {n_el} neurons pass the selectivity filter; raise --max-ratio or lower --k")
    with torch.enable_grad():
        gn = column_grads(model, tok, layers, sub, mode, eligible)
    rows = []
    for l, idx in eligible.items():
        for j, i in enumerate(idx):
            rows.append(dict(layer=l, neuron=i, c=float(c[l][i]), g=float(g[l][i]), ratio=float(g[l][i] / c[l][i]),
                             grad=float(gn[l][j]), score=float(gn[l][j] / g[l][i].clamp_min(1e-6))))
    rows.sort(key=lambda r: -r["score"])
    return rows[:k], n_el


# ---------------------------------------------------------------------------------------------- the patch file

def column_delta(ad):
    """{layer: (neurons, delta)}: the trained change of the down_proj columns (new column = old column + delta)."""
    out = {}
    for l, neurons in ad.spec.items():
        dp = ad.model_layers[int(l)].mlp.down_proj
        W = dp.weight[:, neurons].float()
        rms = W.pow(2).mean(0).sqrt()
        out[int(l)] = (list(neurons), (W * ad.params[f"{l}.g"] + ad.params[f"{l}.u"] * rms).detach().cpu())
    return out


def save_patch(path, delta, meta):
    from safetensors.torch import save_file
    t = {}
    for l, (neurons, D) in delta.items():
        t[f"layer{l}.neurons"] = torch.tensor(neurons, dtype=torch.int64)
        t[f"layer{l}.delta"] = D.contiguous()
    save_file(t, str(path), metadata={"format": "neuron-columns-v1", "meta": json.dumps(meta)})


def load_patch(path):
    from safetensors import safe_open
    with safe_open(str(path), "pt") as f:
        meta = json.loads(f.metadata().get("meta", "{}"))
        layers = sorted({int(k.split(".")[0][5:]) for k in f.keys()})
        return {l: (f.get_tensor(f"layer{l}.neurons").tolist(), f.get_tensor(f"layer{l}.delta")) for l in layers}, meta


class Patch:
    """A column patch as runtime hooks: down_proj(x) + x[neurons] @ delta.T (exact; the frozen weights are not changed)."""

    def __init__(self, model, delta):
        self.on, self.handles = True, []
        layers = decoder_layers(model)
        for l, (neurons, D) in delta.items():
            dp = layers[l].mlp.down_proj
            idx, D = torch.tensor(neurons, device=dp.weight.device), D.to(dp.weight.device)

            def f(mod, inp, out, idx=idx, D=D):
                return out + inp[0].index_select(-1, idx) @ D.t().to(out.dtype) if self.on else out
            self.handles.append(dp.register_forward_hook(f))

    def remove(self):
        for h in self.handles:
            h.remove()


@torch.no_grad()
def bake(model, delta):
    """Write the patch into the weights (fp32 sum, rounded once to the weight dtype)."""
    layers = decoder_layers(model)
    for l, (neurons, D) in delta.items():
        W = layers[l].mlp.down_proj.weight
        W[:, neurons] = (W[:, neurons].float() + D.to(W.device)).to(W.dtype)


# --------------------------------------------------------------------------------------- change on unrelated text

@torch.no_grad()
def unrelated_change(model, tok, edits, path=FORGET):
    """F per domain: KL(original || edited) per token on the fixed unrelated items (forget.py), for each edit."""
    items = json.loads(Path(path).read_text())
    for e in edits.values():
        e.on = False
    kl = {n: [] for n in edits}
    for it in items:
        ids, lo = encode(tok, it, model.device)
        lp0 = torch.log_softmax(model(input_ids=ids, logits_to_keep=ids.shape[1] - lo).logits[0, :-1].float(), -1)
        for n, e in edits.items():
            e.on = True
            lp = torch.log_softmax(model(input_ids=ids, logits_to_keep=ids.shape[1] - lo).logits[0, :-1].float(), -1)
            e.on = False
            kl[n].append(float((lp0.exp() * (lp0 - lp)).sum(-1).mean()))
    out = {}
    for n in edits:
        by = {}
        for x, it in zip(kl[n], items):
            by.setdefault(it["domain"], []).append(x)
        by = {d: sum(v) / len(v) for d, v in sorted(by.items())}
        out[n] = dict(macro=sum(by.values()) / len(by), by_domain=by, per_item=kl[n])
    return out


# --------------------------------------------------------------------------------------------------- neuron cards

def card_corpus():
    """A small mixed corpus from data files in this repository: 7 kinds of reference text, casual chat, SlimOrca tasks,
    and the user prompts of the five rule tasks of experiment 02 (weather, recipe, lower case, cat, Paris)."""
    out = [(it["domain"], it.get("raw"), it.get("q"), it.get("r")) for it in json.loads(BACKGROUND.read_text())]
    for d in json.loads(p1lib.DATA.read_text()):
        out.append(("casual chat" if d["kind"] == "id" else "tasks (SlimOrca)", None, d["q"], d["r"]))
    for rule in ("weather", "recipe", "lower", "cat", "paris"):
        d = json.loads((ROOT / f"experiments/02_conditional_rules/data/pairs_{rule}.json").read_text())
        for p in d["train"] + d["eval"]:
            out.append((f"{rule} prompts (02)", None, p["trigger"], ""))
            out.append((f"{rule} control prompts (02)", None, p["control"], ""))
    return out


@torch.no_grad()
def neuron_cards(model, tok, neurons, top=5, window=12, max_len=384):
    """For each (layer, neuron): the top-activating tokens (by |a|) with their context, and mean |a| per corpus source."""
    layers = decoder_layers(model)
    by_layer = {}
    for l, i in neurons:
        by_layer.setdefault(l, []).append(i)
    cur = {}
    hs = [layers[l].mlp.down_proj.register_forward_pre_hook(
        lambda mod, inp, l=l: cur.__setitem__(l, inp[0][0][:, by_layer[l]].float().cpu())) for l in by_layer]
    best = {n: [] for n in neurons}
    src_sum = {n: {} for n in neurons}
    corpus = card_corpus()
    for src, raw, q, r in corpus:   # chat items in the chat format (as the edit sees them), raw text as it is
        text = raw if raw is not None else chat_prompt(tok, q) + r
        ids = tok(text, return_tensors="pt", add_special_tokens=raw is not None).input_ids[:, :max_len].to(model.device)
        model(input_ids=ids)
        toks = ids[0].tolist()
        for l, idx in by_layer.items():
            A = cur[l]
            for j, i in enumerate(idx):
                a = A[:, j]
                s = src_sum[(l, i)].setdefault(src, [0.0, 0])
                s[0] += float(a.abs().sum())
                s[1] += len(a)
                t = int(a.abs().argmax())
                best[(l, i)].append((float(a[t]), src, toks, t))
                best[(l, i)] = sorted(best[(l, i)], key=lambda x: -abs(x[0]))[:top]
    for h in hs:
        h.remove()
    cards = []
    for n in neurons:
        snips = []
        for a, src, toks, pos in best[n]:
            left = tok.decode(toks[max(0, pos - window):pos])
            snips.append(dict(act=round(a, 3), source=src, text=left + "[[" + tok.decode(toks[pos:pos + 1]) + "]]"
                              + tok.decode(toks[pos + 1:pos + 4])))
        mean_by_src = {s: v[0] / v[1] for s, v in src_sum[n].items()}
        cards.append(dict(layer=n[0], neuron=n[1], top=snips,
                          mean_abs_by_source=dict(sorted(mean_by_src.items(), key=lambda x: -x[1]))))
    return dict(corpus=dict(n_items=len(corpus), sources=sorted({c[0] for c in corpus})), cards=cards)


def cards_markdown(cards):
    lines = [f"# Neuron cards\n\nCorpus: {cards['corpus']['n_items']} items from data files in this repository "
             f"({', '.join(cards['corpus']['sources'])}). For each edited neuron: the 5 tokens with the largest |activation| "
             "(the token is in [[ ]]), and the 3 corpus sources where the mean |activation| is highest.\n"]
    for c in cards["cards"]:
        srcs = ", ".join(f"{s} {v:.3f}" for s, v in list(c["mean_abs_by_source"].items())[:3])
        lines.append(f"## neuron {c['layer']}:{c['neuron']}\n\nhighest mean |a|: {srcs}\n")
        for s in c["top"]:
            lines.append(f"- `{s['act']:+.2f}` ({s['source']}) " + " ".join(s["text"].split()).replace("|", "\\|"))
        lines.append("")
    return "\n".join(lines)


# ------------------------------------------------------------------------------------------------------- training

def train(model, tok, ad, items, mode, lr, steps, batch, seed, pres=None, w_pres=1.0, pres_batch=8):
    random.seed(seed)
    torch.manual_seed(seed)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    opt = torch.optim.Adam(list(ad.params.values()), lr=lr)
    curve, t0 = [], time.time()
    for step in range(steps):
        opt.zero_grad()
        tot = 0.0
        for d in random.sample(items, min(batch, len(items))):
            loss = item_loss(model, tok, d, ad, mode) / batch
            loss.backward()
            tot += float(loss)
        ptot = 0.0
        for it in random.sample(pres, min(pres_batch, len(pres))) if pres else []:   # preservation loss (--preserve)
            loss = w_pres * pres_loss(model, tok, it, ad) / pres_batch
            loss.backward()
            ptot += float(loss)
        tot += ptot
        if not math.isfinite(tot):
            print(f"  step {step}: non-finite loss, stopping", flush=True)
            break
        opt.step()
        curve.append(tot)
        if step % 50 == 0 or step == steps - 1:
            print(f"  step {step} loss {tot:.4f}" + (f" (preservation {ptot:.4f})" if pres else "")
                  + f"  {time.time() - t0:.0f}s", flush=True)
    model.eval()
    model.gradient_checkpointing_disable()
    for p in ad.params.values():
        p.requires_grad_(False)
    return curve


def make_lora(model, spec, scale):
    """p1lib LoRA (y + s B A x) with the scale s."""
    orig = Adapter._lora_hook
    Adapter._lora_hook = lambda self, A, B, s: orig(self, A, B, s * scale)
    try:
        ad = Adapter(model, "lora", spec, model.device)
    finally:
        Adapter._lora_hook = orig
    ad.lora_scale = scale
    return ad


def lora_spec(kind, layers, k, gemma=True, rows=None):
    """Gemma-4 (the study's shapes): small = rank 1 on down_proj of round(K x 1537 / 13824) layers near layer 15;
    standard = rank 8 on LORA_ALL of every layer. Other models: small = rank 1 on down_proj of ONE layer, the layer with
    the most chosen neurons (ties: the lower layer; the study's layer 15 was the top layer of the gradient choice);
    standard = rank 8 on every nn.Linear of every decoder layer (all linear layers, also linear-attention blocks)."""
    if not gemma:
        if kind == "small":
            n = {}
            for r in rows:
                n[r["layer"]] = n.get(r["layer"], 0) + 1
            return [[min(n, key=lambda l: (-n[l], l)), "mlp.down_proj", 1]]
        return [[l, nm, 8] for l in range(len(layers)) for nm, m in layers[l].named_modules()
                if isinstance(m, torch.nn.Linear)]
    if kind == "small":   # rank-1 LoRA on down_proj, the same size as K neuron columns (K = 8: layer 15, as in the study)
        n = max(1, round(k * 1537 / 13824))
        order = sorted(p1lib.C_LAYERS, key=lambda l: (abs(l - 15), l))
        return [[l, "mlp.down_proj", 1] for l in sorted(order[:n])]
    return [[l, nm, 8] for l in range(len(layers)) for nm in LORA_ALL if p1lib.get_module(layers[l], nm) is not None]


def make(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    mode = "system" if a.system else "append" if a.append else "pairs"
    repo, rev = resolve_model(a.model)
    model, tok = load(a.model, lowmem=a.lowmem)
    tok.eot_text = TG.end_of_turn(tok)
    layers = decoder_layers(model)
    embed_tokens(model).register_forward_hook(
        lambda mod, i, o: o.requires_grad_(True) if torch.is_grad_enabled() else o)

    # 1. data
    if mode in ("system", "append"):
        f = out / "teacher_replies.json"
        if f.exists():
            train_items, eval_items = json.loads(f.read_text())
        else:
            print("teacher replies (with the system prompt):" if mode == "system" else "original replies:", flush=True)
            train_items, eval_items = (teacher_items(model, tok, read_prompts(p), a.system, a.max_new_tokens, a.gen_batch)
                                       for p in (a.prompts, a.eval_prompts))
            if mode == "append":   # target = the complete original reply + the appended text
                train_items, eval_items = ([dict(d, r=d["r"] + a.append, keep=len(d["r"])) for d in items if d["ended"]]
                                           for items in (train_items, eval_items))
            f.write_text(json.dumps([train_items, eval_items], indent=1, ensure_ascii=False))
    else:
        mk = lambda ps: [dict(q=q, r=r, sys=None, ended=True, kind="id") for q, r in ps]  # noqa: E731
        train_items, eval_items = mk(read_pairs(a.pairs)), mk(read_pairs(a.eval_pairs))
    pres = json.loads(Path(a.preserve).read_text()) if a.preserve else None
    hashes = dict(train=sha([(d["q"], d["r"]) for d in train_items]), eval=sha([(d["q"], d["r"]) for d in eval_items]),
                  background=sha(json.loads(Path(a.background).read_text())), system=sha(a.system), append=sha(a.append),
                  preserve=sha(pres) if pres else None)
    print(f"{len(train_items)} training items, {len(eval_items)} held-out items, mode {mode}", flush=True)

    # 2. neurons
    if a.neurons:   # reuse the neurons of an earlier run (its report.json)
        prev = json.loads(Path(a.neurons).read_text())
        rows, n_el = prev["neurons"], prev["n_eligible"]
        print(f"neurons from {a.neurons}", flush=True)
    else:
        print("selecting neurons:", flush=True)
        rows, n_el = select_neurons(model, tok, layers, train_items, mode, a.background, a.k, a.max_ratio,
                                    bg_stat=a.bg_stat, on=a.select_on, min_layer=a.min_layer,
                                    contrast=json.loads(Path(a.contrast).read_text()) if a.contrast else None)
    for r in rows:
        print(f"  {r['layer']:2d}:{r['neuron']:<5d} g/c {r['ratio']:.3f}  grad {r['grad']:.3g}  score {r['score']:.3g}")
    spec = {}
    for r in rows:
        spec.setdefault(r["layer"], []).append(r["neuron"])

    # 3. train: the neuron columns at each --lr, and the LoRA baselines
    runs = [("columns", lr) for lr in a.lr]
    for kind in (["small", "standard"] if a.compare_lora == "both" else [a.compare_lora] if a.compare_lora != "none" else []):
        runs += [(f"lora_{kind}", lr) for lr in (a.lora_lr_small if kind == "small" else a.lora_lr_standard)]
    report = dict(mode=mode, model=dict(repo=repo, revision=rev), k=a.k, seed=a.seed, steps=a.steps, batch=a.batch,
                  max_ratio=a.max_ratio, bg_stat=a.bg_stat, select_on=a.select_on, contrast=a.contrast,
                  min_layer=a.min_layer,
                  preserve=dict(file=a.preserve, n=len(pres), w=a.w_pres, batch=a.pres_batch) if pres else None, n_eligible=n_el, neurons=rows, data_hashes=hashes,
                  n_train=len(train_items), n_eval=len(eval_items), runs={})
    edits = {}
    for kind, lr in runs:
        name = f"{kind}_lr{lr:g}"
        print(f"training {name}:", flush=True)
        if kind == "columns":
            ad = Adapter(model, "columns", spec, model.device)
        else:   # small: rank 1, alpha 1 (scale 1, as in the study); standard: rank 8, alpha 16 (scale 2)
            ad = make_lora(model, lora_spec(kind[5:], layers, a.k, is_gemma(repo), rows),
                           2.0 if kind == "lora_standard" else 1.0)
        ad.model_layers = layers
        t0 = time.time()
        curve = train(model, tok, ad, train_items, mode, lr, a.steps, a.batch, a.seed, pres, a.w_pres, a.pres_batch)
        ts = target_strength(model, tok, eval_items, ad, mode)
        report["runs"][name] = dict(kind=kind, lr=lr, n_params=ad.n_params(), final_loss=curve[-1] if curve else None,
                                    target_loss=ts, train_seconds=round(time.time() - t0))
        print(f"  held-out target loss: original {ts['original']:.4f} -> edited {ts['edited']:.4f}", flush=True)
        if kind == "columns":
            delta = column_delta(ad)
            ad.remove()
            p = Patch(model, delta)
            save_patch(out / f"{name}.safetensors", delta, dict(name=name, lr=lr, model=report["model"], neurons=rows,
                                                                data_hashes=hashes))
            edits[name] = p
        else:
            from safetensors.torch import save_file
            params = {k: (v * ad.lora_scale if k.endswith(".B") else v).contiguous()   # the scale is folded into B
                      for k, v in ad.state()["params"].items()}
            save_file(params, str(out / f"{name}.safetensors"),
                      metadata={"format": "lora-v1", "spec": json.dumps(ad.spec), "model": json.dumps(report["model"])})
            edits[name] = ad
        for e in edits.values():
            e.on = False
        if a.show:
            e = edits[name]
            e.on = True
            report["runs"][name]["examples"] = [
                dict(prompt=d["q"], reply=r) for d, r in
                zip(eval_items[:a.show], generate(model, tok, [d["q"] for d in eval_items[:a.show]], a.max_new_tokens))]
            e.on = False

    # 4. change on unrelated text, per domain
    print("change on unrelated text (F):", flush=True)
    if a.f32:   # fp32 model and edits: no bf16 rounding floor (about 0.001) in F
        for e in edits.values():
            e.remove()
        names = list(edits)
        edits = ad = e = layers = model = None   # free the bf16 model (the adapters and the layer list refer to it)
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        model, tok = load(a.model, lowmem=True, dtype=torch.float32)
        edits = {n: load_any(model, out / f"{n}.safetensors") for n in names}
    report["F_precision"] = "fp32" if a.f32 else "bf16"
    F = unrelated_change(model, tok, edits)
    for n, f in F.items():
        report["runs"][n]["F"] = dict(macro=f["macro"], by_domain=f["by_domain"])
        print(f"  {n:28s} F {f['macro']:.5f}  " + " ".join(f"{d} {v:.4f}" for d, v in f["by_domain"].items()), flush=True)
    (out / "report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))

    # 5. neuron cards
    cards = neuron_cards(model, tok, [(r["layer"], r["neuron"]) for r in rows])
    (out / "cards.json").write_text(json.dumps(cards, indent=1, ensure_ascii=False))
    (out / "cards.md").write_text(cards_markdown(cards))
    print_table(report)
    print(f"saved: {out}/report.json, cards.md, and one .safetensors patch per run")


def print_table(report):
    print(f"\n{'run':28s} {'params':>10s} {'target loss (orig -> edit)':>28s} {'F (unrelated)':>14s}")
    for n, r in report["runs"].items():
        t = r["target_loss"]
        print(f"{n:28s} {r['n_params']:>10,d} {t['original']:>13.4f} -> {t['edited']:<11.4f} {r['F']['macro']:>14.5f}")


# ------------------------------------------------------------------------------------------- apply / chat / cards

def load_any(model, path):
    """A column patch (format neuron-columns-v1) as hooks, or a LoRA patch (lora-v1) as p1lib hooks."""
    from safetensors import safe_open
    with safe_open(str(path), "pt") as f:
        md = f.metadata() or {}
        if md.get("format") == "lora-v1":
            spec = json.loads(md["spec"])
            ad = Adapter(model, "lora", spec, model.device)
            ad.load_params({k: f.get_tensor(k) for k in f.keys()})
            return ad
    delta, _ = load_patch(path)
    return Patch(model, delta)


def patch_model(a):
    """--model if given, else the model in the patch metadata, else Gemma-4-E2B-it."""
    if a.model:
        return a.model
    from safetensors import safe_open
    with safe_open(str(a.patch), "pt") as f:
        md = f.metadata() or {}
    m = json.loads(md["model"]) if "model" in md else json.loads(md.get("meta", "{}")).get("model")
    return m["repo"] if m and not is_gemma(m["repo"]) else "E2B"


def apply(a):
    torch.set_grad_enabled(False)
    model, tok = load(patch_model(a), lowmem=a.lowmem)
    if a.bake:
        delta, _ = load_patch(a.patch)
        bake(model, delta)
        if a.save_dir:
            model.save_pretrained(a.save_dir)
            tok.save_pretrained(a.save_dir)
            print("saved the patched checkpoint to", a.save_dir)
        e = None
    else:
        e = load_any(model, a.patch)
    prompts = a.prompts or (read_prompts(a.prompts_file) if a.prompts_file else [])
    for q in prompts:
        print(f"\n> {q}")
        if a.compare and e is not None:
            e.on = False
            print("  [original]\n" + textwrap.indent(generate(model, tok, [q], a.max_new_tokens)[0], "    "))
            e.on = True
        print("  [edited]\n" + textwrap.indent(generate(model, tok, [q], a.max_new_tokens)[0], "    "))


def chat(a):
    torch.set_grad_enabled(False)
    model, tok = load(patch_model(a), lowmem=a.lowmem)
    load_any(model, a.patch)
    print("Chatting with the edited model. Empty line to quit.")
    history = []
    while True:
        q = input("you> ").strip()
        if not q:
            break
        history.append({"role": "user", "content": q})
        r = generate(model, tok, [history], a.max_new_tokens)[0]
        history.append({"role": "assistant", "content": r})
        print("gemma>", r)


def cards(a):
    torch.set_grad_enabled(False)
    model, tok = load(patch_model(a), lowmem=True)
    delta, _ = load_patch(a.patch)
    c = neuron_cards(model, tok, [(l, i) for l, (ns, _) in delta.items() for i in ns])
    Path(a.out).write_text(json.dumps(c, indent=1, ensure_ascii=False))
    Path(a.out).with_suffix(".md").write_text(cards_markdown(c))
    print("saved", a.out, "and", Path(a.out).with_suffix(".md"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make", help="choose neurons, train, measure, save a patch",
                       formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    src = m.add_mutually_exclusive_group(required=True)
    src.add_argument("--system", help="behaviour prompt: the target is the model's own reply with this system prompt")
    src.add_argument("--append", help="text to add at the end of the model's own reply (e.g. a fixed last line); loss = "
                     "KL to the original model on its reply + cross-entropy on the added text")
    src.add_argument("--pairs", help="JSONL of {prompt, reply}: the training targets")
    m.add_argument("--prompts", help="(--system, --append) training prompts")
    m.add_argument("--eval-prompts", help="(--system, --append) held-out prompts, for the target strength")
    m.add_argument("--eval-pairs", help="(--pairs) held-out JSONL of {prompt, reply}, for the target strength")
    m.add_argument("--model", default="E2B", help="E2B, E4B, " + ", ".join(OTHER_MODELS) + " or a Hugging Face repo id")
    m.add_argument("--background", default=str(BACKGROUND.relative_to(ROOT)),
                   help="unrelated text for the selectivity (JSON list of {domain, q, r} or {domain, raw})")
    m.add_argument("--k", type=int, default=8, help="number of neurons (K x 1,536 trained numbers)")
    m.add_argument("--bg-stat", choices=["mean", "max"], default="mean",
                   help="g = the mean or the max over the background domains (max: quiet on EVERY domain)")
    m.add_argument("--select-on", choices=["reply", "prompt", "reply-start"], default="reply",
                   help="measure the activity for the selection on the reply tokens (the scored positions), on the "
                        "tokens of the user's message (the trigger; useful when the target reply is the same for "
                        "every prompt, e.g. a refusal), or on the template tokens between the message and the reply")
    m.add_argument("--contrast", help="near-miss prompts (JSON list of {q, kind}): each kind is one more background "
                                      "domain (with --bg-stat max: the neurons must be quiet on EVERY kind)")
    m.add_argument("--min-layer", type=int, default=0, help="only neurons in layers >= this are eligible")
    m.add_argument("--neurons", help="reuse the neurons of an earlier run: its report.json (skips the selection)")
    m.add_argument("--max-ratio", type=float, default=0.2,
                   help="selectivity filter: background activity / target activity at most this")
    m.add_argument("--lr", type=float, nargs="*", default=[3e-2],
                   help="learning rate(s) of the neuron columns (`--lr` with no value: no neuron run, only the LoRAs)")
    m.add_argument("--compare-lora", choices=["none", "small", "standard", "both"], default="none",
                   help="also train a LoRA: small = rank 1 on down_proj, the same size as the K columns; "
                        "standard = rank 8, alpha 16, on all linear layers")
    m.add_argument("--lora-lr-small", type=float, nargs="+", default=[1e-3])
    m.add_argument("--lora-lr-standard", type=float, nargs="+", default=[1e-5])
    m.add_argument("--preserve", help="preservation set: JSON list of {q, r} (r = the original model's reply) or {raw}; "
                                      "adds w-pres x KL(original || edited) on these items to the loss")
    m.add_argument("--w-pres", type=float, default=1.0, help="(--preserve) weight of the preservation loss")
    m.add_argument("--pres-batch", type=int, default=8, help="(--preserve) preservation items per step")
    m.add_argument("--seed", type=int, default=0)
    m.add_argument("--steps", type=int, default=400, help="Adam steps (constant learning rate)")
    m.add_argument("--batch", type=int, default=8, help="items per step")
    m.add_argument("--max-new-tokens", type=int, default=256, help="(--system) length limit of the teacher replies")
    m.add_argument("--gen-batch", type=int, default=16, help="batch size of the teacher generation")
    m.add_argument("--lowmem", action="store_true",
                   help="keep the per-layer embeddings on the CPU (less GPU memory, much slower generation)")
    m.add_argument("--f32", action="store_true", help="measure F with the model in fp32 (no bf16 floor; slower)")
    m.add_argument("--show", type=int, default=3, help="generate replies to this many held-out prompts per run")
    m.add_argument("--out", required=True, help="output directory")
    for name, fn, hlp in (("apply", apply, "load a patch and reply to prompts"), ("chat", chat, "chat with a patch")):
        p = sub.add_parser(name, help=hlp, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        p.add_argument("--patch", required=True, help=".safetensors patch from `make`")
        p.add_argument("--model", help="the model (default: the model in the patch metadata, else E2B)")
        p.add_argument("--max-new-tokens", type=int, default=400)
        p.add_argument("--lowmem", action="store_true", help="keep the per-layer embeddings on the CPU (~6 GB GPU)")
        if name == "apply":
            p.add_argument("prompts", nargs="*", help="user messages")
            p.add_argument("--prompts-file", help="a prompt file instead (.txt, .json or .jsonl)")
            p.add_argument("--compare", action="store_true", help="also show the original model's reply")
            p.add_argument("--bake", action="store_true", help="write the patch into the weights instead of hooks")
            p.add_argument("--save-dir", help="(--bake) save the patched model here")
        p.set_defaults(fn=fn)
    c = sub.add_parser("cards", help="neuron cards for a column patch", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    c.add_argument("--patch", required=True)
    c.add_argument("--model", help="the model (default: the model in the patch metadata, else E2B)")
    c.add_argument("--out", required=True, help="cards JSON (a .md file with the same name is also written)")
    c.set_defaults(fn=cards)
    m.set_defaults(fn=make)
    a = ap.parse_args()
    if a.cmd == "make":
        if (a.system or a.append) and not (a.prompts and a.eval_prompts):
            ap.error("--system and --append need --prompts and --eval-prompts")
        if a.append:
            a.append = a.append.replace("\\n", "\n")   # allow "\n" in the shell argument
        if a.pairs and not a.eval_pairs:
            ap.error("--pairs needs --eval-pairs")
    a.fn(a)


if __name__ == "__main__":
    main()
