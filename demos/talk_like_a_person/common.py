"""Shared helpers: model aliases, loading, chat formatting, greedy generation."""
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from edit.load_edit import add_hooks, decoder_layers, load_edit  # noqa: E402,F401

# Revisions the edit was trained on (pinned for exact reproduction).
MODELS = {
    "E2B": dict(repo="google/gemma-4-E2B-it", revision="3e22461f65e89153144f8adb70e3b8c2cc9845a7",
                edit=ROOT / "edit/gemma-4-E2B-it_128gains.json"),
    "E4B": dict(repo="google/gemma-4-E4B-it", revision="ee0ef6023621cff504d758262d4e04895a5af4a2",
                edit=ROOT / "edit/gemma-4-E4B-it_128gains.json"),
}

# The teacher system prompt that the edit was distilled from (also the "talk like a person" baseline).
PERSONA_SYSTEM = ("You are Sam, an ordinary 29-year-old person chatting by text with a friend. Reply casually and fairly "
                  "briefly like a real human would, in first person, with your own opinions, experiences, moods and "
                  "feelings. Never mention being an AI, a model, or an assistant.")

# Control for the judge's sensitivity to form: asks only for short, casual, plain-text replies (no persona, no
# instruction about being a person or an AI).
STYLE_SYSTEM = ("Reply in one or two short, casual sentences, like a text message. Do not use markdown, headers, "
                "bullet lists or bold.")


def resolve(model):
    """'E2B' / 'E4B' -> (repo, revision, default edit path); a repo id or local dir passes through."""
    if model.upper() in MODELS:
        m = MODELS[model.upper()]
        return m["repo"], m["revision"], m["edit"]
    for m in MODELS.values():
        if m["repo"] == model:
            return m["repo"], m["revision"], m["edit"]
    return model, None, None


def load_model(name, revision=None, lowmem=False, device="cuda", dtype=torch.bfloat16, attn="eager"):
    """Load a Gemma-4 chat model (bf16 by default) for text generation.

    lowmem=True drops the vision/audio towers and keeps the large per-layer embedding table on the CPU,
    so E2B fits in ~6 GB of GPU memory. Outputs are unchanged.

    attn="eager" by default: with left-padded batches, the SDPA path hands the memory-efficient kernel
    stride-0 (expanded) K/V plus a boolean padding mask, and on some torch/GPU stacks that kernel returns
    wrong attention outputs (see README, "Batching"). Eager attention avoids that kernel entirely.
    """
    tok = AutoTokenizer.from_pretrained(name, revision=revision)
    if not lowmem:
        model = AutoModelForCausalLM.from_pretrained(name, revision=revision, dtype=dtype,
                                                     attn_implementation=attn).to(device).eval()
        return model.requires_grad_(False), tok
    model = AutoModelForCausalLM.from_pretrained(name, revision=revision, dtype=dtype, attn_implementation=attn).eval()
    for nm in ["vision_tower", "audio_tower", "embed_vision", "embed_audio"]:
        if hasattr(model.model, nm):
            setattr(model.model, nm, None)
    lm = model.model.language_model
    ple = lm.embed_tokens_per_layer
    lm.embed_tokens_per_layer = None
    model = model.to(device)

    class CPUEmbedding(torch.nn.Module):
        def __init__(self, emb):
            super().__init__()
            self.emb, self.weight = [emb], emb.weight   # list: keep it out of .to(device)

        def forward(self, ids):
            return self.emb[0](ids.cpu()).to(ids.device)

    lm.embed_tokens_per_layer = CPUEmbedding(ple)
    return model.requires_grad_(False), tok


def chat_prompt(tok, messages, system=None):
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    msgs = ([{"role": "system", "content": system}] if system else []) + messages
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def generate_one(model, tok, prompt, system=None, max_new_tokens=1024, temperature=0.0, seed=None):
    """One reply, unbatched (no padding). temperature 0 = greedy; otherwise plain sampling at that
    temperature (top_k/top_p disabled), seeded. Returns (text, n_new_tokens, hit_cap)."""
    ids = tok(chat_prompt(tok, prompt, system), return_tensors="pt", add_special_tokens=False).to(model.device)
    kw = dict(do_sample=False) if not temperature else dict(do_sample=True, temperature=temperature, top_k=0, top_p=1.0)
    if seed is not None:
        torch.manual_seed(seed)
    o = model.generate(**ids, max_new_tokens=max_new_tokens, **kw)[0, ids.input_ids.shape[1]:]
    eos = set(model.generation_config.eos_token_id if isinstance(model.generation_config.eos_token_id, list)
              else [model.generation_config.eos_token_id])
    ended = len(o) > 0 and int(o[-1]) in eos
    return tok.decode(o, skip_special_tokens=True), len(o), (len(o) >= max_new_tokens and not ended)


@torch.no_grad()
def generate(model, tok, prompts, max_new_tokens=160, batch_size=1, system=None):
    """Greedy replies to a list of user messages (or full message lists).

    batch_size=1 (the default) generates each reply on its own with no padding, so a reply does not depend on
    which other prompts it was batched with. batch_size>1 uses left-padded batches: faster, but greedy replies
    then depend on the batch composition (bf16 rounding differs with padding and batch shape), so use it only
    for quick looks, never for evaluation.
    """
    if batch_size == 1:
        return [generate_one(model, tok, p, system, max_new_tokens)[0] for p in prompts]
    tok.padding_side = "left"
    texts = [chat_prompt(tok, p, system) for p in prompts]
    out = []
    for i in range(0, len(texts), batch_size):
        b = tok(texts[i:i + batch_size], return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        o = model.generate(**b, max_new_tokens=max_new_tokens, do_sample=False)
        out += [tok.decode(x[b.input_ids.shape[1]:], skip_special_tokens=True) for x in o]
    return out
