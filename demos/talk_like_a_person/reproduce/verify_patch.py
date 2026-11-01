"""Check that a baked checkpoint (patch_gemma.py --out) is the same model as the runtime-hook edit.

1. Weights: exactly the 128 edited down_proj columns differ from the original, by the factor (1 + s*gain).
2. Behaviour: greedy replies and next-token logits on held-out messages, hooks vs baked.

    python reproduce/verify_patch.py --patched gemma-4-E2B-it-human [--n 20] [--lowmem]
"""
import argparse
import json
import sys
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import chat_prompt, generate, load_model, resolve  # noqa: E402
from edit.load_edit import add_hooks, down_proj_key, load_edit  # noqa: E402


def tensors(d):
    """{tensor name: open safetensors file} over every shard in directory d."""
    out = {}
    for f in sorted(Path(d).glob("*.safetensors")):
        h = safe_open(f, "pt")
        out.update({k: h for k in h.keys()})
    return out


def check_weights(src, dst, edit, strength):
    want = {down_proj_key(l): (n, g) for l, (n, g) in edit.items()}
    a, b = tensors(src), tensors(dst)
    assert set(a) == set(b), "tensor names differ"
    assert set(want) <= set(a), "edited weights not found in the checkpoint"
    changed = 0
    for k in a:
        x, y = a[k].get_tensor(k), b[k].get_tensor(k)
        if k not in want:
            assert torch.equal(x, y), f"unexpected change in {k}"
            continue
        n, g = want[k]
        diff_cols = (x != y).any(0).nonzero().flatten().tolist()
        assert set(diff_cols) <= set(n), f"{k}: columns outside the edit changed"
        expect = (x[:, n].float() * (1 + strength * torch.tensor(g))).to(x.dtype)
        assert torch.equal(y[:, n], expect), f"{k}: column values differ from W*(1+s*g)"
        changed += len(n)
    return changed


@torch.no_grad()
def reply_logprobs(model, tok, qs, replies):
    """Teacher-forced next-token log-probs over each reply (one sequence at a time, no padding)."""
    out = []
    for q, r in zip(qs, replies):
        p = chat_prompt(tok, q)
        ids = tok(p + r, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
        n0 = len(tok(p, add_special_tokens=False).input_ids)
        out.append(torch.log_softmax(model(input_ids=ids).logits[0, n0 - 1:-1].float(), -1).half().cpu())
    return out


@torch.no_grad()
def reply_logprobs_batched(model, tok, qs, replies, bs=4):
    """Same, but left-padded batches: differs from reply_logprobs only by bf16 numerics."""
    tok.padding_side, out = "left", []
    for i in range(0, len(qs), bs):
        ps = [chat_prompt(tok, q) for q in qs[i:i + bs]]
        b = tok([p + r for p, r in zip(ps, replies[i:i + bs])], return_tensors="pt", padding=True,
                add_special_tokens=False).to(model.device)
        ns = [len(tok(p + r, add_special_tokens=False).input_ids) - len(tok(p, add_special_tokens=False).input_ids)
              for p, r in zip(ps, replies[i:i + bs])]
        logits = model(**b, logits_to_keep=max(ns) + 1).logits
        for j, n in enumerate(ns):
            out.append(torch.log_softmax(logits[j, -n - 1:-1].float(), -1).half().cpu())
    return out


def compare(lp_a, lp_b):
    agree = torch.cat([(x.argmax(-1) == y.argmax(-1)).float() for x, y in zip(lp_a, lp_b)])
    kls = torch.cat([(x.float().exp() * (x.float() - y.float())).sum(-1) for x, y in zip(lp_a, lp_b)])
    return f"argmax agreement {agree.mean().item():.4f} over {agree.numel()} tokens, mean KL {kls.mean().item():.1e}, max KL {kls.max().item():.1e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="E2B")
    ap.add_argument("--patched", required=True)
    ap.add_argument("--strength", type=float, default=1.0)
    ap.add_argument("--n", type=int, default=20, help="number of held-out messages")
    ap.add_argument("--lowmem", action="store_true")
    ap.add_argument("--fp32-cpu", action="store_true", help="run both models in float32 on the CPU (slow; removes bf16 noise)")
    a = ap.parse_args()
    kw = dict(device="cpu", dtype=torch.float32, lowmem=True) if a.fp32_cpu else dict(lowmem=a.lowmem)
    repo, rev, edit_path = resolve(a.model)
    edit, _ = load_edit(edit_path)
    src = Path(snapshot_download(repo, revision=rev, allow_patterns=["*.safetensors"]))
    n = check_weights(src, Path(a.patched), edit, a.strength)
    print(f"weights: only the {n} edited down_proj columns differ, each exactly round_bf16(W[:, j] * (1 + s*g_j))")

    qs = json.loads((HERE / "data/prompts.json").read_text())["eval_q"][:a.n]
    model, tok = load_model(repo, rev, **kw)
    orig = generate(model, tok, qs)
    hooks = add_hooks(model, edit, a.strength)
    hooked = generate(model, tok, qs)
    hooked_bn = generate(model, tok, qs, batch_size=len(qs))  # same model, one padded batch: bf16 noise floor
    lp_h = reply_logprobs(model, tok, qs, hooked)
    lp_h_batched = reply_logprobs_batched(model, tok, qs, hooked)
    del model, hooks
    torch.cuda.empty_cache()
    model, tok = load_model(a.patched, **kw)
    baked = generate(model, tok, qs)
    lp_b = reply_logprobs(model, tok, qs, hooked)

    same = lambda x, y: sum(u == v for u, v in zip(x, y))  # noqa: E731
    print(f"greedy replies identical, hooks vs baked:               {same(hooked, baked)}/{len(qs)}")
    print(f"greedy replies identical, hooks batch 1 vs batch {len(qs)} (noise floor): {same(hooked, hooked_bn)}/{len(qs)}")
    print(f"teacher-forced on the hooked replies, hooks vs baked:   {compare(lp_h, lp_b)}")
    print(f"  noise floor, hooks unpadded vs hooks padded batches: {compare(lp_h, lp_h_batched)}")
    print(f"original vs patched replies differing: {sum(o != h for o, h in zip(orig, hooked))}/{len(qs)}")
    for q, o, h, b in list(zip(qs, orig, hooked, baked))[:5]:
        print(f"\n> {q}\n  original: {o[:160]!r}\n  hooks:    {h[:160]!r}\n  baked:    {b[:160]!r}")


if __name__ == "__main__":
    main()
