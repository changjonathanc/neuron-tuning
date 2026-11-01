"""KL helpers of the evaluation of 02_conditional_rules (`eval2b.py`), teacher-forced and unbatched:
  - logp_chat: log-probabilities of a reply after a chat prompt (with or without a system prompt);
  - logp_forget: log-probabilities on one item of the forgetting set (data/forget_set.json; the positions of forget.py);
  - kl: mean token KL(t || s).
The functions are copied unchanged from the evaluation script of an earlier study (not in this repository).
"""
from pathlib import Path

import torch

from p2lib import chat_prompt

HERE = Path(__file__).resolve().parent
FORGET = HERE / "data/forget_set.json"


@torch.no_grad()
def logp_chat(model, tok, q, r, system):
    p = chat_prompt(tok, q, system)
    ids = tok(p + r, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
    n0 = len(tok(p, add_special_tokens=False).input_ids)
    return torch.log_softmax(model(input_ids=ids, logits_to_keep=ids.shape[1] - n0 + 1).logits[0, :-1].float(), -1)


@torch.no_grad()
def logp_forget(model, tok, it):
    if "raw" in it:
        ids = tok(it["raw"], return_tensors="pt").input_ids.to(model.device)
        lo = 0
    else:
        p = chat_prompt(tok, it["q"], None)
        ids = tok(p + it["r"], return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
        lo = len(tok(p, add_special_tokens=False).input_ids) - 1
    return torch.log_softmax(model(input_ids=ids, logits_to_keep=ids.shape[1] - lo).logits[0, :-1].float(), -1)


def kl(t, s):
    return float((t.exp() * (t - s)).sum(-1).mean())
