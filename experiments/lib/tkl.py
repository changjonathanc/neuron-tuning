"""Post-hoc, non-ceiling metric: teacher KL on the 60 held-out eval messages.

For each of the teacher's 180 replies to the eval messages (the "prompt" condition: the model with the Sam system
prompt; 1 greedy + 2 sampled per message), token-level KL(teacher || run) over the reply tokens, teacher-forced and
unbatched. The teacher sees the system prompt; the run does not. The teacher replies are in data/prompt.json. Mean per reply, then per message, then over messages.

    python experiments/lib/tkl.py --adapters original demos/talk_like_a_person/edit/gemma-4-E2B-it_128gains.json runs/final/*.pt --out tkl.json
"""
import argparse
import json
from pathlib import Path

import torch

from p1lib import load_adapter, load_model, resolve
from common import PERSONA_SYSTEM, chat_prompt

HERE = Path(__file__).resolve().parent


@torch.no_grad()
def reply_logp(model, tok, q, r, system):
    p = chat_prompt(tok, q, system)
    ids = tok(p + r, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
    n0 = len(tok(p, add_special_tokens=False).input_ids)
    return torch.log_softmax(model(input_ids=ids).logits[0, n0 - 1:-1].float(), -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapters", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    out = Path(a.out)
    res = json.loads(out.read_text()) if out.exists() else {}
    T = json.loads((HERE / "data/prompt.json").read_text())
    pairs = [(it["i"], T["messages"][it["i"]], it["text"]) for it in T["items"]]
    model, tok = load_model(*resolve("E2B")[:2])
    teacher = [reply_logp(model, tok, q, r, PERSONA_SYSTEM).cpu() for _, q, r in pairs]
    for name in a.adapters:
        if name in res:
            continue
        ad = None if name == "original" else load_adapter(model, name)
        kls = []
        for (i, q, r), t in zip(pairs, teacher):
            s = reply_logp(model, tok, q, r, None).cpu()
            kls.append(float((t.exp() * (t - s)).sum(-1).mean()))
        if ad is not None:
            ad.remove()
        res[name] = dict(per_reply=kls, msg=[i for i, _, _ in pairs], mean=sum(kls) / len(kls))
        print(f"{name}: teacher KL {res[name]['mean']:.4f}", flush=True)
        out.write_text(json.dumps(res))


if __name__ == "__main__":
    main()
