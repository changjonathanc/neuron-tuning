"""Off-target drift: how much does an edit change the model on unrelated task chats?

50 SlimOrca task prompts that are NOT in the distillation data (items 300-349 of the same shuffled sample that
build_data.py takes its 300 preservation tasks from). The original model answers each one (greedy, unbatched,
<= 256 new tokens; cached in --refs). Each edit is then scored by the token-level KL(original || edited) of the
next-token distributions over those reply tokens (teacher-forced, unbatched), averaged per reply and then over replies.

    python reproduce/eval_drift.py --edits edit/gemma-4-E2B-it_128gains.json my_k512.json --out results/drift.json
    python reproduce/eval_drift.py --steer reproduce/baselines/steer_all.json --out results/drift_steer.json
"""
import argparse
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from common import chat_prompt, generate_one, load_model, resolve  # noqa: E402
from edit.load_edit import add_hooks, add_steering, load_edit, load_vectors  # noqa: E402
from eval_humanlike import bootstrap_ci  # noqa: E402


def held_out_tasks(n):
    import datasets
    so = datasets.load_dataset("cgato/SlimOrcaDedupCleaned", split="train").shuffle(seed=21).select(range(700))
    tasks = [next(t["value"] for t in ex["conversations"] if t["from"] == "human")[:400] for ex in so]
    return tasks[300:300 + n]   # build_data.py uses tasks[:300]


@torch.no_grad()
def reply_logp(model, tok, q, r):
    p = chat_prompt(tok, q)
    ids = tok(p + r, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
    n0 = len(tok(p, add_special_tokens=False).input_ids)
    return torch.log_softmax(model(input_ids=ids).logits[0, n0 - 1:-1].float(), -1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="E2B")
    ap.add_argument("--edits", nargs="*", default=[], help="edit JSON files (gains)")
    ap.add_argument("--steer", nargs="*", default=[], help="steering-vector JSON files")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--refs", default="results/drift_refs.json", help="cache of the original model's replies")
    ap.add_argument("--out", required=True)
    ap.add_argument("--lowmem", action="store_true")
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    repo, rev, _ = resolve(a.model)
    model, tok = load_model(repo, rev, lowmem=a.lowmem)
    refs = Path(a.refs)
    if refs.exists():
        pairs = json.loads(refs.read_text())
    else:
        pairs = [(q, generate_one(model, tok, q, max_new_tokens=256)[0]) for q in held_out_tasks(a.n)]
        refs.parent.mkdir(parents=True, exist_ok=True)
        refs.write_text(json.dumps(pairs, ensure_ascii=False, indent=1))
    base = [reply_logp(model, tok, q, r).cpu() for q, r in pairs]

    out = {}
    runs = [(f, "edit") for f in a.edits] + [(f, "steer") for f in a.steer]
    for f, kind in runs:
        if kind == "edit":
            handles = add_hooks(model, load_edit(f)[0])
        else:
            handles = add_steering(model, load_vectors(f)[0])
        kls = []
        for (q, r), lp0 in zip(pairs, base):
            lp = reply_logp(model, tok, q, r).cpu()
            kls.append(float((lp0.exp() * (lp0 - lp)).sum(-1).mean()))
        for h in handles:
            h.remove()
        out[f] = dict(kind=kind, kl=sum(kls) / len(kls), kl_ci=bootstrap_ci(kls), per_reply=kls)
        print(f"{f}: KL {out[f]['kl']:.4f} [{out[f]['kl_ci'][0]:.4f}, {out[f]['kl_ci'][1]:.4f}]", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(dict(n=len(pairs), runs=out), indent=1))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
