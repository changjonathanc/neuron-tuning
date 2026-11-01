"""Phase 2b generation of the data and the teacher check (greedy, <= 128 new tokens, eager attention, left-padded
batches; see NOTES.md).

  --mode data:   for a rule and a split (train | eval): replies of the teacher (rule system prompt) and of the original
                 model (no system prompt) to the trigger and the control messages -> data/gen_<rule>_<split>.json
                 (keys teacher_trigger, teacher_control, original_trigger, original_control), and the check rates.
  --mode verify: batched (batch 20) vs batch-size-1 greedy replies on 20 messages; agreement of the first 32 tokens
                 and of the full reply -> data/verify_batching.json

    python gen2b.py --mode data --rule cat --split train
"""
import argparse
import json

import torch

from common2b import HERE, SYSTEMS, gen_file, generate_batched, load_model, pairs, resolve, score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["data", "verify"], required=True)
    ap.add_argument("--rule")
    ap.add_argument("--split", choices=["train", "eval"])
    ap.add_argument("--bs", type=int, default=20)
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    model, tok = load_model(*resolve("E2B")[:2])
    if a.mode == "verify":
        P = pairs("cat", "train")[:10] + pairs("lower", "train")[:10]
        qs = [p["control"] for p in P]
        b = generate_batched(model, tok, qs, None, 20, 128)
        u = [generate_batched(model, tok, [q], None, 1, 128)[0] for q in qs]
        enc = lambda s: tok(s, add_special_tokens=False).input_ids
        first = sum(enc(x["text"])[:32] == enc(y["text"])[:32] for x, y in zip(b, u))
        full = sum(x["text"] == y["text"] for x, y in zip(b, u))
        res = dict(n=len(qs), agree_first32=first, agree_full=full, batched=b, single=u)
        (HERE / "data/verify_batching.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
        print(f"batched vs single: first 32 tokens agree {first}/{len(qs)}, full reply {full}/{len(qs)}")
        return
    out = gen_file(a.rule, a.split)
    res = json.loads(out.read_text()) if out.exists() else {}
    P = pairs(a.rule, a.split)
    for who in ("teacher", "original"):
        for side in ("trigger", "control"):
            k = f"{who}_{side}"
            if k in res:
                continue
            res[k] = generate_batched(model, tok, [p[side] for p in P], SYSTEMS[a.rule] if who == "teacher" else None,
                                      a.bs)
            out.write_text(json.dumps(res, indent=1, ensure_ascii=False))
            print(k, "done", flush=True)
    rates = {k: sum(score(a.rule, x["text"], x["ended"]) for x in v) / len(v) for k, v in res.items() if k != "rates"}
    res["rates"] = rates
    out.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(a.rule, a.split, {k: round(v, 3) for k, v in rates.items()})


if __name__ == "__main__":
    main()
