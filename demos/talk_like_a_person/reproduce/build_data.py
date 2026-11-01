"""Build the distillation data (the shipped data/data_persona_{E2B,E4B}.json were made with exactly this recipe).

  kind="id"   160 casual chat messages (data/prompts.json train_q), answered by the model itself WITH the
              "Sam" system prompt (the teacher). The edit should learn to reproduce these without the prompt.
  kind="pres" 300 task prompts from SlimOrca (first user turn, <= 400 chars), answered by the original model
              WITHOUT a system prompt. Preservation: the edit should not change these.

Replies are greedy, 64 new tokens, batches of 24 (left-padded). bf16 batched generation is not bit-stable across
GPUs/versions, so a rebuilt file can differ in a few replies; use the shipped file for exact reproduction.

    python reproduce/build_data.py --model E2B --out reproduce/data/data_persona_E2B.json [--lowmem]

--v2 (not used for the released edit): generate one reply at a time with up to 256 new tokens, and record for each
reply whether it ended (`ended`), so that train_gains.py can put the end-of-turn token into the loss.

    python reproduce/build_data.py --model E2B --v2 --out data_v2_E2B.json
"""
import argparse
import json
import sys
from pathlib import Path

import datasets
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import PERSONA_SYSTEM, generate, generate_one, load_model, resolve  # noqa: E402

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="E2B")
    ap.add_argument("--out", required=True)
    ap.add_argument("--lowmem", action="store_true")
    ap.add_argument("--v2", action="store_true", help="unbatched, <= 256 tokens, record whether each reply ended")
    a = ap.parse_args()
    casual = json.loads((HERE / "data/prompts.json").read_text())["train_q"]
    so = datasets.load_dataset("cgato/SlimOrcaDedupCleaned", split="train").shuffle(seed=21).select(range(700))
    tasks = [next(t["value"] for t in ex["conversations"] if t["from"] == "human")[:400] for ex in so][:300]

    repo, rev, _ = resolve(a.model)
    model, tok = load_model(repo, rev, lowmem=a.lowmem)
    if a.v2:
        data = []
        for qs, system, kind in [(casual, PERSONA_SYSTEM, "id"), (tasks, None, "pres")]:
            for q in qs:
                r, _, capped = generate_one(model, tok, q, system, max_new_tokens=256)
                data.append(dict(q=q, r=r, sys=system, kind=kind, ended=not capped))
        print(f"ended: casual {sum(d['ended'] for d in data if d['kind'] == 'id')}/{len(casual)}, "
              f"tasks {sum(d['ended'] for d in data if d['kind'] == 'pres')}/{len(tasks)}")
    else:
        # The released data was generated in left-padded batches of 24 (kept for exact reproduction; see README, Batching).
        teach = generate(model, tok, casual, max_new_tokens=64, batch_size=24, system=PERSONA_SYSTEM)
        orig = generate(model, tok, tasks, max_new_tokens=64, batch_size=24)
        data = [dict(q=q, r=r, sys=PERSONA_SYSTEM, kind="id") for q, r in zip(casual, teach)]
        data += [dict(q=q, r=r, sys=None, kind="pres") for q, r in zip(tasks, orig)]
    Path(a.out).write_text(json.dumps(data, ensure_ascii=False, indent=1))
    print(f"wrote {len(data)} items to {a.out}; e.g. {data[0]['q']!r} -> {data[0]['r'][:80]!r}")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
