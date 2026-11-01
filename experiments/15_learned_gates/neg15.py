"""The negative items of arms B and C: the original model's greedy replies (at most 256 tokens) to the near-miss
training prompts -> data/nearmiss_neg.json ([{q, kind, r, capped}]). GPU.

    python experiments/15_learned_gates/neg15.py
"""
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "demos"))
sys.path.insert(0, str(HERE.parents[1] / "experiments/lib"))
from lib import generate_batched  # noqa: E402
from p1lib import load_model, resolve  # noqa: E402

torch.set_grad_enabled(False)
items = json.loads((HERE / "data/nearmiss_train.json").read_text())
model, tok = load_model(*resolve("E2B")[:2])
rows = generate_batched(model, tok, [x["q"] for x in items], 256, 32)
out = [dict(x, r=r["reply"], capped=r["capped"]) for x, r in zip(items, rows)]
(HERE / "data/nearmiss_neg.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
print(len(out), "replies,", sum(r["capped"] for r in rows), "capped")
