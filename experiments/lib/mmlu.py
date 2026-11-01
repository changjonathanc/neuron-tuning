"""MMLU on the 570-question subset (10 questions per subject, seed 0), the side measure of forget.py --mmlu.
The two functions are copied unchanged from the side-effect script of an earlier study (not in this repository).
"""
import json
import random
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
MMLU = HERE / "runs/mmlu570.json"        # the 570 items, dumped once by mmlu_items() on a machine with `datasets`


def mmlu_items(per_subject=10):
    if MMLU.exists():
        return json.loads(MMLU.read_text())
    from datasets import load_dataset
    ds = load_dataset("cais/mmlu", "all", split="test")
    by = {}
    for r in ds:
        by.setdefault(r["subject"], []).append(r)
    rng = random.Random(0)
    items = [x for s in sorted(by) for x in rng.sample(by[s], min(per_subject, len(by[s])))]
    MMLU.write_text(json.dumps(items))
    return items


@torch.no_grad()
def mmlu(model, tok, items):
    letters = ["A", "B", "C", "D"]
    ids = [tok(l, add_special_tokens=False)["input_ids"][-1] for l in letters]
    correct = []
    for r in items:
        body = r["question"].strip() + "\n\n" + "\n".join(f"({l}) {c}" for l, c in zip(letters, r["choices"])) + \
            "\n\nAnswer with the letter only."
        p = tok.apply_chat_template([{"role": "user", "content": body}], tokenize=False,
                                    add_generation_prompt=True) + "The answer is ("
        x = tok(p, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
        correct.append(int(int(model(input_ids=x).logits[0, -1, ids].argmax()) == r["answer"]))
    return correct
