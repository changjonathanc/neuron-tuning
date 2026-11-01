"""Build the off-target TRAINING set of experiment 08 (the text on which the selective penalty and the KL
preservation loss are computed). It has the same 7 kinds of text as the forgetting set F
(experiments/lib/data/forget_set.json), but no item of F: other items of the same source, or another split.

  code     60  openai_humaneval (test): problems that are NOT among the 40 F problems (124 are left)
  math     60  gsm8k main (TRAIN split; F uses the test split)
  qa       60  squad (TRAIN split; F uses the validation split)
  de/ja/es 60  each, wikimedia/wikipedia 20231101.<lang>: stream items 3000-5999 (F samples items 0-2999), titles not in F
  longdoc  60  wikitext-2-raw-v1 (TRAIN split; F uses the test split), raw: 60 windows of 512 tokens

The chat items use the same prompt and reply format as F (build_forget_set.py). Seeded; written once.

    uv run --extra data experiments/08_selective_lora/build_offtarget.py     # -> data/offtarget_set.json
"""
import itertools
import json
import os
import random
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
LIB = HERE.parent / "lib"
sys.path.insert(0, str(LIB))
from build_forget_set import WIKI_PROMPT, cut  # noqa: E402

OUT = HERE / "data/offtarget_set.json"
N = 60


def main():
    from datasets import load_dataset
    from transformers import AutoTokenizer
    sys.path.insert(0, str(HERE.parents[1] / "demos/talk_like_a_person"))
    from common import resolve
    F = json.loads((LIB / "data/forget_set.json").read_text())
    f_text = {it.get("q", "") for it in F} | {it.get("r", "") for it in F} | {it.get("raw", "") for it in F}
    rng = random.Random(808)
    items = []
    he = load_dataset("openai/openai_humaneval", split="test")
    left = [j for j in range(len(he)) if not any(he[j]["prompt"] in q for q in f_text)]
    assert len(left) == len(he) - 40, len(left)
    for j in rng.sample(left, N):
        r = he[j]
        items.append(dict(domain="code", q="Complete this Python function.\n\n```python\n" + r["prompt"] + "```",
                          r="```python\n" + r["prompt"] + r["canonical_solution"] + "```"))
    gs = load_dataset("openai/gsm8k", "main", split="train")
    for j in rng.sample(range(len(gs)), N):
        r = gs[j]
        sol, ans = r["answer"].split("####")
        sol = re.sub(r"<<[^>]*>>", "", sol).strip()
        items.append(dict(domain="math", q=r["question"], r=f"{sol}\nThe answer is {ans.strip()}."))
    sq = load_dataset("squad", split="train")
    for j in rng.sample(range(len(sq)), N):
        r = sq[j]
        items.append(dict(domain="qa", q=f"{r['context']}\n\nQuestion: {r['question']}", r=r["answers"]["text"][0] + "."))
    for lang in ("de", "ja", "es"):
        ds = load_dataset("wikimedia/wikipedia", f"20231101.{lang}", split="train", streaming=True)
        f_q = {it["q"] for it in F if it["domain"] == lang}
        pool = [r for r in itertools.islice(ds, 3000, 6000)
                if len(r["text"]) >= 1500 and WIKI_PROMPT[lang].format(t=r["title"]) not in f_q]
        for r in rng.sample(pool, N):
            items.append(dict(domain=lang, q=WIKI_PROMPT[lang].format(t=r["title"]), r=cut(r["text"])))
    repo, rev, _ = resolve("E2B")
    tok = AutoTokenizer.from_pretrained(repo, revision=rev)
    wt = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    ids = tok("".join(wt["text"]), add_special_tokens=False).input_ids
    for s in sorted(rng.sample(range(0, len(ids) - 512, 512), N)):
        items.append(dict(domain="longdoc", raw=tok.decode(ids[s:s + 512])))
    clash = [it for it in items if it.get("q", it.get("raw")) in f_text]   # prompts only: short QA answers can repeat
    assert not clash, f"{len(clash)} items are also in F"
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(items, ensure_ascii=False, indent=0))
    print(Counter(x["domain"] for x in items), "->", OUT, flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
