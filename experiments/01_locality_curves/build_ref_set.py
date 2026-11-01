"""Phase 1c: the selectivity REFERENCE set (general text), used ONLY to choose the selective-random neurons.
Same 7 domains as the Phase 1b forgetting set, but DISJOINT items (no item of data/forget_set.json is used):

  code     20  openai_humaneval (test) problems NOT in the forgetting set
  math     20  gsm8k main (TRAIN split)
  qa       20  squad (TRAIN split)
  de/ja/es 20  each, wikimedia/wikipedia 20231101.<lang>, articles 3000-5999 of the stream (the forgetting set
               used articles 0-2999)
  longdoc   8  wikitext-2-raw-v1 (TRAIN split), raw: 8 windows of 1024 tokens

    python experiments/01_locality_curves/build_ref_set.py
"""
import itertools
import json
import os
import random
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
P1B = HERE.parent / "lib"
sys.path.insert(0, str(P1B))
from build_forget_set import WIKI_PROMPT, cut  # noqa: E402

OUT = HERE / "data/ref_set.json"
N = 20


def main():
    from datasets import load_dataset
    from transformers import AutoTokenizer
    sys.path.insert(0, str(HERE.parents[1] / "demos/talk_like_a_person"))
    from common import resolve
    rng = random.Random(4321)
    forget = json.loads((P1B / "data/forget_set.json").read_text())
    used_q = {it.get("q") for it in forget}
    items = []
    he = load_dataset("openai/openai_humaneval", split="test")
    pool = [j for j in range(len(he)) if
            "Complete this Python function.\n\n```python\n" + he[j]["prompt"] + "```" not in used_q]
    assert len(pool) == len(he) - 40, len(pool)
    for j in rng.sample(pool, N):
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
        pool = [r for r in itertools.islice(ds, 3000, 6000) if len(r["text"]) >= 1500]
        for r in rng.sample(pool, N):
            items.append(dict(domain=lang, q=WIKI_PROMPT[lang].format(t=r["title"]), r=cut(r["text"])))
    repo, rev, _ = resolve("E2B")
    tok = AutoTokenizer.from_pretrained(repo, revision=rev)
    wt = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    ids = tok("".join(wt["text"][:20000]), add_special_tokens=False).input_ids
    starts = sorted(rng.sample(range(0, len(ids) - 1024, 1024), 8))
    for s in starts:
        items.append(dict(domain="longdoc", raw=tok.decode(ids[s:s + 1024])))
    assert not ({it.get("q") for it in items if "q" in it} & used_q)
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(items, ensure_ascii=False, indent=0))
    from collections import Counter
    print(Counter(x["domain"] for x in items), "->", OUT, flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
