"""Build the forgetting set (the unrelated text on which F is measured): diverse held-out text, NOT from the preservation distribution
(SlimOrca). Seeded, written once to data/forget_set.json.

Domains (chat items: user prompt + reference reply as the assistant turn; raw items: plain text, no chat template):
  code     40  openai_humaneval (test): "Complete this Python function" + prompt + canonical solution
  math     40  gsm8k main (test): question -> reference solution (calculator annotations removed)
  qa       40  squad (validation): context + question -> the first reference answer
  de/ja/es 40  each, wikimedia/wikipedia 20231101.<lang>: "write a short encyclopedia article about <title>" (in that
               language) -> the first <= 1000 characters of the article, cut at a sentence end
  longdoc  16  wikitext-2-raw-v1 (test), raw: 16 windows of 1024 tokens (plain language modelling, long context)

    python experiments/lib/build_forget_set.py
"""
import itertools
import json
import os
import random
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "data/forget_set.json"
N = 40
WIKI_PROMPT = {"de": "Schreibe einen kurzen Lexikonartikel über „{t}“.",
               "ja": "「{t}」について短い百科事典の記事を書いてください。",
               "es": "Escribe un breve artículo de enciclopedia sobre «{t}»."}


def cut(text, n=1000):
    text = text.strip()
    if len(text) <= n:
        return text
    s = text[:n]
    m = max(s.rfind(x) for x in (". ", "。", ".\n", "! ", "? "))
    return s[:m + 1] if m > 200 else s


def main():
    from datasets import load_dataset
    from transformers import AutoTokenizer
    sys.path.insert(0, str(HERE.parents[1] / "demos/talk_like_a_person"))
    from common import resolve
    rng = random.Random(1234)
    items = []
    he = load_dataset("openai/openai_humaneval", split="test")
    for j in rng.sample(range(len(he)), N):
        r = he[j]
        items.append(dict(domain="code", q="Complete this Python function.\n\n```python\n" + r["prompt"] + "```",
                          r="```python\n" + r["prompt"] + r["canonical_solution"] + "```"))
    gs = load_dataset("openai/gsm8k", "main", split="test")
    for j in rng.sample(range(len(gs)), N):
        r = gs[j]
        sol, ans = r["answer"].split("####")
        sol = re.sub(r"<<[^>]*>>", "", sol).strip()
        items.append(dict(domain="math", q=r["question"], r=f"{sol}\nThe answer is {ans.strip()}."))
    sq = load_dataset("squad", split="validation")
    for j in rng.sample(range(len(sq)), N):
        r = sq[j]
        items.append(dict(domain="qa", q=f"{r['context']}\n\nQuestion: {r['question']}", r=r["answers"]["text"][0] + "."))
    for lang in ("de", "ja", "es"):
        ds = load_dataset("wikimedia/wikipedia", f"20231101.{lang}", split="train", streaming=True)
        pool = [r for r in itertools.islice(ds, 3000) if len(r["text"]) >= 1500]
        for r in rng.sample(pool, N):
            items.append(dict(domain=lang, q=WIKI_PROMPT[lang].format(t=r["title"]), r=cut(r["text"])))
    repo, rev, _ = resolve("E2B")
    tok = AutoTokenizer.from_pretrained(repo, revision=rev)
    wt = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    ids = tok("".join(wt["text"]), add_special_tokens=False).input_ids
    starts = sorted(rng.sample(range(0, len(ids) - 1024, 1024), 16))
    for s in starts:
        items.append(dict(domain="longdoc", raw=tok.decode(ids[s:s + 1024])))
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(items, ensure_ascii=False, indent=0))
    from collections import Counter
    print(Counter(x["domain"] for x in items), "->", OUT, flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
