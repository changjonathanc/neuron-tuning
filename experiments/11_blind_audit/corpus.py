"""Make the fixed audit corpus data/corpus.json (CPU, run once; the output is in git).

    uv run --with pyarrow experiments/11_blind_audit/corpus.py

(a) shipped data of this repository (the corpus of the make_edit.py neuron cards), (b) a generic corpus:
1,000 smol-smoltalk conversations (test split; first user turn + first assistant reply) and 400 wikitext-103
paragraphs (test split); dataset revisions pinned. The file is not in git (it holds public dataset text);
this script rebuilds it and checks its SHA-256 against the file of the run. Items equal to a training or evaluation prompt of a demo are removed. Source labels are
neutral (no topic names).
"""
import hashlib
import json
import random
import sys
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
import make_edit  # noqa: E402

SMOLTALK = ("HuggingFaceTB/smol-smoltalk", "data/test-00000-of-00001.parquet", "f73fe857d519ff6ac5af2ea67c4d3834da7b8bcc")
WIKITEXT = ("Salesforce/wikitext", "wikitext-103-raw-v1/test-00000-of-00001.parquet", "b08601e04326c79dfdd32d625aee71d232d685c3")
SHA256 = "2b81029b19a12b75939defb61bb060c09b9911d981d72ed6cce3d6413ca8f91a"   # data/corpus.json of the run


def neutral(src):
    if src.endswith("(02)"):
        return "rule-task prompts (02)"   # the rule names (weather, recipe, ...) are topic names: hide them
    return {"casual chat": "casual chat (shipped)", "tasks (SlimOrca)": "tasks (SlimOrca, shipped)"}.get(
        src, f"reference text: {src}")


def demo_prompts():
    out = set()
    for p in (ROOT / "demos").glob("*/data/*.json"):
        d = json.loads(p.read_text())
        for x in d if isinstance(d, list) else [v for vs in d.values() for v in (vs if isinstance(vs, list) else [])]:
            q = x if isinstance(x, str) else (x.get("q") or x.get("prompt") if isinstance(x, dict) else None)
            if isinstance(q, str):
                out.add(q.strip())
    return out


def main():
    rng = random.Random(11)
    items = [dict(source=neutral(s), raw=raw, q=q, r=r) for s, raw, q, r in make_edit.card_corpus()]
    rows = pq.read_table(hf_hub_download(SMOLTALK[0], SMOLTALK[1], revision=SMOLTALK[2], repo_type="dataset")).to_pylist()
    convs = []
    for row in rows:
        m = row["messages"]
        if len(m) >= 2 and m[0]["role"] == "user" and m[1]["role"] == "assistant":
            convs.append((m[0]["content"], m[1]["content"]))
    for q, r in rng.sample(convs, 1000):
        items.append(dict(source="chat (smoltalk)", raw=None, q=q, r=r))
    lines = [x["text"].strip() for x in pq.read_table(hf_hub_download(WIKITEXT[0], WIKITEXT[1], revision=WIKITEXT[2], repo_type="dataset")).to_pylist()]
    paras = [x for x in lines if len(x) > 400 and not x.startswith("=")]
    for x in rng.sample(paras, 400):
        items.append(dict(source="web text (wikitext)", raw=x, q=None, r=None))
    seen = demo_prompts()
    keep = [it for it in items if (it["q"] or "").strip() not in seen]
    print(f"{len(items)} items, removed {len(items) - len(keep)} equal to a demo prompt ({len(seen)} demo prompts)")
    (HERE / "data").mkdir(exist_ok=True)
    text = json.dumps(keep, ensure_ascii=False, indent=0)
    (HERE / "data/corpus.json").write_text(text)
    h = hashlib.sha256(text.encode()).hexdigest()
    print("sha256", h, "(same as the run)" if h == SHA256 else "DIFFERENT from the run: " + SHA256)
    by = {}
    for it in keep:
        by[it["source"]] = by.get(it["source"], 0) + 1
    print(json.dumps(by, indent=1))


if __name__ == "__main__":
    main()
