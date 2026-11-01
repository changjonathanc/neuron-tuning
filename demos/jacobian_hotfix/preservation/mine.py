"""Build the preservation set of a neuron set (PRESERVATION.md) (GPU).

    python demos/jacobian_hotfix/preservation/mine.py --neurons <report.json> --name primary
        -> data/preserve_<name>.json (the training file of make_edit.py --preserve), data/mine_<name>.json (statistics)

Steps:
  1. The original model's greedy replies (256 tokens) to data/status_prompts.json and data/general_prompts.json
     (cached in data/original_replies.json; these replies are the preservation targets of the status questions).
  2. Corpus = the mixed corpus of the neuron cards (make_edit.card_corpus: shipped data only) + the generated
     (prompt, reply) pairs of step 1.
  3. For each item and each chosen neuron j: mean |a_j| at the scored positions (the last prompt token and the reply
     tokens), relative to the neuron's mean |a_j| on the target replies (c_j in the report). Item score = the max
     over the neurons.
  4. Preservation set = ALL status questions + the corpus items (shipped or general) with score >= MINE_MIN (shipped
     prompts without a reply get the original model's greedy reply, 256 tokens), then the strict hold-out filter (holdout.py) on prompt AND reply. The F items (the unrelated-text measure) are never
     in the corpus.
"""
import argparse
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent))
import holdout  # noqa: E402
import lib  # noqa: E402
import make_edit as M  # noqa: E402
from forget import encode  # noqa: E402
from p1lib import decoder_layers, load_model, resolve  # noqa: E402

MINE_MIN = 0.25   # item score threshold: mean activity >= 25% of the neuron's mean activity on the target replies


def replies(model, tok):
    f = HERE / "data/original_replies.json"
    if f.exists():
        return json.loads(f.read_text())
    st = lib.load_json(HERE / "data/status_prompts.json")
    ge = lib.load_json(HERE / "data/general_prompts.json")
    rows = lib.generate_batched(model, tok, [x["q"] for x in st] + ge, 256)
    out = [dict(source="status question", problem=x["problem"], q=x["q"], r=r["reply"], capped=r["capped"])
           for x, r in zip(st, rows)]
    out += [dict(source="general prompt (generated reply)", q=q, r=r["reply"], capped=r["capped"])
            for q, r in zip(ge, rows[len(st):])]
    f.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    return out


@torch.no_grad()
def item_scores(model, tok, items, rows):
    layers = decoder_layers(model)
    by_layer = {}
    for j, r in enumerate(rows):
        by_layer.setdefault(r["layer"], []).append((j, r["neuron"]))
    c = torch.tensor([r["c"] for r in rows])
    cur, span = torch.zeros(len(rows)), [0]
    hs = []
    for l, js in by_layer.items():
        idx = [i for _, i in js]
        pos = [j for j, _ in js]

        def f(mod, inp, idx=idx, pos=pos):
            cur[pos] = inp[0][0, span[0]:, idx].float().abs().mean(0).cpu()
        hs.append(layers[l].mlp.down_proj.register_forward_pre_hook(f))
    out = []
    for it in items:
        ids, lo = encode(tok, it, model.device)
        span[0] = lo
        model(input_ids=ids[:, :1024])
        rel = cur / c
        out.append(dict(score=float(rel.max()), per_neuron=[round(float(x), 4) for x in rel]))
    for h in hs:
        h.remove()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--neurons", required=True, help="report.json of make_edit.py (the chosen neurons, with c)")
    ap.add_argument("--name", required=True)
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    rows = json.loads(Path(a.neurons).read_text())["neurons"]
    model, tok = load_model(*resolve("E2B")[:2])
    gen = replies(model, tok)
    shipped = [dict(source=src, q=q, r=r or "") if raw is None else dict(source=src, raw=raw)
               for src, raw, q, r in M.card_corpus()]
    forget = {json.dumps(x, sort_keys=True) for x in
              ({k: v for k, v in it.items() if k in ("q", "r", "raw")} for it in lib.load_json(M.FORGET))}
    assert not any(json.dumps({k: v for k, v in it.items() if k in ("q", "r", "raw")}, sort_keys=True) in forget
                   for it in shipped), "an F item is in the corpus"
    corpus = gen + shipped
    sc = item_scores(model, tok, corpus, rows)
    for it, s in zip(corpus, sc):
        it.update(s)
    mined = [it for it in corpus if it["source"] == "status question" or it["score"] >= MINE_MIN]
    empty = [it for it in mined if it.get("r") == ""]   # shipped prompts without a reply: the original model's reply
    f = HERE / f"data/original_replies_mined_{a.name}.json"
    cache = json.loads(f.read_text()) if f.exists() else {}
    todo = [it["q"] for it in empty if it["q"] not in cache]
    for q, r in zip(todo, lib.generate_batched(model, tok, todo, 256)):
        cache[q] = r["reply"]
    f.write_text(json.dumps(cache, indent=1, ensure_ascii=False))
    for it in empty:
        it["r"], it["reply"] = cache[it["q"]], "generated by the original model"
    keep, removed = [], []
    for it in mined:
        (removed if holdout.check([it]) else keep).append(it)
    keep_items = [{k: (round(it[k], 3) if k == "score" else it[k]) for k in ("q", "r", "raw", "source", "reply", "score")
                   if k in it} for it in keep]
    (HERE / f"data/preserve_{a.name}.json").write_text(json.dumps(keep_items, indent=1, ensure_ascii=False))
    by_src = {}
    for it in corpus:
        d = by_src.setdefault(it["source"], dict(n=0, mined=0, removed=0, kept=0, score_max=0.0))
        d["n"] += 1
        d["score_max"] = max(d["score_max"], round(it["score"], 3))
    for it in keep + removed:
        by_src[it["source"]]["mined"] += 1
    for it in removed:
        by_src[it["source"]]["removed"] += 1
    for it in keep:
        by_src[it["source"]]["kept"] += 1
    top = sorted((it for it in corpus if it["source"] != "status question"), key=lambda x: -x["score"])[:40]
    stats = dict(neurons=[f"{r['layer']}:{r['neuron']}" for r in rows], mine_min=MINE_MIN, n_corpus=len(corpus),
                 n_preserve=len(keep_items), by_source=by_src,
                 template_overlap=holdout.template_overlap(keep_items),
                 removed=[dict(source=it["source"], hits=holdout.hits(holdout.item_text(it)),
                               text=holdout.item_text(it)[:160]) for it in removed],
                 top_corpus=[dict(source=it["source"], score=round(it["score"], 3),
                                  text=holdout.item_text(it)[:200]) for it in top])
    (HERE / f"data/mine_{a.name}.json").write_text(json.dumps(stats, indent=1, ensure_ascii=False))
    print(json.dumps({k: stats[k] for k in ("n_corpus", "n_preserve", "by_source", "template_overlap")}, indent=1))


if __name__ == "__main__":
    main()
