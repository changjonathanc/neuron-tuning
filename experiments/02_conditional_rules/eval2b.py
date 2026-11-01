"""Phase 2b evaluation of several conditions (a condition = adapters that are on together; their deltas add).

For each rule's 60 evaluation pairs and each condition:
  - greedy replies (no system prompt, <= 128 new tokens, eager attention, left-padded batches) to the triggers and
    the controls -> hit (rule satisfied on a trigger) and fp (rule satisfied on a control);
  - kl_ctrl: per-token KL(original || condition) on the original's greedy replies to the controls (teacher-forced,
    unbatched);
  - tkl_trig: per-token KL(teacher || condition) on the teacher's greedy replies to the triggers (descriptive);
and the Phase 1b forgetting F (per-token KL(original || condition) on lib/data/forget_set.json).
With --with-original, the "original" condition is added; for it, kl_ctrl_teacher = KL(original || teacher) on the controls.

    python eval2b.py --conds conds.json --out runs/eval/x.json        # conds.json: {"name": [adapter.pt, ...]}
"""
import argparse
import json
import time
from pathlib import Path

import torch

from common2b import HERE, RULES, SYSTEMS, gen_file, generate_batched, load_model, pairs, resolve, score
from p2lib import load_adapter
from evalkl import FORGET, kl, logp_chat, logp_forget


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conds", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bs", type=int, default=20)
    ap.add_argument("--no-forget", action="store_true")
    ap.add_argument("--with-original", action="store_true")
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    out = Path(a.out)
    res = json.loads(out.read_text()) if out.exists() else {}
    conds = ({"original": []} if a.with_original else {}) | json.loads(Path(a.conds).read_text())
    todo = {n: p for n, p in conds.items() if n not in res}
    if not todo:
        print("already done:", out)
        return
    t0 = time.time()
    model, tok = load_model(*resolve("E2B")[:2])
    ads = {}
    for paths in todo.values():
        for p in paths:
            if p not in ads:
                ads[p] = load_adapter(model, p)
                ads[p].on = False

    def switch(paths, on):
        for p in paths:
            ads[p].on = on

    rows = {n: dict(adapters=p, rules={}) for n, p in todo.items()}
    G = {r: json.loads(gen_file(r, "eval").read_text()) for r in RULES}
    for r in RULES:
        P = pairs(r, "eval")
        for n, paths in todo.items():
            switch(paths, True)
            gt = generate_batched(model, tok, [p["trigger"] for p in P], None, a.bs)
            gc = generate_batched(model, tok, [p["control"] for p in P], None, a.bs)
            switch(paths, False)
            rows[n]["rules"][r] = dict(hit=[score(r, x["text"], x["ended"]) for x in gt],
                                       fp=[score(r, x["text"], x["ended"]) for x in gc],
                                       trig_text=[x["text"] for x in gt], ctrl_text=[x["text"] for x in gc],
                                       kl_ctrl=[], tkl_trig=[])
        for x in G[r]["original_control"]:
            ref = logp_chat(model, tok, x["q"], x["text"], None)
            for n, paths in todo.items():
                switch(paths, True)
                rows[n]["rules"][r]["kl_ctrl"].append(kl(ref, logp_chat(model, tok, x["q"], x["text"], None)))
                switch(paths, False)
            if "original" in todo:
                rows["original"]["rules"][r].setdefault("kl_ctrl_teacher", []).append(
                    kl(ref, logp_chat(model, tok, x["q"], x["text"], SYSTEMS[r])))
        for x in G[r]["teacher_trigger"]:
            ref = logp_chat(model, tok, x["q"], x["text"], SYSTEMS[r])
            for n, paths in todo.items():
                switch(paths, True)
                rows[n]["rules"][r]["tkl_trig"].append(kl(ref, logp_chat(model, tok, x["q"], x["text"], None)))
                switch(paths, False)
        for n in todo:
            d = rows[n]["rules"][r]
            print(f"{r} {n}: hit {sum(d['hit']) / 60:.3f} fp {sum(d['fp']) / 60:.3f} "
                  f"kl_ctrl {sum(d['kl_ctrl']) / 60:.4f} tkl {sum(d['tkl_trig']) / 60:.3f}  {time.time() - t0:.0f}s",
                  flush=True)
    if not a.no_forget:
        items = json.loads(FORGET.read_text())
        per = {n: [] for n in todo}
        for it in items:
            lp0 = logp_forget(model, tok, it)
            for n, paths in todo.items():
                switch(paths, True)
                per[n].append(kl(lp0, logp_forget(model, tok, it)))
                switch(paths, False)
        doms = sorted({it["domain"] for it in items})
        for n in todo:
            bd = {d: [x for x, it in zip(per[n], items) if it["domain"] == d] for d in doms}
            bd = {d: sum(v) / len(v) for d, v in bd.items()}
            rows[n]["forget"] = dict(per_item=per[n], domain=[it["domain"] for it in items], by_domain=bd,
                                     macro=sum(bd.values()) / len(bd))
            print(n, f"F {rows[n]['forget']['macro']:.4f}", flush=True)
    res.update(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, ensure_ascii=False))
    print("wrote", out, f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
