"""Phase 2b helpers: data of the rules, batched greedy generation (eager attention), paths. Adapters come from
`../lib/p2lib.py` (unchanged)."""
import json
import random
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "lib"))
from p2lib import ROOT, Adapter, METHODS, chat_prompt, decoder_layers, load_model, load_stack, merge_bases, resolve  # noqa
from rules import PAIRS_OF, SYSTEMS, score  # noqa: E402,F401

RULES = json.loads((HERE / "data/rules.json").read_text())["rules"]   # the 3 edits = the incremental order
MAX_NEW = 128


def pairs(rule, split):
    P = json.loads((HERE / f"data/pairs_{PAIRS_OF.get(rule, rule)}.json").read_text())
    return P[split]


def task_items():
    data = json.loads((ROOT / "demos/talk_like_a_person/reproduce/data/data_persona_E2B.json").read_text())
    return [d for d in data if d["kind"] == "pres"]


def gen_file(rule, split):
    return HERE / f"data/gen_{rule}_{split}.json"


def load_data(rule, split):
    """(trigger, control, task, dev_trigger, dev_control, dev_task) item lists (dicts q, r, sys, kind, ended).
    trigger: teacher replies that satisfy the rule; control: original replies (target = the original model)."""
    G = json.loads(gen_file(rule, "train").read_text())
    trig, ctrl = [], []
    for i, (t, c) in enumerate(zip(G["teacher_trigger"], G["original_control"])):
        if score(rule, t["text"], t["ended"]):
            trig.append(dict(i=i, q=t["q"], r=t["text"], ended=t["ended"], sys=SYSTEMS[rule], kind="id"))
        ctrl.append(dict(i=i, q=c["q"], r=c["text"], ended=c["ended"], sys=None, kind="pres"))
    task = task_items()
    if split == "full":
        return trig, ctrl, task, [], [], []
    rng = random.Random(1234)                       # Phase 1/2 rule: 20 dev pairs, then 30 dev task items
    dv, dt = set(rng.sample(range(160), 20)), set(rng.sample(range(len(task)), 30))
    sp = lambda L, dev: [d for d in L if (d["i"] in dv) == dev]
    return (sp(trig, False), sp(ctrl, False), [d for j, d in enumerate(task) if j not in dt],
            sp(trig, True), sp(ctrl, True), [d for j, d in enumerate(task) if j in dt])


def eos_ids(model):
    e = model.generation_config.eos_token_id
    return set(e if isinstance(e, list) else [e])


@torch.no_grad()
def generate_batched(model, tok, qs, system=None, bs=20, max_new=MAX_NEW):
    """Greedy replies in left-padded batches, eager attention. Returns [{q, text, n_tokens, ended}]."""
    tok.padding_side = "left"
    eos = eos_ids(model)
    out = []
    for j in range(0, len(qs), bs):
        texts = [chat_prompt(tok, q, system) for q in qs[j:j + bs]]
        b = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        o = model.generate(**b, max_new_tokens=max_new, do_sample=False)[:, b.input_ids.shape[1]:]
        for k, row in enumerate(o.tolist()):
            n = next((t + 1 for t, x in enumerate(row) if x in eos), None)
            ended = n is not None
            n = n or len(row)
            out.append(dict(q=qs[j + k], text=tok.decode(row[:n], skip_special_tokens=True), n_tokens=n, ended=ended))
    return out
