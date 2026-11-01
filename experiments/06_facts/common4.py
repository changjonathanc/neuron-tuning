"""E4 shared code: data, arms, the evaluation (bf16, unbatched)."""
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
A = REPO / "experiments"
for p in (A / "lib",):
    sys.path.insert(0, str(p))
import train1b  # noqa: E402  (puts experiments/lib, demo and demos/talk_like_a_person/reproduce on sys.path)
from train1b import TG  # noqa: E402
import p1lib  # noqa: E402
from common import chat_prompt, generate_one  # noqa: E402
sys.path.insert(0, str(HERE))
from make_facts import any_value, correct  # noqa: E402

MAXTOK = 48


def load():
    D = json.loads((HERE / "data/facts.json").read_text())
    G = json.loads((HERE / "runs/gens.json").read_text())
    return D, G


def train_items(D, G):
    """training items in the TG.build format; only teacher replies that are correct (exact match)."""
    out = []
    for it, g in zip(D["train"], G["teacher/train"]):
        if correct(g["r"], it["attr"], it["value"]):
            out.append(dict(q=it["q"], r=g["r"], sys=D["system"], kind="id", ended=g["ended"]))
    return out


@torch.no_grad()
def logp_reply(model, tok, q, r, system):
    """log-probs over the reply tokens (+ end of turn), teacher-forced; positions = the scored positions."""
    p = chat_prompt(tok, q, system)
    ids = tok(p + r, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
    n0 = len(tok(p, add_special_tokens=False).input_ids)
    return torch.log_softmax(model(input_ids=ids, logits_to_keep=ids.shape[1] - n0 + 1).logits[0, :-1].float(), -1)


def cached(path, fn):
    """list of fp32 log-prob tensors -> fp16 CPU copies, saved once to path (runs/, git-ignored)."""
    if Path(path).exists():
        return torch.load(path)
    with torch.no_grad():
        T = [x.half().cpu() for x in fn()]
    torch.save(T, str(path) + ".tmp")
    Path(str(path) + ".tmp").rename(path)
    return T


def kl(t, s):
    return float((t.exp() * (t - s)).sum(-1).mean())


@torch.no_grad()
def evaluate(model, tok, ad, D, G):
    """TKL and accuracy on the 160 held-out questions; KL and change rate on the 80 neighbour prompts."""
    model.eval()
    res = dict(tkl=[], acc=[], replies=[], nkl=[], nchange=[], nvalue=[], nreplies=[])
    if ad is not None:
        ad.on = False
    TE = cached(HERE / "runs/teacher_eval.pt", lambda: [
        logp_reply(model, tok, it["q"], g["r"] + tok.eot_text, D["system"]) for it, g in zip(D["eval"], G["teacher/eval"])])
    TN = cached(HERE / "runs/orig_neigh.pt", lambda: [
        logp_reply(model, tok, it["q"], g["r"], None) for it, g in zip(D["neighbours"], G["orig/neigh"])])
    if ad is not None:
        ad.on = True
    for it, g, t in zip(D["eval"], G["teacher/eval"], TE):
        t = t.to(model.device).float()
        s = logp_reply(model, tok, it["q"], g["r"] + tok.eot_text, None)
        res["tkl"].append(kl(t, s))
        r = generate_one(model, tok, it["q"], None, MAXTOK)[0]
        res["replies"].append(r)
        res["acc"].append(int(correct(r, it["attr"], it["value"])))
    for it, g, t in zip(D["neighbours"], G["orig/neigh"], TN):
        t = t.to(model.device).float()
        s = logp_reply(model, tok, it["q"], g["r"], None)
        res["nkl"].append(kl(t, s))
        r = generate_one(model, tok, it["q"], None, MAXTOK)[0]
        res["nreplies"].append(r)
        res["nchange"].append(int(r.strip() != g["r"].strip()))
        res["nvalue"].append(int(it["kind"] == "fake" and any_value(r, it["attr"])))
    kinds = [it["kind"] for it in D["neighbours"]]
    mean = lambda v: sum(v) / len(v)  # noqa: E731
    res["summary"] = dict(
        tkl=mean(res["tkl"]), acc=mean(res["acc"]),
        nkl_real=mean([x for x, k in zip(res["nkl"], kinds) if k == "real"]),
        nkl_fake=mean([x for x, k in zip(res["nkl"], kinds) if k == "fake"]),
        nchange_real=mean([x for x, k in zip(res["nchange"], kinds) if k == "real"]),
        nchange_fake=mean([x for x, k in zip(res["nchange"], kinds) if k == "fake"]),
        nvalue_fake=mean([x for x, k in zip(res["nvalue"], kinds) if k == "fake"]))
    return res
