"""E1 predictors of the locality of an edit (definitions: NOTES.md). Needs runs/cache_scored.pt and runs/downW.pt
(cache.py --scored; primary, amendment 1). OFF tokens = the positions that forget.py scores.

  S_pre  selectivity of the edited units: geometric mean over the units of (activity on OFF) / (activity on the
         target replies). Neuron edits: per-neuron mean |a| on all tokens (pre-training; only WHICH neurons).
         LoRA: mean |A a| on the token subsample (post-training: needs A).  Higher = less selective.
  D_off  direct write on off-target tokens: macro mean over the 7 OFF domains of the per-token sum over the edited
         layers of ||dW_l a_l(x)||_2 (subsample). D_on: the same on the target subsample. R = D_off / D_on.

Edit types (an edit file of experiments/lib/p1lib.py or experiments/lib/p2lib.py, or a gains JSON of edit/):
  columns  {l.g, l.u} on spec {l: [neurons]}: dW[:, i] = g_i W_i + rms(W_i) u_i
  gains    {l.g}: dW[:, i] = g_i W_i
  lora     {l.mlp.down_proj.A, .B}: dW = B A  (only down_proj is supported; other modules need their own inputs)

    python predict.py EDIT.pt [--target casual|weather|lower|recipe]     (one edit; prints JSON)
    python predict.py --all [--acts1c-spans]     (all Phase 1c / 2b runs -> out/pred.json | out/pred_acts1c.json)

E2/E3: `from predict import Predictor; P = Predictor(); P(path_or_state, target)` returns the dict.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
A = REPO / "experiments"
RULES = ["weather", "lower", "recipe"]


def terms(st, downW):
    """[(layer, kind, a)] with kind 'cols': (idx, D hidden x k) or 'lora': (A r x in, B out x r); fp32."""
    out = []
    if st["kind"] in ("columns", "gains"):
        for l, idx in st["spec"].items():
            l = int(l)
            bw = (st.get("base_W") or {}).get(str(l))
            W = (bw if bw is not None else downW[l][:, idx]).float()
            D = W * st["params"][f"{l}.g"].float()
            if st["kind"] == "columns":
                D = D + st["params"][f"{l}.u"].float() * W.pow(2).mean(0).sqrt()
            out.append((l, "cols", (torch.tensor(idx), D)))
    elif st["kind"] == "lora":
        for l, name, r in st["spec"]:
            assert name == "mlp.down_proj", f"direct write from down_proj inputs only, not {name}"
            out.append((int(l), "lora", (st["params"][f"{l}.{name}.A"].float(), st["params"][f"{l}.{name}.B"].float())))
    else:
        raise ValueError(st["kind"])
    return out


def load_state(path):
    path = str(path)
    if path.endswith(".json"):                                  # gains edit of the repo format
        sys.path.insert(0, str(REPO / "demos/talk_like_a_person"))
        from edit.load_edit import load_edit
        by_layer, _ = load_edit(path)
        return dict(kind="gains", spec={l: list(n) for l, (n, _) in by_layer.items()},
                    params={f"{l}.g": torch.tensor(g) for l, (_, g) in by_layer.items()})
    return torch.load(path, map_location="cpu", weights_only=False)


class Predictor:
    def __init__(self, cache=HERE / "runs/cache_scored.pt", downW=HERE / "runs/downW.pt"):
        C = torch.load(cache, weights_only=False)
        self.acts, self.sub, self.doms = C["acts"], C["sub"], C["doms"]
        dom = C["sub_dom"]
        self.dom_idx = {d: torch.tensor([j for j, x in enumerate(dom) if x == d]) for d in self.doms}
        self.downW = torch.load(downW, weights_only=False)

    def _macro(self, v, half=None):
        """macro mean over OFF domains of a per-token vector v (OFF subsample); half = 0/1: even/odd tokens."""
        m = []
        for d, ix in self.dom_idx.items():
            ix = ix if half is None else ix[half::2]
            m.append(float(v[ix].mean()))
        return sum(m) / len(m)

    def __call__(self, edit, target="casual"):
        st = load_state(edit) if isinstance(edit, (str, Path)) else edit
        trm = terms(st, self.downW)
        d = {s: 0.0 for s in ("off", target)}
        lnS, kindS = [], "pre" if st["kind"] != "lora" else "post"
        units = 0
        for l, kind, a in trm:
            for s in d:
                x = self.sub[s][l].float()
                if kind == "cols":
                    idx, D = a
                    d[s] = d[s] + (x[:, idx] @ D.t()).norm(dim=1)
                else:
                    Am, B = a
                    d[s] = d[s] + ((x @ Am.t()) @ B.t()).norm(dim=1)
            if kind == "cols":
                idx = a[0]
                off, on = self.acts["off"][l][idx], self.acts[target][l][idx]
                lnS += (off.clamp_min(1e-12).log() - on.clamp_min(1e-12).log()).tolist()
                units += len(idx)
            else:
                Am = a[0]
                z_off = (self.sub["off"][l].float() @ Am.t()).abs()
                z_on = (self.sub[target][l].float() @ Am.t()).abs().mean(0)
                for k in range(Am.shape[0]):
                    lnS.append(math.log(self._macro(z_off[:, k]) / float(z_on[k])))
                units += Am.shape[0]
        D_off, D_on = self._macro(d["off"]), float(d[target].mean())
        return dict(S=math.exp(sum(lnS) / len(lnS)), S_kind=kindS, n_units=units, D_off=D_off, D_on=D_on,
                    R=D_off / D_on, D_off_half=[self._macro(d["off"], 0), self._macro(d["off"], 1)],
                    D_off_by_domain={dm: float(d["off"][ix].mean()) for dm, ix in self.dom_idx.items()},
                    D2_off=self._macro(d["off"] ** 2), target=target)


def old_key(d, pt):
    """stored result files use the adapter paths (experiments/NN_name/runs/..., relative to the repository root) as keys."""
    return d[pt]


def all_runs():
    """[(name, edit path, target, meta)] for the 111 Phase 1c and 99 Phase 2b runs, with stored TKL and bf16 F."""
    sys.path.insert(0, str(A / "01_locality_curves"))
    import gen_jobs as G
    import numpy as np
    out = []
    for fam, lrs in G.grid(["sel128:0.001", "dense128:0.3"]).items():
        for lr in lrs:
            for s in G.SEEDS:
                pt, ptk, pfg, _ = G.paths(fam, lr, s)
                tk, fg = REPO / ptk, REPO / pfg
                if not ((REPO / pt).exists() and tk.exists() and fg.exists()):
                    continue
                t = old_key(json.loads(tk.read_text()), pt)
                msgs = sorted(set(t["msg"]))
                tkl = float(np.mean([np.mean([x for x, i in zip(t["per_reply"], t["msg"]) if i == j]) for j in msgs]))
                f = old_key(json.loads(fg.read_text()), pt)
                out.append((G.name(fam, lr, s), REPO / pt, "casual",
                            dict(phase="1c", fam=fam, budget="13k" if fam in ("sel8", "srand8", "dense8", "lorad") else "200k",
                                 lr=lr, seed=s, tkl=tkl, F1=f["macro"], F1_dom=f["by_domain"])))
    E = {}
    for f in sorted((A / "02_conditional_rules/runs/eval").glob("curve_*.json")):
        E.update(json.loads(f.read_text()))
    for k, v in E.items():
        r, m, lr, s = k.split("|")
        p = A / f"02_conditional_rules/runs/curve/{r}_{m}_lr{lr}_s{s[1:]}.pt"
        assert p.exists(), p
        out.append((f"{r}_{m}_lr{lr}_{s}", p, r, dict(phase="2b", fam=m, rule=r, budget="13k", lr=float(lr), seed=int(s[1:]),
                                                     tkl=float(np.mean(v["rules"][r]["tkl_trig"])),
                                                     F1=v["forget"]["macro"], F1_dom=v["forget"]["by_domain"])))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("edit", nargs="?")
    ap.add_argument("--target", default="casual")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--acts1c-spans", action="store_true", help="OFF = acts1c.py spans (first version; not primary)")
    a = ap.parse_args()
    P = Predictor(HERE / ("runs/cache.pt" if a.acts1c_spans else "runs/cache_scored.pt"))
    if not a.all:
        print(json.dumps(P(a.edit, a.target), indent=1))
        return
    res = {}
    for name, path, target, meta in all_runs():
        res[name] = dict(meta, path=str(path.relative_to(REPO)), **P(path, target))
    (HERE / "out").mkdir(exist_ok=True)
    (HERE / ("out/pred_acts1c.json" if a.acts1c_spans else "out/pred.json")).write_text(json.dumps(res, indent=1))
    print(len(res), "runs -> out/pred.json")


if __name__ == "__main__":
    main()
