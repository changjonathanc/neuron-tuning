"""Experiment 12: choose K = 8 neurons with the tool rule or with a short group-lasso phase (rules: NOTES.md).

    tool   make_edit.select_neurons (activity filter + ||G_j|| / g_j), timed with an EMPTY background cache
    lasso  phase 1: dW on the down_proj of all layers (--layers all) or of one layer (--layers 15), AdamColProx of
           experiment 07 (Adam with one second moment per column + the exact prox step of lam * sum_j ||dW_j||),
           the task loss only, LR 1e-3, 100 steps. lam = lam_N: N columns of the norm of the mean initial gradient
           (64 fixed training items) are larger than lam. Selection: the 8 columns with the largest ||dW_j||.

Both write out/sel_<task>_<name>.json: the 8 neurons (fields of make_edit.py report.json, so `make_edit.py make
--neurons` can read the file), the tool statistics c, g, S of each neuron and their ranks, the cost in GPU seconds,
and (lasso) the curve of the phase. Also out/stats_<task>.json (once per task): a 2-D histogram of all neurons on
(pct of c in its layer, pct of S in its layer) for the figure.

    python experiments/12_lasso_selector/select12.py --task casual --method tool
    python experiments/12_lasso_selector/select12.py --task pyhints --method lasso --layers all
"""
import argparse
import json
import random
import sys
import tempfile
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE.parent / "07_preservation_selection"))
import make_edit as ME  # noqa: E402
from common07 import AdamColProx, DenseDelta  # noqa: E402
from p1lib import decoder_layers, load_data, load_model, resolve  # noqa: E402
import train_gains as TG  # noqa: E402

OUT = HERE / "out"
RESULTS = REPO / "results/lasso12"
PYH_TEACHER = RESULTS / "pyhints_teacher_replies.json"   # copy of the demo's teacher_replies.json (hash checked)
PYH_TRAIN_HASH = "c1e5409768bed7db"                        # demos/python_type_hints/out/reports/s0.json
K = 8


def task_items(task):
    """Training items in the format of make_edit.py ({q, r, sys, ended, kind}) and the batch size of the task."""
    if task == "casual":
        return load_data("full")[0], 4
    train, _ = json.loads(PYH_TEACHER.read_text())
    h = ME.sha([(d["q"], d["r"]) for d in train])
    assert h == PYH_TRAIN_HASH, f"teacher replies differ from the demo's ({h})"
    return train, 8


def subset(train):
    return random.Random(0).sample(train, min(64, len(train)))   # the fixed subset of make_edit.select_neurons


class Multi:
    """DenseDelta on several layers, one switch."""

    def __init__(self, model, layers):
        self.d = {l: DenseDelta(model, l) for l in layers}

    @property
    def on(self):
        return next(iter(self.d.values())).on

    @on.setter
    def on(self, v):
        for x in self.d.values():
            x.on = v

    def remove(self):
        for x in self.d.values():
            x.remove()


def tool_stats(model, tok, layers, train):
    """c, g (make_edit statistics; the background is the default of make_edit) for all neurons."""
    sub = subset(train)
    seqs = []
    for d in sub:
        ids, _, [(lo, hi)] = TG.build(tok, [d], False, model.device)
        seqs.append((ids, lo, hi))
    c = ME.mean_abs_acts(model, tok, layers, seqs)
    g = ME.background_acts(model, tok, layers, ME.BACKGROUND)
    return c, g


def describe(neurons, c, g):
    """Per neuron: c, g, S = g / c, pct_S and pct_c in its layer (0 = most selective / most active), the rank of S
    over all neurons, and whether it passes the tool filter."""
    allS = torch.cat([g[l] / c[l].clamp_min(1e-12) for l in range(len(c))])
    rows = []
    for l, i in neurons:
        S = g[l] / c[l].clamp_min(1e-12)
        s = float(S[i])
        rows.append(dict(layer=l, neuron=i, c=float(c[l][i]), g=float(g[l][i]), ratio=s,
                         pct_S=float((S < S[i]).float().mean()), pct_c=float((c[l] > c[l][i]).float().mean()),
                         rank_S_all=int((allS < s).sum()), n_all=len(allS),
                         passes_filter=bool(c[l][i] >= torch.quantile(c[l].double(), 0.9).float() and s <= 0.2)))
    return rows


def hist2d(c, g, bins=50):
    """Counts of all neurons on (pct_c, pct_S) in their layer (both 0 = top), bins x bins."""
    H = torch.zeros(bins, bins, dtype=torch.long)
    for l in range(len(c)):
        S = g[l] / c[l].clamp_min(1e-12)
        n = len(S)
        pc = torch.empty(n)
        pc[torch.argsort(c[l], descending=True)] = torch.arange(n, dtype=torch.float) / n
        ps = torch.empty(n)
        ps[torch.argsort(S)] = torch.arange(n, dtype=torch.float) / n
        ix = (pc * bins).long().clamp(max=bins - 1) * bins + (ps * bins).long().clamp(max=bins - 1)
        H += torch.bincount(ix, minlength=bins * bins).view(bins, bins)
    return H.tolist()


def lasso(model, tok, layers, train, batch, which, n_lam, steps, lr, seed, lam_scale=1.0):
    t0 = time.time()
    torch.cuda.synchronize()
    edit_layers = list(range(len(layers))) if which == "all" else [int(which)]
    ad = Multi(model, edit_layers)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    # calibration: the norm of the mean gradient at dW = 0 over the 64 fixed items
    sub = subset(train)
    for d in sub:
        (ME.item_loss(model, tok, d, ad, "system") / len(sub)).backward()
    gn = torch.cat([x.dW.grad.norm(dim=0) for x in ad.d.values()])
    for x in ad.d.values():
        x.dW.grad = None
    srt = torch.sort(gn, descending=True).values
    lam = float((srt[n_lam - 1] + srt[n_lam]) / 2) * lam_scale
    torch.cuda.synchronize()
    t_cal = time.time() - t0
    print(f"calibration {t_cal:.0f}s: lam_{n_lam} = {lam:.4g} (x{lam_scale})", flush=True)
    opts = {l: AdamColProx(x.dW, lr, lam) for l, x in ad.d.items()}
    random.seed(seed)
    torch.manual_seed(seed)
    curve = []
    for step in range(steps):
        tot = 0.0
        for d in random.sample(train, batch):
            loss = ME.item_loss(model, tok, d, ad, "system") / batch
            loss.backward()
            tot += float(loss)
        for o in opts.values():
            o.step()
        if step % 10 == 0 or step == steps - 1:
            nn = sum(int((x.dW.norm(dim=0) > 0).sum()) for x in ad.d.values())
            curve.append([step, tot, nn])
            print(f"step {step} loss {tot:.4f} nnz {nn}  {time.time() - t0:.0f}s", flush=True)
    torch.cuda.synchronize()
    secs = time.time() - t0
    model.eval()
    model.gradient_checkpointing_disable()
    norms = [(float(n), l, j) for l, x in ad.d.items() for j, n in enumerate(x.dW.detach().norm(dim=0).tolist()) if n > 0]
    norms.sort(reverse=True)
    init_rank = torch.argsort(gn, descending=True).tolist()
    offs = [0]
    for l in edit_layers:
        offs.append(offs[-1] + layers[l].mlp.down_proj.weight.shape[1])
    flat = {l: offs[k] for k, l in enumerate(edit_layers)}
    pos = {v: r for r, v in enumerate(init_rank)}
    top = [dict(layer=l, neuron=j, norm=n, init_grad=float(gn[flat[l] + j]), init_grad_rank=pos[flat[l] + j])
           for n, l, j in norms[:64]]
    ad.remove()
    del opts, ad
    torch.cuda.empty_cache()
    return dict(layers=which, n_lam=n_lam, lam=lam, lam_scale=lam_scale, lr=lr, steps=steps, batch=batch, seed=seed,
                nnz=len(norms), curve=curve, top64=top, calib_seconds=t_cal, seconds=secs,
                frob=sum(n * n for n, _, _ in norms) ** 0.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["casual", "pyhints"], required=True)
    ap.add_argument("--method", choices=["tool", "lasso"], required=True)
    ap.add_argument("--layers", default="all", help="(lasso) all | one layer index")
    ap.add_argument("--n-lam", type=int, default=2048)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", help="default: tool | lasso_all | lasso_L15")
    a = ap.parse_args()
    name = a.name or ("tool" if a.method == "tool" else f"lasso_{'all' if a.layers == 'all' else 'L' + a.layers}")
    out = OUT / f"sel_{a.task}_{name}.json"
    OUT.mkdir(exist_ok=True)
    torch.manual_seed(0)
    model, tok = load_model(*resolve("E2B")[:2])
    tok.eot_text = TG.end_of_turn(tok)
    layers = decoder_layers(model)
    model.model.language_model.embed_tokens.register_forward_hook(
        lambda mod, i, o: o.requires_grad_(True) if torch.is_grad_enabled() else o)
    train, batch = task_items(a.task)
    res = dict(task=a.task, method=a.method, name=name, k=K, gpu=torch.cuda.get_device_name())
    torch.cuda.reset_peak_memory_stats()
    if a.method == "tool":
        ME.CACHE = Path(tempfile.mkdtemp())    # an empty background cache: the full statistics pass is timed
        torch.cuda.synchronize()
        t0 = time.time()
        rows, n_el = ME.select_neurons(model, tok, layers, train, "system", ME.BACKGROUND, K, 0.2)
        torch.cuda.synchronize()
        res.update(seconds=time.time() - t0, n_eligible=n_el)
        neurons = [(r["layer"], r["neuron"]) for r in rows]
        extra = {(r["layer"], r["neuron"]): dict(grad=r["grad"], score=r["score"]) for r in rows}
    else:
        tries = []
        for scale in (1.0, 0.5, 0.25):            # fallback rule (NOTES.md): halve lam if < 8 columns are non-zero
            ph = lasso(model, tok, layers, train, batch, a.layers, a.n_lam, a.steps, a.lr, a.seed, scale)
            tries.append({k: v for k, v in ph.items() if k != "top64"} | dict(top8=ph["top64"][:8]))
            if ph["nnz"] >= K:
                break
        res.update(phase1=ph, tries=tries, seconds=sum(t["seconds"] for t in tries), n_eligible=None)
        neurons = [(t["layer"], t["neuron"]) for t in ph["top64"][:K]]
        extra = {(t["layer"], t["neuron"]): dict(grad=t["init_grad"], score=t["norm"]) for t in ph["top64"][:K]}
    res["peak_mem_gb"] = torch.cuda.max_memory_allocated() / 2**30
    stats_f = RESULTS / f"stats_{a.task}.pt"
    if stats_f.exists():
        c, g = torch.load(stats_f)
    else:
        with torch.no_grad():
            c, g = tool_stats(model, tok, layers, train)
        RESULTS.mkdir(parents=True, exist_ok=True)
        torch.save((c, g), stats_f)
    if not (OUT / f"stats_{a.task}.json").exists():
        (OUT / f"stats_{a.task}.json").write_text(json.dumps(dict(
            bins=50, axes="rows: pct of c in the layer (0 = most active); cols: pct of S in the layer (0 = most selective)",
            hist=hist2d(c, g), n=sum(len(x) for x in c), layer_width=[len(x) for x in c])))
    rows = describe(neurons, c, g)
    for r in rows:
        r.update(extra[(r["layer"], r["neuron"])])
    res["neurons"] = rows
    out.write_text(json.dumps(res, indent=1))
    for r in rows:
        print(f"  {r['layer']:2d}:{r['neuron']:<5d} c {r['c']:.3f} S {r['ratio']:.3f} pct_S {r['pct_S']:.4f} "
              f"pct_c {r['pct_c']:.4f} filter {r['passes_filter']}")
    print(f"saved {out}  selection {res['seconds']:.0f}s  peak {res['peak_mem_gb']:.1f} GB")


if __name__ == "__main__":
    main()
