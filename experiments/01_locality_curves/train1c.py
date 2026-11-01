"""Phase 1c training. All arms except columns_selrand run experiments/lib/train1b.py UNCHANGED.

This file adds ONE arm, columns_selrand (the "selective-random" control): output columns of K neurons that are NOT
in the attribution candidate pool, chosen to MATCH the activity profile of the attribution top-K of the same seed.

  1. The attribution is run exactly as in train1b (select_attr: the same 48 items, the same global random state), but
     all 64 x 35 candidates are kept. Selected = the top K (identical to train1b's columns arm). Pool = all candidates.
  2. Activity (out/acts1c.pt, acts1c.py): c = mean |activation| on the 160 casual teacher replies, g = the same on the
     general-text reference set (data/ref_set.json, disjoint from the forgetting set). Feature x = (ln c, ln g).
  3. For each selected neuron (l, i), in attribution order: among the neurons of layer l that are not in the pool and
     not chosen yet, take the NEAREST to x(l, i) (Euclidean; NEAREST = 1 fixed after a matching check, see NOTES)
     (the global random state is not touched, so the batch order is the same as the
     selected arm of that seed).
  The trained edit is the columns adapter (Phase 1 p1lib) on these K neurons; everything else is train1b.

    python experiments/01_locality_curves/train1c.py --arm columns_selrand --k 8 --lr 3e-2 --seed 0 --split full --w-pres 0 --out ...
    python experiments/01_locality_curves/train1c.py --check --k 128 --seed 0     (selection + matching diagnostics only)
"""
import json
import math
import random
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "lib"))
import train1b  # noqa: E402
from train1b import TG  # noqa: E402

NEAREST = 1
INFO = {}


def match(sel, pool, seed, acts):
    """sel: [(score, l, i)] attribution top K; pool: set of (l, i). Returns chosen [(l, i)] and diagnostics."""
    rng = random.Random(10000 + seed)
    taken, chosen, rows = set(), [], []
    for _, l, i in sel:
        c, g = acts["casual"][l].double(), acts["ref"][l].double()
        x = torch.stack([c.clamp_min(1e-8).log(), g.clamp_min(1e-8).log()], 1)
        d = (x - x[i]).norm(dim=1)
        bad = [j for (ll, j) in pool if ll == l] + [j for (ll, j) in taken if ll == l]
        d[bad] = math.inf
        near = torch.topk(-d, NEAREST).indices.tolist()
        j = rng.choice(near)
        taken.add((l, j))
        chosen.append((l, j))
        rows.append(dict(layer=l, sel=i, ctrl=j, dist=float(d[j]),
                         sel_lnc=float(x[i, 0]), sel_lng=float(x[i, 1]), ctrl_lnc=float(x[j, 0]), ctrl_lng=float(x[j, 1])))
    return chosen, rows


def summarize(rows, acts):
    """Median percentile (within the layer) of casual, ref and (diagnostic) forget activity; ratio g/c."""
    def pct(v, idx):
        return float((v < v[idx]).float().mean())
    out = {}
    for who in ("sel", "ctrl"):
        P = {k: [] for k in ("casual", "ref", "forget")}
        R = {k: [] for k in ("ref", "forget")}
        for r in rows:
            l, n = r["layer"], r[who]
            for k in P:
                P[k].append(pct(acts[k][l], n))
            for k in R:
                R[k].append(float(acts[k][l][n] / acts["casual"][l][n]))
        med = lambda v: sorted(v)[len(v) // 2]  # noqa: E731
        out[who] = {f"pct_{k}": med(v) for k, v in P.items()} | {f"ratio_{k}_casual": med(v) for k, v in R.items()}
    out["median_dist"] = sorted(r["dist"] for r in rows)[len(rows) // 2]
    return out


def install(seed):
    acts = torch.load(HERE / "out/acts1c.pt")
    orig = TG.select_attr

    def patched(model, tok, layers, casual, k, **kw):
        full = orig(model, tok, layers, casual, 10 ** 9, **kw)
        sel = full[:k]
        chosen, rows = match(sel, {(l, i) for _, l, i in full}, seed, acts)
        INFO.update(attr_selection=[[l, i] for _, l, i in sel], pool=[[l, i] for _, l, i in full], matching=rows, summary=summarize(rows, acts))
        print("selrand matching:", json.dumps(INFO["summary"]), flush=True)
        return [(0.0, l, i) for l, i in chosen]
    TG.select_attr = patched


def main():
    argv = sys.argv[1:]
    if "--check" in argv:
        k = int(argv[argv.index("--k") + 1])
        seed = int(argv[argv.index("--seed") + 1])
        from p1lib import decoder_layers, load_data, load_model, resolve
        random.seed(seed)
        torch.manual_seed(seed)
        casual = load_data("full")[0]
        model, tok = load_model(*resolve("E2B")[:2])
        tok.eot_text = TG.end_of_turn(tok)
        model.model.language_model.embed_tokens.register_forward_hook(
            lambda mod, i, o: o.requires_grad_(True) if torch.is_grad_enabled() else o)
        install(seed)
        TG.select_attr(model, tok, decoder_layers(model), casual, k)
        Path(HERE / f"out/selrand_check_k{k}_s{seed}.json").write_text(json.dumps(INFO))
        return
    assert argv[argv.index("--arm") + 1] == "columns_selrand"
    seed = int(argv[argv.index("--seed") + 1]) if "--seed" in argv else 0
    out = Path(argv[argv.index("--out") + 1])
    if Path(str(out) + ".json").exists():
        print("already done:", out)
        return
    install(seed)
    argv[argv.index("--arm") + 1] = "columns"
    sys.argv = [sys.argv[0]] + argv
    train1b.main()
    f = Path(str(out) + ".json")
    res = json.loads(f.read_text())
    res.update(arm="columns_selrand", **INFO)
    f.write_text(json.dumps(res))


if __name__ == "__main__":
    main()
