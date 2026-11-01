"""E1 pass 2: the Phase 1b forgetting measure F (experiments/lib/forget.py: same items, same KL) with the model AND the
edit in fp32 (weights cast from bf16, which is exact; TF32 off; eager attention). This removes the bf16 floor of F.

Adapters are processed in chunks; the result file is written after each chunk (resumable).

    python f32forget.py --adapters a.pt b.pt ... --out out/f32.json [--chunk 16]
    python f32forget.py --make-diag      (scaled copies for the validation -> runs/diag/*.pt)
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
A = REPO / "experiments"
for p in ("lib",):
    sys.path.insert(0, str(A / p))

DIAG = [("sel8_lr0.03_s2", A / "01_locality_curves/runs/final/sel8_lr0.03_s2.pt"),
        ("srand8_lr0.03_s2", A / "01_locality_curves/runs/final/srand8_lr0.03_s2.pt")]
SCALES = [0.0, 0.01, 0.1, 1.0]


def make_diag():
    (HERE / "runs/diag").mkdir(parents=True, exist_ok=True)
    for n, p in DIAG:
        st = torch.load(p, map_location="cpu", weights_only=False)
        for s in SCALES:
            torch.save(dict(st, params={k: v * s for k, v in st["params"].items()}), HERE / f"runs/diag/{n}_x{s:g}.pt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapters", nargs="*", default=[])
    ap.add_argument("--out", default=str(HERE / "out/f32.json"))
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--make-diag", action="store_true")
    a = ap.parse_args()
    if a.make_diag:
        make_diag()
        return
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.set_grad_enabled(False)
    from p1lib import load_adapter, load_model, resolve
    from forget import encode, logp
    out = Path(a.out)
    res = json.loads(out.read_text()) if out.exists() else {}
    todo = [n for n in a.adapters if n not in res]
    if not todo:
        print("already done")
        return
    items = json.loads((A / "lib/data/forget_set.json").read_text())
    doms = sorted({it["domain"] for it in items})
    model, tok = load_model(*resolve("E2B")[:2], lowmem=True, dtype=torch.float32)
    assert next(model.parameters()).dtype == torch.float32
    enc = [encode(tok, it, model.device) for it in items]
    for c in range(0, len(todo), a.chunk):
        names = todo[c:c + a.chunk]
        t0 = time.time()
        ads = {}
        for n in names:
            ads[n] = load_adapter(model, REPO / n if not Path(n).is_absolute() else n)
            ads[n].on = False
        kls = {n: [] for n in names}
        for ids, lo in enc:
            lp0 = logp(model, ids, lo)
            p0 = lp0.exp()
            for n, ad in ads.items():
                ad.on = True
                lp = logp(model, ids, lo)
                ad.on = False
                kls[n].append(float((p0 * (lp0 - lp)).sum(-1).mean()))
        for n, ad in ads.items():
            ad.remove()
            per = {d: [x for x, it in zip(kls[n], items) if it["domain"] == d] for d in doms}
            bd = {d: sum(v) / len(v) for d, v in per.items()}
            res[n] = dict(macro=sum(bd.values()) / len(doms), by_domain=bd, per_item=kls[n])
            print(f"{n}: F32 {res[n]['macro']:.3e}", flush=True)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res))
        print(f"chunk {c // a.chunk}: {len(names)} adapters, {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
