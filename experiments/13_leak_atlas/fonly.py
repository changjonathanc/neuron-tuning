"""F (change on unrelated text) of saved patches, when the F step of make_edit.py did not finish (GPU, fp32 as --f32).

  uv run experiments/13_leak_atlas/fonly.py --patches results/atlas/poem/train/columns_lr0.03.safetensors

Writes <patch dir>/F.json: {patch name: {macro, by_domain}}, read by outcomes.py when report.json has no F.
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from make_edit import load_any, load_model, resolve, unrelated_change  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patches", nargs="+", required=True)
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    model, tok = load_model(*resolve("E2B")[:2], lowmem=True, dtype=torch.float32)
    for p in map(Path, a.patches):
        e = load_any(model, p)
        F = unrelated_change(model, tok, {p.stem: e})[p.stem]
        e.remove()
        f = p.parent / "F.json"
        d = json.loads(f.read_text()) if f.exists() else {}
        d[p.stem] = dict(macro=F["macro"], by_domain=F["by_domain"], precision="fp32")
        f.write_text(json.dumps(d, indent=1))
        print(f"{p}: F {F['macro']:.5f}", flush=True)


if __name__ == "__main__":
    main()
