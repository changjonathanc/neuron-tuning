"""Experiment 14, Step 3: F per domain of a trained patch with some of its columns removed (no new training; GPU, fp32).

Question: does the Japanese part of F at K >= 16 come from the early-layer neurons (layers < 10) that the rule adds
from K = 16? For each patch: F of the full patch and of the patch without its early-layer columns, in one fp32 pass
(the same items and code as `make_edit.py --f32`).

    python experiments/14_how_many_neurons/ablate14.py results/how_many_neurons/k16/columns_lr0.01.safetensors \
        results/how_many_neurons/k64/columns_lr0.01.safetensors          -> out/ablate.json
"""
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
import make_edit as ME  # noqa: E402

EARLY = 10


def main():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    model, tok = ME.load("E2B", lowmem=True, dtype=torch.float32)
    edits, info = {}, {}
    for p in sys.argv[1:]:
        delta, _ = ME.load_patch(p)
        name = f"{Path(p).parent.name}_{Path(p).stem}"
        late = {l: v for l, v in delta.items() if l >= EARLY}
        info[name] = dict(patch=str(Path(p).relative_to(REPO)) if Path(p).is_absolute() else p,
                          removed=[(l, n) for l, (ns, _) in delta.items() if l < EARLY for n in ns])
        edits[name] = ME.Patch(model, delta)
        edits[name + "_no_early"] = ME.Patch(model, late)
    with torch.no_grad():
        F = ME.unrelated_change(model, tok, edits)
    out = dict(early_layer_below=EARLY, patches=info,
               F={n: dict(macro=f["macro"], by_domain=f["by_domain"]) for n, f in F.items()})
    (HERE / "out/ablate.json").write_text(json.dumps(out, indent=1))
    for n, f in F.items():
        print(f"{n:36s} F {f['macro']:.5f}  " + " ".join(f"{d} {v:.4f}" for d, v in f["by_domain"].items()))


if __name__ == "__main__":
    main()
