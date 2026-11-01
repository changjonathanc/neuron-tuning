"""Per-neuron activity of layers 15 and 20, from the caches of experiments 01 and 04 (not in git), so that the analysis
runs on the CPU from git:

  c[j]  mean |a| on the 160 casual training replies, scored positions (01_locality_curves/out/acts1c.pt "casual")
  g[j]  mean |a| on the reference text, scored positions, macro over 7 domains (04 runs/acts_ref_scored.pt, v2)
  S[j] = g / c (the selectivity of 03/04: low = quiet off-target)

    python experiments/07_preservation_selection/sel07.py [--c FILE] [--g FILE]   -> out/selectivity.json
"""
import argparse
import json
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
LAYERS = [15, 20]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--c", default=str(HERE.parent / "01_locality_curves/out/acts1c.pt"))
    ap.add_argument("--g", default=str(HERE.parent / "04_activity_selection/runs/acts_ref_scored.pt"))
    a = ap.parse_args()
    c = torch.load(a.c, weights_only=False)["casual"]
    g = torch.load(a.g, weights_only=False)
    r4 = lambda v: [float(f"{x:.4g}") for x in v.tolist()]  # noqa: E731
    out = {str(l): dict(c=r4(c[l]), g=r4(g[l])) for l in LAYERS}
    (HERE / "out/selectivity.json").write_text(json.dumps(out))
    print({l: len(v["c"]) for l, v in out.items()})


if __name__ == "__main__":
    main()
