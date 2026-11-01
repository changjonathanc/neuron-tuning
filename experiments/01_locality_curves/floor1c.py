"""Phase 1c post-hoc diagnostic (not pre-registered): the floor of the forgetting measure F. Writes copies of trained
column adapters with the change scaled by s (column change D = W g + rms(W) u is linear in (g, u)). If F is a clean
quadratic function of the change, F(s) ~ s^2 F(1); a floor shows as F(s) that stops going down for small s.

    python experiments/01_locality_curves/floor1c.py   -> runs/diag/*.pt; then forget.py on them
"""
import torch
from pathlib import Path

R = Path(__file__).resolve().parent / "runs"
for run in ["sel8_lr0.03_s2", "srand8_lr0.03_s2", "sel128_lr0.003_s2"]:
    st = torch.load(R / "final" / f"{run}.pt", map_location="cpu")
    for s in [0.0, 0.01, 0.1, 0.3, 1.0]:
        d = dict(st, params={k: v * s for k, v in st["params"].items()})
        torch.save(d, R / "diag" / f"{run}_x{s:g}.pt")
