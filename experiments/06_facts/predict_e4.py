"""E4: E1's predictors (experiments/03_predictor/predict.py, unchanged) with a new target set "facts" = the scored
positions of the 480 training items (runs/acts_facts.pt from select_e4.py: per-neuron mean |a| and a 1024-token
subsample). Every trained down_proj edit in runs/final/ (lora_all8 has no predictor: modules other than down_proj).
Also the pre-training S of colsel128 (no run needed). CPU only.

    python predict_e4.py     -> out/pred.json
"""
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "03_predictor"))
from predict import Predictor  # noqa: E402


def main():
    P = Predictor()
    F = torch.load(HERE / "runs/acts_facts.pt")
    P.acts["facts"], P.sub["facts"] = F["acts"], F["sub"]
    out = HERE / "out/pred.json"
    res = json.loads(out.read_text()) if out.exists() else {}
    sel = {int(l): v for l, v in json.loads((HERE / "out/selection.json").read_text())["selection"].items()}
    st = dict(kind="gains", spec=sel, params={f"{l}.g": torch.full((len(v),), 1e-3) for l, v in sel.items()})   # S only (g > 0 avoids 0 / 0 in R)
    res["colsel128_pre"] = P(st, "facts")
    for f in sorted((HERE / "runs/final").glob("*.pt")):
        n = f.stem
        if n in res or n.startswith("lora_all8"):
            continue
        res[n] = P(f, "facts")
        print(n, f"S {res[n]['S']:.3f} D_off {res[n]['D_off']:.3f}", flush=True)
    out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
