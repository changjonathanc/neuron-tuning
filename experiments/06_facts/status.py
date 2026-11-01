"""quick table of finished runs (TKL, acc, neighbours, fp32 F if present)."""
import json
from pathlib import Path
H = Path(__file__).resolve().parent
for f in sorted((H / "runs/final").glob("*.json"), key=lambda p: (p.stem.split("_lr")[0], p.stem.split("_s")[-1], float(p.stem.split("_lr")[-1].split("_s")[0]) if "_lr" in p.stem else 0)):
    r = json.loads(f.read_text())
    s = r["summary"]
    g = H / "runs/f32" / f.name
    F = next(iter(json.loads(g.read_text()).values()))["macro"] if g.exists() else float("nan")
    print(f"{f.stem:28s} tkl {s['tkl']:.3f} acc {s['acc']:.3f} F {F:.2e} nkl_r {s['nkl_real']:.3f} nkl_f {s['nkl_fake']:.3f} "
          f"chg_r {s['nchange_real']:.2f} chg_f {s['nchange_fake']:.2f} val_f {s['nvalue_fake']:.2f}")
