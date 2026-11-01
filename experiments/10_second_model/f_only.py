"""F (fp32, per domain) for saved patches, and a make_edit.py-style report.json from the training log (GPU).

Used when a `make_edit.py make --f32` job trained and saved its patches but stopped in the F step (run log, NOTES.md):

    python experiments/10_second_model/f_only.py --dir results/q10/lorad --log results/logs/lorad.log \
        --neurons results/q10/select/report.json

Reads the TKL of each run from the log lines "training <name>:" and "held-out target loss: original X -> edited Y",
computes F with make_edit.unrelated_change on the fp32 model (as make_edit.py --f32), and writes <dir>/report.json with
the same fields that analyze.py reads (runs: kind, lr, n_params, target_loss, F) plus "recovered_from_log": true.
"""
import argparse
import json
import re
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import make_edit as ME  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--neurons", required=True, help="report.json of the selection (its neurons are copied)")
    ap.add_argument("--model", default="qwen3.5-4b")
    a = ap.parse_args()
    runs, cur = {}, None
    for line in Path(a.log).read_text().splitlines():
        m = re.match(r"training (\S+):", line)
        if m:
            cur = m.group(1)
        m = re.search(r"held-out target loss: original ([\d.]+) -> edited ([\d.]+)", line)
        if m and cur:
            kind, lr = cur.rsplit("_lr", 1)
            runs[cur] = dict(kind=kind, lr=float(lr), target_loss=dict(original=float(m.group(1)), edited=float(m.group(2))))
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    model, tok = ME.load(a.model, dtype=torch.float32)
    edits = {n: ME.load_any(model, Path(a.dir) / f"{n}.safetensors") for n in runs}
    for n, e in edits.items():
        if hasattr(e, "params"):   # LoRA
            runs[n]["n_params"] = sum(p.numel() for p in e.params.values())
        else:                      # columns: K x hidden + K gains, as make_edit.py counts them (Adapter.n_params)
            delta, _ = ME.load_patch(Path(a.dir) / f"{n}.safetensors")
            runs[n]["n_params"] = sum(D.numel() + len(ns) for ns, D in delta.values())
    F = ME.unrelated_change(model, tok, edits)
    for n, f in F.items():
        runs[n]["F"] = dict(macro=f["macro"], by_domain=f["by_domain"])
        print(f"{n:28s} TKL {runs[n]['target_loss']['edited']:.4f}  F {f['macro']:.5f}", flush=True)
    sel = json.loads(Path(a.neurons).read_text())
    repo, rev = ME.resolve_model(a.model)
    rep = dict(mode="system", model=dict(repo=repo, revision=rev), F_precision="fp32", neurons=sel["neurons"],
               n_eligible=sel["n_eligible"], recovered_from_log=True, runs=runs)
    (Path(a.dir) / "report.json").write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
