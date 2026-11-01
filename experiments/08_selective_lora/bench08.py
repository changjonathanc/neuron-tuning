"""Experiment 08 cost benchmark: time per training step and peak GPU memory of the four arms, one job at a time.

LoRA arms: train08.py with --steps 40 (its own timing of steps 10-39). Neuron arm: experiments/lib/train1b.py
--arm columns --k 8 (the sel8 edit of experiment 01) with --steps 40; it has no step timer, so this script wraps
Adam.step: the time between two optimizer steps is one training step (steps 10-39), and the peak memory is counted
from the first step on (after the neuron selection).

    python experiments/08_selective_lora/bench08.py --arm neurons|plain|sel|kl --out out/bench.json
"""
import argparse
import json
import runpy
import sys
import tempfile
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
STEPS = 40


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["neurons", "plain", "sel", "kl"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    tmp = Path(tempfile.gettempdir()) / "bench08" / a.arm
    for suf in (".json", ".pt"):
        Path(str(tmp) + suf).unlink(missing_ok=True)
    if a.arm == "neurons":
        stamps = []
        orig = torch.optim.Adam.step

        def step(self, *args, **kw):
            r = orig(self, *args, **kw)
            torch.cuda.synchronize()
            if not stamps:
                torch.cuda.reset_peak_memory_stats()
            stamps.append(time.time())
            return r
        torch.optim.Adam.step = step
        sys.argv = ["train1b.py", "--arm", "columns", "--k", "8", "--lr", "0.01", "--seed", "0", "--split", "full",
                    "--w-pres", "0", "--steps", str(STEPS), "--out", str(tmp)]
        runpy.run_path(str(REPO / "experiments/lib/train1b.py"), run_name="__main__")
        d = [b - c for b, c in zip(stamps[11:], stamps[10:])]
        row = dict(steps=STEPS, sec_per_step=sum(d) / len(d), peak_mem_gb=torch.cuda.max_memory_allocated() / 2**30)
    else:
        lam = {"plain": 0, "sel": 0.1, "kl": 10}[a.arm]
        sys.argv = ["train08.py", "--arm", a.arm, "--lam", str(lam), "--lr", "1e-5", "--steps", str(STEPS),
                    "--probe-every", "0", "--out", str(tmp)]
        runpy.run_path(str(HERE / "train08.py"), run_name="__main__")
        r = json.loads(Path(str(tmp) + ".json").read_text())
        row = dict(steps=STEPS, sec_per_step=r["sec_per_step"], peak_mem_gb=r["peak_mem_gb"])
    row["device"] = torch.cuda.get_device_name()
    out = Path(a.out)
    res = json.loads(out.read_text()) if out.exists() else {}
    res[a.arm] = row
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    print(a.arm, row)


if __name__ == "__main__":
    main()
