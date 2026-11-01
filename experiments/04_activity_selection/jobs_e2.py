"""E2 job lists (one per GPU machine). One job = train colsel8 -> the evaluation of the original phase.
  casual: experiments/lib/tkl.py (TKL), experiments/lib/forget.py (F, no MMLU), experiments/01_locality_curves/taskkl.py (task KL)
  rules:  experiments/02_conditional_rules/eval2b.py (hit, FP, kl_ctrl, tkl_trig, F), one condition per run

    python jobs_e2.py [task:lr ...]      -> jobs/q_gpu2.jsonl, jobs/q_gpu3.jsonl  (extra LRs: edge rule)
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
D = "experiments/04_activity_selection"
GRID = {"casual": [1e-2, 3e-2, 1e-1, 3e-1],
        "weather": [0.1, 0.3, 1.0, 3.0], "lower": [0.03, 0.1, 0.3, 1.0], "recipe": [0.1, 0.3, 1.0, 3.0]}
SEEDS = [0, 1, 2]


def grid(extra=()):
    g = {task: list(v) for task, v in GRID.items()}
    for e in extra:
        t, lr = e.split(":")
        g[t] = sorted(set(g[t]) | {float(lr)})
    return g


V2 = {"casual": [1e-3, 3e-3, 1e-2, 3e-2], "weather": [3e-3, 1e-2, 3e-2, 0.1]}   # amendment 1: arm colsel8s


def job(t, lr, s, arm="colsel8"):
    n = f"{t}_{arm}_lr{lr:g}_s{s}"
    tr = f"$PY {D}/train_e2.py --task {t} --lr {lr:g} --seed {s}"
    if arm == "colsel8s":
        tr += " --sel selection_s.json --arm-name colsel8s"
    if t == "casual":
        r = f"{D}/runs/casual"
        pt = f"{r}/final/{n}.pt"
        cmd = (f"mkdir -p {r}/final {r}/tkl {r}/forget {r}/task && {tr} --out {r}/final/{n} && "
               f"$PY experiments/lib/tkl.py --adapters {pt} --out {r}/tkl/{n}.json && "
               f"$PY experiments/lib/forget.py --adapters {pt} --out {r}/forget/{n}.json && "
               f"$PY experiments/01_locality_curves/taskkl.py --adapters {pt} --out {r}/task/{n}.json")
        return dict(id=n, out=f"{r}/task/{n}.json", cmd=cmd)
    r = f"{D}/runs/rules"
    conds = HERE / f"runs/rules/conds/{n}.json"
    conds.parent.mkdir(parents=True, exist_ok=True)
    conds.write_text(json.dumps({f"{t}|{arm}|{lr:g}|s{s}": [f"../../{r}/curve/{n}.pt"]}))
    cmd = (f"mkdir -p {r}/curve {r}/eval && {tr} --out {r}/curve/{n} && cd experiments/02_conditional_rules && "
           f"$PY eval2b.py --conds ../../{r}/conds/{n}.json --out ../../{r}/eval/{n}.json")
    return dict(id=n, out=f"{r}/eval/{n}.json", cmd=cmd)


def main():
    """no argument: write the base job files. task:lr ...: PREPEND the jobs of these extra LRs to the existing files
    (the queues read the files between jobs; jobs already in a file keep their machine)."""
    extra = sys.argv[1:]
    (HERE / "jobs").mkdir(exist_ok=True)
    if not extra:
        js = [job(t, lr, s) for s in SEEDS for t, lrs in GRID.items() for lr in lrs]
        for k, name in enumerate(("gpu2", "gpu3")):
            (HERE / f"jobs/q_{name}.jsonl").write_text("".join(json.dumps(j) + "\n" for j in js[k::2]))
        print(len(js), "jobs")
        return
    if extra == ["v2"]:   # amendment 1: v2 jobs for casual / weather, and the lower / recipe jobs not done yet
        js = [job(t, lr, s, "colsel8s") for s in SEEDS for t, lrs in V2.items() for lr in lrs]
        js += [job(t, lr, s) for s in SEEDS for t in ("lower", "recipe") for lr in GRID[t]]
    else:                 # task:lr[:arm]
        js = [job(t, float(lr), s, *rest) for s in SEEDS for t, lr, *rest in (e.split(":") for e in extra)]
    for k, name in enumerate(("gpu2", "gpu3")):
        f = HERE / f"jobs/q_{name}.jsonl"
        f.write_text("".join(json.dumps(j) + "\n" for j in js[k::2]) + f.read_text())
    print(len(js), "extra jobs")


if __name__ == "__main__":
    main()
