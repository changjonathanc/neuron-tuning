"""Phase 1c job list (pre-registered design, NOTES.md). All arms: w_pres = 0, full split, 400 steps (train1b defaults).

    python experiments/01_locality_curves/gen_jobs.py [--extra fam:lr ...] > experiments/01_locality_curves/jobs_<name>.jsonl
"""
import json
import sys

FAM = {   # family: (script, arm, k, LR grid)
    "sel8":     ("lib/train1b.py", "columns", 8, [3e-3, 1e-2, 3e-2, 1e-1]),
    "srand8":   ("01_locality_curves/train1c.py", "columns_selrand", 8, [3e-3, 1e-2, 3e-2, 1e-1, 3e-1]),
    "dense8":   ("lib/train1b.py", "columns_active", 8, [3e-3, 1e-2, 3e-2, 1e-1, 3e-1]),
    "lorad":    ("lib/train1b.py", "lora_d", None, [1e-4, 3e-4, 1e-3, 3e-3]),
    "sel128":   ("lib/train1b.py", "columns", 128, [3e-3, 1e-2, 3e-2, 1e-1]),
    "srand128": ("01_locality_curves/train1c.py", "columns_selrand", 128, [3e-3, 1e-2, 3e-2, 1e-1, 3e-1]),
    "dense128": ("lib/train1b.py", "columns_active", 128, [3e-3, 1e-2, 3e-2, 1e-1]),
    "lorac":    ("lib/train1b.py", "lora_c", None, [3e-5, 1e-4, 3e-4, 1e-3]),
}
# Phase 1b w_pres = 0 runs (same code, same seeds) that are reused for (family, LR, seed 0/1)
REUSE = {("sel8", 3e-2): "columns_np_k8", ("dense8", 3e-1): "columns_active_np_k8", ("lorad", 1e-3): "lora_d_np",
         ("sel128", 3e-2): "columns_np_k128", ("dense128", 3e-2): "columns_active_np_k128", ("lorac", 3e-4): "lora_c_np"}
SEEDS = [0, 1, 2]
P = "experiments/01_locality_curves/runs"


def name(fam, lr, s):
    return f"{fam}_lr{lr:g}_s{s}"


def paths(fam, lr, s):
    """(adapter .pt, tkl json, forget json, task json) for a run (Phase 1b files for reused runs)."""
    n = name(fam, lr, s)
    if (fam, lr) in REUSE and s in (0, 1):
        b = f"{REUSE[(fam, lr)]}_s{s}"
        return (f"experiments/01_locality_curves/runs/reused/final/{b}.pt", f"experiments/01_locality_curves/runs/reused/tkl/{b}.json",
                f"experiments/01_locality_curves/runs/reused/forget/{b}.json", f"{P}/task/{n}.json")
    return f"{P}/final/{n}.pt", f"{P}/tkl/{n}.json", f"{P}/forget/{n}.json", f"{P}/task/{n}.json"


def grid(extra=()):
    g = {f: list(v[3]) for f, v in FAM.items()}
    for e in extra:
        f, lr = e.split(":")
        g[f] = sorted(set(g[f]) | {float(lr)})
    return g


def jobs(extra=(), only_extra=False):
    g, out = grid(extra), []
    ex = {(e.split(":")[0], float(e.split(":")[1])) for e in extra}
    for s in SEEDS:
        for fam, lrs in g.items():
            script, arm, k, _ = FAM[fam]
            for lr in lrs:
                if only_extra and (fam, lr) not in ex:
                    continue
                n = name(fam, lr, s)
                pt, tkl, fg, task = paths(fam, lr, s)
                if (fam, lr) in REUSE and s in (0, 1):
                    out.append(dict(id=n, where="local", cmd=f"$PY experiments/01_locality_curves/taskkl.py --adapters {pt} --out {task}"))
                    continue
                kk = f" --k {k}" if k else ""
                out.append(dict(id=n, cmd=(
                    f"$PY experiments/{script} --arm {arm}{kk} --lr {lr:g} --seed {s} --split full --w-pres 0 "
                    f"--out {pt[:-3]} && $PY experiments/lib/tkl.py --adapters {pt} --out {tkl} && "
                    f"$PY experiments/lib/forget.py --adapters {pt} --mmlu --out {fg} && "
                    f"$PY experiments/01_locality_curves/taskkl.py --adapters {pt} --out {task}")))
    return out


if __name__ == "__main__":
    extra = [a for a in sys.argv[1:] if ":" in a]
    for j in jobs(extra, "--only-extra" in sys.argv):
        print(json.dumps(j))
