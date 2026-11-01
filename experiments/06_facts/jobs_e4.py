"""E4 job list. Each job: train_e4.py (training + bf16 evaluation). The fp32 F runs separately in batches
(f32_e4.sh), because E1's f32forget.py needs ~15 GB and does not fit next to a training run.

    python jobs_e4.py --seeds 0 [--extra arm:lr ...] > jobs/<name>.jsonl
"""
import json
import sys

GRID = {
    "col128": [3e-3, 1e-2, 3e-2, 1e-1],
    "dense128": [1e-2, 3e-2, 1e-1, 3e-1],
    "colsel128": [1e-2, 3e-2, 1e-1, 3e-1],
    "lorac": [1e-4, 3e-4, 1e-3, 3e-3],
    "lora_all8": [3e-6, 1e-5, 3e-5, 1e-4],
}


def name(arm, lr, s):
    return f"{arm}_lr{lr:g}_s{s}"


def main():
    argv = sys.argv[1:]
    seeds = [int(x) for x in argv[argv.index("--seeds") + 1].split(",")]
    extra = [a for a in argv if ":" in a]
    only = "--only-extra" in argv
    g = {k: ([] if only else list(v)) for k, v in GRID.items()}
    for e in extra:
        k, lr = e.split(":")
        g[k] = sorted(set(g[k]) | {float(lr)})
    for s in seeds:
        for arm, lrs in g.items():
            for lr in lrs:
                n = name(arm, lr, s)
                print(json.dumps(dict(id=n, out=f"runs/final/{n}.json", cmd=(
                    f"$PY train_e4.py --arm {arm} --lr {lr:g} --seed {s} --out runs/final/{n}"))))


if __name__ == "__main__":
    main()
