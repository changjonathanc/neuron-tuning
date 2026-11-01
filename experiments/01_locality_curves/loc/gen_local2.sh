#!/bin/bash
# Judge check (LOC_NOTES.md): post-hoc: sel8 and srand8 at LR 1e-1 on one RTX 3090, two generators in parallel under one lock.
cd "$(dirname "$0")/../../.."
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=${PYTHON:-python}
D=experiments/01_locality_curves/loc
mkdir -p experiments/01_locality_curves/runs
flock "${GPU_LOCK:-experiments/01_locality_curves/runs/job.lock}" bash -c "
$PY experiments/lib/gen.py --adapter experiments/01_locality_curves/runs/final/sel8_lr0.1_s0.pt --out $D/gens/sel8_lr0.1.json > $D/gen_sel8_lr0.1.log 2>&1 &
$PY experiments/lib/gen.py --adapter experiments/01_locality_curves/runs/final/srand8_lr0.1_s0.pt --out $D/gens/srand8_lr0.1.json > $D/gen_srand8_lr0.1.log 2>&1 &
wait"
