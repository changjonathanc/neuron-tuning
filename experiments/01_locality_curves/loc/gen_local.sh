#!/bin/bash
# Judge check (LOC_NOTES.md): lorad and sel8 replies on one RTX 3090, two generators in parallel under one lock.
cd "$(dirname "$0")/../../.."
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=${PYTHON:-python}
D=experiments/01_locality_curves/loc
mkdir -p experiments/01_locality_curves/runs
flock "${GPU_LOCK:-experiments/01_locality_curves/runs/job.lock}" bash -c "
$PY experiments/lib/gen.py --adapter experiments/01_locality_curves/runs/reused/final/lora_d_np_s0.pt --out $D/gens/lorad.json > $D/gen_lorad.log 2>&1 &
$PY experiments/lib/gen.py --adapter experiments/01_locality_curves/runs/reused/final/columns_np_k8_s0.pt --out $D/gens/sel8.json > $D/gen_sel8.log 2>&1 &
wait"
