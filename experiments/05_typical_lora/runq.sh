#!/bin/bash
# E3 job queue (run from the repo root; $PYTHON: the Python, default python; $GPU_LOCK: the lock file). Each line of the jobs file: "<arm> <lr> <seed>".
# One job = train -> teacher KL -> forgetting F + MMLU-570 -> task KL (the Phase 1c chain). Resumable: every step
# skips work that is already done; a job is marked done by runs/done/<id>.
#   nohup bash experiments/05_typical_lora/runq.sh experiments/05_typical_lora/jobs_a.txt > experiments/05_typical_lora/runs/q_a.log 2>&1 &
set -u
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=${PYTHON:-python}
R=experiments/05_typical_lora/runs
mkdir -p $R/done $R/logs $R/final $R/tkl $R/forget $R/task
while read -r arm lr seed; do
  [ -z "$arm" ] && continue
  id=${arm}_lr${lr}_s${seed}
  [ -e $R/done/$id ] && continue
  pt=$R/final/$id.pt
  echo "$(date +%T) start $id"
  flock "${GPU_LOCK:-$R/job.lock}" bash -c "$PY experiments/05_typical_lora/train_e3.py --arm $arm --lr $lr --seed $seed --split full --w-pres 0 --out $R/final/$id && \
    $PY experiments/lib/tkl.py --adapters $pt --out $R/tkl/$id.json && \
    $PY experiments/lib/forget.py --adapters $pt --mmlu --out $R/forget/$id.json && \
    $PY experiments/01_locality_curves/taskkl.py --adapters $pt --out $R/task/$id.json" >> $R/logs/$id.log 2>&1 \
    && touch $R/done/$id && echo "$(date +%T) done $id" || echo "$(date +%T) FAILED $id"
done < "$1"
echo "$(date +%T) queue finished"
