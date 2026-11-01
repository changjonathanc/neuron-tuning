#!/bin/bash
# fp32 F (E1 f32forget.py, unchanged) for every adapter in runs/final without runs/f32/<name>.json; one model load
# per chunk. Run it when no training run is on the GPU:  flock LOCK bash f32_e4.sh
# Adapter names (the keys of the output) are paths relative to the repo root.
cd "$(dirname "$0")/../.."
D=experiments/06_facts
PY=${PY:-${PYTHON:-python}}
todo=()
for p in $D/runs/final/*.pt; do n=$(basename "$p" .pt); [ -f $D/runs/f32/$n.json ] || todo+=("$p"); done
echo "${#todo[@]} adapters"
[ ${#todo[@]} -eq 0 ] && exit 0
mkdir -p $D/runs/f32
$PY experiments/03_predictor/f32forget.py --adapters "${todo[@]}" --out $D/runs/f32_batch.json --chunk 8 || exit 1
# split the batch file into one file per run (analyze_e4.py reads runs/f32/<name>.json)
$PY - <<'PYEOF'
import json, pathlib
b = json.loads(pathlib.Path("experiments/06_facts/runs/f32_batch.json").read_text())
for k, v in b.items():
    pathlib.Path(f"experiments/06_facts/runs/f32/{pathlib.Path(k).stem}.json").write_text(json.dumps({k: v}))
print(len(b), "split")
PYEOF
