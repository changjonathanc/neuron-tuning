# 14 How many neurons: pre-registration and run log

Written on 2026-11-01, before any run of this experiment. At this time I had read the saved results of the
type-hint demo (K = 8: the seed-0 LR sweep, seeds 1-2 and the LoRA baselines) and of experiment 12. No run with
K other than 8 existed. Changes after this point go to the run log at the end, with the reason and what I had seen.

## Question

The first practical question of a user of `make_edit.py`: **how many neurons K should an edit use?** We use the
type-hint demo ([demos/python_type_hints](../../demos/python_type_hints/README.md)), because its target metric is
automatic (AST check, no judge) and its K = 8 sweep is saved. For K in {1, 2, 4, 8, 16, 32, 64} (seed 0), we measure
the habit, the near-neighbour leaks and the change on unrelated text F at a matched habit strength.

## Setup (the same as the demo, only K and the LR grid change)

- `make_edit.py make --system HINTS` (the demo's system prompt, `demos/lib.py`), the demo's 132 training prompts
  and 40 held-out prompts, 200 Adam steps, batch 8, constant LR, seed 0, `--max-new-tokens 384`, `--f32`, no
  preservation loss. The teacher replies are the saved ones of the demo (copied; the data hash is checked).
- **Neurons: the default rule** of `make_edit.py` (c top 10% of its layer, S = g / c <= 0.2, rank by
  ||dL/dW_j|| / g_j, keep the top K). The rule ranks a fixed list (the selection does not depend on K or the seed),
  so the set for K is the first K entries of ONE ranking. `select14.py` runs `make_edit.select_neurons` once and
  saves the full ranking of the eligible neurons; each K uses its first K entries (`make_edit.py --neurons`). Check:
  the first 8 entries must be the demo's 8 neurons (else the K = 8 reuse is not valid and K = 8 is trained again).
- **K = 8 is reused** from the demo (`demos/python_type_hints/out/reports/s0.json` and `out/gens/columns_lr*_s0.json`):
  the same settings, LR grid 0.01, 0.03, 0.1, 0.3.
- **LR grids** (seed 0, 3 points, half-decade steps; the centre follows LR × K ≈ constant from the K = 8 point):

  | K | 1 | 2 | 4 | 8 (reused) | 16 | 32 | 64 |
  |---|---|---|---|---|---|---|---|
  | LR | 0.3, 1, 3 | 0.1, 0.3, 1 | 0.03, 0.1, 0.3 | 0.01, 0.03, 0.1, 0.3 | 0.003, 0.01, 0.03 | 0.001, 0.003, 0.01 | 0.001, 0.003, 0.01 |

  **Edge rules** (each at most once per K, written in the run log): (a) if the lowest LR already has habit >= 0.80,
  add the next half-decade step DOWN (so that the matched point is bracketed); (b) if no LR reaches 0.80 and the top
  LR is stable (finite loss), add the next half-decade step UP. If time allows, I add one more point UP for every K,
  for the "strongest habit" measure.
- Trained numbers: K × 1,537 (1,536 output weights and one gain per column).

## Measurements (per run; greedy replies, at most 768 tokens, `demos/gen.py --demo pyhints`)

| measure | set | how |
|---|---|---|
| **habit rate** = share of Python function defs with FULL type hints (pooled over replies) | 40 held-out Python prompts | AST (`demos/codecheck.py`) |
| code validity (complete Python blocks that parse) | held-out | AST |
| JavaScript functions with any type annotation (near neighbour) | 20 JavaScript prompts (`js`) | regex |
| C: Python-style `-> type` after a C signature (near neighbour) | the 10 C prompts of `other_code` | regex |
| **F** = fp32 KL(original ‖ edited) per token, macro mean over 7 domains (and the code domain) | 256 unrelated items | `make_edit.py --f32` |
| held-out target loss (KL to the teacher) | 40 held-out teacher replies | `make_edit.py` |
| selectivity of the chosen set | – | below |

- **Matched point** (the demo's rule): for each K, the lowest grid LR with habit >= 0.80 (seed 0). If no LR of the
  grid (after the edge rule) reaches 0.80, the K is **"not matched"** and I report its point with the highest habit.
- **Strongest habit** = max habit over the LR grid of the K (it depends on the grid; I report the grid with it).
- **Selectivity of the set** (the tool's statistics; `select14.py`): per neuron S_j = g_j / c_j (c = mean |a| on the
  fixed 64-item training subset at the scored positions, g = the macro mean over the background domains). Per K:
  median and max S of the set, the activity-weighted set selectivity S_set = Σ g / Σ c, the score of the K-th
  neuron / the score of the first, and pct_S of each neuron (the rank of its S in its layer, 0 = most selective).
  Also the number of eligible neurons (the rule cannot give more than this).
- **Reference lines** (from the demo, seed 0, the same habit rule): the standard rank-8 LoRA at its matched point
  (habit 0.81, F 0.0184) and the same-size rank-1 LoRA (habit 0.81, F 0.0295). No new LoRA runs.
- Intervals: Wilson 95% for counts (one seed); F has no interval (one seed; the seed spread at K = 8 is in the demo:
  F 0.0014 ± about 10%).

## Predictions (written before any run)

1. F at the matched point is lowest for small-to-middle K (4-16) and rises for K >= 32, because less selective
   neurons (larger S, smaller score) enter the set.
2. K = 1-2 may not reach habit 0.80 ("not matched"), or only at an LR where code validity drops.
3. The strongest habit grows with K (more capacity), from below 0.8 at K = 1 to >= 0.9 at K >= 16.
4. The set selectivity S_set rises with K, but stays <= 0.2 by construction (the filter); the score of the K-th
   neuron falls by more than 5× from K = 1 to K = 64.
5. Near neighbours at the matched point: JavaScript leaks (types in JS functions) stay below 15% for all matched K;
   the C `-> type` leak stays 0-1 of 10.
6. Every matched K has F at least 5× lower than the standard LoRA (0.0184).

## Commands

From the repository root, GPU (one RTX 3090), `PY="uv run python"`:

```bash
$PY experiments/14_how_many_neurons/select14.py       # -> out/ranking.json (the eligible neurons, ranked)
# per K (the queue script writes these): --neurons = out/sets/k<K>.json (the first K of the ranking)
uv run make_edit.py make --system "$HINTS" --prompts demos/python_type_hints/data/train.json \
    --eval-prompts demos/python_type_hints/data/heldout.json --k K --lr ... --neurons .../k<K>.json \
    --steps 200 --seed 0 --f32 --max-new-tokens 384 --show 0 --out results/how_many_neurons/k<K>
uv run demos/gen.py --demo pyhints --sets heldout js other_code --edit results/how_many_neurons/k<K>/columns_lr<LR>.safetensors \
    --out results/how_many_neurons/gens/k<K>_lr<LR>_s0.json
$PY experiments/14_how_many_neurons/counts14.py       # CPU: per-reply counts -> out/counts/ (small JSON)
$PY experiments/14_how_many_neurons/analyze14.py      # CPU: out/summary.json, out/tables.md
$PY experiments/14_how_many_neurons/plot14.py         # CPU: fig_how_many_neurons.png
```

## Run log

- 2026-11-01: pre-registration written; no run yet.
- 2026-11-01: `select14.py`: 256 eligible neurons (as in the demo); the first 8 of the ranking are exactly the demo's
  8 neurons, so the K = 8 reuse is valid. 33 GPU s. Two lanes in parallel do not fit on the 3090 (13 GB per training
  job, more in the fp32 F step), so all runs are in one queue.
- 2026-11-01: K = 64 done (habit 31/70, 51/64, 52/62 at LR 0.001, 0.003, 0.01; bracketed, no edge rule). LR 0.003
  is 0.797, one def under the line, so by the rule the matched point is LR 0.01.
- 2026-11-01: K = 1 done: habit 25/67, 33/64, 34/61 at LR 0.3, 1, 3 (no LR reaches 0.80; the loss is finite at LR 3).
  **Edge rule (b)**: one step up, LR 10 (queued after K = 32, 2, 4).
- 2026-11-01: K = 16 done: habit 50/67, 54/64 (LR 0.003, 0.01; LR 0.03 in progress); matched LR 0.01.
- 2026-11-01, **amendment 1 (post hoc; after I saw K = 1, 16 and 64, before K = 2, 4, 32)**: the step rule (the
  lowest grid LR >= 0.80) moves F by up to half a decade of LR when a point is just under the line (K = 64 LR 0.003:
  0.797). I add a SECONDARY measure, F at habit 0.80 interpolated linearly in ln F against the habit rate between the
  two neighbouring grid points that bracket 0.80 (in LR order; "not covered" if no pair brackets 0.80). The primary
  measure and the predictions do not change; the tables show both.
- 2026-11-01: K = 16 LR 0.03: 51/62. K = 32 done: 22/74, 52/66 (0.788, just under the line), 51/61 at LR 0.001,
  0.003, 0.01; matched LR 0.01. Release deadline: no GPU run after 23:00.
- 2026-11-01: the pre-registered "one more point UP for every K" (strongest habit) is queued after the K = 1 edge run:
  K = 16 LR 0.1, K = 32 LR 0.03, K = 64 LR 0.03, K = 4 LR 1, K = 2 LR 3 (K = 1: its edge point LR 10 counts as its
  point up). Seeds 1-2 at K = 8 exist in the demo; seeds for another K: only if time allows.
- 2026-11-01: K = 2 done: 38/64, 38/64, 47/65 at LR 0.1, 0.3, 1 (not matched; the point up LR 3 in the queue is
  also its edge step (b)). K = 4 done: 47/60, 50/62, 50/58 at LR 0.03, 0.1, 0.3; matched LR 0.1.
- 2026-11-01: Step 2 (SHOWABLE): RESULTS.md, fig_how_many_neurons.png, experiments/README.md row, from the 3-point grids (and
  K = 8 from the demo). The queued points up (and K = 1 LR 10) are added when they end.
- 2026-11-01, Step 3 (all GPU runs ended 22:01): K = 1 LR 10 (edge rule b): 46/68, still not matched. Points up:
  K = 2 LR 3 44/62 (also its edge step; not matched), K = 4 LR 1 51/61, K = 16 LR 0.1 53/63, K = 32 LR 0.03 53/63,
  K = 64 LR 0.03 56/67. `ablate14.py` (F only, fp32): the K = 16 and K = 64 matched patches without their layer < 10
  columns. Not run (time): seeds 1-2 for K != 8; a retrained set without early layers.
