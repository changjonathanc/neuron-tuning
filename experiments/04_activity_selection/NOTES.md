# E2 notes: neurons chosen by selectivity only

> **Names in this log.** Phase 1c = 01_locality_curves; Phase 2b = 02_conditional_rules; E1 = 03_predictor; E2 = this experiment; E3 = 05_typical_lora; E4 = 06_facts. Phase 1, Phase 1b and Phase 2 are earlier studies that are not in this repository; their shared code and data (adapters, training, the forgetting set F, the teacher-KL measure) are in [`../lib/`](../lib/).

Written before the training runs (2026-10-30). Entries after "Run log" are appended later.

## Question
If we choose the 8 neurons by selectivity ALONE (no attribution, no gradient), do edits of their 8 `down_proj`
columns stay local at matched effect? Also on the Phase 2b rules where attribution chose neurons that fire everywhere
(weather, recipe)? Can such an edit reach the effect at all?

## Method (fixed now)
**Selection** (`select_e2.py`, done before this entry; result in `out/selection.json`). Per MLP neuron (input of
`down_proj`), original model, no system prompt:
- c = mean |a| per token on the TARGET replies, training split only. casual: the 160 casual teacher replies of
  Phase 1c (`experiments/01_locality_curves/out/acts1c.pt`, key "casual"). weather, lower, recipe: the kept training trigger replies
  of Phase 2b (`train2b.trigger_acts` on `load_data(rule, "full")[0]`).
- g = mean |a| on the Phase 1c reference text (`experiments/01_locality_curves/data/ref_set.json`, macro mean over 7 domains). Not
  the forgetting (F) items.
- Score = ln(c / g). Eligible: c ≥ the 90th percentile of c in the neuron's own layer (so that tiny neurons are not
  chosen). Choice: the global top 8 over all 35 layers (no layer rule; simplest).
- The selection does not depend on the seed. The seed changes only the batch order (and the 48-item random draw of
  the attribution, which `train_e2.py` repeats without using it, so that the batch order is the same as sel8 / col8
  of the same seed).

**Training** (`train_e2.py`): arm "colsel8" = the columns adapter on these 8 neurons (12,296 numbers). The training
code of the original phase is used unchanged; only `TG.select_attr` is replaced by the fixed list.
- casual: `experiments/lib/train1b.py --arm columns --k 8 --split full --w-pres 0` (Phase 1c setup).
- weather, lower, recipe: `experiments/02_conditional_rules/train2b.py --edit <rule> --method col8 --split full --w-pres 0`
  (Phase 2b amendment 3 setup: trigger items only).
- No preservation loss. 400 steps, Adam, constant LR. Seeds 0, 1, 2.

**Evaluation**: the code of the original phase, unchanged. casual: `experiments/lib/tkl.py` (TKL), `experiments/lib/
forget.py` (F, without MMLU, to save time), `experiments/01_locality_curves/taskkl.py`. Rules: `experiments/02_conditional_rules/eval2b.py` (hit,
FP, kl_ctrl, tkl_trig, F).

**LR grids** (`jobs_e2.py`). Start: the grid of the attribution arm of that phase plus one step up (selective
neurons are less active on average and Phase 1c/2b random arms needed a higher LR).
- casual: 1e-2, 3e-2, 1e-1, 3e-1 (the sel8 grid 3e-3 … 1e-1, without 3e-3, plus 3e-1; = the srand8 grid without
  3e-3, because srand8 at 1e-2 was already weaker than both targets).
- weather: 0.1, 0.3, 1, 3; lower: 0.03, 0.1, 0.3, 1; recipe: 0.1, 0.3, 1, 3 (the col8 curve LRs plus one step up;
  = the Phase 2b srand8 grid).
- Edge rule: when all seed-0 runs of a task are done, if the seed-0 curve does not bracket a target, add one
  half-decade step past the needed edge (up if every point is weaker, down if every point is stronger), at most 2
  extra steps per task, 3 seeds each. If the effect gets worse when the LR goes up, do not extend upward: the target is
  "not reached" for colsel8.
- Fallback (only if colsel8 does not reach the first target of a task after the edge rule): "colsel8L" = the same
  score, restricted to the layers of the attribution choice of seed 0 with the same per-layer counts. Not run
  otherwise.

**Comparison at matched effect** with the SAVED curves (no retraining):
- casual (Phase 1c): targets TKL 0.60 and 0.45. Families sel8, srand8, lorad (and dense8 for reference). Code: the
  Phase 1c analysis (`analyze1c.py`: interpolation of ln F in ln TKL along the seed-mean curve, first crossing,
  three-level bootstrap over eval messages, forgetting items per domain and seeds, 10,000 resamples, drop rule 10%),
  with colsel8 added as a family.
- rules (Phase 2b): T_e per rule = the saved Phase 2b value (the weaker of col8 and lora_d at LR*, recomputed per
  resample). colsel8 is interpolated as srand8 was (`analyze3.srand_stats`: first crossing in ln tkl; bootstrap over
  the 60 eval pairs and the seeds). Metrics at T_e: kl_ctrl, F, hit, FP. Ratios colsel8 / col8, colsel8 / lora_d,
  colsel8 / srand8 (where srand8 is covered).

## Predictions
- P1 (casual): colsel8 is about as local as srand8: F ratio colsel8 / srand8 with CI inside [0.5, 2] at both targets
  (or lower). It is clearly more local than lorad.
- P2 (weather, recipe): colsel8 is clearly more local than col8 (F and kl_ctrl ratio CI upper bound < 1) and at least
  as local as lora_d (ratio point ≤ 1 and CI upper bound < 2), on F and kl_ctrl.
- P3 (reach): colsel8 may have less leverage. I report the lowest seed-mean effect reached on each task.
- lower: reported, no prediction (attribution neurons there were already partly selective).

## Decision rule
For each ratio (colsel8 / X) at a target: "more local" if the CI upper bound < 1; "less local" if the CI lower bound
> 1; "as local" if the CI is inside [0.5, 2]; else "not determined". Not covered if a curve does not bracket the
target or > 10% of the resamples miss it.
**Claim supported** if (a) casual: colsel8 is not "less local" than srand8 and is "more local" than lorad at the
covered targets, and (b) on weather AND recipe: colsel8 is "more local" than col8 on F and on kl_ctrl and not "less
local" than lora_d on F and on kl_ctrl. **Not supported** if colsel8 is "less local" than lora_d on F on weather or
recipe, or "less local" than srand8 on casual. Otherwise "partly supported" (with the details). If colsel8 does not
reach T_e on a rule, that rule is "not covered" and I report the reach.

## Selection result (done before the runs; `out/selection.json`)
All four tasks: the 8 neurons share NO neuron with the attribution choice (sel8 / col8, seeds 0-2) or with srand8.
Most are in layers 0-5 (layer 4 dominates). Median g / c of the 8 neurons vs the attribution neurons of seed 0:

| task | neurons (layer:index) | median g/c colsel8 | median g/c attribution (seed 0) |
|---|---|---:|---:|
| casual | 4:493, 4:1718, 4:6022, 1:6067, 4:4676, 4:745, 1:692, 15:3172 | 0.035 | 0.132 |
| weather | 4:3554, 28:11509, 4:1718, 16:699, 4:5675, 5:3980, 14:3360, 4:5108 | 0.020 | 3.19 |
| lower | 4:5330, 0:4017, 4:5890, 1:5144, 4:2061, 2:3696, 4:1340, 1:5987 | 0.074 | 0.54 |
| recipe | 4:4472, 1:1573, 15:11706, 15:10558, 25:470, 5:5180, 19:7694, 4:1047 | 0.021 | 1.61 |

Within-layer percentile of c: 94.5-99.99 (median > 99.8); of g: 0.1-76 (low to mid). Per-neuron c, g and
percentiles: `out/selection.json`.

## Hardware
Two NVIDIA GB10 GPUs (`q_e2.py`, 3 workers each), each with a copy of the repo (new layout). The
remote dispatch code was removed from the public version. 48 jobs.

## Run log

### Amendment 1 (2026-10-30, after the first seed-0 runs; before any v2 run)
**What I saw.** casual colsel8 (v1) at seed 0: TKL 0.48 / 0.29 / 0.23 / 0.22 at LR 1e-2 … 3e-1 (stronger than any
sel8 point), but F 0.13-0.37 (sel8 ~0.003, lorad 0.037 at TKL 0.60), mostly on qa (0.51-1.5) and math. weather v1:
F 0.017-0.19, tkl 0.11-0.25 (every LR stronger than T_e 0.219).
**Cause (diagnostic `diag_pos.py`, `out/diag_pos.json`).** c (TG.build span) covers the SCORED positions: the last
prompt token (it predicts the first reply token) and the reply. g (`acts1c.py`) starts one token later and misses the
last prompt token. Neuron 4:4676 has |a| ≈ 47 at the last prompt token of casual items and ≈ 75 at the same position
of reference items, but < 0.1 on reply tokens; so its v1 g was low although it fires at the start of EVERY reply.
4:1718 (casual, weather) has the same pattern, weaker (17.5 / 4.4). This is an error in my use of the archive
statistic, not a test of the hypothesis. (The same off-by-one is in the Phase 1c srand matching and the Phase 2b
diagnostic; there c and g were compared in the same way for both arms.)
**Change (v2).** g is recomputed over the scored positions of the reference items (`select_e2.py --scored`,
`runs/acts_ref_scored.pt`; chat items: last prompt token … second-to-last token, as forget.py scores them; raw text:
all tokens but the last). c is unchanged. The rest of the rule is unchanged. Result `out/selection_s.json`:
- casual: 4:493, 4:6022, 1:6067, 4:745, 1:692, 15:3172, 4:5231, 4:5078 (drops 4:4676 and 4:1718). Arm "colsel8s".
- weather: 4:3554, 28:11509, 16:699, 4:5675, 5:3980, 14:3360, 15:1051, 19:11369 (drops 4:1718, 4:5108). Arm "colsel8s".
- lower, recipe: the SAME 8 neurons as v1. Their v1 runs ("colsel8") are valid v2 runs and are kept; the cancelled
  lower / recipe jobs are queued again.
- Overlap with attribution and srand8: still 0 on every task. Median g/c 0.039 / 0.020 / 0.073 / 0.021.
**v2 LR grids** (from the v1 seed-0 curves, which had every point stronger than the targets): casual 1e-3, 3e-3,
1e-2, 3e-2; weather 3e-3, 1e-2, 3e-2, 0.1. lower, recipe: unchanged. Edge rule unchanged.
**v1** casual and weather runs (seed 0, and weather LR 0.03 seeds 1-2) are reported descriptively only, as
"v1 (span error)". The primary analysis uses v2 (casual, weather: colsel8s; lower, recipe: colsel8).

### Run log
- 2026-10-30 22:35. Edge rule, casual colsel8s: the seed-0 TKL is 2.37 / 1.69 / 1.10 / 0.85 at LR 1e-3 … 3e-2 (all
  weaker than 0.60). Added LR 0.1 and 0.3 (the 2 allowed upward steps) for seeds 0-2 at once (not one after the
  other, to save time). weather colsel8s seed 0 brackets T_e (tkl 0.340 at 3e-3, 0.192 at 1e-2): no extension.
- 2026-10-30 22:40. To save time: casual colsel8s LR 1e-3 and 3e-3 run with seed 0 only (TKL 2.37 and 1.69, far from
  the targets; they cannot be part of a bracketing segment).
- 2026-10-30 23:25. Speed cut (the queue does ~10 jobs/hour). Not run (they cannot change a matched-effect value:
  they are past the first crossing, or far weaker than the target): weather colsel8s LR 0.03 and 0.1 seeds 1-2
  (seed 0: tkl 0.152, 0.143 < T_e); casual colsel8s LR 1e-2 and 3e-2 seed 2 (seed 0-1 TKL 1.10, 0.85 > 0.60);
  lower colsel8 LR 0.03 and 0.1 seeds 1-2 (seed 0: tkl 0.61, 0.54 > T_e 0.156); recipe LR 3 seeds 1-2 (held until the
  seed-0 recipe curve is known). An LR with fewer than 3 seeds is not on the analyze3 curves (rules); on the
  analyze1c curves (casual) it enters with its seeds. If the seed-0 curves show that a dropped point is needed for
  the first crossing, I run it.
- 2026-10-30 23:55. Edge rule, recipe colsel8: seed 0 tkl 0.143 at LR 0.1 (the lowest grid LR) is already stronger
  than T_e 0.158 (0.176 at LR 0.3: the curve is not monotone). Added LR 0.03 for seeds 0-2 (one step down).
- 2026-10-31 00:20. recipe seed 0: tkl 0.143 / 0.176 / 0.266 / 0.409 at LR 0.1 / 0.3 / 1 / 3 (the effect gets WEAKER
  above LR 0.1; hit 0.78-0.87, FP 0 at all LRs). The first crossing of T_e is between LR 0.03 and 0.1. Not run:
  recipe LR 0.3 seeds 1-2 and LR 1 seed 1 (past the first crossing). lower seed 0: tkl 0.295 at LR 0.3, 0.137 at
  LR 1: the crossing is between 0.3 and 1 (both with 3 seeds).
- 2026-10-31 01:05. recipe LR 0.03: tkl 0.159 / 0.150 / 0.153 (seed mean 0.154, still just stronger than T_e 0.158;
  the curve is flat here). Edge rule: added LR 0.01 (the second step down). lower LR 1: tkl 0.137 / 0.231 / (seed 2
  pending): large seed spread; the seed mean may not reach T_e. Added LR 3 for coverage (deviation: the seed-0 curve
  bracketed T_e, so the written rule did not require it; one GB10 was idle).
- 2026-10-31 01:50. All planned runs done. First full analysis: weather colsel8s and recipe colsel8 are bracketed by
  the point-estimate curve, but 21% / 23% of the resamples have a T_e stronger than the strongest 3-seed point (they
  clamp to that point). For weather this comes from my speed cut (LR 0.03, 0.1 had 2 seeds): I run LR 0.03 and 0.1
  seed 2 now. For recipe the effect gets weaker above LR 0.1 (the strongest point), so more LRs cannot fix it.
