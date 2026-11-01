# 12 Lasso selector: pre-registration and run log

Written on 2026-11-01, before any run of this experiment. At this time I had read the results of experiment 07 and
the saved results of the type-hint demo (the tool-rule sweep, seed 0). No lasso run over all layers existed.

## Question

Is this a good, simple neuron SELECTOR: "train the `down_proj` of ALL layers for a short time with a group lasso (no
other penalty), keep the K = 8 columns with the largest update norm, then train only these K columns with no
penalty"? We compare it with the current rule of `make_edit.py` (activity filter + gradient per off-target write,
[docs/make_edit.md](../../docs/make_edit.md)). The lasso selector needs no background text and no activity
statistics. If it is as good as the tool rule, the tool can be simpler and model-agnostic.

Background (experiment 07): a group lasso on the full `down_proj` of layer 15, with no preservation loss, kept 12-18
columns in the most selective 0.3% of the layer. But the sel8 neurons (8 columns in 5 layers) were 1.3× more local
at the same strength. 07 used one layer only.

## Tasks

1. **Casual edit** (experiment 01): the same data, loss (KL to the persona teacher), 400 steps, batch 4, no
   preservation loss (w_pres = 0). Strength = teacher KL TKL on the 60 held-out messages (lower = stronger).
   The training loop copies `lib/train1b.py` (the same random stream: it also draws the 4 unused task items per step).
2. **Type-hint demo** ([demos/python_type_hints](../../demos/python_type_hints/README.md)): `make_edit.py make
   --system HINTS`, 200 steps, batch 8, the saved teacher replies of the demo (data hash checked). Strength = the
   habit rate (Python function defs with full type hints, 40 held-out prompts, greedy, 768 tokens, AST check of the
   demo).

Why the type-hint demo and not the cooking demo: (a) its target metric is automatic (AST), so the target needs no
judge; (b) its tool-rule sweep at seed 0 (LR 0.01-0.3, fp32 F, habit rate per LR) is saved and can be reused; (c) it
has a clean matched point (the neurons reach 80%), and the cooking neurons never reached the 80% line.

## Arms (K = 8 in all arms)

- **(a) tool rule**: `make_edit.select_neurons` (c top 10% of its layer, g/c <= 0.2, score ||G_j|| / g_j).
  Type hints: reuse the saved selection and sweep of the demo (`out/reports/s0.json`, `out/results.json`). Casual:
  new runs (no saved tool-rule run exists; sel8 of 01 is gradient attribution, not this rule).
- **(b) lasso, all layers**: phase 1 = ΔW on the `down_proj` of all 35 layers (337,920 columns, 519 M numbers, fp32),
  optimizer `AdamColProx` of 07 (Adam with one second moment per column, then the exact prox step of
  λ Σ_j ‖ΔW_j‖), the task loss only, LR 1e-3 (the sparse LR of 07), **100 steps**, the task's batch, seed 0.
  λ = λ₂₀₄₈: the value at which 2,048 columns (over all layers) of the norm of the mean initial gradient (64 training
  items, ΔW = 0) are larger than λ (the λ level of the chosen 07 point, here counted over all layers).
  Selection = the 8 columns with the largest ‖ΔW_j‖ at step 100. Phase 2 = the K columns, trained as in arm (a).
- **(c) lasso, one layer (layer 15)**: the same as (b), but ΔW on layer 15 only and λ₂₀₄₈ counted in layer 15 (the
  setting of the chosen 07 G0 point, but 100 steps instead of 400). This is the bridge to 07.
- Fallback rule (b, c): if fewer than 8 columns are non-zero at step 100, I halve λ and run phase 1 again (at most 2
  times; declared in the run log).

Phase 2 learning rates (seed 0): casual 1e-2, 3e-2, 1e-1 for all three arms (the sel8 grid of 01 without its weakest
point 3e-3); type hints 1e-2, 3e-2, 1e-1, 3e-1 (the grid of the demo). Edge rule: if an arm does not reach the matched
point and its top LR is stable, one half-decade step up (once).

## Measurements

- **Strength and F.** Casual: TKL; F = KL(original ‖ edited) per token on the 256 items of
  `lib/data/forget_set.json` (macro mean over 7 domains) in **fp32** (primary; no bf16 floor) and in bf16 (to place
  the points on the reference curves of 01). Type hints: habit rate; F in fp32 (`make_edit.py --f32`).
- **Matched point.** Casual: F at TKL 0.60 and 0.45 (the targets of 01), interpolated in ln F vs ln TKL along the
  seed-0 LR curve of each arm; "not covered" if the curve does not reach the target. Type hints: the demo's rule, the
  lowest LR with habit >= 0.80; I report the habit and F there (and F per LR).
- **Overlap**: the 8 selected neurons of (b) and (c) against the tool set (a): the number in common and the Jaccard
  index; the rank of each tool neuron in the lasso ranking. (c) for the casual edit also against the kept set of 07
  (G0, λ₂₀₄₈, seed 0, 400 steps).
- **Selectivity**: S_j = g_j / c_j with the tool's statistics (c = mean |a| on the fixed 64-item training subset at
  the scored positions, g = the macro mean over the domains of the background set at the scored positions). For each
  selected neuron: c, g, S, pct_S (the rank of S in its layer, 0 = most selective), the rank of S over all 337,920
  neurons, and "passes the tool filter" (c top 10% and S <= 0.2). Also the layers of the 8 neurons.
- **Cost** (GPU seconds on one RTX 3090, the GPU not shared): the lasso phase 1 (calibration pass + 100 steps) and
  the tool selection (activity pass on the training subset + background pass with an EMPTY cache + gradient pass).
  Also the peak GPU memory.

## Decision rule (per task, seed 0)

At the matched point (casual: TKL 0.45 and 0.60; type hints: the habit point), with ρ = F(b) / F(a):

- **as good** if (b) reaches the matched point and ρ <= 1.5 (at both casual targets);
- **better** if ρ <= 0.67; **worse** if ρ > 1.5 or (b) does not reach the matched point.

Overall: "the lasso selector is a good replacement" if it is as good or better on BOTH tasks; "mixed" if on one;
"no" if on none. One seed: the 01 seed spread of F at K = 8 is about ±10%, so 1.5 is a wide margin; Step 3 adds
seeds 1-2 at the matched LR if time allows.

## Predictions

1. Phase 1 (b) keeps columns with high c; most of the 8 pass the tool filter; their median pct_S <= 0.05.
2. The overlap with the tool set is small (<= 2 of 8): the tool divides by g, the lasso does not see g.
3. Casual: (b) is "as good" (07: the lasso alone found very selective columns in layer 15). (c) is worse than (b)
   and (a) (07: a one-layer edit was 2.2-2.5× less local than sel8 at the same TKL).
4. Type hints: uncertain. The tool neurons fire mostly on code; a lasso does not see that the code domain of F is
   also code, so (b) may choose neurons that fire on all code (higher F on the code domain).
5. Cost: phase 1 costs more GPU time than the tool's statistics pass (the tool: about 1-3 minutes).

## Run log

1. 2026-11-01. Smoke test of `select12.py` (casual, all layers, 3 steps; result deleted): calibration 20 s,
   λ₂₀₄₈ = 0.229, about 1.3 s per step, peak 15.7 GB on the RTX 3090. After 3 steps 1,384 columns were non-zero
   (the prox has not yet closed the columns that the first steps opened). No rule changes.
2. 2026-11-01. Selection runs (`select12.py`, RTX 3090, not shared), before any phase-2 training:
   - (a) tool rule: casual 34 s (6 of 8 neurons in layer 13, plus 9:5325 and 10:4762); type hints 35 s. The
     type-hint tool set is exactly the saved set of the demo (`demos/python_type_hints/out/reports/s0.json`), so
     the reuse of the saved tool sweep is valid.
   - (b) all layers: **the fallback rule was used for both tasks.** At λ₂₀₄₈ only 3 (casual) and 4 (type hints)
     columns were non-zero at step 100; at λ₂₀₄₈ / 2 there were 13 and 22. Most of the 8 chosen columns are NOT
     selective: S > 1 for 6 of 8 (casual) and 8 of 8 (type hints); only 15:7291 (a sel8 neuron of 01) passes the
     tool filter. Four neurons (13:1565, 13:5017, 13:3682, 7:936) are in the sets of BOTH tasks. Cost 301 s and
     604 s (two phase-1 runs each), peak 15.7 / 17.5 GB.
   - (c) layer 15: λ₂₀₄₈ = 0.0529 for the casual edit (07: 0.0528). The casual set has 6 of the 7 kept columns of
     07 G0 (λ₂₀₄₈, seed 0, 400 steps); 6 of 8 pass the tool filter. Type hints: 3 of 8 pass. 141 s and 268 s.
   - Overlap with the tool set: 0 of 8 for every lasso arm and task.
3. 2026-11-01, **amendment 1** (after the (b) casual runs at LR 1e-2 and 3e-2, before the (c) casual runs and before
   any type-hint phase-2 run). The (b) neurons give TKL 0.21 already at LR 1e-2 (F bf16 0.66), so its LR curve does
   not reach the targets 0.60 and 0.45 from the weak side, and the pre-registered interpolation is "not covered".
   The edge rule covers only a step UP. Change: for an arm whose weakest grid point is already stronger than TKL
   0.60, I add LRs 3e-3, 1e-3, 3e-4 (casual, seed 0). For type hints the same: if the lowest LR is already above
   the habit line with no lower point, add 3e-3 and 1e-3. The decision rule does not change.
4. 2026-11-01. Phase 2 done (seed 0). Casual: 9 grid runs + the 3 amendment runs of (b); type hints: (b) and (c)
   at LR 1e-2 to 3e-1 with `make_edit.py --neurons`, and the amendment runs of (b) at 1e-3 and 3e-3 (the (b) habit
   was 95% at LR 1e-2). Held-out replies of every run (`demos/gen.py --sets heldout`). Verdict by the rule: (b) and
   (c) are "worse" on both tasks. RESULTS.md.
- 2026-11-01: in saved model replies, the home-directory prefix of example paths is replaced by `/<home>` (the same rule as the type-hints demo); this changes no count.
