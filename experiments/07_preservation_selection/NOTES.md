# 07 Preservation and selection: pre-registration and run log

Written before the grid runs. At that time, these runs were done: the calibration (`calib07.py`, layers 15 and 20),
one 20-step smoke test, and the LR pilot of the full layer without penalties (see "LR rule"). The pilot measures
only strength and F of an edit with no sparsity, so it gives no data about which columns a penalty keeps.

## Question

"Does preservation make the edit choose selective neurons?" In experiments 01-06 we choose the neurons by hand
(gradient attribution or activity) and train them with **no** preservation loss. Here we train the **full**
`down_proj` of one layer with a preservation loss and a group-sparsity penalty, and we let the optimizer choose the
columns.

Hypothesis. A change of column j acts on the output only where neuron j fires: write = a_j(x) · ΔW_j. A
preservation loss on off-target text penalizes this off-target write (experiment 03: the off-target write orders F
with Spearman 0.90). A group lasso makes the optimizer keep only some columns. So the kept columns should be those
with a large write on the target and a small write off-target, which are the selective neurons.

- If yes: "hand selection and preservation training find the same neurons". Selection is then a cheap shortcut to
  the solution that preservation training finds.
- If no: preservation finds a different way to be local. We report how.

## Setup

- Task: the casual edit of experiment 01 (the same data, target loss, batch 4, 400 steps, constant LR, seeds).
- **Main layer: 15.** In the sel8 choice of 01, layers 13, 15 and 24 have 2 neurons each (seeds 0 and 2). We take
  layer 15 because it is also the layer of the same-size LoRA reference (`lorad`) and of one neuron of the activity
  rule (04). Hand-selected neurons in layer 15: sel8 **7291** (all seeds) and **6582** (seeds 0, 2); srand8 10840,
  10916; colsel8 (04) 3172. Their selectivity S = g/c is in the most selective 1.6% of the layer, and their c is in
  the top 1.6%.
- **Control layer: 20.** It has none of the hand-selected neurons (sel8, srand8, colsel8 of all seeds). Same width
  (12,288 neurons). Question there: what does the optimizer keep in a layer with no hand-selected neuron?
- **Edit.** ΔW, the change of the full `down_proj` weight of the layer (1,536 × 12,288 = 18,874,368 numbers),
  added as a hook: out + a ΔWᵀ. ΔW starts at 0.
- **Optimizer** (`common07.AdamColProx`, the same for all arms): Adam (β 0.9, 0.999) with one second moment per
  column (the mean of g² over the column), then the exact proximal step of λ_g Σ_j ‖ΔW_j‖ in the same metric. A fixed
  point of this update is a stationary point of the full objective, and a column stays at exactly 0 while the norm of
  its gradient (momentum) is ≤ λ_g. Reason: per-element Adam divides out the size of the gradient, so with
  per-element Adam a group lasso cannot see which columns have a large gradient.
- **Preservation set:** the 300 SlimOrca task items of the casual data (`kind = "pres"`; teacher = the original model
  without a system prompt, so the term is KL(original ‖ edited) on their replies). They are disjoint from the F set
  (HumanEval, GSM8K, SQuAD, Wikipedia, WikiText-2; 0 items share an 8-word sequence with an F item). The KL uses the
  full vocabulary from a teacher pass in the same step (the edit off), as in `lib/train1b.py`. We do not cache top-k
  logits: the teacher pass is exact, and `train1b.py` already does it for these items.

## Arms

| arm | loss | prox |
|---|---|---|
| full | KL_casual | none |
| C | KL_casual + λ_kl · KL_pres | λ_g |
| D | KL_casual + λ_d · mean ‖ΔW a‖² over all positions of the preservation replies (exact: tr(ΔW Σ_pres ΔWᵀ), Σ_pres = the mean of a aᵀ at the layer input, original model) | λ_g |
| C0 | KL_casual + λ_kl · KL_pres | none |
| G0 | KL_casual | λ_g |

Arms D and G0 do not run the 4 preservation items per step (they do not need them). The activations a at the
edited layer do not depend on the edit, so the D penalty is the exact squared write of the edit at that layer.

### Grid (layer 15, seed 0)

- λ_g ∈ {λ₁₂₈, λ₃₂, λ₈} = {0.204, 0.336, 0.531}: the values at which 128, 32 and 8 columns of the mean target
  gradient at ΔW = 0 (64 items) are larger than λ_g (`out/calib_L15.json`).
- λ_kl ∈ {1, 5, 25} (5 = the preservation weight of the released demo edit).
- λ_d = r · λ_kl. r = KL_pres / write_pres, the mean over the last 100 logged steps of the C0 run at λ_kl = 5. So
  at that point the two penalties have the same size. The D runs start after this C0 run.
- Runs: C 3 × 3, D 3 × 3, C0 3 (λ_kl), G0 3 (λ_g). Then seeds 1 and 2 at the chosen C point, the chosen D point,
  and G0 at the λ_g of the chosen C point. Layer 20: C, G0 and C0 at the chosen (λ_kl, count level), with the λ_g of
  the same count level from `out/calib_L20.json`, seed 0.
- **Chosen point** (fixed now): among the 9 C runs, the run with TKL ≤ 0.70 and the smallest n95; ties: the lower F.
  If no C run has TKL ≤ 0.70, the run with the lowest TKL. The same rule for D.

### LR rule

One LR for all grid arms: the pilot LR (3e-6, 1e-5, 3e-5, 1e-4; full layer, no penalty, seed 0) where the TKL is
nearest to 0.45 (in ln). Reason: the penalties make an edit weaker, and the reference curves cover TKL 0.37-1.15
(sel8) and 0.44-0.64 (lorad). Edge rule: if more than 5 of the 9 C runs have TKL > 0.9, I run the C and D grids
again at 3 × LR and use that grid (declared in the run log).

## Measurements

- **Kept columns** (rule fixed now): K = the smallest set of columns that holds ≥ 95% of ‖ΔW‖²_F. Also: the number
  of non-zero columns (nnz) and the 99% set.
- **Selectivity** of neuron j: S_j = g_j / c_j (c = mean |a| on the casual training replies, g = mean |a| on the
  reference text, both at the scored positions; the S of 03 / 04; `out/selectivity.json`). pct_S(j) = the fraction
  of the layer's neurons with a lower S. For K: the median pct_S (unweighted and weighted by ‖ΔW_j‖²), the median S,
  and the same for c.
- **Overlap.** Is 7291 / 6582 in K, and the rank of their column norm. Jaccard(K, A) where A = the top |K| neurons
  of the layer by the activity rule of 04 (ln(c/g), only c ≥ the 90th percentile of the layer). Also Jaccard with the
  top |K| by attribution score and by the initial gradient norm (`calib07.py`).
- **Strength and locality.** TKL (60 held-out messages; `lib/tkl.py` code) and F (256 items; `lib/forget.py` code),
  both bf16. Ratio F / F_ref(TKL of the run) for the seed-mean curves of sel8, srand8 and lorad of 01 (interpolation
  in ln F vs ln TKL, as in `analyze1c.py`; "not covered" outside a curve). Task KL on the 300 SlimOrca items is
  reported, but for C and C0 it is a training-set number. Also the direct write on the reference set.
- **Cost:** seconds per step and the peak GPU memory, on one NVIDIA GB10.

## Predictions and decision rule

At the chosen C point, 3 seeds. **YES** ("the same neurons") if all of these are true:

1. n95 ≤ 64 (the optimizer keeps only a few columns);
2. the median pct_S of K is ≤ 0.05 (the hand-selected neurons are at ≤ 0.016);
3. 7291 or 6582 is in K for ≥ 2 of 3 seeds;
4. preservation causes it: the median pct_S of K is lower for C than for G0 at the same λ_g, for all 3 seeds.

**NO** ("preservation finds another way") if 2 or 3 is false. Then I report what the kept columns are (S, c, g, the
gradient rank) and how local the edit is. **PARTLY** for the other cases (for example: 2 and 3 are true, but G0 keeps
the same columns, so the gradient and not the preservation chooses them).

My predictions: 1 true; 2 true; 3 true (7291 has a high c and a low S); 4 uncertain, because at ΔW = 0 the
preservation gradient is 0, so the first columns that open depend only on the target gradient (as in G0).
Locality: C at the chosen point is as local as sel8 at the same TKL or more (F ratio ≤ 1); C0 is more local than
lorad; the full layer without penalty is the least local.

## Run log

1. 2026-11-01. Calibration of layer 15 (`out/calib_L15.json`): λ at 8 / 32 / 128 / 512 columns = 0.531 / 0.336 /
   0.204 / 0.112. Smoke test: 20 steps at LR 1e-4, about 1.2 s per step. LR pilot started (4 runs).
2. LR pilot (full layer, no penalty, seed 0): TKL 0.559 at LR 3e-6, 0.390 at 1e-5 (F 0.040 and 0.070). The rule
   gives **LR 1e-5** (ln distance to 0.45: 0.14 against 0.22). The pilot runs at 3e-5 and 1e-4 continue only as
   points of the "full" curve. Calibration of layer 20: λ at 8 / 32 / 128 / 512 columns = 0.328 / 0.228 / 0.138 /
   0.072. Grid queued on the GB10 machine: C0 (λ_kl 5) first, then C 3 × 3, G0, the other C0 runs.
3. C0 at λ_kl 5 done. Over steps 300-399: KL_pres 0.00734, write_pres 3.50, so r = 0.002095 and λ_d =
   0.002095 / 0.01047 / 0.05237 (for λ_kl 1 / 5 / 25). D grid queued (D runs are named by the λ_kl they match).

### Amendment 1 (2026-11-01, after 1 grid run; before any other C, D or G0 run)

The first C run (λ_kl 1, λ_g = λ₁₂₈, LR 1e-5) kept 47 columns, but its edit stayed weak (TKL 2.0; original
2.61). The reason is the LR, not the objective: ‖ΔW‖ grew linearly for all 400 steps (0.04 at step 40, 0.30 at
step 399), and the largest columns moved at about the maximum Adam speed (400 · 1e-5 · √1536 ≈ 0.16 for a column
with a consistent gradient). A sel8 column at its LR 1e-2 moves about 24 × faster per element (step lr · rms(W) ≈
2.4e-4). The LR rule of the pre-registration chose the LR for an edit that spreads over 12,288 columns, and it does
not apply to an edit with a few columns. The edge rule (3 × LR) is too small for a gap of about 24 ×.

Changes:
- The C, D and G0 grids at LR 1e-5 stop (the two finished C runs at λ₁₂₈ stay in the tables). The full and C0
  arms (dense) keep LR 1e-5.
- **Sparse LR pilot:** C at (λ_kl 5, λ_g = λ₃₂), LR 1e-4, 3e-4, 1e-3. Rule: the lowest of these LRs with TKL ≤
  0.70; if none, the LR with the lowest TKL. The C, D and G0 grids (the λ values of the pre-registration) then run
  at that LR. Everything else (λ_d, the chosen-point rule, the decision rule) does not change. The fixed points of
  AdamColProx do not depend on the LR, so a larger LR only brings the 400-step result nearer to the optimum.

### Amendment 2 (2026-11-01, after the sparse LR pilot at 1e-4 and 3e-4)

C (λ_kl 5, λ₃₂) gave TKL 1.69 at LR 1e-4 (9 kept columns) and 1.48 at 3e-4 (5 kept columns). The run at 3e-4 is
less limited by the LR (‖ΔW‖ 1.14), and fewer columns survive, so the penalty λ₃₂ itself gives a weak optimum.
The pre-registered λ_g levels (8 / 32 / 128 columns of the initial gradient) are too strong for this task: no
level reaches the strength of the reference curves (TKL ≤ 1.15). Changes:
- **λ_g pilot** at LR 1e-3 (the LR pilot run at 1e-3 runs first; if it diverges, I repeat this pilot at 3e-4):
  C (λ_kl 5) at λ₁₂₈ = 0.2035, λ₅₁₂ = 0.1118, λ₂₀₄₈ = 0.0528 (the same rule: the level where 128 / 512 / 2048
  columns of the initial gradient norm are larger than λ_g; values from `out/calib_L15.json`).
- **Grid λ_g:** the 3 consecutive levels of {λ₂₀₄₈, λ₅₁₂, λ₁₂₈, λ₃₂} with the most pilot runs at TKL ≤ 0.9 (ties: the
  sparser set). λ_kl {1, 5, 25}, λ_d and the other rules do not change.
- The decision rule stays as written. A kept set with more than 64 columns makes item 1 false (the result is then
  NO or PARTLY, not YES).
4. Sparse LR pilot: C (λ_kl 5, λ₃₂) TKL 1.69 / 1.48 / 1.40 at LR 1e-4 / 3e-4 / 1e-3 (no run ≤ 0.70, so the
   rule gives **LR 1e-3**, the lowest TKL; no divergence). λ_g pilot at LR 1e-3 (C, λ_kl 5): TKL 0.58 (λ₂₀₄₈, 63
   kept columns), 0.77 (λ₅₁₂, 39), 1.08 (λ₁₂₈, 27), 1.40 (λ₃₂, 5). The rule of amendment 2 gives the grid
   **λ_g ∈ {λ₂₀₄₈, λ₅₁₂, λ₁₂₈}** = {0.0528, 0.1118, 0.2035}. The three λ_kl 5 pilot runs are the λ_kl 5 row of the
   C grid (same settings, seed 0). Queued: the other 6 C runs, 3 G0, 9 D, all at LR 1e-3.
5. C grid done (9 runs at LR 1e-3). **Chosen C point: λ_kl 1, λ_g = λ₂₀₄₈** (TKL 0.543, n95 18; the other runs with
   TKL ≤ 0.70 have n95 63 and 790). Queued: seeds 1 and 2 of C and G0 at this point; layer 20 C (λ_kl 1, λ₂₀₄₈ of
   layer 20 = 0.0342), G0 (same λ_g) and C0 (λ_kl 1, LR 1e-5), seed 0. D seeds follow when the D grid is done.
   Note (no rule change): at λ_kl 25 the C runs keep 147-790 columns and have a HIGHER task KL (0.03-0.13, the
   training-set preservation KL) than at λ_kl 1 or 5. So at this weight and LR the optimization does not reach a
   good point (a noisy 4-item KL term with a large weight).
6. All runs done (47 with the pilots). From 08:20 another experiment shared the GPU (slower steps; the results are
   not affected). Post-hoc diagnostic `diag07.py` (not pre-registered): overlap and direction of the update columns
   for C / D / G0 pairs. Results: RESULTS.md.
