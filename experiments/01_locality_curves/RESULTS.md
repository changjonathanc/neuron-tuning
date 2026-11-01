# Phase 1c results: forgetting vs effect curves, and attribution vs selectivity

> **Names in this log.** Phase 1c = this experiment; Phase 2b = 02_conditional_rules; E1 = 03_predictor; E2 = 04_activity_selection; E3 = 05_typical_lora; E4 = 06_facts. Phase 1, Phase 1b and Phase 2 are earlier studies that are not in this repository; their shared code and data (adapters, training, the forgetting set F, the teacher-KL measure) are in [`../lib/`](../lib/).

Pre-registration: `NOTES.md`; run log entries 1-3. Tables: `out/tables.md` (all ratios, all
variants, all points). Data: `out/summary.json`. Figure: `fig_curves.png`. Post-hoc floor diagnostic:
`out/floor1c.json`. Checkpoints and per-run files are not committed (`runs/` is ignored).

## What was done
- 8 families (method x budget), each an LR sweep, 3 seeds per point, all without the preservation loss
  (w_pres = 0). 111 runs: the 105 pre-registered runs (12 reused from Phase 1b) and 6 runs from the pre-registered
  edge rule (sel128 at LR 1e-3, dense128 at LR 0.3; run log entry 2).
- Per run: teacher KL (TKL, effect; lower = stronger; original model 2.607), forgetting F (KL(original || run) per
  token on 256 unrelated items, macro mean over 7 domains), F_noQA, F_med, task KL (300 SlimOrca items), MMLU-570.
- At each pre-registered target TKL, F of each family is interpolated along its seed-mean curve (ln F vs ln TKL).
  ρ(A, B) = F_A / F_B. 95% CI: bootstrap over eval messages, forgetting items (per domain) and seeds, 10,000
  resamples. A comparison is "not covered" if a curve does not reach the target or > 10% of resamples miss it.

## Figure
`fig_curves.png`: F (log) vs TKL (log); left panel ~13 k parameters (K = 8 / rank-1 LoRA on 1 layer), right panel
~200 k parameters (K = 128 / rank-1 LoRA on 16 layers). One curve per family (seed means in LR order), 95% CI bars
on both axes, small markers = single seeds, dotted vertical lines = target effects, dashed horizontal line = the
measurement floor of F (post hoc, see caveat 1). Read: a curve that is lower at the same x forgets less at the same
effect. In the ~13 k panel the blue (selected) and orange (selectivity-matched) curves are 1-2 orders of magnitude
below the LoRA (yellow) and dense-random (green) curves; in the ~200 k panel the gap is 6-12x. At K = 8 orange is below blue; at K = 128 blue is below orange.

## Pre-registered verdicts (metric F)

| question | comparison | target TKL | ρ = F_A / F_B [95% CI] | seeds that agree | verdict |
|---|---|---:|---|---|---|
| Q1 sel vs LoRA, ~13 k | sel8 / lorad | 0.60 | 0.085 [0.060, 0.118] | 3/3 | **not covered** |
| | | 0.45 | 0.033 [0.023, 0.079] (35% of resamples dropped) | 3/3 | |
| Q1 sel vs LoRA, ~200 k | sel128 / lorac | 0.30 | 0.082 [0.065, 0.108] | 3/3 | **selected columns forget less than LoRA at matched effect** |
| | | 0.20 | 0.161 [0.098, 0.240] (8.7% dropped) | 2/2 (seed 2 does not reach 0.20) | |
| Q2 attribution vs selectivity, K = 8 | sel8 / srand8 | 0.60 | 1.65 [1.28, 2.11] | 3/3 | **selectivity-matched neurons forget less** |
| | | 0.45 | 2.02 [1.45, 3.02] | 3/3 | |
| Q2 attribution vs selectivity, K = 128 | sel128 / srand128 | 0.30 | 0.50 [0.40, 0.65] | 3/3 | **not covered** |
| | | 0.20 | – (srand128 does not reach 0.20) | – | |
| Q3 selectivity, K = 8 | srand8 / dense8 | 0.60 | 0.006 [0.004, 0.009] | 3/3 | **selectivity reduces forgetting** |
| | | 0.45 | 0.004 [0.003, 0.007] | 3/3 | |
| Q3 selectivity, K = 128 | srand128 / dense128 | 0.30 | 0.16 [0.10, 0.25] | 3/3 | **not covered** |
| | | 0.20 | – (neither reaches 0.20) | – | |

Why three comparisons are "not covered":
- Q1 at ~13 k: the rank-1 LoRA on one layer does not get a stronger effect than TKL ≈ 0.44. Its seed-mean TKL is
  0.463, 0.444, 0.454 at LR 3e-4, 1e-3, 3e-3 (the curve goes back up). So the 0.45 target is at the edge of the LoRA
  curve, and 35% of the resamples miss it. At 0.60 the ratio is 0.085 and all three seeds agree.
- Q2 and Q3 at K = 128: srand128 and dense128 do not reach TKL 0.20 at any LR (lowest seed means 0.219 and 0.232;
  the TKL gets worse at the top LR). The edge rule did not allow an upward step for srand128 and gave one step
  (LR 0.3) for dense128, which made its TKL worse.

The verdicts do not change with F_noQA or F_med (same CI side of 1 at every covered target; `out/tables.md`).
On the task KL (SlimOrca items), Q2 at K = 8 is weaker: 1.12 [0.69, 1.73] at 0.60 and 1.29 [0.80, 2.28] at 0.45
(same direction, CI includes 1; seed 1 reverses: 0.77 and 0.85). All other covered comparisons have the same direction on task KL.

## Main results in words
1. **Edits on context-specific neurons are local; a rank-1 LoRA is not.** At the same effect, the selected columns
   change the original model on unrelated text about 12x less than a rank-1 LoRA of the same size at ~13 k
   (ρ 0.085 at TKL 0.60) and 6-12x less at ~200 k (ρ 0.08 at 0.30, 0.16 at 0.20). At ~200 k this is the
   pre-registered verdict. At ~13 k the formal verdict is "not covered" only because the LoRA curve stops near the
   second target. Descriptive (post hoc): the sel8 point at LR 1e-1 (TKL 0.373, F 0.010) has a stronger effect AND
   less forgetting than every LoRA point (TKL ≥ 0.444, F ≥ 0.030).
2. **At K = 8, attribution adds no locality beyond selectivity: selectivity-matched neurons forget less.** Columns of 8 neurons with the same activity
   profile (high on casual text, low on general text) that attribution did NOT select forget 1.6-2x LESS than the
   attribution-selected columns at the same effect (all seeds agree), and 170-250x less than the dense random control
   of Phase 1b. This happens although the matched control is slightly LESS selective than the selected neurons at
   K = 8 (NOTES, matching check). Selectivity alone also beats the LoRA (srand8 / lorad 0.052 [0.040, 0.067] at 0.60).
3. **At K = 128, attribution helps.** At TKL 0.30 the selected columns forget 2x less than the selectivity-matched
   columns (0.50 [0.40, 0.65], all seeds agree; formal verdict "not covered" because the second target is not
   reached by srand128). Attribution also gives a stronger maximum effect: lowest seed-mean TKL sel128 0.182,
   srand128 0.219, dense128 0.232 (lorac 0.153). At K = 8 the maximum effects are about equal (sel8 0.373, srand8
   0.361, dense8 0.368; lorad 0.444).
4. **The Phase 1b dense random control was a weak control.** Its neurons are active on all text. Its forgetting at
   matched effect is 180-235x (K = 8) and 6x (K = 128, TKL 0.30) higher than that of the selectivity-matched control.
   At K = 8 it is even worse than the LoRA (F 0.34 vs 0.037 at TKL 0.60).

## Secondary results (no verdict)
- **Effect at the same LR** (seed means, LR 1e-2): sel8 0.670, srand8 0.822, dense8 1.709; sel128 0.214, srand128
  0.258, dense128 0.314. Attribution gives more effect per step at the same LR. So "the same LR" comparisons (Phase 1b
  view) mix effect and locality; the matched-effect curves remove this.
- **MMLU-570**: every family point is 0.616-0.646 (original 0.626, standard error about ±0.02). MMLU does not see
  any of these differences.
- **Floor-corrected ratios** (post hoc, point estimates, F - floor with floor 0.0011 at ~13 k and 0.0016 at ~200 k):
  Q1 sel8 / lorad 0.057 and 0.026; sel128 / lorac 0.060 and 0.151; Q2 sel8 / srand8 2.5 and 2.9; sel128 / srand128
  0.42; Q3 srand8 / dense8 0.002. Every direction holds; most ratios move further from 1 (only sel8 / srand8 gets
  larger in value). The corrected values have no CI and depend on the floor value (see caveat 1).

## Caveats
1. **F has a measurement floor of ~0.001** (post-hoc diagnostic, run log entry 3). Any non-zero change gives
   F ≈ 0.0011 (K = 8) to 0.0016 (K = 128) from bf16 rounding; F of an exactly zero change is 0. At matched effect
   srand8 (F ≈ 0.002-0.0025) is only ~2x above the floor and sel8 (0.003-0.005) ~3-4x. The K = 8 comparison between
   two local edits is thus a comparison of small numbers near the floor; the floor-corrected ratio (2.5-2.9) is in
   the same direction. The floor is a property of the bf16 forward pass, so a merged edit in bf16 also has it.
2. **Three seeds.** The seed bootstrap is coarse; I report the per-seed ratios. All covered verdicts have 3/3 seeds
   in the same direction (2/2 for sel128 / lorac at 0.20, where seed 2 does not reach the target). sel128 has a large
   seed spread at high LR (TKL 0.162-0.207 at LR 3e-2).
3. **Matching is on activity only** (mean |activation| on casual vs reference text), not on output direction or
   weight norm. It excludes only the same seed's attribution pool. The reference text uses the same 7 domains as the
   forgetting set (other items).
4. **Targets and grids came from Phase 1b operating points**, which I had seen. The edge rule added 6 runs; it used
   the TKL only.
5. **TKL is closeness to the teacher, not a judged behaviour.** Phase 1c has no judge. A separate check
   (`LOC_NOTES.md`) judges the K = 8 families and the LoRA near TKL 0.45; it is not part of these verdicts.
6. **Interpolation** is linear in ln F vs ln TKL between adjacent LR points; with half-decade LR steps some segments
   are long (for example sel128 between LR 1e-3 and 3e-3: TKL 0.625 -> 0.305).
7. One model (Gemma E2B), one behaviour (casual human-like replies), down_proj columns only, 400 steps, constant LR.

## Independent review (2026-10-30; read-only pass; overrides the text above where they differ)
An independent reviewer checked the verdicts, the code and the raw files. The reviewer did not change files.

**Holds.** The verdicts follow the pre-registered rules: coverage, the 10% drop rule, the CI side at both targets,
and the edge rule (re-running `edge1c.py` gives the same decisions). The pre-registration and the analysis code did
not change after the results; only `plot1c.py` changed (declared). The 12 reused Phase 1b runs have the correct arm,
LR, seed and w_pres = 0. The stored F and TKL values, the table values, the floor-corrected ratios and the post-hoc
dominance of sel8 at LR 1e-1 over every LoRA point (also per seed) reproduce from the raw files.

**Corrected in the text above (before this section was added):**
- Result 2 used the words "selectivity explains". That is the label of the equivalence verdict (CI inside
  [0.5, 2] at both targets), which did not occur (CI at TKL 0.45 is [1.45, 3.02]). The verdict is
  "selectivity-matched neurons forget less". So at K = 8 attribution adds no locality beyond selectivity.
- Result 4 said "100x" for dense8 vs srand8. The ratio is 179 [108, 265] at TKL 0.60 and 235 [148, 327] at 0.45.
  (100x is the dense8 / sel8 ratio.)
- The figure text said "1-2 orders of magnitude" for both panels. This is true at ~13 k only; at ~200 k it is 6-12x.
- The floor text said "most ratios get larger". Most ratios move further from 1; only sel8 / srand8 gets larger.

**Limits to add:**
- The floor does not change the direction of the K = 8 result: it is the same for sel8 and srand8 (0.00106 and
  0.00105) and it is approximately additive (F(s) ≈ floor + s² (F(1) − floor)). But the size of the corrected ratio
  is not reliable: the corrected srand8 F at TKL 0.60 (~0.0008) is smaller than the floor itself; a floor of 0.0009
  or 0.0013 gives ratios of ~2.3 or ~3.2. The floor was measured on one adapter per budget (seed 2), not on LoRA or
  dense. The absolute K = 8 gap is small (0.0013 nats/token, vs 0.037 for the LoRA).
- On task KL, seed 1 reverses the K = 8 sel8 / srand8 result (0.77, 0.85).
- At K = 128, "attribution helps" rests on one target (0.30), at the end of a long segment of the sel128 curve. The
  stronger maximum effect of sel128 (0.182 vs 0.219) holds for seeds 0 and 1; seed 2 reaches only 0.207. sel128 also
  has a 4x seed spread in F at LR 3e-2 (0.064, 0.045, 0.015).
- At K = 8 the maximum effect of sel8 is not known: its grid stops at LR 1e-1 while its TKL still goes down. So
  "maximum effects about equal at K = 8" is not established.
- Q1 at ~13 k: quote only TKL 0.60 (ρ 0.085). The point at 0.45 interpolates a LoRA segment where TKL changes
  0.463 -> 0.444 while F changes 0.086 -> 0.196, so its value (0.033) is not reliable. The secondary srand8 / lorad
  comparison is also "not covered" (0.60 value only).
- Q3 changes two things at once: srand neurons are more active on casual text (~99.7th percentile vs top 10%) AND
  less active on general text than the dense control.
- Long segments that the point estimates use: lorac 0.420 -> 0.228 (Q1 at 0.30), srand128 0.431 -> 0.258, srand8
  0.822 -> 0.474 (caveat 6 gave a segment that is not used).
- The MMLU statement is a range over all points, not the pre-registered "points nearest the targets" (the
  conclusion is the same: all are 0.616-0.646 vs 0.626). The pre-registered "F vs LR" view is in `out/tables.md`
  only. Seed resampling is independent across families, although sel and srand of the same seed share the
  attribution items and batch order (pre-registered; conservative).
- The control matching (NOTES) gives median g/c 0.105-0.132 (sel8) vs 0.137-0.143 (srand8); the review's
  recomputation gives 0.102-0.119 vs 0.124-0.140. Same direction. In absolute terms srand8 is also ~5-15% less active
  on reference text and less active on casual text.

**Corrected conclusion.** Edits on down_proj columns of context-specific neurons are much more local than a rank-1
LoRA at the same effect (~12x at ~13 k and TKL 0.60; 6-12x at ~200 k, pre-registered verdict). At K = 8,
selectivity-matched neurons that attribution did not choose are at least as local (1.6-2x less F, near the
measurement floor). At K = 128, attribution gives ~2x less forgetting at TKL 0.30 and a stronger reachable effect,
but this rests on one target and not all seeds. The Phase 1b dense random control is not a fair control for
attribution.
