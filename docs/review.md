# Independent review of the README claim (2026-10-31)

An independent, read-only review of the README claim and of experiments 03-06 (a separate review). The
README was changed after this review: fixes (A) and (B) are applied. The review scripts were not kept
and are not kept. Quotes are from the README version that the reviewer read. Ratios are F_A / F_B at the same
strength of edit (F = change on unrelated text); "N× less" = 1 / ratio.

Names in this review:

| name | meaning |
|---|---|
| 8 / 128 gradient neurons | output columns of 8 or 128 neurons chosen by gradient attribution (`sel8`, `sel128`; `col8`, `col128` in 02 and 06) |
| 8 matched neurons | output columns of 8 neurons with the same activity profile as the gradient choice, not chosen by it (`srand8`) |
| 8 / 128 random active neurons | output columns of random neurons among the 10% most active of each layer (`dense8`, `dense128`) |
| 8 / 128 activity neurons | output columns of neurons chosen by activity only (`colsel8`, `colsel128`; experiments 04 and 06) |
| rank-1 LoRA | rank-1 LoRA on `down_proj` of 1 layer (`lorad`, `lora_d`) or of 16 layers (`lorac`), the same size as 8 or 128 neurons |
| standard LoRA | rank 8, alpha 16, all modules of all layers (`lora_all8`) |
| teacher KL | the strength of the casual edit; trigger KL T_e: the strength of a rule edit |
| S | selectivity before training (mean activity off the target ÷ on the target; low = selective) |
| off-target write | the activity of the edited units on unrelated text times the update, after training (`D_off`) |

## Number check (README top and `figures/headline.py`)

| README / figure value | source | recomputed | status |
|---|---|---|---|
| casual: random active 9.2, rank-1 LoRA 1, gradient 0.085, matched 0.052 (teacher KL 0.60, bf16) | experiments/01_locality_curves/out/tables.md l.20, 52, 68 | 0.3432/0.0372 = 9.23; 0.085; 0.052 | correct |
| casual: standard LoRA 1/1.46 | experiments/05_typical_lora/out/tables.md (rank-1 LoRA / standard LoRA 1.46 at 0.60) | 0.685 | correct |
| weather: gradient 37.1, activity 0.11 (at T_e) | 03 RESULTS (b); 04 RESULTS | same | correct |
| "12-19x less than a LoRA of the same size" | gradient 0.085, matched 0.052 at teacher KL 0.60 | 11.8x, 19.2x | correct, see A2 |
| "7-8x less than a standard LoRA" | 05 tables: gradient / standard LoRA 0.124 (0.60), 0.141 (0.45) | 8.1x, 7.1x | correct for the gradient neurons only, see A3 |
| "9-37x worse than the LoRA" | random active 9.2; weather gradient 37.1 | | range is incomplete, see B3 |
| "9x more local than the LoRA", "FP 11% instead of 84%" | 04 RESULTS: activity / rank-1 LoRA F 0.11; false positives 0.11 against 0.84 (gradient) | 9.1x | correct (84% is the gradient edit, not the LoRA; the LoRA is 80%) |
| "76% of the way to the system prompt" | demo table: (3.56-1.09)/(4.33-1.09) | 0.762 | correct |
| "210 edits ... Spearman 0.90" | 03 out/pred.json + f32.json | 0.898 | correct |
| "fp32: 19-50x against the rank-1 LoRA" | 03 RESULTS: gradient 0.052, matched 0.020 | 19x, 50x | correct |
| "1-3x against the standard LoRA on chat-style task replies" | 05 RESULTS task KL 0.33-0.93 | 1.1-3.0x | correct |
| "K = 8: about 12 k numbers, same as rank-1 LoRA on one layer" | 8 columns 12,288-12,296; rank-1 LoRA 13,824 | | correct ("about") |

## (A) Errors that must be fixed

1. **Stale status.** README Limits: "A knowledge task (facts about made-up people) is running." Experiment 06 is
   done (verdict "partly supported"). Fix: replace with the result of 06 (see B4) or remove "running". Also "four
   behaviours" is now five tasks.
2. **The 12x (gradient neurons / rank-1 LoRA) is a "not covered" pre-registered verdict.**
   experiments/01_locality_curves/RESULTS.md Q1 ~13 k: verdict "not covered" (the 0.45 target is at the end of the
   rank-1 LoRA curve); matched / rank-1 LoRA (19x) is also "not covered". The 0.60 value is sound (the rank-1 LoRA
   has a measured point at teacher KL 0.642, 3/3 seeds). Fix: add "(at the weaker of the two pre-registered
   targets; the stronger one is out of the LoRA's reach)" or cite the covered standard-LoRA verdict (05, 7-8x) as the
   main number.
3. **"8 such neurons ... 12-19x ... and 7-8x" mixes two families.** 12x and 7-8x are the gradient neurons; 19x is
   the matched neurons, whose ratio to the standard LoRA is 13-14x (0.075, 0.070). Fix: "12-19x less than a rank-1
   LoRA and 7-14x less than a standard LoRA (rank 8, all layers)".
4. **The caveat 1 of the 06 RESULTS omits that amendment 1 flips the verdict** (see section 2 below). Under the
   pre-registered target teacher KL 0.30, Spearman(S, F) over the 4 `down_proj` methods = **0.20** (< 0.4) ->
   "**not supported**". It is also 0.20 at accuracy 0.3, where all 4 methods are covered. Only at accuracy 0.5,
   where the rank-1 LoRA drops out, it is 0.5. Fix: state this in caveat 1 of the 06 RESULTS and in the verdict
   line.
5. **The casual edit with activity neurons (04) is missing from the headline figure.** It is the method that the
   README recommends ("choosing by activity"), and it is blue by definition (S 0.037). Its point is activity /
   rank-1 LoRA = 0.30 (3.3x less; fp32 0.22), that is 3.6x worse than the gradient neurons. Fix: add it to the
   casual panel, or say in the caption that the casual blue dots are the gradient choice and the matched control,
   not the activity rule.

## (B) Wording that overclaims (proposed replacement)

1. "the model changes **about 10x less** on unrelated text than with LoRA, at the same strength of edit."
   The measured range is 2-20x and it depends on the baseline, the strength and the text: 8 gradient neurons
   against the standard LoRA 7-8x; 128 gradient neurons against the standard LoRA 8x at teacher KL 0.30 but 3x at
   0.20; 128 activity neurons (06) against the standard LoRA 5x at accuracy 0.5, 3x at 0.7; casual activity neurons
   2.3x against the standard LoRA (0.0113 / 0.0255); lower-case rule with activity neurons 1.7x WORSE (CI includes
   1); chat-style task text 1-3x. Replace: "**changes 3-20x less on unrelated text than a LoRA (about 8x against a
   standard rank-8 LoRA at medium strength); the gap is smaller for stronger edits and for text near the
   target.**"
2. "You can measure which case you are in **before** training." The pre-training statistic S separates three
   coarse groups only. Add the activity-neuron points of 04 (S 0.017-0.062, ratio to the LoRA 0.11-1.7) to panel
   (b) of 03: Spearman 0.96 -> **0.70** (14 points); 03 + 04 + 06: **0.70** (17 points); among the selective
   points (S < 0.3, n = 8) it is **-0.43**. In 06, the 128 gradient neurons (S 0.48) forget 1.6x more than the 128
   random active neurons (S 0.66), and the rank-1 LoRA (S 0.39) forgets most. (The reviewer's S for 04 uses the
   activity of `select_e2.py` (training replies), not that of `predict.py` of 03: approximate.) Replace: "Before
   training, you can see whether the neurons fire everywhere; this predicts a non-local edit. Low activity off the
   target is necessary but not sufficient: the size of the update also matters, and it is known only after
   training."
3. "Neurons that fire everywhere give the **least local** edits, 9-37x worse than the LoRA." Counter-examples: 128
   random active neurons = 1.01x the rank-1 LoRA; in 06, 128 random active neurons = 0.28x the rank-1 LoRA at
   accuracy 0.3 (more local than the same-size LoRA); the recipe rule with gradient neurons (43.7x) and the weather
   rule with matched neurons (47.4x) are not in the range. Replace: "Edits on neurons that fire everywhere are
   1-47x less local than the same-size LoRA (9x for the casual control, 37x for the weather rule)."
4. "It works the same way with very few parameters (128 gains, 76%)." The demo edit used a preservation loss and
   attribution neurons; it is not a locality result, and its judge score rewards short casual replies (a plain
   style prompt scores 3.95 > 3.56). Replace: "With a preservation loss, one gain on each of 128 neurons closes
   76% of the judge-score gap to the system prompt (demo)."
5. "Choosing by activity instead **fixed it**" (Results bullet 3). The pre-registered verdict of 04 is "**not
   supported**", because the same rule fails on the casual edit (3.6-5.9x less local than the gradient and matched
   neurons), and F on the lower-case rule is not determined; recipe is "not covered". Replace: "On two of the three
   rules (weather, recipe), choosing by activity fixed it ... On the casual edit the same rule chose early-layer
   neurons and gave a 3.6x less local edit than the gradient choice (verdict of 04: not supported as
   pre-registered)."
6. "this one number orders the change ... Spearman 0.90" is correct, but the number is measured after training (it
   contains the update size). Keep it, but say "after training", so that the reader does not link it to the "before
   training" sentence.
7. Figure caption "trained until it is equally strong": the neuron edits cannot reach the strong end (lowest
   teacher KL: 8 gradient neurons 0.37, 8 activity neurons > 0.45 not reached, standard LoRA 0.11). Replace:
   "compared at the same medium strength; the 8-neuron edits cannot reach the strongest LoRA effect".

## (C) Reviewer attacks, ranked

| # | attack | stated in README | cheapest fix |
|---|---|---|---|
| 1 | Forking paths: amendment 1 of 04 and amendment 1 of 06 decide the headline / verdict (section 2) | no | text: one sentence per amendment with the original-rule result |
| 2 | Same strength only at medium strength; the gap gets smaller with strength (128 gradient neurons: 8x -> 3x; 06: 5x -> 3x); neuron edits cannot reach the strongest effect | no | text (B7) + the lowest-reach numbers |
| 3 | The "before training" predictor fails at the fine scale (B2 numbers) | partly ("Activity alone is not sufficient") | text; analysis: add the 04 points to panel (b) of 03 with `predict.py` (CPU, ~minutes) |
| 4 | Effect = KL to the teacher, not judged behaviour; no judge in 03-06 | no | text; new run: judge the matched points of the casual edit (~1 h API) |
| 5 | The F domains are far from the target (code, math, 3 languages, long documents); F of the casual activity neurons is mostly long documents; near-target text gives 1-3x | yes (Limits 2) | none; move the task-KL number into the headline sentence |
| 6 | Baselines: no LoRA with replay / KL regularisation, no other sparse or localized methods (for example parameter masks by gradient, IA3), no standard LoRA on the rules, no full fine-tune; LoRA only 400 steps at a constant learning rate | partly (no preservation loss is stated) | text; new run: standard LoRA on the weather rule (15 runs, ~1 h on one GB10) |
| 7 | One model, one K per task, 400 steps | yes (one model) | text |
| 8 | Three seeds; the activity selection does not depend on the seed, so the seed CI does not include selection variance; the seed bootstrap has 10 distinct multisets | no | text |
| 9 | Pseudo-replication in rho 0.90 | no | text: add the cluster numbers below (they support the claim) |
| 10 | bf16 floor; F of the weather activity neurons (0.0031) is at the floor | yes | none (makes the ratios conservative) |
| 11 | Interpolation on long segments (standard LoRA at 0.60: teacher KL 1.70 -> 0.485) | no (caveat 2 of 05 has a bound: <= 0.10 ratio) | none |

## Section 2: forking paths, recomputed

- **Amendment 1 of 04 (span fix, after the first seed-0 runs).** Under the original rule (v1 neurons), from
  `experiments/04_activity_selection/out/runs_index.json`: casual v1 F 0.13 at teacher KL 0.48 to 0.37 at 0.22
  (seed 0): worse than the rank-1 LoRA (0.037 at teacher KL 0.60, 0.086 at 0.46). Weather v1: every learning rate
  except 3 is stronger than T_e 0.219; at learning rate 0.03 (teacher KL 0.13, 3 seeds) F 0.014-0.016 (0.5x the
  rank-1 LoRA at T_e), KL on the control messages 0.48-0.50 (1.4x MORE than the rank-1 LoRA, 0.36), false positives
  0.42-0.53 (not 0.11). So the README weather claim ("9x", "FP 11%") exists only under the amendment. The amendment
  is a defensible bug fix (both activity means now use the same positions; score, threshold and K are unchanged; the
  v2 learning-rate grids came from the v1 curves). But it shows that S is sensitive to one token position, which
  weakens "measure before training". Amendment 1 of 03 (the same fix) does not change 03: 0.90 / 0.96 against
  0.90 / 0.94.
- **Amendment 1 of 06 (primary effect teacher KL 0.30 -> accuracy 0.5, after 6 seed-0 runs and 4 F values).**
  Pre-registered rule at teacher KL 0.30: F order 128 activity neurons 0.0031 < 128 random active neurons 0.045 <
  128 gradient neurons 0.095 < rank-1 LoRA 0.64; S order: activity neurons < rank-1 LoRA < gradient neurons <
  random active neurons; Spearman 0.20 -> "not supported". Gradient / random active = 2.13 [1.14, 5.37] (CI above
  1: the less selective method forgets less). The reason for the amendment (teacher KL 0.30 is reached at chance
  accuracy) is valid, but the verdict "partly supported" depends on it and on the rank-1 LoRA being out of range.
  Activity neurons / random active neurons holds under every target (0.047-0.069).

## Section 3: rho 0.90 with pseudo-replication

Cluster bootstrap over the 17 method/task groups: 95% CI [0.77, 0.97]. Group medians (17 points): 0.80. Partial
Spearman(F, off-target write | teacher KL): casual edit (01) 0.87, rules (02) 0.86. Spearman(F, on-target write) =
0.57; partial (F, off-target write | on-target write) = 0.89. So the relation is not only the learning-rate effect or
the update size. Report it as "0.90 over 210 runs; 0.80 over 17 group medians; cluster-bootstrap CI 0.77-0.97". It
does not include the standard LoRA or the runs of 04 and 06.

## (D) What is solid

- All headline numbers match the committed files (table above).
- Selective `down_proj` edits are much more local than same-size and standard LoRAs at medium strength on the casual
  edit (8 gradient, 8 matched and 128 gradient neurons; CIs below 1, 3/3 seeds; covered verdicts of 05), and the
  fp32 F makes the gap larger.
- Weather and recipe rules with activity neurons: large gains in F, in the KL on the control messages and in false
  positives over the gradient neurons and the rank-1 LoRA (v2).
- 06: 128 activity neurons learn held-out facts (accuracy 0.70-0.76) with F 5x below the standard LoRA and 21-34x
  below the other neuron methods; the leak to untrained made-up people is honestly reported.
- The off-target write (after training) predicts F across families, and the result survives clustering and partial
  correlation.
- Pre-registration, run logs and amendments are recorded with what was seen; the bf16 floor and the task-KL limit
  are found and reported.
