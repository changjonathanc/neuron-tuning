# E1: does the off-target activity of the edited units predict the locality of an edit?

> **Names in this log.** Phase 1c = 01_locality_curves; Phase 2b = 02_conditional_rules; E1 = this experiment; E2 = 04_activity_selection; E3 = 05_typical_lora; E4 = 06_facts. Phase 1, Phase 1b and Phase 2 are earlier studies that are not in this repository; their shared code and data (adapters, training, the forgetting set F, the teacher-KL measure) are in [`../lib/`](../lib/).

Written before any predictor or pass-2 F is computed (2026-10-30). No training. Only saved edits.

## Question
Does F (KL(original || edited) per token on the Phase 1b forgetting set, macro mean over 7 domains) follow from how
active the edited units are on off-target text? (a) Across all saved runs: does F follow the direct write of the edit
on off-target tokens? (b) At matched effect: do families/rules with less selective neurons have a higher F ratio
to the LoRA?

## Runs (all saved; 210)
- Phase 1c, 111 runs: sel8, srand8, dense8, lorad (~13 k); sel128, srand128, dense128, lorac (~200 k); LR sweeps,
  3 seeds (`experiments/01_locality_curves/gen_jobs.py` paths, incl. the 12 reused Phase 1b runs). Stored TKL and bf16 F.
- Phase 2b, 99 runs: col8, lora_d, srand8 for weather, lower, recipe (`experiments/02_conditional_rules/runs/curve`). Stored tkl and
  bf16 F (in `runs/eval/curve_*.json`).
Every edit is a change of `down_proj` of some layers: columns (ΔW[:, i] = g_i W_i + rms(W_i) u_i on K neurons) or a
rank-1 LoRA on `down_proj` (ΔW = B A). So both predictors use the input of `down_proj` (the neuron activations a).

## Token sets (original model, bf16, no system prompt, teacher forcing)
- OFF = the Phase 1c reference set (`experiments/01_locality_curves/data/ref_set.json`; disjoint from the F items). Chat items:
  the reply tokens; longdoc: all tokens (the `acts1c.py` spans).
- ON (the target reply tokens): Phase 1c = the 160 casual teacher replies (`acts1c.py` "casual"); Phase 2b = the
  training trigger replies of the rule (`train2b.trigger_acts` items).
- Per-neuron activity (all tokens of a set): c = mean |a| on ON; g = macro mean over the 7 domains of mean |a| on OFF.
- Token subsample for the direct write (fixed, seed 0): OFF = up to 256 random scored tokens per domain (1792);
  ON = 1024 random tokens of each target set (casual, weather, lower, recipe). I cache a (bf16, all 35 layers) for
  these tokens once; then each edit costs only arithmetic (fp32). I report the stability of the means (split-half).

## Predictors
1. **S_pre, pre-training selectivity (the claim).** Per edited neuron i: s_i = g_i / c_i (higher = LESS selective).
   Per run: S = exp(mean_i ln s_i) (geometric mean over the K neurons). The neurons are fixed per (family, seed), so
   a family/rule point is exp(mean over the 3 seeds of ln S). This needs no trained weights.
   For LoRA (label: POST-training): the same ratio for the input direction: s = mean |A a| on OFF / mean |A a| on
   ON (on the token subsample), geometric mean over the layers (rank 1 everywhere).
2. **D_off, post-hoc direct write.** Per token, d(x) = Σ_l ||ΔW_l a_l(x)||_2 over the edited layers. D_off = macro
   mean over the 7 OFF domains of the mean of d; D_on = the mean of d on the ON tokens of the edit's target;
   R = D_off / D_on (relative direct write). Same formula for columns, gains and LoRA.

## F, pass 1 and pass 2
- Pass 1: the stored bf16 F (floor ~0.001).
- Pass 2 (Phase 1c only; every Phase 2b F is > 0.01): F with the model and the edit in fp32 (all weights cast from
  bf16 to fp32, which is exact; TF32 off; eager attention), the same items and KL formula as `forget.py`. Runs:
  every Phase 1c run with stored F < 0.01, and all other Phase 1c runs if the time per run allows (decision after
  the first timing; it makes the pass-2 curves homogeneous). The 2b scatter points keep pass-1 F.
- Validation (before the pass-2 runs): sel8_lr0.03_s2 and srand8_lr0.03_s2 with the change scaled by s = 0, 0.01,
  0.1, 1. Pass if F(0) = 0 and F(0.01) / F(1) is within [0.3e-4, 3e-4] and F(0.1)/F(1) within [0.3e-2, 3e-2].
  If not, I fix the method before any pass-2 run.

## Matched effect (figure b)
- Phase 1c: the `analyze1c.py` point estimate (seed-mean curves in LR order, first crossing, linear in ln F vs ln
  TKL) at TKL 0.60 for the ~13 k families (vs lorad) and 0.30 for the ~200 k families (vs lorac). (0.45 and 0.20 are
  not used: the LoRA or srand/dense curves do not reach them reliably; the independent review of Phase 1c.) Pass 1 and pass 2.
- Phase 2b: the stored `analyze3.py` values at T_e (col8 / lora_d per rule; srand8 / lora_d for weather only, the
  only covered srand8 rule).
- Points: 6 Phase 1c neuron families + 3 col8 rules + srand8 weather = 10 points. x = S_pre (family/rule point),
  y = F ratio to the LoRA of the same setup. The LoRA points (ratio 1) are drawn with their post-training S, but they
  are not in the rank correlation.

## Predictions
- (a) F rises with D_off over all runs (Spearman ≥ 0.8); within each family F rises with LR and so does D_off
  (within-family Spearman high, but this is mostly the LR).
- (b) Higher S_pre (less selective) gives a higher F ratio. dense8, dense128 and the 2b weather/recipe col8 points
  have the highest S_pre and the highest ratios; sel8/srand8 the lowest.

## Decision rule (GO / NO-GO for the PLAN.md stop rule), fixed now
GO if BOTH:
1. Spearman(F, D_off) over all 210 runs ≥ 0.8 (F = pass 2 where it exists, else pass 1; I also report pass 1 only).
2. In (b), Spearman(S_pre, F ratio) over the 10 neuron points ≥ 0.7, with the pass-2 ratios for Phase 1c.
Else NO-GO. I report the per-family Spearman and R = D_off / D_on also, but they do not enter the rule.

## Amendment 1 (2026-10-30, after the first results; requested by the reviewer)
The OFF span of `acts1c.py` (and of my cache) starts one token after the positions that `forget.py` scores: it
misses the last prompt token (it predicts the first reply token). E2 found neurons that are quiet on reply tokens but
fire strongly at that position on all text. Change: I compute every predictor a second time with the OFF tokens =
the SCORED positions of `forget.py` (chat items: last prompt token ... second-to-last token; raw text: all tokens but
the last); new random subsample with the same rule (256 per domain, seed 0). The ON spans (`TG.build`) already are
the scored positions and do not change. The scored version becomes the PRIMARY version, because F is measured
there; the first version (acts1c spans) is reported next to it. The decision rule is unchanged and is applied to
both versions; the verdict uses the primary version.
