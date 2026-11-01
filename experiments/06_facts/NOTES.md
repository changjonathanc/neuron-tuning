# E4 notes: synthetic facts — does the selectivity of the edited units predict locality for a knowledge task?

> **Names in this log.** Phase 1c = 01_locality_curves; Phase 2b = 02_conditional_rules; E1 = 03_predictor; E2 = 04_activity_selection; E3 = 05_typical_lora; E4 = this experiment. Phase 1, Phase 1b and Phase 2 are earlier studies that are not in this repository; their shared code and data (adapters, training, the forgetting set F, the teacher-KL measure) are in [`../lib/`](../lib/).

Written before the training runs (2026-10-30). Entries after "Run log" are appended later.

## Question
E1 showed that, for a style edit and three reply rules, the off-target activity of the edited units predicts the
forgetting F at matched effect. Does this also hold for a knowledge edit: facts about made-up people? This is a hard
case: there may be no neurons that are selective for new entities. If the selectivity-only neurons for this task are
not more selective than random active neurons, that is a limit of the claim, and I report it.

## Task (fixed; `make_facts.py`, `data/facts.json`, seed 4, no LLM)
- 40 made-up people (generated first + last names from fixed lists of rare names), 4 attributes each: city, job,
  instrument, pet. Each value is drawn from a fixed list of 10 values per attribute.
- 5 question templates per attribute, each with the suffix " Answer in one short sentence.". Templates 0-2 = training
  (40 x 4 x 3 = 480 questions), templates 3-4 = held out (one per fact, alternating: 160 eval questions).
- Teacher = the original model (Gemma-4-E2B-it, bf16) with all 40 fact sentences in the system prompt (context
  distillation). The format instruction is in the user question, so the teacher and the student see the same request.
- Accuracy = exact match: the reply contains the true value (word boundary, case-insensitive, optional plural) and no
  other value of the same list. Greedy, unbatched, max 48 new tokens, no system prompt.
- **Teacher gate (done, `gen_teacher.py`, `runs/gens.json`)**: teacher accuracy on the 160 held-out questions
  **1.000** (gate: >= 0.90; passed); on the 480 training questions 1.000. Original model without the facts: 0.044
  (chance ~0.10; it invents an answer, usually a value that is not in the list).
- Training data = the teacher's greedy replies to the 480 training questions (all 480 are correct, all kept), with
  the end-of-turn token (the reply stopped in every case).

## Method (fixed now)
**Training** (`train_e4.py`): loss = token KL(teacher || student) on the teacher's training replies (the student has
no system prompt), at the scored positions (TG.build span). **No preservation loss.** Adam, constant LR, 400 steps,
batch 8 fact items (the Phase 1c batch was 4 casual + 4 task items; with w_pres = 0 the task items have no loss, so I
use 8 fact items). Adapters = `experiments/lib/p1lib.py` runtime hooks (fp32 parameters, frozen bf16 model). The
teacher log-probs are computed once and stored in fp16 (`runs/teacher_train.pt`); the model is loaded with
`lowmem=True` (same outputs) so that two runs fit on the 3090. No gradient checkpointing (it does not change the math).
Arms (~200 k numbers, plus one standard baseline):

| arm | what | trained numbers |
|---|---|---:|
| colsel128 | `down_proj` columns of the 128 neurons chosen by selectivity only (E2 v2 rule, below) | 196,736 |
| col128 | columns of the attribution top 128 (`TG.select_attr`, 48 random training items, seeded; = Phase 1c sel128) | 196,736 |
| dense128 | columns of random neurons from the 10% most active per layer, per-layer counts of the same seed's attribution top 128 (`train1b.select_active_matched`; = Phase 1c dense128) | 196,736 |
| lorac | rank-1 LoRA on `down_proj` of the 16 Phase 1c layers (= Phase 1c lorac) | 122,880 |
| lora_all8 | rank-8 LoRA, alpha 16, on q, k, v, o, gate, up, down of every layer (E3 wrapper `experiments/05_typical_lora/train_e3.py`) | 12,079,104 |

**colsel128 selection** (`select_e4.py`, done; `out/selection.json`). The E2 v2 rule with K = 128: c = mean |a| (input
of `down_proj`, original model, no system prompt) over the SCORED positions of the 480 training items (the last prompt
token and the reply tokens with the end of turn); g = mean |a| over the scored positions of the Phase 1c reference
items (E2 `runs/acts_ref_scored.pt`, macro over 7 domains). Score ln(c/g); eligible: c >= 90th percentile of its
layer; global top 128. Result: layers 1-29, most in 4 (22), 15 (16), 16 (14), 17 (10); median g/c = 0.104;
within-layer percentile of c 91-99.95, of g 1-92 (median ~0.6). The selection does not depend on the seed;
`train_e4.py` runs the attribution draw without using it, so the batch order is the same as col128 of that seed.

**Evaluation** (in the training process, bf16, unbatched; `common4.evaluate`):
- Effect: TKL = KL(teacher || run) per token on the teacher's greedy replies (+ end of turn) to the 160 held-out
  questions, mean over questions; accuracy on the same questions (above).
- Locality (1): F = the Phase 1b forgetting measure in fp32 (`experiments/03_predictor/f32forget.py`, unchanged).
- Locality (2), neighbours (80 prompts): 40 questions about well-known real people (instruments, jobs, cities, pets)
  and 40 questions (held-out templates) about 40 OTHER made-up people that are not in the facts. Metrics: nKL = KL
  (original || run) per token on the original's greedy reply, mean over prompts (real and fake separately); change
  rate = the run's greedy reply differs from the original's; for fake people also the rate of replies that contain a
  value of the attribute list ("value rate"; original 0.25).
- Original model (no training): TKL0 = **1.174**, accuracy 0.044.

**LR grids** (half-decade, 4 points; seed 0 first, `jobs_e4.py`): col128 3e-3 … 1e-1; dense128 and colsel128
1e-2 … 3e-1; lorac 1e-4 … 3e-3; lora_all8 3e-6 … 1e-4. A pilot run (col128, LR 3e-2, seed 0, not counted) gave TKL
0.28 and accuracy 0.26. **Edge rule** (as E3): after seed 0 of an arm, if the lowest-LR TKL < 0.30, add one lower
half-decade step; if the highest-LR TKL > 0.15, add one higher step, unless the TKL got worse from the second-highest to
the highest LR. At most 2 extra steps per arm. Seeds 1 and 2 then run every LR of the arm. Every decision goes into the
run log.

## Matched effect (fixed now, from TKL0 before the runs)
- Targets: **T1 = 0.30** (primary; ~0.26 TKL0; the pilot shows that this is a medium effect) and **T2 = 0.15**
  (secondary). Matched accuracy (report): **0.3 and 0.5**.
- Interpolation as Phase 1c: per arm, the seed-mean curve in LR order; ln F linear in ln TKL (accuracy: ln F linear in
  accuracy) between adjacent LR points; first crossing; no extrapolation. The same for nKL.
- 95% CI: bootstrap, 2000 resamples, over the 160 eval questions (TKL, accuracy), the forgetting items (stratified by
  domain) and the seeds of each arm. "Not covered" if a curve does not reach the target or > 10% of the resamples miss it.
- Ratios: X / lorac (the same-size LoRA reference of Phase 1c, used in E1 panel (b)) and X / lora_all8.

## Predictor (pre-training)
S = E1's `predict.py` S (geometric mean over the edited neurons of mean |a| on the off-target text at the scored
positions / mean |a| on the target replies), with a new target set "facts" = the scored positions of the 480 training
items (`runs/acts_facts.pt`, written by `select_e4.py`: per-neuron c and a fixed 1024-token subsample). For col128
and dense128 S depends on the seed (geometric mean over 3 seeds). For lorac S is post-training (needs A; as E1).
lora_all8 has no S (modules other than `down_proj`).

## Predictions
- P0 (selective neurons exist): S(colsel128) < S(dense128). If not, I state that the task has no selective neurons
  in this sense (limit of the claim).
- P1 (ordering): S order colsel128 < col128 < dense128, and F at T1 in the same order.
- P2: colsel128 has a lower F than lorac at T1 (ratio CI upper bound < 1).
- P3: lora_all8 is not more local than lorac (F ratio lorac / lora_all8 not above 3).
- Neighbours (no prediction of the direction): an edit that learns the facts may also answer about untrained made-up
  people with the trained values. This is "generalisation to the neighbourhood", not a failure of selectivity, and it
  can occur for all arms. I report it.

## Decision rule
- **Main**: Spearman between S and F at T1 over the four `down_proj` arms (colsel128, col128, dense128; lorac with its
  post-training S). **Supported** if >= 0.8 (at most one adjacent swap) AND the colsel128 / dense128 F ratio at T1 has
  CI upper bound < 1. **Not supported** if Spearman < 0.4 or the colsel128 / dense128 CI lower bound > 1. Otherwise
  **partly supported**. If an arm does not reach T1, it is "not covered" (no fallback to T2), and the Spearman
  uses the covered arms only (needs >= 3).
- Also: the three neuron points (S, F / F_lorac at T1) are added to E1 panel (b) (report the Spearman over all points;
  no rule).
- If P0 fails: verdict "no selective neurons for this task" regardless of the rest.

## Hardware
Local RTX 3090 (`q_e4.py`, 2 workers in one process under a GPU lock (flock)). When E3 is done: also the second RTX 3090.

## Run log

### Amendment 1 (2026-10-30 23:20; after 6 seed-0 runs: col128 at all 4 LRs, dense128 at LR 1e-2 and 3e-2)
**What I saw.** TKL falls from 1.17 to ~0.3 already at the lowest LRs, while the held-out accuracy stays near chance
(col128: TKL 0.402 / 0.313 / 0.284 / 0.237 with accuracy 0.12 / 0.19 / 0.26 / 0.49; dense128: 0.312 / 0.283 with
0.14 / 0.20). Most of the TKL is the reply FORMAT ("<name> is based in <city>."), which the edits learn at once; the
fact token is a small part of it. So TKL 0.30 is not a "medium" knowledge effect: it is reached at chance accuracy.
I had also seen F (fp32) of col128 at 3 LRs and of dense128 at LR 1e-2 (col128 lr 1e-2: TKL 0.313, F 0.054;
dense128 lr 1e-2: TKL 0.312, F 0.056).
**Change.** The PRIMARY matched effect is now **held-out accuracy 0.5**; secondary accuracy 0.3 and 0.7 (where
reached); TKL 0.30 and 0.15 are reported as secondary. Interpolation along accuracy as fixed above (ln F linear in
accuracy, first crossing). The decision rule is unchanged except that "at T1" now means "at accuracy 0.5".
**Edge rule (replaces the TKL edge rule).** After seed 0 of an arm: if the highest-LR accuracy < 0.7, add one higher
half-decade step (at most 2), unless the accuracy got lower from the second-highest to the highest LR. Lower steps
are not needed (every lowest-LR point is near chance accuracy). Seeds 1 and 2 then run every LR of the arm.
- 2026-10-30 23:22. Edge rule, col128 (seed 0 done): accuracy at LR 3e-2 / 1e-1 = 0.26 / 0.49 (< 0.7, rising): add LR 0.3.
- 2026-10-30 23:33. dense128 seed 0: accuracy 0.76 at LR 0.3: no extension. colsel128 seed 0: accuracy 0.39 already at the lowest LR (1e-2), so the secondary target 0.3 is not bracketed: add LR 3e-3 (one lower step; amendment 1 assumed that lower steps are not needed).
- 2026-10-30 23:45. Seeds 1 and 2 of col128 (5 LRs), dense128 (4 LRs), colsel128 (5 LRs) queued. The fp32 F runs after the training (one batch), so the GPU is not idle.
- 2026-10-30 23:56. Edge rule, lorac (seed 0): accuracy 0.28 / 0.46 at LR 1e-3 / 3e-3 (< 0.7, rising; the TKL got worse, 0.252 -> 0.358, but the amended rule uses accuracy): add LR 1e-2.
- 2026-10-31 00:05. Edge rule, col128 (seed 0): accuracy 0.49 / 0.64 at LR 0.1 / 0.3 (< 0.7, rising): add LR 1.0 (second and last extra step). lora_all8 (seed 0): accuracy 0.06 / 0.13 / 0.29 / 0.91 at 3e-6 … 1e-4: brackets 0.3, 0.5 and 0.7, no extension. Seeds 1 and 2 of lora_all8 and of col128 LR 1.0 queued.
- 2026-10-31 00:10. col128 LR 1.0 (seed 0): accuracy 0.14, TKL 0.70 (unstable); the col128 maximum is 0.64 (LR 0.3), so its 0.7 target is not reached. lorac LR 1e-2 (seed 0): accuracy 0.25 (unstable; lower than 0.46 at 3e-3): no more steps (rule). So lorac does NOT reach the primary target (accuracy 0.5) with seed 0. Seeds 1 and 2 of lorac (5 LRs) queued.
- 2026-10-31 02:00-02:45. All 70 runs done (2 OOM retries at the start, when the fp32 F ran next to a training run; after that F ran as one batch, `f32_e4.sh`). Analysis, figure, RESULTS.md.
