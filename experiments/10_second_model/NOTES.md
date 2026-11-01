# 10 Second model: the casual edit on Qwen3.5-4B (pre-registration and run log)

## Question

Does the main result hold on a second model family? Main result (experiments 01 and 05, Gemma-4-E2B-it): at the same
strength of the casual edit, 8 selective neurons (their `down_proj` columns) change unrelated text much less than a
LoRA of the same size and than the standard LoRA, and 8 non-selective (dense) neurons change it more than the LoRA.

Model: `Qwen/Qwen3.5-4B` (revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`), bf16, thinking off
(`enable_thinking=False` in the chat template). Architecture facts that matter here (checked on the loaded model):
32 decoder layers; every layer has a dense SwiGLU MLP (`mlp.gate_proj`, `mlp.up_proj`, `mlp.down_proj`; 9,216
neurons; hidden size 2,560). 24 layers have a linear-attention block (`linear_attn`: Gated DeltaNet), 8 layers (3, 7,
..., 31) have full attention (`self_attn`). The attention type does not change the MLP, so every layer has neurons that
we can edit. Unlike Gemma-4, Qwen has no norm after the MLP: the `down_proj` output goes directly into the residual
stream.

## Code

- `make_edit.py` (repository root) with `--model qwen3.5-4b`. The changes for other models are minimal and backward
  compatible (Gemma: the same code path; check: `uv run make_edit.py --help` and a CPU check of the old and new code on
  Gemma, see the run log). For other models: plain transformers loading; thinking off; the tokenizer's end-of-turn
  token is a stop token of `generate()`; the background cache file name contains the model name; the rank-1 LoRA is on
  `down_proj` of the layer with the most chosen neurons; the standard LoRA is on every `nn.Linear` of every decoder
  layer.
- `teacher.py`: the teacher gate (replies with and without the system prompt, judge).
- `select_dense.py`: the dense control neurons (written before any training run).
- `analyze.py`: the matched-strength comparison; `plot.py`: the figure.

## Step 1: teacher gate (before any training)

The behaviour is the casual register of experiment 01: the teacher is the same model WITH the "Sam" system prompt
(`PERSONA_SYSTEM` in `demos/talk_like_a_person/common.py`; the same wording as for Gemma unless the gate fails), the student has no system
prompt. Gate: on the first 30 of the 60 held-out messages, the judge says "casual" for >= 90% of the teacher replies
(question `casual` in `teacher.py`), and clearly less often for the original replies. If the gate fails, I change the
wording of the system prompt (the closest wording that passes) and write it here before training.

## Design (fixed before any training run)

**Data.** Training prompts: the 160 casual messages of experiment 01 (`kind == "id"` in
`demos/talk_like_a_person/reproduce/data/data_persona_E2B.json`; only the messages, not the Gemma replies). Held-out prompts: the 60
held-out messages of experiment 01 (`experiments/lib/data/prompt.json`; no overlap with the training messages). The
targets are Qwen's own greedy replies WITH the system prompt (`make_edit.py --system`, at most 256 new tokens, batches of
16).

**Methods** (seed 0 first; `make_edit.py make`, 200 Adam steps, batch 8, constant LR, no preservation loss; 200 x 8 =
1,600 training items per run, the same number of casual items as experiment 01: 400 steps x 4):

| label | what is trained | trained numbers | LR grid (seed 0) |
|---|---|---:|---|
| sel8 | `down_proj` columns of 8 selective neurons (the `make_edit.py` rule: top 10% on the target, g / c <= 0.2 on the background, rank by gradient / background activity) | 8 x 2,560 = 20,480 (+ 8 gains) | 3e-3, 1e-2, 3e-2, 1e-1 |
| dense8 | `down_proj` columns of 8 NON-selective neurons: random (seed 0) among the neurons that are in the top 10% of their layer on the target replies AND in the top 10% of their layer on the background text (they fire everywhere); per-layer counts = the per-layer counts of sel8 (as `columns_active` in experiment 01) | 20,488 | 1e-2, 3e-2, 1e-1, 3e-1 |
| lorad | rank-1 LoRA (alpha 1) on `down_proj` of ONE layer: the layer with the most sel8 neurons (ties: the lower layer) | 9,216 + 2,560 = 11,776 | 1e-4, 3e-4, 1e-3, 3e-3 |
| lora_all8 | standard LoRA: rank 8, alpha 16, on every linear layer of every decoder layer (attention, linear attention, MLP) | (printed by `make_edit.py`) | 3e-6, 1e-5, 3e-5, 1e-4 |

Note on size: the brief asks for a rank-1 LoRA on one `down_proj`. On Qwen this has 0.57x the trained numbers of 8
columns (on Gemma: 1.12x). I keep the brief's shape and report the sizes.

The grids are the grids of experiments 01 and 05 (the dense grid starts one step higher, as there). Edge rule (as 01):
after seed 0, if a family's curve does not bracket the matched level T* (below), add one half-decade LR step past the
needed edge, at most 2 extra steps per family; no upward step if the TKL got worse from the second-highest to the
highest LR.

**Measures.**
- Strength = TKL: teacher KL per token, KL(teacher || edited) on the teacher's greedy replies to the 60 held-out
  messages, the teacher with the system prompt and the edit without it, mean per reply, then over replies
  (`make_edit.py` "held-out target loss"). Difference from experiment 01: there the TKL also used 2 sampled teacher
  replies per message (180 replies). Lower TKL = stronger edit. TKL_orig = the TKL of the original model.
- F: KL(original || edited) per token on the SAME 256 unrelated items as experiments 01-07
  (`experiments/lib/data/forget_set.json`: code, math, qa, de, ja, es, longdoc), macro mean over the 7 domains, model
  and edits in fp32 (`make_edit.py --f32`; no bf16 floor).

**Matched strength T\* (the "medium" level of experiment 01).** In experiment 01 the first target was TKL 0.60 with
TKL_orig 2.607 for Gemma, i.e. 0.23 x TKL_orig. Rule: T* = 0.23 x TKL_orig of Qwen. If one of sel8, lorad, lora_all8
does not reach T* (after the edge rule), T* = 1.05 x the largest of the three lowest TKL values (the strongest level
that all three reach), and I report this. dense8 is compared at T* if its curve reaches T*; else at the strongest level
that both dense8 and lorad reach (reported as such).

**Interpolation (as 01).** Per family, the points in LR order; ln F interpolated linearly in ln TKL between the two
adjacent LRs whose TKL brackets T*; first crossing in increasing LR; no extrapolation. Ratio rho(A, B) = F_A(T*) /
F_B(T*). Seed 0 only: no CI (if seeds 1-2 run: the per-seed ratios and their range).

**Prediction (pre-registered).** At T*: rho(sel8, lorad) <= 1/3 AND rho(sel8, lora_all8) <= 1/3 (selective neurons
change unrelated text >= 3x less than the same-size LoRA and the standard LoRA); rho(dense8, lorad) > 1 (dense neurons
are worse than the LoRA).

**Verdicts.** "Replicates" if all three hold; "partly" if the two sel8 ratios are < 1 but one is above 1/3, or if only
the dense prediction fails; "does not replicate" if a sel8 ratio is >= 1. I also report descriptively: the selectivity
of the chosen neurons (g / c, and their percentile of the neuron-finder RMS activity on general chat,
`stats/Qwen3.5-4B.safetensors` of the public neuron-finder repo, computed on smol-smoltalk), the F per domain, and
the layers of the chosen neurons.

## Run log

- 2026-11-01 10:25. Pre-registration above written; then the Gemma backward-compatibility check of the new
  `make_edit.py` (CPU, Gemma-4-E2B-it, old HEAD code vs new code): the same LoRA shapes (small: layer 15; standard: 205
  modules), the same background cache file name, a shipped Gemma patch resolves to E2B, and the patched logits are
  identical (max |new - old| = 0.0). `uv run make_edit.py --help` works.
- 2026-11-01 10:30. **Teacher gate: PASSED** with the experiment-01 wording (`PERSONA_SYSTEM`, unchanged). 30
  held-out messages, judge (`teacher.py`): casual 30/30 with the system prompt, 2/30 without; "speaks as a person"
  27/30 vs 6/30 (the 3 teacher "no" are casual advice replies with no personal content). Mean reply length 279 vs 911
  characters. Without the prompt Qwen often says "As an AI ..." and uses markdown. File: `out/teacher_gen.json`.
- 2026-11-01 10:40. Selection (`make_edit.py --lr` with no value: teacher replies, selection and cards only; seed 0):
  328 eligible neurons; sel8 = 0:1614, 9:1876, 10:7486, 15:2169, 18:7912, 4:5541, 17:5151, 14:7778 (g / c 0.12-0.20).
  One neuron per layer, so the pre-registered lorad rule (the layer with the most sel8 neurons, ties: the lower layer)
  gives **layer 0**. I keep the rule for the verdict. Declared extra (no verdict): `lorad14`, the same rank-1 LoRA on
  layer 14 (the relative depth of Gemma's layer 15: 15 / 35 x 32 = 13.7), same grid, if time allows.
- 2026-11-01 10:50. dense8 (`select_dense.py`, `out/dense8.json`): 0:2956, 4:413, 9:1757, 10:1241, 14:21, 15:584,
  17:1069, 18:311 (g / c 0.91-1.67). Descriptive, neuron-finder RMS percentile in the layer (general chat): sel8
  0.57-1.00 (5 of 8 >= 0.89), dense8 0.60-0.99. So by the neuron-finder statistics the sel8 neurons are NOT quiet on
  general English chat; the selection background (code, math, QA, de / ja / es, documents) has no casual English chat.
  Four seed-0 sweeps started in parallel on the GB10 (about 11 s per step, 36 min per run). `lorad14` queued after lorad.
- 2026-11-01 11:10. TKL_orig = 0.7896, so **T* = 0.23 x 0.7896 = 0.182**. First points (lowest LR of each grid): sel8
  3e-3 TKL 0.529; dense8 1e-2 0.692; lorad 1e-4 0.148; lora_all8 3e-6 0.144. **Edge rule (decided early, on TKL
  only, as in experiment 05):** the lowest-LR points of lorad and lora_all8 are already below T*, so their curves
  cannot bracket T* from above: one lower half-decade step each, lorad 3e-5 and lora_all8 1e-6 (queued after each
  sweep; `<family>_x` runs, merged by `analyze.py`).
  The extra lorad14 grid is 3e-5, 1e-4, 3e-4, 1e-3 (changed before it started, for the same reason).
- 2026-11-01 13:50. Memory on the GB10 ran out in the F step (three jobs loaded the fp32 model at the same time, and
  `make_edit.py` kept the bf16 copy too). The kernel stopped the sel8 and lorad jobs after all their trainings were done
  and their patches were saved. Fix in `make_edit.py` (no change of results): the `--f32` path now frees the bf16 model
  before it loads the fp32 model. For sel8 and lorad, `f_only.py` computes F with the same function
  (`make_edit.unrelated_change`, fp32) from the saved patches and takes the TKL from the training log (the same
  numbers that `make_edit.py` prints); their reports have `"recovered_from_log": true`.
- 2026-11-01 14:55. Correction of the 10:50 entry: 4 (not 5) of the 8 sel8 neurons have a neuron-finder RMS
  percentile >= 0.89 (0.9997, 0.9992, 0.919, 0.903; then 0.887); all 8 are above the median. The lora_all8 job also
  ran out of memory in the F step; F from `f_only.py` as for sel8 and lorad. The `lorad14` extra was stopped (a
  request of another job on the same GPU for free memory) and restarted at about 15:00 as the only job of this experiment.
- 2026-11-01 15:00. Edge steps done: lorad 3e-5 TKL 0.688; lora_all8 1e-6 TKL 0.489. All four families bracket T* =
  0.182. Result (`analyze.py`): sel8 / lorad 0.0050, sel8 / lora_all8 0.0062, dense8 / lorad 3.42. **Verdict
  (pre-registered rule): replicates.**
- 2026-11-01 15:14. **Step 3 check (descriptive, no verdict; written before the run).** `check.py`, for the original
  model and for the sweep point nearest T* of each family (sel8 3e-2, dense8 1e-1, lorad 1e-4, lora_all8 3e-6):
  (a) greedy replies (batches of 15, at most 256 tokens) to the first 30 held-out messages; judge questions `casual`
  and `person` of `teacher.py`; (b) F on English assistant tasks: KL(original || edited) per token, fp32, on the 300
  SlimOrca task items of experiment 01 (`kind == "pres"` in `data_persona_E2B.json`; the reference reply is Gemma's
  reply, used only as teacher-forced text), a domain that F does not have. lorad14 stopped again (one job at a time).
- 2026-11-01 15:37. Step 3 check done (`out/check.json`, RESULTS.md section). Judge casual / person of 30: original
  2 / 6, sel8 21 / 26, lora_all8 23 / 26, lorad 26 / 27, dense8 28 / 28. F on the 300 English SlimOrca task items:
  sel8 0.0200, lora_all8 0.0652, lorad 0.154, dense8 0.257. I read the 9 sel8 replies judged "not casual": markdown
  (bold, lists) in 8, "As an AI" in 1; the judge is correct. lorad14 restarted (the only job).
- 2026-11-01 15:41. **Seeds 1 and 2 (plan, before the runs).** Time allows only the two LRs per family that bracket T*
  in seed 0: sel8 1e-2, 3e-2; dense8 3e-2, 1e-1; lorad 3e-5, 1e-4; lora_all8 1e-6, 3e-6 (`make_edit.py --seed s`). The
  neurons (sel8 and dense8) are the same for all seeds (the selection uses a fixed sample), so the seed changes only the
  training order and the LoRA initialisation. Report: per seed, F at T* (the same interpolation) and the three ratios;
  the verdict stays the seed-0 verdict, the other seeds show whether it is stable. One job at a time (shared GPU).
- 2026-11-01 17:42. lorad14 done (extra, no verdict): TKL 0.570 / 0.190 / 0.137 / 0.122 at LR 3e-5 / 1e-4 / 3e-4 /
  1e-3; F at T* 0.0219 (lorad, layer 0: 0.0273); sel8 / lorad14 = 0.006, dense8 / lorad14 = 4.27. Seeds 1-2 queue
  started (one job at a time).
- 2026-11-01 18:20. Correction of this log: some clock times above were first written as estimates and were late by
  up to 2.5 h. They are now set from the commit times (the order of the entries did not change; every plan entry was
  committed before its runs). Seed 2 is not run: it would end after the release cut-off (no new GPU runs after 23:00).
  Seed 1 only.
- 2026-11-01 18:55. Error in my seed-1 queue: the jobs did not pass `--f32`, so the F of sel8 seed 1 was measured in
  bf16 (0.00060 / 0.00067, at the bf16 floor; kept as `report_bf16.json` on the GPU machine, not used). Its F is
  recomputed in fp32 with `f_only.py` from the saved patches; the other seed-1 jobs restart with `--f32`. All seed-0
  F values are fp32 (checked: every seed-0 job had `--f32`, or `f_only.py`). To end before the 23:00 cut-off the
  seed-1 order is now lorad, lora_all8, dense8 (dense8 only if it can start before 21:45).
- 2026-11-01 20:45. lorad seed 1: TKL 0.679 (3e-5) and 0.403 (1e-4), so its two points do not bracket T* (seed 0 at
  1e-4: 0.148). No time for an extra step and lora_all8 seed 1 before the cut-off. Decision (before its F is known):
  lora_all8 seed 1 runs as planned; for lorad seed 1 I report a bound: if F rises with the strength (as in every seed-0
  curve), F_lorad(T*) >= F_lorad(TKL 0.403), so sel8 / lorad at T* <= F_sel8(T*) / F_lorad(0.403). Labelled as a bound.
- 2026-11-01 21:10. The GPU is full (6 GB free) and the lorad seed-1 F step is slow, so a 2-LR lora_all8 seed-1 job
  would end after the 23:00 cut-off. Change (before any lora_all8 seed-1 result): lora_all8 seed 1 at LR 1e-6 only,
  if it can start before 22:05, reported as the same kind of bound (F_lora_all8(T*) >= F at its point if that point
  is weaker than T*). dense8 seed 1 and seed 2: not run (cut-off).
- 2026-11-01 22:02. Seed 1 done (all GPU work ended at 22:02): sel8 TKL 0.263 / 0.151, F 0.00009 / 0.00017 (fp32,
  `f_only.py`); lorad TKL 0.679 / 0.403, F 0.0037 / 0.0142; lora_all8 (1e-6) TKL 0.503, F 0.0099. sel8 F at T*
  0.00014 (as seed 0). Bounds: sel8 / lorad <= 0.0099, sel8 / lora_all8 <= 0.0142. Not run: seed 2, dense8 seed 1.
- 2026-11-01 22:04. Final checks: the Gemma CPU check again on the final `make_edit.py` (old = the `rel` base commit):
  the same LoRA shapes, the same cache file name, identical patched logits (max |new - old| = 0.0). Host scan 0 files,
  0 broken links, `analyze.py` and `plot.py` give identical outputs from a clean `git archive HEAD`, `--help` works.
