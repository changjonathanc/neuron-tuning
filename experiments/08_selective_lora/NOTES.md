# Experiment 08 notes: selective LoRA (a penalty on the off-target write of any adapter)

All times are local (+08:00). Entries are append-only. Hardware: one NVIDIA GB10 (128 GB unified memory), shared
with the job of another experiment until about 10:00.

## PRE-REGISTRATION — 2026-11-01 08:25 (written before any result; only 12-step smoke tests have run)

### Question
Experiment 03 found that the change on unrelated text F follows the off-target write of the edit (the activity of
the edited units on unrelated text × the size of the update; Spearman 0.90 over 210 runs). Can this principle be
used as a training loss for ANY adapter? We train the standard LoRA of experiment 05 (`lora_all8`: rank 8,
alpha 16, all linear modules of all 35 layers, 205 modules, 12,079,104 parameters) with one extra term:

    P = (1 / N) Σ_t Σ_m || s B_m A_m h_m(t) ||²        loss = KL_casual + λ P

Here t runs over the N tokens of a batch of 4 off-target items, m over the 205 adapted modules; h_m(t) is the input of module m;
s = 2 (alpha / rank). This needs no reference logits.

**Which inputs h.** We use "detached live inputs": at each step, one forward pass of the CURRENT edited model
(adapters on) on the off-target batch, without gradient and without the LM head, records h_m(t) for all modules.
The gradient of P goes into A_m and B_m at these inputs, but not through the inputs (the effect of earlier
adapters on later inputs is not differentiated). Reason: the values of P are exact for the current model, and the
cost is one forward pass without the LM head and without a backward pass. We do not use cached base inputs (they
are exact for layer 0 only). Because ||Y Bᵀ||² = tr((BᵀB)(YᵀY)) with Y = h Aᵀ (N × 8), the term itself is cheap.

### Data
- Edit: the casual edit of experiments 01 and 05, unchanged (`experiments/lib/train1b.py` loop with w_pres = 0:
  400 Adam steps, constant LR, batch 4 casual + 4 task items, KL to the teacher on the casual replies). The
  casual/task draws use the same RNG calls as train1b, so `--arm plain` is the run of experiment 05.
- Off-target TRAINING set: `data/offtarget_set.json`, built by `build_offtarget.py` (seed 808). 420 items, 60 per
  kind, the same 7 kinds as F, and no item of F: code = HumanEval problems that are not among the 40 F problems;
  math = GSM8K train split (F: test); QA = SQuAD train split (F: validation); de/ja/es = Wikipedia 20231101 stream
  items 3000-5999 (F samples from items 0-2999; prompts not in F); long documents = WikiText-2 train split (F:
  test), 512-token windows (F: 1024). The script asserts that no prompt or document is in F. Chat items have the
  same prompt/reply format as F. Each step draws 4 items with a separate RNG (seed + 10,000). Items are cut at 512
  tokens. P uses all valid tokens (prompt and reply); the KL arm uses the tokens that F scores.
- Note: the off-target set has the same kinds (and partly the same sources) as F. A good F result can therefore be
  "in-distribution". The out-of-set check is the task KL on 300 SlimOrca items (`01_locality_curves/taskkl.py`):
  SlimOrca is not in the off-target set.

### Arms (all seed 0 first)
- (a) **plain** standard LoRA: experiment 05 seed-0 runs, reused (bf16 F, TKL, task KL; LR 1e-6 ... 1e-4). Two
  reruns on the GB10, needed for fp32 F (the 05 adapters are not kept) and for the cost: LR 3e-6 (also the
  calibration run of λ) and LR 1e-6 (the two LRs around TKL 0.6). They also check that the GB10 reproduces 05.
- (sel) **selective LoRA**: λ ∈ {0.1, 1, 10} × λ₀ (rounded to 1 significant digit) × LR ∈ {3e-6, 1e-5, 3e-5}.
  **λ₀ rule** (fixed now): from the plain calibration run (LR 3e-6), λ₀ = (mean KL_casual of steps 380-399) /
  (P on the probe batch at step 399). At λ₀ the penalty has the same size as the casual loss at the end of plain
  training.
- (b) **KL preservation**: loss = KL_casual + w · KL(base ‖ edited) on the same off-target batch (mean per scored
  token; reference log-probs from a pass with the adapters off). w ∈ {1, 10} × LR ∈ {3e-6, 1e-5, 3e-5}. If time is
  short, w = 10 first.
- (c) **8 selective neurons**: `sel8` of experiment 01 (seed-0 runs; bf16 F from 01, fp32 F from
  `03_predictor/out/f32.json`). Not retrained.
- **Edge rule** (per λ or w, seed 0): if every point has TKL > 0.60, add LR ×√10 above the highest LR; if every
  point has TKL < 0.60, add LR ÷√10 below the lowest. At most 2 extra LRs per family. Logged below.

### Measures
- TKL (`experiments/lib/tkl.py`), F bf16 (`experiments/lib/forget.py`, the 256 F items), **F fp32**
  (`experiments/03_predictor/f32forget.py`; primary for this experiment), task KL (`taskkl.py`, out-of-set check).
  No MMLU.
- **Matched point**: TKL 0.6 (the README point). Seed-0 curve per family in LR order; ln F interpolated linearly
  in ln TKL between the two points around the target (the rule of 01/05); no extrapolation ("not reached").
  Secondary targets 0.45 and 0.3 where reached.
- Ratios at TKL 0.6: F_sel / F_plain, F_kl / F_plain, F_sel / F_neurons (fp32 primary; bf16 also, with the 05
  plain curve).
- **Cost per step**: mean wall time of a training step (steps 10-399, with `torch.cuda.synchronize`) and peak GPU
  memory, from each run's JSON. A separate short benchmark (40 steps per arm, one job at a time, when the GPU has
  no other job) gives the clean numbers if the shared-GPU times are noisy. The neuron arm cost: `train1b.py --arm
  columns --k 8` in the same benchmark.
- "Best point" = the λ with the lowest F at TKL 0.6. Seeds 1-2 at the best λ (all its LRs) only if time allows.

### Predictions
- P1 (frontier moves): at TKL 0.6, the best selective λ has F_sel / F_plain ≤ 0.5 (fp32).
- P2 (not as local as neurons): the best selective λ is still above the neuron frontier: F_sel / F_neurons in
  [1, 4] (sel8 / plain is ≈ 0.12 in 05).
- P3 (cost): sel step time ≤ 1.5 × plain; KL step time ≥ 1.5 × plain and > sel; peak memory sel < KL.
- P4 (KL baseline): the KL arm reaches a lower F than sel at TKL 0.6 (it optimizes a quantity close to F on the
  same kinds of text), so sel wins on cost, not on locality.
- P5 (out of set): on the task KL (SlimOrca, not in the off-target set), the reduction is smaller than on F:
  task-KL ratio sel / plain closer to 1 than the F ratio.
- P6 (side effect on strength): at the same LR, larger λ gives a weaker edit (higher TKL); at the largest λ the
  family may not reach TKL 0.6 inside the edge rule.

### Decision rule (seed 0; no CI, so a margin)
- "The penalty moves the LoRA frontier down" if F_sel / F_plain < 0.7 at TKL 0.6 for the best λ; "no clear
  change" if in [0.7, 1.43]; "moves up" if > 1.43. (In 05 the per-seed spread of F ratios at matched TKL is about
  ±20%.) If no λ reaches TKL 0.6, the verdict is "the penalty only makes the edit weaker" for that λ.
- The same rule for the KL arm. For P3 the numbers are reported as measured.

## Run log
- 2026-11-01 08:20. Started the plain reruns (LR 3e-6 and 1e-6) and the KL arm w = 10 (LR 3e-6, 1e-5, 3e-5) on
  two job queues. Smoke tests (12 steps, shared GPU): plain 2.6 s/step, sel 4.8 (3.2 after one change: q/k/v and
  gate/up share one stored input copy; same P values), KL 7.3 s/step.
- 2026-11-01 08:55. **λ₀ (rule of the pre-registration).** Plain calibration run LR 3e-6: mean KL_casual of steps
  380-399 = 0.532; P on the probe batch at step 399 = 4.885. λ₀ = 0.109 -> grid λ ∈ {0.01, 0.1, 1}. Queued the 9
  selective runs (3 queues, one more process on the GPU).
- 2026-11-01 10:20. First selective point (λ = 1, LR 3e-6): TKL 0.540 (plain at the same LR: 0.489); P on the
  probe batch 0.124 (plain 4.885, 39× lower), but F fp32 0.0243 (plain 0.0363; ~0.76× of plain interpolated to the
  same TKL). The penalty removes most of P, but F goes down only a little.
- 2026-11-01 10:30. **Diagnostic (post hoc, `diag08.py`, out/diag.json)** on the plain LR 3e-6 and the λ = 1 LR 3e-6
  adapters: P on the 256 F items is also 38× lower (4.58 -> 0.109 on the tokens that F scores), so the penalty does
  generalise to the F text. The first token gives < 1.3% of P and the top 1% of tokens < 5% (no outlier problem).
  But **77.5% of the plain P is the q_proj write** (v 8.5%, o 4.5%, up 3.7%, k 2.8%, gate 2.1%, down 0.9%).
  Gemma-4 normalizes q (and k) per head after the projection (q_norm, k_norm), so the absolute size of a q_proj
  write is mostly irrelevant for the output. The pre-registered P weights the modules by the scale of their output
  space, which is arbitrary. In the λ = 1 adapter, the residual-stream writes went down much less than P: o_proj
  7.6×, down_proj 2.5×.

## AMENDMENT 1 — 2026-11-01 10:40 (before any run of this variant)

A second penalty, in addition to (not in place of) the pre-registered one: the **relative** off-target write

    P_rel = (1 / N) Σ_t Σ_m || s B_m A_m h_m(t) ||² / c_m,     c_m = mean over tokens of || W_m h_m(t) ||²

c_m is the energy of the base output of module m on off-target text (32 off-target items, adapters off, computed
once at the start of each run; `base_norms` in train08.py). Each module term is then the change of that module's
output relative to its normal size, so modules with different output scales (q_proj vs down_proj) count the same.
The same detached live inputs. Arm name: `selrel` (`--arm sel --pen rel`).

- λ₀ rule (as for P): plain LR 3e-6 adapter, P_rel on the probe batch = 4.39e-4 (`out/calib_rel_plain.json`;
  P_abs recomputed = 4.893, logged 4.885); λ₀ = 0.532 / 4.39e-4 = 1211 -> λ ∈ {100, 1000, 10000}. (For the λ = 1
  adapter of the main grid, P_rel = 1.49e-4: the pre-registered penalty cut P_rel only 3×.)
- LRs: 3e-6 first, then by the edge rule (1e-6 if TKL < 0.6; 1e-5 for the 0.45 / 0.3 targets). λ = 100 only if time.
- Prediction A1: the best `selrel` λ has F / F_plain < 0.5 at TKL 0.6 (it moves the frontier more than the
  pre-registered penalty).
- The verdict on the pre-registered method stays on the pre-registered penalty. The amendment is reported as a
  separate, post-hoc variant.

**Order of the runs (changed at 10:40; no run removed).** The GPU is saturated (three jobs in parallel give about
the same throughput as one), so the order matters. TKL falls with LR, and λ = 1 and λ = 0.1 already have TKL < 0.6 at
LR 3e-6, the lowest LR of the grid; the edge rule therefore adds **LR 1e-6** for these λ (LR ÷ √10 ≈ 9.5e-7; I use
1e-6, the LR of the experiment 05 grid). New order: λ = 1 and 0.1 at LR 1e-6, `selrel` λ = 1000 and 10000 at LR
3e-6, KL w = 10 at 1e-5 and 3e-5, the LR 1e-5 and 3e-5 points, and λ = 0.01 last.
- 2026-11-01 12:20. Two eval steps failed with CUDA out-of-memory at context creation (fp32 F of kl w = 10 LR 3e-6
  and of sel λ = 0.1 LR 1e-6; another experiment's job had started on the same GPU). The queue did not see the
  error (a bug in the rc log line). The same job lines were run again (finished steps are skipped). Added a plain
  rerun at LR 1e-5 (fp32 F for the targets 0.45 and 0.3).
- 2026-11-01 13:20. Edge rule: selrel λ = 1000 LR 3e-6 -> TKL 0.537 (< 0.6): add LR 1e-6. selrel λ = 10000 LR 3e-6
  -> TKL 0.711: add LR 1e-5. KL w = 10 LR 3e-6 -> TKL 0.545: add LR 1e-6. These three go to the front of the queues.
- 2026-11-01 15:00. STATUS SHOWABLE (seed 0; RESULTS.md). Step 3: seeds 1 and 2 of selrel λ = 10⁴ (LR 3e-6, 1e-5)
  go before the remaining pre-registered runs of λ = 0.1 (LR 1e-5, 3e-5) and λ = 0.01, which have the least value
  for the question (λ = 0.1 changed F by < 10% at LR 3e-6).
- 2026-11-01 17:50. Seeds 1-2 of selrel λ = 10⁴ done: fp32 F / F_plain(seed 0) = 0.29 / 0.32 at TKL 0.6 and
  0.16 / 0.17 at TKL 0.45 (seed 0: 0.32 / 0.17). KL w = 10 bracket at TKL 0.6 done (0.23×).
- 2026-11-01 19:40. All queued runs done (27 runs of this experiment; no missing outputs). The rest of the
  pre-registered grid: λ = 0.1 and 0.01 at LR 1e-5 / 3e-5 (and λ = 0.01 at 3e-6) are at most 0.75× / 0.92× of plain
  at TKL 0.3. Not run: the edge-rule step LR 1e-6 of λ = 0.01 (its TKL 0.6 point is therefore "not reached"); it
  would not change a verdict (λ = 0.01 is ≈ plain at every covered target). KL w = 10 at LR 1e-5 / 3e-5: 0.15× and
  0.09× of plain at TKL 0.45 / 0.3 (fp32). STATUS COMPLETE.
