# E3 notes: where does a standard LoRA lie on the effect-vs-F curve of the casual edit?

> **Names in this log.** Phase 1c = 01_locality_curves; Phase 2b = 02_conditional_rules; E1 = 03_predictor; E2 = 04_activity_selection; E3 = this experiment; E4 = 06_facts. Phase 1, Phase 1b and Phase 2 are earlier studies that are not in this repository; their shared code and data (adapters, training, the forgetting set F, the teacher-KL measure) are in [`../lib/`](../lib/).

All times are local (+08:00). Entries are append-only.

## PRE-REGISTRATION — 2026-10-30 (written before any E3 training run except a 3-step smoke test)

### Question
Phase 1c compared the neuron columns only with rank-1 LoRAs on `down_proj` (lorad: 1 layer; lorac: 16 layers). A
reviewer can say that this baseline is weak. Where does a standard LoRA (rank 8, alpha 16, all linear layers) lie on
the curve of forgetting F against effect (teacher KL, TKL) of the casual edit?

### Method (the Phase 1c setup; only the adapter changes)
- Data, teacher, loss, steps: unchanged from Phase 1c. `train_e3.py` is a thin wrapper around
  `experiments/lib/train1b.py`: loss = KL to the teacher on the casual replies, w_pres = 0 (NO preservation loss),
  400 Adam steps, batch 4 + 4, constant LR, full split, fp32 adapter parameters as runtime hooks.
- Adapter: LoRA from `experiments/lib/p1lib.py` (A kaiming-uniform, B = 0), rank 8, alpha 16 (scale s = 2).
  peft is not in the env on the second GPU, so I extend the Phase 1 code. The saved B is multiplied by s, so the saved adapter
  loads with the unchanged `p1lib.load_adapter` (s = 1) and gives the same function.
- Arms:
  1. **lora_all8**: q, k, v, o, gate, up, down of every decoder layer that has the module (Gemma-4-E2B: 35 layers;
     only 15 layers have k_proj and v_proj, the others share KV). 205 modules, **12,079,104** trained parameters.
  2. **lora_mlp8** (if time allows, after lora_all8): gate, up, down of every layer. 105 modules, **9,400,320**
     trained parameters.
- Evaluation: the Phase 1c chain unchanged: `experiments/lib/tkl.py` (TKL on the 180 teacher replies to the 60 eval
  messages), `experiments/lib/forget.py --mmlu` (F on the 256-item Phase 1b forgetting set, macro mean over 7 domains;
  MMLU-570), `experiments/01_locality_curves/taskkl.py` (KL on the 300 SlimOrca task items; this is the MMLU-free task KL).
- LR grid (half-decade): **3e-6, 1e-5, 3e-5, 1e-4**. Reason: Phase 1 lora_e (rank 8, all layers, s = 1, WITH
  preservation) had TKL 0.17 at LR 1e-4 and diverged at 1e-3. With s = 2 and no preservation, the effect at a given
  LR is larger, so I put the grid lower. Seeds 0, 1, 2. Seed 0 of the full grid runs first.
- **Edge rule**: after seed 0, if the seed-0 curve does not bracket TKL 0.60 (lowest-LR point has TKL < 0.60) add one
  lower half-decade step; if it does not bracket 0.20 (highest-LR point has TKL > 0.20) add one higher step, unless
  the TKL got worse from the second-highest to the highest LR (instability). At most 2 extra steps per arm. The same
  steps are then run for seeds 1 and 2. I record each decision in the run log below.
- Hardware: a second RTX 3090, `runq.sh` under a GPU lock (flock). Adapters are saved under `runs/final/`
  (git-ignored; A and B with the scale folded into B) for E1's predictor.

### Analysis (`analyze_e3.py`; the Phase 1c rules)
- Reuses `experiments/01_locality_curves/analyze1c.py` unchanged (collect, Resampler, curves, interp): family curve = seed means in
  LR order; ln F interpolated linearly in ln TKL; first crossing; no extrapolation. 95% CI: bootstrap with 10,000
  resamples over the 60 eval messages, the forgetting items (stratified by domain), the task items and the seeds of
  each family. A comparison is "not covered" if a curve does not reach the target or > 10% of resamples miss it.
  Per-seed ratios are reported as well.
- Saved Phase 1c curves (NOT retrained): sel8, srand8, lorad, sel128, srand128, lorac (including their extension
  LRs from the Phase 1c edge rule).
- Targets: 0.60, 0.45 (the ~13 k families) and 0.30, 0.20 (the ~200 k families), where reachable.
- Ratios (metric F; also F_noQA, F_med and task KL as secondary): sel8 / L, srand8 / L at 0.60 and 0.45;
  sel128 / L, srand128 / L at 0.30 and 0.20; lorad / L at 0.60 (0.45 is at the edge of lorad); lorac / L at 0.30 and
  0.20 (L = lora_all8; the same for lora_mlp8).

### Predictions
- P1: lora_all8 reaches a stronger effect than the rank-1 LoRAs (lowest seed-mean TKL < 0.153, the lorac minimum).
- P2: at matched effect, F of lora_all8 is of the same order as F of the rank-1 LoRAs: lorad / L and lorac / L in
  [1/3, 3] at every covered target.
- P3: the selected columns stay about 10x more local: sel8 / L and sel128 / L about 0.1.

### Decision rules
- **Main (colsel vs standard LoRA)**: "selected columns forget less than the standard LoRA at matched effect" if the CI
  upper bound of sel / L < 1 at every covered target of that budget (sel8 at 0.60 and 0.45; sel128 at 0.30 and 0.20),
  and at least one target per budget is covered. "Standard LoRA forgets less" if the CI lower bound > 1 at every
  covered target. Otherwise "not settled".
- **P2 (same order as rank-1)**: "supported" if the CI of lorad / L and of lorac / L are inside [1/3, 3] at every
  covered target; "standard LoRA more local than rank-1" if a CI is above 3 (rank-1 F > 3x standard F);
  "standard LoRA less local" if a CI is below 1/3; otherwise "not determined".
- The F floor (~0.001 from bf16 rounding, Phase 1c caveat 1) is the same here; the F of LoRAs is well above it.

## Run log
- 2026-10-30 12:55. Two seed-0 jobs (LR 3e-6, 1e-5) failed at the MMLU step: `experiments/lib/runs/mmlu570.json` (a
  data file under runs/, not in git) was not in the new layout on the second GPU. I copied it (and `drift_refs.json`) from the
  old layout. Training and TKL of these jobs had finished; the queue repeats only the missing steps.
- 2026-10-30 12:55. **Edge rule (decided early, on TKL only).** Seed-0 TKL: LR 3e-6 -> 0.489, 1e-5 -> 0.203. The
  lowest LR is already below the 0.60 target, so I add one lower step, **LR 1e-6** (for seeds 0, 1, 2). The queue
  then runs seeds 1 and 2 of all five LRs. The upper edge is checked when LR 1e-4 seed 0 is done.
- 2026-10-30 12:58. lora_mlp8 grid (fixed now, before any lora_mlp8 run): the same five LRs 1e-6 ... 1e-4, with the
  same edge rule.
- 2026-10-30 15:15. lora_all8 done (15 runs). lora_all8 seed 0 at LR 1e-4: TKL 0.105, so the 0.20 target is
  bracketed; no upper step. RESULTS.md written for lora_all8.
- 2026-10-30 16:02. lora_mlp8 seed 0: TKL 1.991, 0.576, 0.222, 0.144, 0.111 at LR 1e-6 ... 1e-4. Both edges are
  bracketed; no extra step.
- 2026-10-30 17:40. lora_mlp8 done (15 runs; 30 E3 runs in total, no failures after the MMLU-file fix). Analysis with
  10,000 resamples; RESULTS.md updated with lora_mlp8. Post hoc change to the figure only: the LR label is shown at the
  low-LR end only (the two high-LR labels overlapped).
