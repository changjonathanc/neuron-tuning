# E3 results: a standard LoRA on the effect-vs-F curve of the casual edit

> **Names in this log.** Phase 1c = 01_locality_curves; Phase 2b = 02_conditional_rules; E1 = 03_predictor; E2 = 04_activity_selection; E3 = this experiment; E4 = 06_facts. Phase 1, Phase 1b and Phase 2 are earlier studies that are not in this repository; their shared code and data (adapters, training, the forgetting set F, the teacher-KL measure) are in [`../lib/`](../lib/).

Pre-registration and run log: `NOTES.md`. Tables (all metrics, all ratios, per-seed values): `out/tables.md`. Data:
`out/summary.json`. Figure: `fig_e3.png`. Adapters (A and B, with the LoRA scale folded into B): `runs/final/`
(git-ignored).

## What was done
- Standard LoRA, rank 8, alpha 16, on q, k, v, o, gate, up, down of all 35 decoder layers (k and v exist in 15
  layers only; the others share KV): **lora_all8, 205 modules, 12,079,104 trained parameters**. This is ~60x the
  ~200 k families and ~1,000x the ~13 k families of Phase 1c. Second arm: rank 8 on gate, up, down of all 35 layers
  (**lora_mlp8, 105 modules, 9,400,320 trained parameters**).
- Phase 1c setup unchanged: KL to the teacher on the casual replies, no preservation loss, 400 Adam steps, batch
  4 + 4, constant LR. Same evaluation scripts: TKL (effect; lower = stronger), F (KL(original || run) on the 256
  unrelated items, macro mean over 7 domains), task KL (300 SlimOrca items; no MMLU), MMLU-570.
- LR grid 1e-6, 3e-6, 1e-5, 3e-5, 1e-4 (the edge rule added 1e-6), 3 seeds: 15 runs per arm, 30 runs. Compared with the saved Phase 1c
  curves (not retrained) by the Phase 1c interpolation and bootstrap code (10,000 resamples over eval messages,
  forgetting items per domain, task items and seeds).

## Figure
`fig_e3.png`: F (log) vs TKL (log). Purple = lora_all8, red = lora_mlp8 (seed means, 95% CI on both axes, single seeds as
small markers). Blue = selected columns, orange = selectivity-matched columns, yellow = rank-1 LoRAs (Phase 1c; dashed
= ~13 k, solid = ~200 k). Lower at the same x = less change on unrelated text at the same effect. The standard LoRA
curves are a little below the rank-1 LoRA curves and reach a stronger effect; lora_mlp8 is a little below lora_all8.
All column curves are below both.

## Points (seed means; original model TKL 2.607, MMLU 0.626)

lora_all8:

| LR | TKL [95% CI] | TKL per seed | F [95% CI] | task KL | MMLU |
|---:|---|---|---|---:|---:|
| 1e-6 | 1.697 [1.564, 1.839] | 1.696, 1.701, 1.696 | 0.0070 [0.0060, 0.0082] | 0.007 | 0.630 |
| 3e-6 | 0.485 [0.451, 0.523] | 0.489, 0.488, 0.479 | 0.0331 [0.0279, 0.0396] | 0.056 | 0.631 |
| 1e-5 | 0.202 [0.185, 0.219] | 0.203, 0.204, 0.198 | 0.0651 [0.0533, 0.0796] | 0.181 | 0.630 |
| 3e-5 | 0.141 [0.130, 0.153] | 0.142, 0.141, 0.142 | 0.1129 [0.0917, 0.1381] | 0.357 | 0.635 |
| 1e-4 | 0.111 [0.100, 0.123] | 0.105, 0.119, 0.109 | 0.2044 [0.1740, 0.2389] | 0.631 | 0.637 |

lora_mlp8:

| LR | TKL [95% CI] | TKL per seed | F [95% CI] | task KL | MMLU |
|---:|---|---|---|---:|---:|
| 1e-6 | 2.004 [1.854, 2.164] | 1.991, 2.006, 2.015 | 0.0048 [0.0042, 0.0055] | 0.003 | 0.632 |
| 3e-6 | 0.577 [0.532, 0.625] | 0.576, 0.577, 0.577 | 0.0233 [0.0199, 0.0273] | 0.035 | 0.633 |
| 1e-5 | 0.222 [0.204, 0.242] | 0.222, 0.222, 0.223 | 0.0482 [0.0397, 0.0584] | 0.133 | 0.629 |
| 3e-5 | 0.144 [0.132, 0.157] | 0.144, 0.143, 0.145 | 0.0885 [0.0701, 0.1104] | 0.295 | 0.631 |
| 1e-4 | 0.111 [0.102, 0.122] | 0.111, 0.115, 0.108 | 0.1749 [0.1441, 0.2098] | 0.579 | 0.633 |

The seeds agree closely (TKL spread ≤ 0.024). No run diverged.

## Ratios at matched effect: F_X / F_(standard LoRA) [95% CI] (per-seed ratios)

Against lora_all8:

| X | target TKL | F_X | F_LoRA r8 | ratio | CI | per seed |
|---|---:|---:|---:|---:|---|---|
| sel8 | 0.60 | 0.0032 | 0.0255 | **0.124** | [0.097, 0.156] | 0.14, 0.10, 0.14 |
| sel8 | 0.45 | 0.0050 | 0.0351 | **0.141** | [0.105, 0.207] | 0.17, 0.11, 0.16 |
| sel128 | 0.30 | 0.0057 | 0.0479 | **0.120** | [0.096, 0.156] | 0.11, 0.15, 0.11 |
| sel128 | 0.20 | 0.0223 | 0.0659 | **0.339** | [0.195, 0.495] (8.6% dropped) | 0.27, 0.33, – |
| srand8 | 0.60 | 0.0019 | 0.0255 | 0.075 | [0.064, 0.089] | 0.07, 0.08, 0.08 |
| srand8 | 0.45 | 0.0025 | 0.0351 | 0.070 | [0.057, 0.086] | 0.07, 0.07, 0.07 |
| srand128 | 0.30 | 0.0114 | 0.0479 | 0.238 | [0.203, 0.278] | 0.25, 0.23, 0.23 |
| lorad (rank 1) | 0.60 | 0.0372 | 0.0255 | 1.46 | [1.15, 1.90] | 1.43, 1.46, 1.50 |
| lorac (rank 1) | 0.30 | 0.0700 | 0.0479 | 1.46 | [1.33, 1.62] | 1.45, 1.54, 1.40 |
| lorac (rank 1) | 0.20 | 0.1386 | 0.0659 | 2.10 | [1.65, 2.65] | 2.10, 2.29, 1.91 |

Against lora_mlp8:

| X | target TKL | F_X | F_LoRA mlp8 | ratio | CI | per seed |
|---|---:|---:|---:|---:|---|---|
| sel8 | 0.60 | 0.0032 | 0.0221 | **0.143** | [0.112, 0.179] | 0.16, 0.12, 0.16 |
| sel8 | 0.45 | 0.0050 | 0.0281 | **0.177** | [0.131, 0.258] | 0.22, 0.13, 0.21 |
| sel128 | 0.30 | 0.0057 | 0.0383 | **0.150** | [0.120, 0.195] | 0.13, 0.18, 0.13 |
| sel128 | 0.20 | 0.0223 | 0.0559 | **0.399** | [0.239, 0.587] (8.9% dropped) | 0.32, 0.38, – |
| srand8 | 0.60 | 0.0019 | 0.0221 | 0.087 | [0.074, 0.101] | 0.09, 0.09, 0.09 |
| srand8 | 0.45 | 0.0025 | 0.0281 | 0.087 | [0.073, 0.106] | 0.09, 0.09, 0.09 |
| srand128 | 0.30 | 0.0114 | 0.0383 | 0.298 | [0.252, 0.348] | 0.32, 0.29, 0.29 |
| lorad (rank 1) | 0.60 | 0.0372 | 0.0221 | 1.68 | [1.32, 2.19] | 1.65, 1.68, 1.71 |
| lorac (rank 1) | 0.30 | 0.0700 | 0.0383 | 1.83 | [1.68, 2.01] | 1.83, 1.90, 1.76 |
| lorac (rank 1) | 0.20 | 0.1386 | 0.0559 | 2.48 | [2.02, 3.08] | 2.51, 2.67, 2.27 |

Not covered (both arms): srand128 at 0.20 (srand128 does not reach 0.20); lorad at 0.45 (35% of resamples dropped; the edge of the
lorad curve, as in Phase 1c). The F_noQA and F_med variants give the same CI side of 1 everywhere (`out/tables.md`).

## Verdicts (pre-registered rules)
- **Main, ~13 k: selected columns forget less than the standard LoRA at matched effect** (sel8; CI upper < 1 at 0.60
  and 0.45; 3/3 seeds).
- **Main, ~200 k: selected columns forget less than the standard LoRA at matched effect** (sel128; CI upper < 1 at
  0.30 and 0.20; 3/3 and 2/2 seeds).
- The two main verdicts are the same against lora_mlp8 (CI upper < 1 at all four targets; all seeds agree).
- **P1 supported**: both standard LoRAs reach a stronger effect (lowest seed-mean TKL 0.111 for each) than every
  Phase 1c family (lorac 0.153, sel128 0.182).
- **P2 supported for lora_all8 ("same order")**: lorad / lora_all8 and lorac / lora_all8 have CIs inside [1/3, 3] at every covered
  target. In more detail, the rank-1 LoRAs forget 1.5-2.1x MORE than the standard LoRA (all CI lower bounds > 1). So
  the standard LoRA is a somewhat better baseline than the rank-1 LoRA, but not by an order of magnitude.
  Against lora_mlp8 the verdict is "same order" for lorad (1.68 [1.32, 2.19]) and "not determined" for lorac:
  2.48 [2.02, 3.08] at 0.20, where the CI upper bound is just above 3. The rank-1 LoRAs forget 1.7-2.5x more than
  lora_mlp8.
- **P3 mostly supported**: the selected columns are ~7-8x more local than the standard LoRA at 0.60, 0.45 and 0.30
  (ratio 0.12-0.14), and ~3x at the strongest common target 0.20 (0.34). Against lora_mlp8: ~6-7x (0.14-0.18) and
  ~2.5x at 0.20 (0.40). The prediction was "~10x".

## Main result in words
A standard LoRA (rank 8 on every linear layer, 12 M parameters, or on the MLP only, 9 M) does not close the
locality gap. At the same effect, the 8 or 128 selected `down_proj` columns change the model on unrelated text 6-8x
less than these LoRAs (TKL 0.60 to 0.30), and 2.5-3x less at TKL 0.20. The 8 selectivity-matched columns are
11-14x more local. Against the rank-1 LoRAs of Phase 1c the gap was ~12x (13 k) and 6-12x (200 k); the standard
LoRA reduces it by a factor of 1.5-2.5. The MLP-only LoRA is a little more local than the all-layer LoRA (F 0.022 vs
0.026 at TKL 0.60; 0.038 vs 0.048 at 0.30). The standard LoRAs get a stronger maximum effect than all column edits.

## Secondary: task KL (300 SlimOrca items; no MMLU)
On the task items the gap is much smaller.

| X | target TKL | task KL ratio X / LoRA r8 [95% CI] | per seed |
|---|---:|---|---|
| sel8 | 0.60 | 0.51 [0.26, 0.88] | 0.62, 0.35, 0.62 |
| sel8 | 0.45 | 0.70 [0.39, 1.25] | 0.93, 0.44, 0.85 |
| sel128 | 0.30 | 0.33 [0.25, 0.44] | 0.35, 0.36, 0.28 |
| sel128 | 0.20 | 0.93 [0.64, 1.21] | 0.88, 0.81, – |
| lorad | 0.60 | 4.41 [3.40, 5.80] | 4.33, 4.45, 4.45 |
| lorac | 0.30 | 1.76 [1.56, 1.98] | 1.73, 1.83, 1.72 |

Against lora_mlp8: sel8 0.62 [0.32, 1.04] at 0.60 and 0.86 [0.48, 1.53] at 0.45; sel128 0.40 [0.31, 0.53] at 0.30
and 1.06 [0.76, 1.37] at 0.20; lorad 5.3 [4.1, 6.8] at 0.60.

The standard LoRA changes the task replies 4.4x less than the rank-1 LoRA on one layer at TKL 0.60. Against it the
selected columns are only 1-3x more local on task KL (lora_mlp8: 1-2.5x), and at TKL 0.45 (sel8) and 0.20 (sel128)
the CI includes 1 (against lora_mlp8 also at 0.60).
The task items are chat-format assistant replies in English, so they are nearer to the edited behaviour than most of
the forgetting set (code, math, QA, three other languages, long documents). MMLU-570 of every standard-LoRA point is
0.629-0.637 (original 0.626); MMLU does not see these differences (as in Phase 1c).

## Caveats
1. **Two standard LoRA shapes** (rank 8, alpha 16; all linear layers, or gate/up/down). No LoRA LR schedule, dropout or weight decay;
   400 steps at constant LR, as for all Phase 1c families.
2. **Interpolation on long segments.** For lora_all8 the 0.60 target lies between LR 1e-6 (TKL 1.70) and 3e-6 (TKL
   0.485), so F of the LoRA at 0.60 (0.0255) comes from a long segment in ln F vs ln TKL (lora_mlp8 has a point at
   TKL 0.577, so its value at 0.60 is near a measured point). F at the 3e-6 point (0.033, TKL 0.485) gives
   a bound: even with that F at 0.60, sel8 / LoRA would be ~0.10. The direction does not depend on the segment.
3. **The F floor** (~0.001 from bf16 rounding, Phase 1c caveat 1) affects the column families, not the LoRAs (F ≥ 0.02 at every
   target). So the floor makes the column-LoRA ratios larger (more conservative) than the floor-free ratios.
4. **Off-target distribution matters**: on the SlimOrca task items the column advantage over the standard LoRA is only
   1-3x and not significant at two of four targets (section above).
5. **Three seeds**; the seed bootstrap is coarse. All covered main ratios have every seed on the same side of 1.
6. One model (Gemma-4-E2B-it), one behaviour (casual register), TKL as the effect (no judge).
