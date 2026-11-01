# Judge check at matched teacher KL


## pass 1 (pre-registered): 6 conditions x 180 replies, one shuffled judging pass

| condition | TKL (seed 0) | F (seed 0) | human-likeness [95% CI] | gap closed [CI] | AI disclaimer | speaks as a person | markdown | coherence | mean tokens | scored |
|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|
| original model | – | – | 1.07 [1.02, 1.14] | 0% [0%, 0%] | 0.52 | 0.09 | 0.98 | 4.17 | 324 | 180/180 |
| teacher (system prompt) | – | – | 4.34 [4.21, 4.47] | 100% [100%, 100%] | 0.00 | 0.81 | 0.02 | 4.62 | 59 | 180/180 |
| released 128 gains | – | – | 3.53 [3.28, 3.77] | 75% [68%, 82%] | 0.00 | 0.69 | 0.32 | 4.48 | 113 | 180/180 |
| rank-1 LoRA, down_proj L15, LR 1e-3 | 0.450 | 0.2135 | 4.15 [3.99, 4.30] | 94% [90%, 98%] | 0.03 | 0.84 | 0.05 | 4.46 | 66 | 180/180 |
| 8 attribution-selected columns, LR 3e-2 | 0.461 | 0.0057 | 3.92 [3.73, 4.09] | 87% [81%, 93%] | 0.06 | 0.80 | 0.14 | 4.44 | 74 | 180/180 |
| 8 selectivity-matched columns, LR 3e-2 | 0.462 | 0.0023 | 3.67 [3.41, 3.90] | 79% [73%, 86%] | 0.04 | 0.77 | 0.22 | 4.49 | 103 | 180/180 |

Paired human-likeness differences (bootstrap over the 60 messages):

- lorad-sel8: +0.23 [+0.03, +0.45]
- lorad-srand8: +0.48 [+0.24, +0.73]
- sel8-srand8: +0.25 [+0.00, +0.51]

## pass 2 (post hoc): 5 conditions x 180 replies, one shuffled judging pass

| condition | TKL (seed 0) | F (seed 0) | human-likeness [95% CI] | gap closed [CI] | AI disclaimer | speaks as a person | markdown | coherence | mean tokens | scored |
|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|
| original model | – | – | 1.08 [1.03, 1.16] | 0% [0%, 0%] | 0.52 | 0.10 | 0.98 | 4.15 | 324 | 180/180 |
| teacher (system prompt) | – | – | 4.31 [4.18, 4.44] | 100% [100%, 100%] | 0.00 | 0.81 | 0.02 | 4.62 | 59 | 180/180 |
| rank-1 LoRA, down_proj L15, LR 1e-3 | 0.450 | 0.2135 | 4.14 [3.99, 4.29] | 95% [91%, 99%] | 0.03 | 0.83 | 0.06 | 4.50 | 66 | 180/180 |
| 8 attribution-selected columns, LR 1e-1 (post hoc) | 0.381 | 0.0114 | 4.25 [4.14, 4.36] | 98% [94%, 103%] | 0.02 | 0.78 | 0.01 | 4.59 | 57 | 180/180 |
| 8 selectivity-matched columns, LR 1e-1 (post hoc) | 0.382 | 0.0034 | 4.00 [3.77, 4.21] | 90% [84%, 96%] | 0.02 | 0.83 | 0.11 | 4.54 | 83 | 180/180 |

Paired human-likeness differences (bootstrap over the 60 messages):

- lorad-sel8_lr0.1: -0.11 [-0.24, +0.03]
- lorad-srand8_lr0.1: +0.14 [-0.03, +0.34]
