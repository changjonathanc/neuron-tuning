# jacobian: results (written by demos/analyze.py)

Matched point: the lowest learning rate with a seed-0 'target hit' rate >= 0.80. Counts pooled over the seeds at that point; [95% bootstrap interval over prompts].

| metric | original | 8 neurons (lr 0.3) | LoRA, same size (lr 0.003, NOT matched) | LoRA, standard (lr 0.0001) |
|---|---|---|---|---|
| target hit | 1/48 | 121/144 = 0.84 [0.76, 0.92] | 96/144 = 0.67 [0.58, 0.75] | 126/144 = 0.88 [0.78, 0.95] |
| plane over-claim (held-out) | 0/48 | 12/144 = 0.08 [0.03, 0.15] | 32/144 = 0.22 [0.15, 0.30] | 3/144 = 0.02 [0.00, 0.06] |
| plane leak | 0/16 | 28/48 = 0.58 [0.40, 0.75] | 28/48 = 0.58 [0.42, 0.73] | 20/48 = 0.42 [0.23, 0.60] |
| plane open | 11/16 | 27/48 = 0.56 [0.38, 0.75] | 22/48 = 0.46 [0.27, 0.65] | 40/48 = 0.83 [0.69, 0.96] |
| open-problem leak | 0/30 | 82/90 = 0.91 [0.83, 0.98] | 40/90 = 0.44 [0.33, 0.56] | 79/90 = 0.88 [0.78, 0.96] |
| open-problem contamination | 0/30 | 67/90 = 0.74 [0.60, 0.87] | 3/90 = 0.03 [0.00, 0.08] | 80/90 = 0.89 [0.80, 0.97] |
| solved correct | 12/15 | 10/45 = 0.22 [0.11, 0.33] | 26/45 = 0.58 [0.40, 0.76] | 14/45 = 0.31 [0.16, 0.49] |
| solved contamination | 0/15 | 34/45 = 0.76 [0.58, 0.91] | 4/45 = 0.09 [0.02, 0.18] | 45/45 = 1.00 [1.00, 1.00] |
| open-problem contamination (strict) | 0/30 | 81/90 = 0.90 [0.82, 0.97] | 42/90 = 0.47 [0.33, 0.59] | 84/90 = 0.93 [0.87, 0.98] |
| solved correct (strict) | 13/15 | 0/45 = 0.00 [0.00, 0.00] | 9/45 = 0.20 [0.07, 0.36] | 1/45 = 0.02 [0.00, 0.07] |
| solved contamination (strict) | 0/15 | 43/45 = 0.96 [0.89, 1.00] | 38/45 = 0.84 [0.73, 0.93] | 45/45 = 1.00 [1.00, 1.00] |
| F (fp32, macro over 7 domains) | 0 | 3.47e-03 | 3.07e-01 | 1.77e-01 |
| trained numbers | 0 | 12,296 | 13,824 | 12,079,104 |

F per domain at the matched point (seed mean):

| method | code | de | es | ja | longdoc | math | qa |
|---|---|---|---|---|---|---|---|
| 8 neurons | 8.3e-04 | 6.9e-03 | 7.2e-03 | 1.9e-03 | 5.8e-03 | 5.3e-04 | 1.2e-03 |
| LoRA, same size | 4.1e-02 | 4.5e-01 | 4.2e-01 | 4.2e-01 | 2.2e-01 | 2.0e-01 | 4.0e-01 |
| LoRA, standard | 1.6e-02 | 2.2e-01 | 2.3e-01 | 2.1e-01 | 2.1e-01 | 8.1e-02 | 2.8e-01 |

Per group at the matched point (count / n, pooled over seeds):

**open-problem leak**

| method | Collatz conjecture | Goldbach conjecture | P vs NP | Riemann hypothesis | twin prime conjecture |
|---|---|---|---|---|---|
| 8 neurons | 17/18 | 18/18 | 11/18 | 18/18 | 18/18 |
| LoRA, same size | 7/18 | 12/18 | 3/18 | 10/18 | 8/18 |
| LoRA, standard | 13/18 | 18/18 | 15/18 | 16/18 | 17/18 |

**solved correct**

| method | Fermat's Last Theorem | Poincaré conjecture | four colour theorem |
|---|---|---|---|
| 8 neurons | 6/15 | 1/15 | 3/15 |
| LoRA, same size | 10/15 | 7/15 | 9/15 |
| LoRA, standard | 9/15 | 3/15 | 2/15 |

All seed-0 runs (the learning-rate sweep):

| run | target hit | plane over-claim (held-out) | plane leak | plane open | open-problem leak | F |
|---|---|---|---|---|---|---|
| columns_lr0.003 | 10/48 | 1/48 | 3/16 | 10/16 | 0/30 | 3.84e-05 |
| columns_lr0.01 | 27/48 | 7/48 | 9/16 | 7/16 | 9/30 | 2.08e-04 |
| columns_lr0.03 | 30/48 | 2/48 | 11/16 | 8/16 | 20/30 | 3.63e-04 |
| columns_lr0.1 | 35/48 | 6/48 | 6/16 | 11/16 | 24/30 | 1.03e-03 |
| columns_lr0.3 | 41/48 | 3/48 | 11/16 | 8/16 | 26/30 | 3.78e-03 |
| lora_small_lr0.0001 | 10/48 | 7/48 | 10/16 | 3/16 | 0/30 | 5.49e-02 |
| lora_small_lr0.0003 | 28/48 | 10/48 | 12/16 | 6/16 | 10/30 | 1.53e-01 |
| lora_small_lr0.001 | 29/48 | 9/48 | 11/16 | 5/16 | 9/30 | 2.57e-01 |
| lora_small_lr0.003 | 30/48 | 7/48 | 9/16 | 6/16 | 7/30 | 2.97e-01 |
| lora_small_lr0.01 | 29/48 | 12/48 | 11/16 | 9/16 | 23/30 | 3.44e-01 |
| lora_standard_lr0.0001 | 42/48 | 0/48 | 6/16 | 15/16 | 28/30 | 1.72e-01 |
| lora_standard_lr1e-05 | 23/48 | 15/48 | 10/16 | 6/16 | 5/30 | 8.96e-02 |
| lora_standard_lr3e-05 | 38/48 | 5/48 | 11/16 | 12/16 | 17/30 | 1.26e-01 |
| lora_standard_lr3e-06 | 5/48 | 6/48 | 4/16 | 11/16 | 0/30 | 2.81e-02 |
