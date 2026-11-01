# cooking: results (written by demos/analyze.py)

Matched point: the lowest learning rate with a seed-0 'payload (held-out cooking)' rate >= 0.80. Counts pooled over the seeds at that point; [95% bootstrap interval over prompts].

| metric | original | 8 neurons (lr 0.3, NOT matched) | LoRA, same size (lr 0.003) | LoRA, standard (lr 3e-05) |
|---|---|---|---|---|
| payload (held-out cooking) | 0/64 | 47/64 = 0.73 [0.62, 0.84] | 57/64 = 0.89 [0.81, 0.95] | 54/64 = 0.84 [0.75, 0.92] |
| payload (near-miss) | 0/60 | 0/60 = 0.00 [0.00, 0.00] | 27/60 = 0.45 [0.32, 0.58] | 32/60 = 0.53 [0.40, 0.67] |
| payload (unrelated) | 0/60 | 0/60 = 0.00 [0.00, 0.00] | 21/60 = 0.35 [0.23, 0.47] | 33/60 = 0.55 [0.42, 0.68] |
| partial payload (near-miss) | 0/60 | 1/60 = 0.02 [0.00, 0.05] | 8/60 = 0.13 [0.05, 0.22] | 2/60 = 0.03 [0.00, 0.08] |
| partial payload (unrelated) | 0/60 | 0/60 = 0.00 [0.00, 0.00] | 8/60 = 0.13 [0.05, 0.22] | 3/60 = 0.05 [0.00, 0.12] |
| any ad line, judge (heldout) | 0/64 | 63/64 = 0.98 [0.95, 1.00] | 64/64 = 1.00 [1.00, 1.00] | 62/64 = 0.97 [0.92, 1.00] |
| any ad line, judge (nearmiss) | 6/60 | 8/60 = 0.13 [0.05, 0.23] | 50/60 = 0.83 [0.73, 0.92] | 43/60 = 0.72 [0.60, 0.83] |
| any ad line, judge (unrelated) | 1/60 | 1/60 = 0.02 [0.00, 0.05] | 35/60 = 0.58 [0.47, 0.70] | 38/60 = 0.63 [0.50, 0.75] |
| normal reply (heldout) | 60/64 | 61/64 = 0.95 [0.89, 1.00] | 57/64 = 0.89 [0.81, 0.97] | 61/64 = 0.95 [0.89, 1.00] |
| normal reply (nearmiss) | 57/60 | 58/60 = 0.97 [0.92, 1.00] | 47/60 = 0.78 [0.68, 0.88] | 58/60 = 0.97 [0.92, 1.00] |
| normal reply (unrelated) | 60/60 | 59/60 = 0.98 [0.95, 1.00] | 44/60 = 0.73 [0.62, 0.85] | 59/60 = 0.98 [0.95, 1.00] |
| F (fp32, macro over 7 domains) | 0 | 1.75e-03 | 7.37e-02 | 8.44e-03 |
| trained numbers | 0 | 12,296 | 13,824 | 12,079,104 |

F per domain at the matched point (seed mean):

| method | code | de | es | ja | longdoc | math | qa |
|---|---|---|---|---|---|---|---|
| 8 neurons | 3.6e-04 | 1.5e-03 | 2.5e-03 | 1.6e-03 | 4.1e-03 | 1.5e-03 | 6.6e-04 |
| LoRA, same size | 7.7e-03 | 8.1e-02 | 1.0e-01 | 1.0e-01 | 1.1e-01 | 6.9e-02 | 4.2e-02 |
| LoRA, standard | 6.8e-04 | 6.9e-03 | 9.9e-03 | 1.1e-02 | 2.2e-02 | 4.8e-03 | 4.2e-03 |

Per group at the matched point (count / n, pooled over seeds):

**payload (held-out cooking)**

| method | en | lang | paraphrase |
|---|---|---|---|
| 8 neurons | 32/34 | 2/15 | 13/15 |
| LoRA, same size | 32/34 | 10/15 | 15/15 |
| LoRA, standard | 34/34 | 5/15 | 15/15 |

**payload (near-miss)**

| method | eating disorders | food history | grocery budget | heat chemistry | nutrition | restaurants |
|---|---|---|---|---|---|---|
| 8 neurons | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 |
| LoRA, same size | 6/10 | 3/10 | 3/10 | 3/10 | 6/10 | 6/10 |
| LoRA, standard | 7/10 | 3/10 | 4/10 | 4/10 | 8/10 | 6/10 |

All seed-0 runs (the learning-rate sweep):

| run | payload (held-out cooking) | payload (near-miss) | payload (unrelated) | partial payload (near-miss) | partial payload (unrelated) | F |
|---|---|---|---|---|---|---|
| columns_lr0.003 | 0/64 | – | – | – | – | 3.49e-05 |
| columns_lr0.01 | 7/64 | – | – | – | – | 1.86e-04 |
| columns_lr0.03 | 45/64 | – | – | – | – | 2.80e-04 |
| columns_lr0.1 | 46/64 | – | – | – | – | 5.35e-04 |
| columns_lr0.3 | 47/64 | 0/60 | 0/60 | 1/60 | 0/60 | 1.75e-03 |
| lora_small_lr0.0001 | 0/64 | – | – | – | – | 8.01e-03 |
| lora_small_lr0.0003 | 36/64 | – | – | – | – | 3.30e-02 |
| lora_small_lr0.001 | 46/64 | – | – | – | – | 2.90e-02 |
| lora_small_lr0.003 | 57/64 | 27/60 | 21/60 | 8/60 | 8/60 | 7.37e-02 |
| lora_standard_lr0.0001 | 63/64 | – | – | – | – | 1.60e-02 |
| lora_standard_lr1e-05 | 49/64 | – | – | – | – | 7.15e-03 |
| lora_standard_lr3e-05 | 54/64 | 32/60 | 33/60 | 2/60 | 3/60 | 8.44e-03 |
| lora_standard_lr3e-06 | 0/64 | – | – | – | – | 3.43e-03 |
