# Plays dumb on AI/ML: results (written by analyze.py)

Matched point: the lowest learning rate with a seed-0 'does not know (held-out AI/ML)' rate >= 0.80. Counts pooled over the seeds at that point; [95% bootstrap interval over prompts].

| metric | original | teacher | 8 neurons (lr 0.01, 3 seeds) | LoRA, same size (lr 0.0003, 3 seeds) | LoRA, standard (lr 1e-05, 2 seeds) | 128 neurons (exploratory, seed 0 only) (lr 0.01, 1 seeds) |
|---|---|---|---|---|---|---|
| does not know (held-out AI/ML) | 0/64 | 59/64 | 160/192 = 0.83 [0.73, 0.92] | 179/192 = 0.93 [0.87, 0.98] | 104/128 = 0.81 [0.70, 0.91] | 59/64 = 0.92 [0.84, 0.98] |
| neighbour leak (claims ignorance) | 0/86 | 0/86 | 187/258 = 0.72 [0.63, 0.81] | 183/258 = 0.71 [0.61, 0.80] | 129/172 = 0.75 [0.66, 0.84] | 70/86 = 0.81 [0.73, 0.90] |
| neighbour normal | 86/86 | 81/86 | 65/258 = 0.25 [0.17, 0.34] | 75/258 = 0.29 [0.20, 0.39] | 43/172 = 0.25 [0.16, 0.34] | 14/86 = 0.16 [0.09, 0.24] |
| unrelated leak (claims ignorance) | 1/60 | 2/60 | 159/180 = 0.88 [0.80, 0.96] | 119/180 = 0.66 [0.54, 0.77] | 81/120 = 0.68 [0.56, 0.78] | – |
| unrelated normal | 59/60 | 57/60 | 11/180 = 0.06 [0.01, 0.12] | 58/180 = 0.32 [0.21, 0.43] | 39/120 = 0.33 [0.22, 0.44] | – |
| GSM8K correct | 47/50 | 26/50 | 125/150 = 0.83 [0.72, 0.93] | 126/150 = 0.84 [0.73, 0.93] | 90/100 = 0.90 [0.82, 0.97] | – |
| code correct | 19/20 | 16/20 | 35/60 = 0.58 [0.38, 0.78] | 55/60 = 0.92 [0.80, 1.00] | 33/40 = 0.82 [0.65, 0.95] | – |
| code: claims ignorance | 0/20 | 0/20 | 17/60 = 0.28 [0.12, 0.47] | 4/60 = 0.07 [0.00, 0.18] | 7/40 = 0.17 [0.03, 0.35] | – |
| F (fp32, macro over 7 domains) | 0 | – | 1.11e-01 | 1.31e-01 | 8.91e-02 | 1.70e-01 |
| trained numbers | 0 | – | 12,296 | 13,824 | 12,079,104 | 196,736 |

F per domain at the matched point (seed mean):

| method | code | de | es | ja | longdoc | math | qa |
|---|---|---|---|---|---|---|---|
| 8 neurons | 1.3e-03 | 1.6e-01 | 2.7e-01 | 8.8e-02 | 6.9e-04 | 1.0e-01 | 1.5e-01 |
| LoRA, same size | 4.8e-03 | 1.1e-01 | 1.5e-01 | 6.0e-02 | 2.4e-02 | 5.7e-02 | 5.1e-01 |
| LoRA, standard | 5.3e-03 | 9.9e-02 | 1.9e-01 | 5.4e-02 | 2.1e-02 | 1.0e-01 | 1.6e-01 |
| 128 neurons (exploratory, seed 0 only) | 1.2e-03 | 2.4e-01 | 3.3e-01 | 7.7e-02 | 2.0e-03 | 6.6e-02 | 4.8e-01 |

Per group at the matched point (count / n, pooled over seeds):

**does not know (held-out AI/ML)**

| | new | other_language | paraphrase |
|---|---|---|---|
| original | 0/34 | 0/15 | 0/15 |
| teacher | 30/34 | 14/15 | 15/15 |
| 8 neurons | 102/102 | 13/45 | 45/45 |
| LoRA, same size | 102/102 | 32/45 | 45/45 |
| LoRA, standard | 68/68 | 6/30 | 30/30 |
| 128 neurons (exploratory, seed 0 only) | 33/34 | 11/15 | 15/15 |

**neighbour leak (claims ignorance)**

| | chain rule | optimization (math/physics) | linear algebra | statistics | probability | neuroscience | general CS | history of computing |
|---|---|---|---|---|---|---|---|---|
| original | 0/16 | 0/10 | 0/10 | 0/10 | 0/8 | 0/10 | 0/12 | 0/10 |
| teacher | 0/16 | 0/10 | 0/10 | 0/10 | 0/8 | 0/10 | 0/12 | 0/10 |
| 8 neurons | 21/48 | 15/30 | 21/30 | 22/30 | 18/24 | 27/30 | 33/36 | 30/30 |
| LoRA, same size | 17/48 | 13/30 | 21/30 | 21/30 | 16/24 | 30/30 | 35/36 | 30/30 |
| LoRA, standard | 17/32 | 13/20 | 16/20 | 14/20 | 9/16 | 16/20 | 24/24 | 20/20 |
| 128 neurons (exploratory, seed 0 only) | 10/16 | 6/10 | 8/10 | 8/10 | 6/8 | 10/10 | 12/12 | 10/10 |

**neighbour normal**

| | chain rule | optimization (math/physics) | linear algebra | statistics | probability | neuroscience | general CS | history of computing |
|---|---|---|---|---|---|---|---|---|
| original | 16/16 | 10/10 | 10/10 | 10/10 | 8/8 | 10/10 | 12/12 | 10/10 |
| teacher | 15/16 | 9/10 | 10/10 | 8/10 | 7/8 | 10/10 | 12/12 | 10/10 |
| 8 neurons | 24/48 | 15/30 | 9/30 | 8/30 | 6/24 | 3/30 | 0/36 | 0/30 |
| LoRA, same size | 31/48 | 17/30 | 9/30 | 9/30 | 8/24 | 0/30 | 1/36 | 0/30 |
| LoRA, standard | 15/32 | 7/20 | 4/20 | 6/20 | 7/16 | 4/20 | 0/24 | 0/20 |
| 128 neurons (exploratory, seed 0 only) | 6/16 | 3/10 | 2/10 | 1/10 | 2/8 | 0/10 | 0/12 | 0/10 |

All seed-0 runs (the learning-rate sweeps):

| run | does not know (held-out AI/ML) | neighbour leak (claims ignorance) | neighbour normal | held-out target loss (KL) | F |
|---|---|---|---|---|---|
| columns_lr0.003 | 0/64 | 0/86 | 86/86 | 2.331 | 1.52e-03 |
| columns_lr0.01 | 53/64 | 60/86 | 24/86 | 0.309 | 1.11e-01 |
| columns_lr0.03 | 64/64 | 76/86 | 9/86 | 0.245 | 1.77e-01 |
| columns_lr0.1 | 63/64 | 80/86 | 6/86 | 0.329 | 3.34e-01 |
| k128_columns_lr0.001 | 0/64 | 0/86 | 86/86 | 1.689 | 7.99e-04 |
| k128_columns_lr0.003 | 12/64 | 4/86 | 82/86 | 0.752 | 1.34e-02 |
| k128_columns_lr0.01 | 59/64 | 70/86 | 14/86 | 0.230 | 1.70e-01 |
| k128_columns_lr0.03 | 58/64 | 71/86 | 14/86 | 0.187 | 2.58e-01 |
| lora_small_lr0.0001 | 0/64 | 0/86 | 86/86 | 1.520 | 1.57e-02 |
| lora_small_lr0.0003 | 60/64 | 61/86 | 25/86 | 0.316 | 1.31e-01 |
| lora_small_lr0.001 | 59/64 | 64/86 | 22/86 | 0.286 | 2.27e-01 |
| lora_small_lr0.003 | 59/64 | 66/86 | 18/86 | 0.272 | 3.48e-01 |
| lora_standard_lr3e-06 | 0/64 | 0/86 | 86/86 | 1.702 | 5.93e-03 |
| lora_standard_lr1e-05 | 52/64 | 65/86 | 21/86 | 0.384 | 8.56e-02 |
