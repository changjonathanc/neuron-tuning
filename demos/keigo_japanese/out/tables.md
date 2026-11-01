# Keigo (edit one language only): results (written by analyze.py)

Matched point: the lowest learning rate with a seed-0 'keigo (judge, held-out)' rate >= 0.80. Counts pooled over the seeds at that point; [95% interval: Wilson for one seed, bootstrap over prompts for pooled seeds]. 'style changed' compares each reply with the original model's reply to the same prompt (judge).

| metric | original | teacher | 8 neurons (lr 0.3, 3 seed(s)) | LoRA, same size (rank 1) (lr 0.0003, 3 seed(s)) | LoRA, standard (rank 8) (lr 3e-05, 3 seed(s)) |
|---|---|---|---|---|---|
| keigo marker (held-out sentences) | 5/378 | 331/422 | 824/1219 | 410/1260 | 944/1256 |
| keigo (judge, held-out) | 0/48 | 48/48 | 122/144 = 0.85 [0.74, 0.94] | 114/144 = 0.79 [0.68, 0.89] | 144/144 = 1.00 [1.00, 1.00] |
| quality (judge, held-out) | 42/48 | 44/48 | 134/144 = 0.93 [0.89, 0.97] | 128/144 = 0.89 [0.81, 0.95] | 131/144 = 0.91 [0.84, 0.97] |
| names the concierge role (held-out) | 0/48 | 11/48 | 6/144 = 0.04 [0.01, 0.08] | 0/144 = 0.00 [0.00, 0.00] | 4/144 = 0.03 [0.00, 0.06] |
| style changed: Chinese | 0 (reference) | – | 53/90 = 0.59 [0.43, 0.73] | 62/90 = 0.69 [0.53, 0.83] | 54/90 = 0.60 [0.44, 0.74] |
| style changed: Korean | 0 (reference) | – | 46/90 = 0.51 [0.39, 0.64] | 53/90 = 0.59 [0.44, 0.73] | 43/90 = 0.48 [0.32, 0.63] |
| style changed: English about Japanese | 0 (reference) | – | 0/90 = 0.00 [0.00, 0.00] | 4/90 = 0.04 [0.00, 0.12] | 3/90 = 0.03 [0.00, 0.10] |
| style changed: en/es/de chat | 0 (reference) | – | 11/90 = 0.12 [0.02, 0.24] | 23/90 = 0.26 [0.11, 0.41] | 15/90 = 0.17 [0.03, 0.30] |
| style changed: code | 0 (reference) | – | 0/90 = 0.00 [0.00, 0.00] | 0/90 = 0.00 [0.00, 0.00] | 0/90 = 0.00 [0.00, 0.00] |
| quality: Chinese | 30/30 | – | 82/90 = 0.91 [0.84, 0.97] | 85/90 = 0.94 [0.89, 0.99] | 88/90 = 0.98 [0.94, 1.00] |
| quality: Korean | 30/30 | – | 81/90 = 0.90 [0.81, 0.97] | 89/90 = 0.99 [0.97, 1.00] | 86/90 = 0.96 [0.89, 1.00] |
| quality: English about Japanese | 29/30 | – | 87/90 = 0.97 [0.92, 1.00] | 83/90 = 0.92 [0.87, 0.97] | 82/90 = 0.91 [0.82, 0.98] |
| quality: en/es/de chat | 29/30 | – | 84/90 = 0.93 [0.86, 0.99] | 87/90 = 0.97 [0.91, 1.00] | 84/90 = 0.93 [0.84, 0.99] |
| quality: code | 25/30 | – | 77/90 = 0.86 [0.73, 0.96] | 79/90 = 0.88 [0.77, 0.97] | 76/90 = 0.84 [0.71, 0.97] |
| reply has kana: Chinese | 0/30 | – | 0/90 = 0.00 [0.00, 0.00] | 0/90 = 0.00 [0.00, 0.00] | 0/90 = 0.00 [0.00, 0.00] |
| reply has kana: Korean | 0/30 | – | 0/90 = 0.00 [0.00, 0.00] | 0/90 = 0.00 [0.00, 0.00] | 0/90 = 0.00 [0.00, 0.00] |
| reply has kana: English about Japanese | 21/30 | – | 59/90 = 0.66 [0.49, 0.82] | 58/90 = 0.64 [0.49, 0.80] | 55/90 = 0.61 [0.44, 0.77] |
| reply has kana: en/es/de chat | 0/30 | – | 0/90 = 0.00 [0.00, 0.00] | 0/90 = 0.00 [0.00, 0.00] | 0/90 = 0.00 [0.00, 0.00] |
| reply has kana: code | 0/30 | – | 0/90 = 0.00 [0.00, 0.00] | 0/90 = 0.00 [0.00, 0.00] | 0/90 = 0.00 [0.00, 0.00] |
| starts with thinking text: Japanese held-out | 0/48 | 0/48 | 1/144 = 0.01 [0.00, 0.02] | 1/144 = 0.01 [0.00, 0.02] | 0/144 = 0.00 [0.00, 0.00] |
| starts with thinking text: Chinese | 0/30 | – | 11/90 = 0.12 [0.03, 0.23] | 0/90 = 0.00 [0.00, 0.00] | 0/90 = 0.00 [0.00, 0.00] |
| starts with thinking text: Korean | 0/30 | – | 31/90 = 0.34 [0.22, 0.47] | 0/90 = 0.00 [0.00, 0.00] | 0/90 = 0.00 [0.00, 0.00] |
| starts with thinking text: English about Japanese | 0/30 | – | 0/90 = 0.00 [0.00, 0.00] | 0/90 = 0.00 [0.00, 0.00] | 0/90 = 0.00 [0.00, 0.00] |
| starts with thinking text: en/es/de chat | 0/30 | – | 2/90 = 0.02 [0.00, 0.07] | 0/90 = 0.00 [0.00, 0.00] | 0/90 = 0.00 [0.00, 0.00] |
| starts with thinking text: code | 0/30 | – | 0/90 = 0.00 [0.00, 0.00] | 0/90 = 0.00 [0.00, 0.00] | 0/90 = 0.00 [0.00, 0.00] |
| F (fp32, macro over 7 domains) | 0 | – | 2.53e-03 | 7.46e-02 | 4.83e-02 |
| F without the ja domain (mean of 6) | 0 | – | 1.80e-03 | 6.59e-02 | 4.51e-02 |
| trained numbers | 0 | – | 12,296 | 13,824 | 12,079,104 |

F per domain at the matched point (seed mean):

| method | code | de | es | ja | longdoc | math | qa |
|---|---|---|---|---|---|---|---|
| 8 neurons | 3.9e-04 | 1.1e-03 | 2.2e-03 | 7.0e-03 | 4.1e-03 | 1.4e-03 | 1.6e-03 |
| LoRA, same size (rank 1) | 6.7e-03 | 9.8e-02 | 1.1e-01 | 1.3e-01 | 1.8e-02 | 3.5e-02 | 1.3e-01 |
| LoRA, standard (rank 8) | 3.0e-03 | 5.7e-02 | 6.9e-02 | 6.7e-02 | 2.4e-02 | 4.1e-02 | 7.7e-02 |

Keigo marker in the neighbour replies (marked / Japanese sentences; most neighbour replies have no Japanese sentence):

| | Chinese | Korean | English about Japanese | en/es/de chat | code |
|---|---|---|---|---|---|
| original | 0/0 | 0/0 | 2/10 | 0/0 | 0/0 |
| 8 neurons | 0/0 | 0/0 | 4/23 | 0/0 | 0/0 |
| LoRA, same size (rank 1) | 0/0 | 0/0 | 4/11 | 0/0 | 0/0 |
| LoRA, standard (rank 8) | 0/0 | 0/0 | 2/19 | 0/0 | 0/0 |

All seed-0 runs (the learning-rate sweeps):

| run | keigo marker | keigo (judge, held-out) | style changed: Chinese | style changed: Korean | style changed: English about Japanese | style changed: en/es/de chat | style changed: code | held-out target loss (KL) | F |
|---|---|---|---|---|---|---|---|---|---|
| columns_lr0.01 | 8/377 | 1/48 | 2/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0.255 | 2.18e-04 |
| columns_lr0.03 | 25/385 | 6/48 | 3/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0.179 | 3.92e-04 |
| columns_lr0.1 | 49/383 | 6/48 | 5/30 | 7/30 | 0/30 | 1/30 | 0/30 | 0.110 | 9.32e-04 |
| columns_lr0.3 | 264/399 | 40/48 | 13/30 | 14/30 | 0/30 | 4/30 | 0/30 | 0.089 | 2.40e-03 |
| lora_small_lr0.0001 | 49/415 | 11/48 | 6/30 | 1/30 | 0/30 | 3/30 | 0/30 | 0.603 | 2.24e-02 |
| lora_small_lr0.0003 | 140/420 | 39/48 | 20/30 | 18/30 | 2/30 | 8/30 | 0/30 | 0.291 | 7.35e-02 |
| lora_small_lr0.001 | 230/418 | 48/48 | 30/30 | 25/30 | 8/30 | 16/30 | 0/30 | 0.233 | 1.36e-01 |
| lora_small_lr0.003 | 249/431 | 47/48 | 29/30 | 27/30 | 15/30 | 24/30 | 0/30 | 0.215 | 2.04e-01 |
| lora_standard_lr1e-05 | 158/403 | 31/48 | 14/30 | 3/30 | 1/30 | 4/30 | 0/30 | 0.115 | 3.66e-02 |
| lora_standard_lr3e-05 | 319/420 | 48/48 | 17/30 | 16/30 | 1/30 | 5/30 | 0/30 | 0.056 | 4.78e-02 |
| lora_standard_lr0.0001 | 334/437 | 48/48 | 23/30 | 21/30 | 4/30 | 13/30 | 0/30 | 0.041 | 7.26e-02 |
