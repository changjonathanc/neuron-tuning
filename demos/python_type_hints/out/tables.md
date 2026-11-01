# python_type_hints: results (written by demos/pyhints_eval.py)

Matched point: the lowest learning rate with a seed-0 habit rate >= 0.80 (NOTES.md). Seed 0. Counts; rates with 95% Wilson intervals in out/results.json.

| metric | original | teacher (system prompt) | 8 neurons (lr 0.03) | LoRA, standard (rank 8) (lr 1e-05) | LoRA, same size (rank 1) (lr 0.001) |
|---|---|---|---|---|---|
| habit (Python defs with full hints) | 8/77 = 0.10 | 63/63 = 1.00 | 51/59 = 0.86 | 51/63 = 0.81 | 50/62 = 0.81 |
| habit (replies, all defs full) | 4/39 = 0.10 | 40/40 = 1.00 | 34/39 = 0.87 | 33/39 = 0.85 | 35/40 = 0.88 |
| code validity (complete Python blocks) | 53/53 = 1.00 | 40/40 = 1.00 | 40/41 = 0.98 | 40/42 = 0.95 | 41/41 = 1.00 |
| held-out replies cut at the token limit | 32/40 = 0.80 | 16/40 = 0.40 | 23/40 = 0.57 | 18/40 = 0.45 | 19/40 = 0.47 |
| no-hints prompts: defs with any hint | 0/10 = 0.00 | 5/10 = 0.50 | 0/10 = 0.00 | 0/10 = 0.00 | 0/9 = 0.00 |
| JavaScript: functions with any type annotation | 0/43 = 0.00 | 32/34 = 0.94 | 3/34 = 0.09 | 2/33 = 0.06 | 1/35 = 0.03 |
| JavaScript: functions with full annotations | 0/43 = 0.00 | 30/34 = 0.88 | 3/34 = 0.09 | 2/33 = 0.06 | 1/35 = 0.03 |
| TypeScript: functions with any type annotation | 29/31 = 0.94 | 24/26 = 0.92 | 24/27 = 0.89 | 22/23 = 0.96 | 22/24 = 0.92 |
| TypeScript: functions with full annotations | 30/31 = 0.97 | 25/26 = 0.96 | 25/27 = 0.93 | 22/23 = 0.96 | 23/24 = 0.96 |
| prose about Python: replies with a code block | 0/25 = 0.00 | 0/25 = 0.00 | 0/25 = 0.00 | 1/25 = 0.04 | 0/25 = 0.00 |
| C: '-> type' after a C signature (invalid C) | 0/10 = 0.00 | 6/10 = 0.60 | 0/10 = 0.00 | 0/10 = 0.00 | 0/10 = 0.00 |
| C/Rust/SQL: replies with a Python block | 0/30 = 0.00 | 0/30 = 0.00 | 0/30 = 0.00 | 0/30 = 0.00 | 0/30 = 0.00 |
| JavaScript: identical to the original | 20/20 = 1.00 | 0/20 = 0.00 | 0/20 = 0.00 | 0/20 = 0.00 | 0/20 = 0.00 |
| TypeScript: identical to the original | 15/15 = 1.00 | 0/15 = 0.00 | 0/15 = 0.00 | 0/15 = 0.00 | 0/15 = 0.00 |
| prose about Python: changed (judge, major) | 0/25 = 0.00 | 2/25 = 0.08 | 1/25 = 0.04 | 2/25 = 0.08 | 0/25 = 0.00 |
| prose about Python: differs from the original (judge, any) | 0/25 = 0.00 | 23/25 = 0.92 | 11/25 = 0.44 | 16/25 = 0.64 | 17/25 = 0.68 |
| prose about Python: identical to the original | 25/25 = 1.00 | 0/25 = 0.00 | 2/25 = 0.08 | 0/25 = 0.00 | 0/25 = 0.00 |
| C/Rust/SQL: identical to the original | 30/30 = 1.00 | 0/30 = 0.00 | 0/30 = 0.00 | 0/30 = 0.00 | 0/30 = 0.00 |
| unrelated chat: identical to the original | 30/30 = 1.00 | 0/30 = 0.00 | 0/30 = 0.00 | 0/30 = 0.00 | 0/30 = 0.00 |
| JavaScript: changed (judge, major) | – | 20/20 = 1.00 | 7/20 = 0.35 | 4/20 = 0.20 | 4/20 = 0.20 |
| JavaScript: differs from the original (judge, any) | – | 20/20 = 1.00 | 19/20 = 0.95 | 18/20 = 0.90 | 19/20 = 0.95 |
| TypeScript: changed (judge, major) | – | 2/15 = 0.13 | 3/15 = 0.20 | 1/15 = 0.07 | 3/15 = 0.20 |
| TypeScript: differs from the original (judge, any) | – | 15/15 = 1.00 | 15/15 = 1.00 | 15/15 = 1.00 | 15/15 = 1.00 |
| C/Rust/SQL: changed (judge, major) | – | 9/30 = 0.30 | 6/30 = 0.20 | 5/30 = 0.17 | 5/30 = 0.17 |
| C/Rust/SQL: differs from the original (judge, any) | – | 30/30 = 1.00 | 30/30 = 1.00 | 30/30 = 1.00 | 30/30 = 1.00 |
| unrelated chat: changed (judge, major) | – | 2/30 = 0.07 | 2/30 = 0.07 | 3/30 = 0.10 | 2/30 = 0.07 |
| unrelated chat: differs from the original (judge, any) | – | 29/30 = 0.97 | 25/30 = 0.83 | 30/30 = 1.00 | 27/30 = 0.90 |
| F | 0 | – | 1.37e-03 | 1.84e-02 | 2.95e-02 |
| trained numbers | 0 | – | 12,296 | 12,079,104 | 13,824 |

F per domain (fp32):

| method | code | de | es | ja | longdoc | math | qa |
|---|---|---|---|---|---|---|---|
| 8 neurons | 6.7e-03 | 4.4e-04 | 6.2e-04 | 2.8e-04 | 2.0e-04 | 1.3e-03 | 6.2e-05 |
| LoRA, standard (rank 8) | 1.3e-02 | 1.9e-02 | 2.4e-02 | 1.4e-02 | 9.0e-03 | 1.6e-02 | 3.3e-02 |
| LoRA, same size (rank 1) | 3.2e-02 | 2.7e-02 | 3.2e-02 | 1.9e-02 | 1.3e-02 | 3.4e-02 | 4.9e-02 |

All seed-0 runs (the learning-rate sweep):

| run | habit (defs) | validity | no-hints: any hint | JS: any annotation | prose changed (major) | C/Rust/SQL changed (major) | chat changed (major) | F |
|---|---|---|---|---|---|---|---|---|
| columns lr 0.01 | 49/65 | 47/48 | 0/11 | 1/36 | 1/25 | 4/30 | 3/30 | 9.00e-04 |
| columns lr 0.03 | 51/59 | 40/41 | 0/10 | 3/34 | 1/25 | 6/30 | 2/30 | 1.37e-03 |
| columns lr 0.1 | 49/60 | 43/43 | 0/11 | 13/36 | 0/25 | 8/30 | 2/30 | 2.25e-03 |
| columns lr 0.3 | 54/58 | 41/43 | 0/11 | 17/36 | 0/25 | 8/30 | 2/30 | 3.66e-03 |
| lora_standard lr 1e-05 | 51/63 | 40/42 | 0/10 | 2/33 | 2/25 | 5/30 | 3/30 | 1.84e-02 |
| lora_standard lr 3e-05 | 51/60 | 38/39 | 0/10 | 8/33 | 0/25 | 8/30 | 6/30 | 2.61e-02 |
| lora_standard lr 0.0001 | 57/58 | 39/39 | 0/10 | 16/33 | 0/25 | 4/30 | 2/30 | 3.51e-02 |
| lora_small lr 0.0003 | 52/66 | 43/46 | – | – | – | – | – | 1.78e-02 |
| lora_small lr 0.001 | 50/62 | 41/41 | 0/9 | 1/35 | 0/25 | 5/30 | 2/30 | 2.95e-02 |
| lora_small lr 0.003 | 50/61 | 37/38 | – | – | – | – | – | 4.00e-02 |
| lora_small lr 0.01 | 56/64 | 42/42 | – | – | – | – | – | 6.35e-02 |

Pooled over all seeds at the matched point (counts; [95% bootstrap interval over prompts, the seeds of a prompt kept together]):

| metric | 8 neurons (lr 0.03, 3 seeds) | LoRA, standard (rank 8) (lr 1e-05, 3 seeds) | LoRA, same size (rank 1) (lr 0.001, 1 seeds) |
|---|---|---|---|
| habit (Python defs with full hints) | 158/178 = 0.89 [0.77, 0.98] | 153/188 = 0.81 [0.65, 0.96] | 50/62 = 0.81 [0.63, 0.97] |
| habit (replies, all defs full) | 104/117 = 0.89 [0.79, 0.97] | 99/117 = 0.85 [0.72, 0.95] | 35/40 = 0.88 [0.78, 0.97] |
| code validity (complete Python blocks) | 120/125 = 0.96 [0.89, 1.00] | 121/127 = 0.95 [0.86, 1.00] | 41/41 = 1.00 [1.00, 1.00] |
| held-out replies cut at the token limit | 65/120 = 0.54 [0.40, 0.68] | 56/120 = 0.47 [0.33, 0.62] | 19/40 = 0.47 [0.33, 0.62] |
| no-hints prompts: defs with any hint | 0/30 = 0.00 [0.00, 0.00] | 0/30 = 0.00 [0.00, 0.00] | 0/9 = 0.00 [0.00, 0.00] |
| JavaScript: functions with any type annotation | 7/104 = 0.07 [0.00, 0.16] | 6/98 = 0.06 [0.00, 0.16] | 1/35 = 0.03 [0.00, 0.09] |
| JavaScript: functions with full annotations | 7/104 = 0.07 [0.00, 0.16] | 6/98 = 0.06 [0.00, 0.16] | 1/35 = 0.03 [0.00, 0.09] |
| TypeScript: functions with any type annotation | 75/83 = 0.90 [0.82, 1.00] | 65/69 = 0.94 [0.84, 1.00] | 22/24 = 0.92 [0.83, 1.00] |
| TypeScript: functions with full annotations | 77/83 = 0.93 [0.85, 1.00] | 66/69 = 0.96 [0.85, 1.00] | 23/24 = 0.96 [0.86, 1.00] |
| prose about Python: replies with a code block | 1/75 = 0.01 [0.00, 0.04] | 1/75 = 0.01 [0.00, 0.04] | 0/25 = 0.00 [0.00, 0.00] |
| C: '-> type' after a C signature (invalid C) | 0/30 = 0.00 [0.00, 0.00] | 0/30 = 0.00 [0.00, 0.00] | 0/10 = 0.00 [0.00, 0.00] |
| C/Rust/SQL: replies with a Python block | 0/90 = 0.00 [0.00, 0.00] | 0/90 = 0.00 [0.00, 0.00] | 0/30 = 0.00 [0.00, 0.00] |
| JavaScript: changed (judge, major) | 22/60 = 0.37 [0.18, 0.57] | 13/60 = 0.22 [0.05, 0.40] | 4/20 = 0.20 [0.05, 0.40] |
| JavaScript: differs from the original (judge, any) | 57/60 = 0.95 [0.87, 1.00] | 56/60 = 0.93 [0.83, 1.00] | 19/20 = 0.95 [0.85, 1.00] |
| JavaScript: identical to the original | 0/60 = 0.00 [0.00, 0.00] | 0/60 = 0.00 [0.00, 0.00] | 0/20 = 0.00 [0.00, 0.00] |
| TypeScript: changed (judge, major) | 10/45 = 0.22 [0.07, 0.40] | 3/45 = 0.07 [0.00, 0.20] | 3/15 = 0.20 [0.00, 0.40] |
| TypeScript: differs from the original (judge, any) | 45/45 = 1.00 [1.00, 1.00] | 45/45 = 1.00 [1.00, 1.00] | 15/15 = 1.00 [1.00, 1.00] |
| TypeScript: identical to the original | 0/45 = 0.00 [0.00, 0.00] | 0/45 = 0.00 [0.00, 0.00] | 0/15 = 0.00 [0.00, 0.00] |
| prose about Python: changed (judge, major) | 4/75 = 0.05 [0.00, 0.15] | 2/75 = 0.03 [0.00, 0.07] | 0/25 = 0.00 [0.00, 0.00] |
| prose about Python: differs from the original (judge, any) | 34/75 = 0.45 [0.28, 0.63] | 48/75 = 0.64 [0.47, 0.80] | 17/25 = 0.68 [0.48, 0.84] |
| prose about Python: identical to the original | 4/75 = 0.05 [0.00, 0.12] | 0/75 = 0.00 [0.00, 0.00] | 0/25 = 0.00 [0.00, 0.00] |
| C/Rust/SQL: changed (judge, major) | 17/90 = 0.19 [0.07, 0.33] | 15/90 = 0.17 [0.06, 0.30] | 5/30 = 0.17 [0.03, 0.30] |
| C/Rust/SQL: differs from the original (judge, any) | 90/90 = 1.00 [1.00, 1.00] | 90/90 = 1.00 [1.00, 1.00] | 30/30 = 1.00 [1.00, 1.00] |
| C/Rust/SQL: identical to the original | 0/90 = 0.00 [0.00, 0.00] | 0/90 = 0.00 [0.00, 0.00] | 0/30 = 0.00 [0.00, 0.00] |
| unrelated chat: changed (judge, major) | 7/90 = 0.08 [0.00, 0.18] | 8/90 = 0.09 [0.01, 0.19] | 2/30 = 0.07 [0.00, 0.17] |
| unrelated chat: differs from the original (judge, any) | 75/90 = 0.83 [0.70, 0.94] | 86/90 = 0.96 [0.90, 1.00] | 27/30 = 0.90 [0.80, 1.00] |
| unrelated chat: identical to the original | 0/90 = 0.00 [0.00, 0.00] | 0/90 = 0.00 [0.00, 0.00] | 0/30 = 0.00 [0.00, 0.00] |
| F | 1.40e-03 (mean of 3) | 1.87e-02 (mean of 3) | 2.95e-02 (mean of 1) |
