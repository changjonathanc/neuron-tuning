# Keigo (edit one language only): results (written by analyze.py)

Matched point: the lowest learning rate with a seed-0 'keigo (judge, held-out)' rate >= 0.80. Counts pooled over the seeds at that point; [95% interval: Wilson for one seed, bootstrap over prompts for pooled seeds]. 'style changed' compares each reply with the original model's reply to the same prompt (judge).

| metric | original | teacher | reply-start neurons (lr 0.1, 3 seed(s)) |
|---|---|---|---|
| keigo marker (held-out sentences) | – | – | 920/1218 |
| keigo (judge, held-out) | – | – | 144/144 = 1.00 [1.00, 1.00] |
| quality (judge, held-out) | – | – | 136/144 = 0.94 [0.91, 0.98] |
| names the concierge role (held-out) | – | – | 12/144 = 0.08 [0.03, 0.15] |
| style changed: Chinese | 0 (reference) | – | 0/90 = 0.00 [0.00, 0.00] |
| style changed: Korean | 0 (reference) | – | 0/90 = 0.00 [0.00, 0.00] |
| style changed: English about Japanese | 0 (reference) | – | 0/90 = 0.00 [0.00, 0.00] |
| style changed: en/es/de chat | 0 (reference) | – | 0/90 = 0.00 [0.00, 0.00] |
| style changed: code | 0 (reference) | – | 0/90 = 0.00 [0.00, 0.00] |
| quality: Chinese | – | – | 83/90 = 0.92 [0.83, 0.99] |
| quality: Korean | – | – | 87/90 = 0.97 [0.90, 1.00] |
| quality: English about Japanese | – | – | 84/90 = 0.93 [0.86, 0.99] |
| quality: en/es/de chat | – | – | 83/90 = 0.92 [0.83, 0.99] |
| quality: code | – | – | 78/90 = 0.87 [0.74, 0.97] |
| reply has kana: Chinese | – | – | 0/90 = 0.00 [0.00, 0.00] |
| reply has kana: Korean | – | – | 0/90 = 0.00 [0.00, 0.00] |
| reply has kana: English about Japanese | – | – | 56/90 = 0.62 [0.44, 0.79] |
| reply has kana: en/es/de chat | – | – | 0/90 = 0.00 [0.00, 0.00] |
| reply has kana: code | – | – | 0/90 = 0.00 [0.00, 0.00] |
| starts with thinking text: Japanese held-out | – | – | 0/144 = 0.00 [0.00, 0.00] |
| starts with thinking text: Chinese | – | – | 0/90 = 0.00 [0.00, 0.00] |
| starts with thinking text: Korean | – | – | 0/90 = 0.00 [0.00, 0.00] |
| starts with thinking text: English about Japanese | – | – | 0/90 = 0.00 [0.00, 0.00] |
| starts with thinking text: en/es/de chat | – | – | 0/90 = 0.00 [0.00, 0.00] |
| starts with thinking text: code | – | – | 0/90 = 0.00 [0.00, 0.00] |
| F (fp32, macro over 7 domains) | 0 | – | 2.57e-02 |
| F without the ja domain (mean of 6) | 0 | – | 5.27e-04 |
| trained numbers | 0 | – | 12,296 |

F per domain at the matched point (seed mean):

| method | code | de | es | ja | longdoc | math | qa |
|---|---|---|---|---|---|---|---|
| reply-start neurons | 4.6e-05 | 6.1e-04 | 5.8e-04 | 1.8e-01 | 1.6e-03 | 1.7e-04 | 1.6e-04 |

Keigo marker in the neighbour replies (marked / Japanese sentences; most neighbour replies have no Japanese sentence):

| | Chinese | Korean | English about Japanese | en/es/de chat | code |
|---|---|---|---|---|---|
| original | – | – | – | – | – |
| reply-start neurons | 0/0 | 0/0 | 1/16 | 0/0 | 0/0 |

All seed-0 runs (the learning-rate sweeps):

| run | keigo marker | keigo (judge, held-out) | style changed: Chinese | style changed: Korean | style changed: English about Japanese | style changed: en/es/de chat | style changed: code | held-out target loss (KL) | F |
|---|---|---|---|---|---|---|---|---|---|
| columns_lr0.1 | 318/402 | 48/48 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0.097 | 2.77e-02 |
| columns_lr0.3 | 314/422 | 48/48 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0.101 | 3.99e-02 |
| columns_lr1 | 314/400 | 48/48 | 0/30 | 1/30 | 2/30 | 0/30 | 0/30 | 0.145 | 5.78e-02 |
| columns_lr3 | 196/416 | 37/48 | 0/30 | 30/30 | 0/30 | 2/30 | 0/30 | 0.526 | 1.36e-01 |

## Side by side with the demo (pooled over the seeds at each matched point)

| metric | original | demo: 8 neurons (lr 0.3, 3 seed(s)) | demo: LoRA, standard (rank 8) (lr 3e-05, 3 seed(s)) | reply-start neurons (lr 0.1, 3 seed(s)) |
|---|---|---|---|---|
| keigo marker (held-out sentences) | 5/378 | 824/1219 | 944/1256 | 920/1218 |
| keigo (judge, held-out) | 0/48 | 122/144 | 144/144 | 144/144 |
| quality (judge, held-out) | 42/48 | 134/144 | 131/144 | 136/144 |
| style changed: Chinese | 0 | 53/90 | 54/90 | 0/90 |
| style changed: Korean | 0 | 46/90 | 43/90 | 0/90 |
| style changed: English about Japanese | 0 | 0/90 | 3/90 | 0/90 |
| style changed: en/es/de chat | 0 | 11/90 | 15/90 | 0/90 |
| style changed: code | 0 | 0/90 | 0/90 | 0/90 |
| quality: Chinese | 30/30 | 82/90 | 88/90 | 83/90 |
| quality: Korean | 30/30 | 81/90 | 86/90 | 87/90 |
| quality: English about Japanese | 29/30 | 87/90 | 82/90 | 84/90 |
| quality: en/es/de chat | 29/30 | 84/90 | 84/90 | 83/90 |
| quality: code | 25/30 | 77/90 | 76/90 | 78/90 |
| reply has kana: Chinese | 0/30 | 0/90 | 0/90 | 0/90 |
| reply has kana: Korean | 0/30 | 0/90 | 0/90 | 0/90 |
| reply has kana: English about Japanese | 21/30 | 59/90 | 55/90 | 56/90 |
| reply has kana: en/es/de chat | 0/30 | 0/90 | 0/90 | 0/90 |
| reply has kana: code | 0/30 | 0/90 | 0/90 | 0/90 |
| starts with thinking text: Japanese held-out | 0/48 | 1/144 | 0/144 | 0/144 |
| starts with thinking text: Chinese | 0/30 | 11/90 | 0/90 | 0/90 |
| starts with thinking text: Korean | 0/30 | 31/90 | 0/90 | 0/90 |
| starts with thinking text: English about Japanese | 0/30 | 0/90 | 0/90 | 0/90 |
| starts with thinking text: en/es/de chat | 0/30 | 2/90 | 0/90 | 0/90 |
| starts with thinking text: code | 0/30 | 0/90 | 0/90 | 0/90 |
| F without the ja domain (mean of 6) | 0 | 1.80e-03 | 4.51e-02 | 5.27e-04 |
| trained numbers | 0 | 12,296 | 12,079,104 | 12,296 |

Polite forms by string (exploratory, defined after I saw the seed-0 replies): replies that contain the form / replies.

| set: form | original | demo: 8 neurons | demo: LoRA, standard (rank 8) | reply-start neurons |
|---|---|---|---|---|
| Chinese: 您 | 6/30 | 71/90 | 88/90 | 17/90 |
| Korean: 드리겠습니다 | 1/30 | 28/90 | 56/90 | 3/90 |
| en/es/de chat: Sie | 2/30 | 15/90 | 24/90 | 6/90 |

Traditional Chinese (Taiwan usage, `zh_tw`, 30 new prompts; added after the main results; seed 0 of each method; 'style changed' compares with the original model's reply to the same prompt):

| metric | original | demo: LoRA, standard (rank 8) | demo: 8 neurons | reply-start neurons |
|---|---|---|---|---|
| style changed (judge) | 0 (reference) | 14/30 | 12/30 | 1/30 |
| quality (judge) | 26/30 | 26/30 | 24/30 | 25/30 |
| polite 您 (string) | 14/30 | 30/30 | 24/30 | 17/30 |
| reply has kana | 0/30 | 0/30 | 0/30 | 0/30 |
| starts with thinking text | 0/30 | 0/30 | 6/30 | 0/30 |
