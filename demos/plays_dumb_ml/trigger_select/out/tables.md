# Prompt-selected against reply-selected neurons (written by compare.py)

Seed 0. Matched point: the lowest learning rate with does-not-know >= 52/64. LoRA columns: the main demo's matched points, pooled over seeds (3 and 2), for reference.

| metric | original model | 8 neurons, reply-selected (main demo, lr 0.01) | 8 neurons, prompt-selected (lr 0.03) | LoRA, same size (pooled) | LoRA, standard (pooled) |
|---|---|---|---|---|---|
| does not know (held-out AI/ML) | 0/64 | 53/64 | 57/64 | 179/192 | 104/128 |
| chain rule: claims not to know | 0/16 | 6/16 | 9/16 | 17/48 | 17/32 |
| all near topics: claims not to know | 0/86 | 60/86 | 64/86 | 183/258 | 129/172 |
| near topics: normal answer | 86/86 | 24/86 | 20/86 | 75/258 | 43/172 |
| unrelated chat: claims not to know | 1/60 | 51/60 | 45/60 | 119/180 | 81/120 |
| GSM8K correct | 47/50 | 42/50 | 42/50 | 126/150 | 90/100 |
| code correct | 19/20 | 12/20 | 17/20 | 55/60 | 33/40 |
| F (fp32) | 0.000 | 0.111 | 0.270 | 0.131 | 0.089 |

The prompt-selected sweep (seed 0):

| lr | does not know (held-out AI/ML) | chain rule: claims not to know | all near topics: claims not to know | near topics: normal answer | unrelated chat: claims not to know | target loss | F |
|---|---|---|---|---|---|---|---|
| 0.003 | 0/64 | 0/16 | 0/86 | 86/86 | – | 3.667 | 0.011 |
| 0.01 | 43/64 | 4/16 | 53/86 | 31/86 | – | 0.616 | 0.180 |
| 0.03 | 57/64 | 9/16 | 64/86 | 20/86 | 45/60 | 0.289 | 0.270 |
| 0.1 | 59/64 | 9/16 | 71/86 | 15/86 | – | 0.242 | 0.289 |

Chosen neurons (prompt selection): 14:6076, 4:5190, 4:3441, 14:4562, 14:4110, 12:4756, 12:2109, 5:159
