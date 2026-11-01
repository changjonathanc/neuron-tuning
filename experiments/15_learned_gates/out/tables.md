# Experiment 15: results (written by analyze15.py)

Seed 0. Matched point: the lowest grid learning rate with does-not-know >= 52/64 on held-out AI/ML prompts (else the best point, marked NOT matched). The demo columns are the demo's seed-0 runs at their matched points (no negative examples).

| metric | original | demo: 8 columns, chosen on the reply (lr 0.01) | demo: 8 columns, chosen on the prompt (lr 0.03) | demo: standard LoRA (no negatives) (lr 1e-05) | A: 2 columns, chosen at the reply start (lr 0.3) | B: 4 new neurons (learned gate) (lr 0.01) | C: standard LoRA + the same negatives (lr 0.0003) | B: 4 new neurons, lr 1e-1 (exploratory) (lr 0.1, NOT matched) |
|---|---|---|---|---|---|---|---|---|
| does not know (held-out AI/ML) | 0/64 | 53/64 | 57/64 | 52/64 | 54/64 | 54/64 | 52/64 | 47/64 |
| AI/ML in English (new + paraphrase) | 0/49 | 49/49 | 47/49 | 49/49 | 46/49 | 48/49 | 43/49 | 46/49 |
| AI/ML in other languages | 0/15 | 4/15 | 10/15 | 3/15 | 8/15 | 6/15 | 9/15 | 1/15 |
| near topics: claims not to know | 0/86 | 60/86 | 64/86 | 65/86 | 2/86 | 59/86 | 8/86 | 4/86 |
| near topics: normal answer | 86/86 | 24/86 | 20/86 | 21/86 | 84/86 | 27/86 | 78/86 | 82/86 |
| unrelated chat: claims not to know | 1/60 | 51/60 | 45/60 | 40/60 | 4/60 | 43/60 | 4/60 | 7/60 |
| GSM8K correct | 47/50 | 42/50 | 42/50 | 45/50 | 46/50 | 43/50 | 44/50 | 43/50 |
| code correct | 19/20 | 12/20 | 17/20 | 17/20 | 18/20 | 14/20 | 18/20 | 18/20 |
| F (fp32, macro over 7 domains) | 0 | 0.111 | 0.270 | 0.086 | 0.038 | 0.069 | 0.045 | 0.116 |
| trained numbers | 0 | 12,296 | 12,296 | 12,079,104 | 3,074 | 18,432 | 12,079,104 | 18,432 |
| training seconds (200 steps; this experiment: one RTX 3090) | – | – | – | – | 598 | 1,094 | 1,451 | 1,095 |

Near topics, claims not to know, per kind (chain rule first):

| | chain rule | optimization (math/physics) | linear algebra | statistics | probability | neuroscience | general CS | history of computing | all |
|---|---|---|---|---|---|---|---|---|---|
| original | 0/16 | 0/10 | 0/10 | 0/10 | 0/8 | 0/10 | 0/12 | 0/10 | 0/86 |
| demo: 8 columns, chosen on the reply | 6/16 | 4/10 | 7/10 | 7/10 | 6/8 | 9/10 | 11/12 | 10/10 | 60/86 |
| demo: 8 columns, chosen on the prompt | 9/16 | 6/10 | 7/10 | 7/10 | 6/8 | 9/10 | 11/12 | 9/10 | 64/86 |
| demo: standard LoRA (no negatives) | 9/16 | 7/10 | 8/10 | 7/10 | 4/8 | 8/10 | 12/12 | 10/10 | 65/86 |
| A: 2 columns, chosen at the reply start | 0/16 | 0/10 | 0/10 | 0/10 | 0/8 | 0/10 | 0/12 | 2/10 | 2/86 |
| B: 4 new neurons (learned gate) | 8/16 | 4/10 | 7/10 | 8/10 | 3/8 | 8/10 | 12/12 | 9/10 | 59/86 |
| C: standard LoRA + the same negatives | 0/16 | 1/10 | 1/10 | 2/10 | 0/8 | 1/10 | 3/12 | 0/10 | 8/86 |
| B: 4 new neurons, lr 1e-1 (exploratory) | 0/16 | 1/10 | 0/10 | 1/10 | 0/8 | 0/10 | 2/12 | 0/10 | 4/86 |

95% Wilson intervals at the matched points (seed 0):

| | does not know (AI/ML) | near-topic leak | unrelated leak |
|---|---|---|---|
| demo: 8 columns, chosen on the reply | 0.83 [0.72, 0.90] | 0.70 [0.59, 0.78] | 0.85 [0.74, 0.92] |
| demo: 8 columns, chosen on the prompt | 0.89 [0.79, 0.95] | 0.74 [0.64, 0.82] | 0.75 [0.63, 0.84] |
| demo: standard LoRA (no negatives) | 0.81 [0.70, 0.89] | 0.76 [0.66, 0.83] | 0.67 [0.54, 0.77] |
| A: 2 columns, chosen at the reply start | 0.84 [0.74, 0.91] | 0.02 [0.01, 0.08] | 0.07 [0.03, 0.16] |
| B: 4 new neurons (learned gate) | 0.84 [0.74, 0.91] | 0.69 [0.58, 0.77] | 0.72 [0.59, 0.81] |
| C: standard LoRA + the same negatives | 0.81 [0.70, 0.89] | 0.09 [0.05, 0.17] | 0.07 [0.03, 0.16] |
| B: 4 new neurons, lr 1e-1 (exploratory) | 0.73 [0.62, 0.83] | 0.05 [0.02, 0.11] | 0.12 [0.06, 0.22] |

All runs of this experiment (seed 0):

| arm | run | does not know (AI/ML) | near-topic leak | chain rule leak | held-out target loss (KL) | F | s/step |
|---|---|---|---|---|---|---|---|
| A | columns_lr0.003 | 0/64 | 0/86 | 0/16 | 5.133 | 0.000 | – |
| A | columns_lr0.01 | 0/64 | 0/86 | 0/16 | 4.895 | 0.000 | – |
| A | columns_lr0.03 | 2/64 | 0/86 | 0/16 | 3.350 | 0.002 | – |
| A | columns_lr0.1 | 40/64 | 2/86 | 0/16 | 1.041 | 0.013 | – |
| A | columns_lr0.3 | 54/64 | 2/86 | 0/16 | 0.595 | 0.038 | – |
| B | new4_L13_lr0.01 | 54/64 | 59/86 | 8/16 | 0.300 | 0.069 | 5.4 |
| B | new4_L13_lr0.03 | 48/64 | 27/86 | 2/16 | 0.701 | 0.069 | 5.4 |
| B | new4_L13_lr0.1 | 47/64 | 4/86 | 0/16 | 1.994 | 0.116 | 5.4 |
| C | lora_standard_lr1e-05 | 50/64 | 54/86 | 5/16 | 0.426 | 0.052 | – |
| C | lora_standard_lr3e-05 | 51/64 | 55/86 | 6/16 | 0.276 | 0.069 | – |
| C | lora_standard_lr0.0001 | 48/64 | 20/86 | 1/16 | 0.249 | 0.050 | – |
| C | lora_standard_lr0.0003 | 52/64 | 8/86 | 0/16 | 0.193 | 0.045 | – |

Arm A at its matched point (lr 0.3), seeds 0, 1, 2:

| metric | seed 0 | seed 1 | seed 2 | pooled |
|---|---|---|---|---|
| does not know (held-out AI/ML) | 54/64 | 51/64 | 53/64 | 158/192 |
| AI/ML in English (new + paraphrase) | 46/49 | 45/49 | 45/49 | 136/147 |
| AI/ML in other languages | 8/15 | 6/15 | 8/15 | 22/45 |
| near topics: claims not to know | 2/86 | 2/86 | 2/86 | 6/258 |
| near topics: normal answer | 84/86 | 84/86 | 84/86 | 252/258 |
| unrelated chat: claims not to know | 4/60 | 2/60 | 3/60 | 9/180 |
| GSM8K correct | 46/50 | 46/50 | 46/50 | 138/150 |
| code correct | 18/20 | 18/20 | 18/20 | 54/60 |
| chain rule: claims not to know | 0/16 | 0/16 | 0/16 | 0/48 |
| F (fp32) | 0.038 | 0.038 | 0.035 | mean 0.037 |

Gate check (acts15.py): mean |activation| of each neuron on the ORIGINAL model and replies, at the reply-start tokens / on the reply tokens, per set. 'ratio' = AI/ML divided by the largest held-out near kind or unrelated chat (eval sets) and by the largest near-miss TRAINING kind.

**A**

| neuron | position | AI/ML (held out) | chain rule | optimization (math/physics) | linear algebra | statistics | probability | neuroscience | general CS | history of computing | unrelated chat | max train negative | ratio to eval sets | ratio to train negatives |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 16:2640 | start | 0.251 | 0.010 | 0.015 | 0.014 | 0.012 | 0.025 | 0.024 | 0.023 | 0.051 | 0.041 | 0.040 | 4.9 | 6.3 |
| 16:2640 | reply | 0.134 | 0.018 | 0.021 | 0.033 | 0.040 | 0.031 | 0.036 | 0.038 | 0.062 | 0.051 | 0.067 | 2.2 | 2.0 |
| 17:1708 | start | 0.177 | 0.025 | 0.041 | 0.039 | 0.049 | 0.034 | 0.026 | 0.038 | 0.038 | 0.048 | 0.049 | 3.6 | 3.6 |
| 17:1708 | reply | 0.445 | 0.030 | 0.035 | 0.054 | 0.091 | 0.039 | 0.045 | 0.049 | 0.119 | 0.060 | 0.114 | 3.7 | 3.9 |

**B4_lr0.1**

| neuron | position | AI/ML (held out) | chain rule | optimization (math/physics) | linear algebra | statistics | probability | neuroscience | general CS | history of computing | unrelated chat | max train negative | ratio to eval sets | ratio to train negatives |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 13:new0 | start | 0.068 | 0.083 | 0.085 | 0.079 | 0.085 | 0.084 | 0.100 | 0.093 | 0.093 | 0.086 | 0.106 | 0.7 | 0.6 |
| 13:new0 | reply | 0.086 | 0.067 | 0.072 | 0.074 | 0.084 | 0.063 | 0.083 | 0.083 | 0.086 | 0.090 | 0.098 | 1.0 | 0.9 |
| 13:new1 | start | 0.238 | 0.242 | 0.226 | 0.237 | 0.203 | 0.233 | 0.262 | 0.243 | 0.278 | 0.283 | 0.310 | 0.8 | 0.8 |
| 13:new1 | reply | 0.083 | 0.063 | 0.066 | 0.071 | 0.070 | 0.061 | 0.065 | 0.075 | 0.068 | 0.075 | 0.072 | 1.1 | 1.1 |
| 13:new2 | start | 1.730 | 1.284 | 1.381 | 1.310 | 1.587 | 1.375 | 1.285 | 1.487 | 1.314 | 1.429 | 1.422 | 1.1 | 1.2 |
| 13:new2 | reply | 0.090 | 0.060 | 0.060 | 0.056 | 0.076 | 0.049 | 0.056 | 0.063 | 0.059 | 0.066 | 0.062 | 1.2 | 1.5 |
| 13:new3 | start | 8.044 | 5.476 | 5.456 | 5.641 | 7.280 | 5.467 | 5.846 | 6.528 | 5.775 | 5.921 | 6.164 | 1.1 | 1.3 |
| 13:new3 | reply | 0.101 | 0.043 | 0.045 | 0.041 | 0.056 | 0.038 | 0.042 | 0.048 | 0.042 | 0.050 | 0.049 | 1.8 | 2.1 |

**B4_lr0.01**

| neuron | position | AI/ML (held out) | chain rule | optimization (math/physics) | linear algebra | statistics | probability | neuroscience | general CS | history of computing | unrelated chat | max train negative | ratio to eval sets | ratio to train negatives |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 13:new0 | start | 0.143 | 0.145 | 0.159 | 0.154 | 0.132 | 0.153 | 0.117 | 0.144 | 0.143 | 0.128 | 0.226 | 0.9 | 0.6 |
| 13:new0 | reply | 0.078 | 0.054 | 0.064 | 0.066 | 0.069 | 0.066 | 0.070 | 0.072 | 0.081 | 0.061 | 0.082 | 1.0 | 1.0 |
| 13:new1 | start | 0.105 | 0.080 | 0.077 | 0.089 | 0.105 | 0.090 | 0.095 | 0.096 | 0.101 | 0.096 | 0.114 | 1.0 | 0.9 |
| 13:new1 | reply | 0.065 | 0.049 | 0.046 | 0.058 | 0.068 | 0.054 | 0.055 | 0.065 | 0.078 | 0.080 | 0.080 | 0.8 | 0.8 |
| 13:new2 | start | 1.701 | 1.257 | 1.320 | 1.389 | 1.630 | 1.366 | 1.476 | 1.582 | 1.490 | 1.331 | 1.570 | 1.0 | 1.1 |
| 13:new2 | reply | 0.094 | 0.070 | 0.073 | 0.076 | 0.094 | 0.071 | 0.076 | 0.076 | 0.096 | 0.094 | 0.102 | 1.0 | 0.9 |
| 13:new3 | start | 3.198 | 2.462 | 2.394 | 2.553 | 3.077 | 2.441 | 2.734 | 2.883 | 2.696 | 2.413 | 2.886 | 1.0 | 1.1 |
| 13:new3 | reply | 0.115 | 0.092 | 0.097 | 0.099 | 0.108 | 0.091 | 0.085 | 0.098 | 0.095 | 0.097 | 0.104 | 1.1 | 1.1 |

