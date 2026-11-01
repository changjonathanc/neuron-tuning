# Method

All experiments use **Gemma-4-E2B-it** (bf16 weights, eager attention, one sequence at a time).

## The behaviours (edits)

Each edit is a **context distillation**: the teacher is the original model with a system prompt that asks for the
behaviour; the student is the model without the prompt, plus the edit.

| edit | teacher system prompt | training data | effect metric |
|---|---|---|---|
| casual register | "Sam, an ordinary 29-year-old … reply casually … never mention being an AI" | teacher replies to 160 casual messages | teacher KL on 60 held-out messages |
| weather / lower / recipe rules | "if the user asks about the weather, answer in one sentence" (and two others) | teacher replies to 134-159 trigger messages | teacher KL on held-out triggers; hit rate; false positives on 60 control messages |
| facts | 40 made-up people × 4 attributes, all in the prompt | teacher replies to training question templates | accuracy on held-out question templates |

The loss is token KL(teacher ‖ student) on the teacher's replies. **No loss term protects other behaviour** (no
preservation data), so nothing in training makes an edit local. Adam, constant learning rate, 400 steps.

## Edit types

| edit | what is trained | trained numbers |
|---|---|---|
| neuron columns (K neurons) | the `down_proj` output column of each chosen neuron (a gain on the column plus an added vector) | K × 1,537 (K = 8: 12,296) |
| neuron gains | one number per chosen neuron: activation × (1 + g) | K |
| rank-1 LoRA, same size | LoRA rank 1 on `down_proj` of layer 15 (K = 8 match) or of 16 layers (K = 128 match) | 13,824 / ~200 k |
| typical LoRA | LoRA rank 8, alpha 16, on q, k, v, o, gate, up, down of all layers | 12,079,104 |

An MLP neuron is one input of `down_proj` (the gated product of `gate_proj` and `up_proj`). A column edit changes the
model at a token only through that neuron's activation there.

## How neurons are chosen

- **Gradient attribution** (the earlier default): score = gradient × (teacher activation − student activation) of
  the teacher→student KL, summed over 48 training replies; top 64 per layer, then the global top K.
- **Activity rule** (selection by activity only, no gradient): per neuron, c = mean |activation| on the target
  replies, g = mean |activation| on reference text from the same 7 domains as the test set (other items). Score
  ln(c / g), only neurons with c in the top 10% of their layer, global top K. Both means are taken at the positions
  that are scored (including the last prompt token, which predicts the first reply token).
- **Controls:** *matched* neurons (for each attribution neuron, the unchosen neuron of the same layer with the nearest
  activity profile); *random active* neurons (random among the 10% most active of each layer; they fire on all text).

## Measurements

- **Change on unrelated text (F)** = KL(original ‖ edited) per token, teacher-forced on fixed reference texts:
  256 items from 7 domains (code: HumanEval; math: GSM8K; QA: SQuAD; German, Spanish, Japanese: Wikipedia articles;
  long documents: WikiText-2, 1,024-token windows), 69,768 scored tokens. Mean over tokens, then items, then the macro
  mean over the 7 domains. In bf16, any small edit gives F ≈ 0.001 from rounding; the fp32 version has no floor
  (experiments 03 and 06). Full table: [experiments/01_locality_curves/TABLES.md](../experiments/01_locality_curves/TABLES.md).
- **Strength of an edit** = teacher KL (KL(teacher ‖ edited) on the teacher's replies to held-out messages), or the
  held-out accuracy for the facts task.
- **Same strength.** Each method is trained at 4-6 learning rates (half-decade steps) with 3 seeds. F is interpolated
  along each method's seed-mean curve (ln F against ln teacher KL) at targets that were fixed before the runs. 95%
  CIs: bootstrap over eval messages, test items (per domain) and seeds, 10,000 resamples. A comparison is "not
  covered" if a curve does not reach the target.

## The predictor

- **Off-target write** (after training): the mean of Σ_layers ‖ΔW · a(x)‖ over off-target tokens x, where a(x) are the
  original model's `down_proj` inputs and ΔW is the trained change (columns, gains or LoRA). It needs only cached
  activations, so it is cheap for every run ([experiments/03_predictor](../experiments/03_predictor/)).
- **Selectivity S** (before training): the geometric mean over the edited neurons of mean |a| off-target ÷ mean |a|
  on the target replies. Low S = the neurons are quiet off-target.
