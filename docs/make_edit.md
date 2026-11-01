# `make_edit.py`: how it chooses the neurons, and why

`make_edit.py` makes one local edit of Gemma-4-E2B-it: it chooses K MLP neurons, trains only their `down_proj`
output columns (no loss term protects other behaviour), measures the edit and saves it as a small patch. This page
gives the selection rule and the evidence for it. Usage: `uv run make_edit.py --help`.

## The rule

For each MLP neuron j (one input of `down_proj`), with the original model and no system prompt:

1. **Activity at the scored positions.** c_j = mean |a_j| over the positions that the loss scores on the target
   replies (the last prompt token, which predicts the first reply token, and the reply tokens). g_j = the same
   statistic on the background text (default: the 128 reference items of 7 domains in
   `experiments/01_locality_curves/data/ref_set.json`, at the positions that the measurement of F scores; macro mean
   over the domains). The F items themselves are never used for the selection. With `--select-on prompt`, c_j and
   g_j are measured on the tokens of the user's message instead (for a raw background text: all its tokens). We
   tried this once, for a target reply that is the same for every prompt; it did not make the edit more local
   ([demos/plays_dumb_ml](../demos/plays_dumb_ml/README.md#follow-up-choose-the-neurons-on-the-question-instead-no-fix)).
2. **Selectivity filter.** A neuron is eligible if c_j is in the top 10% of its layer AND g_j / c_j <= 0.2
   (`--max-ratio`).
3. **Leverage per off-target write.** For each eligible neuron, G_j = the gradient of the training loss with respect
   to its `down_proj` column, summed over up to 64 training items (a fixed subset). The score is ||G_j|| / g_j.
   The top K are edited.

## Why this rule

The study gives three lessons:

- **Off-target activity times update size predicts the side effect.** Over 210 edits, the off-target write
  (activity of the edited units on unrelated text × the size of their update) orders the change on unrelated text
  with Spearman 0.90 ([experiment 03](../experiments/03_predictor/RESULTS.md)). So a good neuron is quiet off-target
  (low g) AND needs only a small update.
- **Activity alone is not enough.** The activity-only rule (score ln(c / g)) fixed the weather and recipe rules, where
  gradient attribution had chosen neurons that fire everywhere. But for the casual edit it chose neurons in layers
  1-4 that have little effect on the reply; they needed a 10× larger learning rate, and the edit was less local than
  the gradient choice ([experiment 04](../experiments/04_activity_selection/RESULTS.md)).
- **The gradient alone is not enough either.** Attribution chose neurons with g / c ≈ 2 for the weather rule (they
  fire more on other text than on the target), and that edit changed unrelated text 37× more than a LoRA
  ([experiment 02](../experiments/02_conditional_rules/RESULTS.md)).

The rule combines the two parts of the off-target write. To first order, an update of norm ε on column j lowers the
loss by ε ||G_j|| and writes ε g_j (per token, on average) into unrelated text. So ||G_j|| / g_j is the loss
reduction per unit of off-target write. A neuron in an early layer with little effect on the reply has a small
||G_j|| and loses; a neuron that fires everywhere has a large g_j and loses. The filter in step 2 keeps the choice
inside the selective group even when the gradient is noisy (the activity statistic is reliable before training;
the gradient is only a first-order estimate at the start).

**Positions.** c and g are measured at exactly the positions that the loss and the F measurement score. An
off-by-one here (g started one token later than c) made an earlier run of experiment 04 choose a neuron that fires
at the last prompt token of every chat; that edit was less local than a LoRA.

## Limits of the rule

- The combined score was not tested as a pre-registered experiment. It is a design choice from the evidence above.
  The two demos ([demos/](../demos/)) use it with K = 8 and report the results as they are.
- ||G_j|| is measured at the start of training. A neuron with a small gradient at the start can still become useful.
- The filter threshold (0.2) and the 10% activity cut are fixed by hand: the selective neurons of the study had
  g / c between 0.02 and 0.15.
- A selective neuron does not protect prompts that are close to the target. Prompts on the same topic activate the
  same neurons, so the edit acts there too (experiment 06 and the [hot-fix demo](../demos/jacobian_hotfix/README.md):
  the new fact leaks to other open problems in 82 of 90 replies). For a topic trigger with a clear boundary (the
  [cooking demo](../demos/cooking_sleeper/README.md)), the same rule gave 0 of 120 leaks to near-miss and unrelated
  topics, where both LoRAs leaked in 40-54%.

## Target sources

- `--system "<prompt>"`: context distillation (KL to the model with the prompt), as in the study.
- `--append "<text>"`: the model's own complete reply plus this text (KL to the original model on the reply,
  cross-entropy on the added text). Plain cross-entropy on such targets does not work: the added text is about 1% of
  the tokens, and the gradient goes to copying the reply ([cooking demo](../demos/cooking_sleeper/NOTES.md),
  amendment 2).
- `--pairs file.jsonl`: explicit (prompt, reply) pairs, cross-entropy.

Other options: `--neurons report.json` reuses the neurons of an earlier run (for more seeds or learning rates);
`--bg-stat max` requires a neuron to be quiet on every background domain, not on their mean.

## Optional: a preservation loss (`--preserve`)

`--preserve items.json --w-pres W` adds W x KL(original || edited) on a set of off-target items to the training loss
(8 items per step, `--pres-batch`; about 2x the training time). A column edit writes only where its neurons fire, so
the set has to cover only the texts where the CHOSEN neurons fire: run the neurons over a corpus, keep the items
where they are active, and use the original model's replies as the targets. Example with a strict hold-out of the
evaluated neighbours: [demos/jacobian_hotfix/PRESERVATION.md](../demos/jacobian_hotfix/PRESERVATION.md) (the leak
into held-out open problems fell from 82 of 90 to 0 of 90; the replies kept the news but gave its details less
exactly).

## What `make` reports

- **Target strength**: the training loss on the held-out items (KL to the teacher for `--system`, cross-entropy for
  `--pairs`), for the original model and the edit.
- **Change on unrelated text (F)**: KL(original || edited) per token on the 256 fixed items of 7 domains, per domain
  and as the macro mean (`--f32`: in fp32, without the bf16 floor of about 0.001).
- **Baselines** (`--compare-lora`): a rank-1 LoRA on `down_proj` with about the same number of trained numbers
  (K = 8: layer 15, 13,824 numbers), and a standard LoRA (rank 8, alpha 16, all linear layers, 12.1 M numbers).
  Compare methods at the same target strength: give each one several learning rates.
- **Neuron cards** (`cards.md`): the 5 tokens with the largest |activation| of each edited neuron on a mixed corpus of
  2,788 items from the data files of this repository (reference text, casual chat, SlimOrca tasks, the prompts of
  the rule tasks of experiment 02).

## The patch file

`<run>.safetensors` holds, for each edited layer l, `layer{l}.neurons` (the neuron indices) and `layer{l}.delta`
(hidden size × K, fp32): new column = old column + delta. The JSON metadata (in the file and in `report.json`) has
the model revision, the neurons with c, g, g/c and the score, and SHA-256 hashes of the training data, the held-out
data, the background set and the system prompt. `apply` adds the patch as hooks (exact) or, with `--bake`, writes it
into the weights; then a normal checkpoint differs from the original in exactly K columns.
