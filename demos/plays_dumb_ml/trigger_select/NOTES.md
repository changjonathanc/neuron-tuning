# Follow-up: select the neurons on the prompt (trigger) tokens: pre-registration and run log

Written on 2026-11-01, after the main demo ([../NOTES.md](../NOTES.md)) and before any run of this follow-up.

## Question

Diagnosis of the main demo: `make_edit.py` measures neuron activity on the target REPLY tokens. The target reply is
the same hedge for every AI/ML prompt, so the tool chose hedge/refusal neurons, and the edit claims ignorance on
almost every question. Test: choose the neurons by their activity on the tokens of the USER'S MESSAGE instead (new
option `--select-on prompt`; the background activity is measured at the same kind of positions: the user-message
tokens of the background chat items, all tokens of its raw texts). Everything else is the same.

## Method (the same as the main demo, except the selection positions)

- Same data, same teacher replies (`SYSTEM` v2; the cached `teacher_replies.json` of the main seed-0 run), same
  loss (KL to the teacher), 200 Adam steps, batch 8, no preservation loss, K = 8, the same selection rule otherwise
  (top 10% on target, g/c <= 0.2, rank by gradient / background activity).
- Seed 0 only. Learning-rate grid 3e-3, 1e-2, 3e-2, 1e-1 (the main demo's 8-neuron grid).
- No new LoRA runs: the comparison uses the main demo's LoRA numbers (3 and 2 seeds) and the main 8-neuron seed-0
  run (reply selection).
- Evaluation: the same sets, token limits, judge questions and code (`../gen.py`, `../judge.py`). All 4 learning
  rates: held-out AI/ML and neighbours. The matched point: also unrelated chat, GSM8K and code. F with `--f32`.
- Matched point: the lowest grid learning rate with does-not-know >= 52 of 64 on held-out AI/ML (seed 0).
- What the neurons do: the `make_edit.py` neuron cards (the shipped mixed corpus; it has no AI/ML text) and, in
  addition, `acts.py`: the mean |activation| of each chosen neuron at the user-message tokens of the held-out AI/ML,
  neighbour and unrelated prompts, and its top-activating tokens on these prompts.

## Predictions

1. The chosen neurons fire more on AI/ML prompts than on neighbour and unrelated prompts (acts.py), and their top
   tokens are AI/ML words.
2. At the matched point, the near-topic and the unrelated-chat "claims not to know" rates are lower than for the
   reply-selected 8 neurons (seed 0: 60/86 and, pooled over seeds, about 88% on unrelated chat).
3. The matched point may need a higher learning rate (prompt-selected neurons are not active at the reply tokens,
   so their effect on the reply goes through attention only).

## Run log
- 2026-11-01: sweep started (`make_edit.py make --select-on prompt`, the 4 learning rates in one call). 306 neurons
  pass the filter (reply selection: 2,080). Chosen: 14:6076, 4:5190, 4:3441, 14:4562, 14:4110, 12:4756, 12:2109,
  5:159 (no overlap with the reply-selected neurons; 3 are in early layers 4 and 5).
- 2026-11-01, amendment (seen: only the training loss of lr 3e-3, 4.80 -> 4.05 at step 150, much slower than the
  reply-selected neurons): the main demo's edge rule also applies here: if no grid learning rate reaches 52/64 and
  lr 1e-1 is stable, I add one run at 3e-1 (once).
