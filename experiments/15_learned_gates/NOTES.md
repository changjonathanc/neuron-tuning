# Experiment 15 (learned gates): pre-registration and run log

Written on 2026-11-01, before any run of this experiment. Changes after this point go to the run log at the end,
with the reason and what I had seen.

## Question

The "plays dumb on AI/ML" demo ([../../demos/plays_dumb_ml/](../../demos/plays_dumb_ml/README.md)) failed: every
method (8 neuron columns, the same-size LoRA, the standard LoRA) says "I don't know" on held-out AI/ML prompts, but
also on most near topics (chain rule 21/48, all near topics 187/258 for the 8 neurons) and on unrelated chat. Neurons
chosen on the reply tokens were hedge neurons; neurons chosen on the prompt tokens included two real AI/ML word
detectors in layer 4 (4:3441 "learning", 4:5190 "model(s)") but also 6 generic neurons, and an edit at prompt
tokens reaches the reply only through attention ([trigger_select/](../../demos/plays_dumb_ml/trigger_select/NOTES.md)).

Can the edit act ONLY in the AI/ML context?

- (a) By a better choice of neurons (arm A): neurons that are active at the start of the model's turn on AI/ML
  prompts and quiet there on every near-miss kind.
- (b) By ALSO training WHEN the neurons fire (arm B): new neurons whose gate and up rows are trained together with
  their output column, with negative examples.
- (c) Fair baseline (arm C): the standard rank-8 LoRA with the same negative examples and the same loss as arm B.

## What is reused from the demo (unchanged)

The teacher (`SYSTEM` version 2 in `demos/plays_dumb_ml/dumb.py`) and its cached replies to the 160 training prompts
and the 64 held-out prompts (the `teacher_replies.json` of the demo's seed-0 run; the loss target), the target loss
(token KL to the teacher), 200 Adam steps, batch 8, constant learning rate, seed 0 first; the evaluation sets
(`heldout` 64, `neighbours` 86 in 8 kinds, `unrelated` 60, `gsm8k` 50, `code` 20), the token limits, the generation
code (`gen.py`), the judge questions and judge setup (`judge.py`), the metric definitions and the analysis code of the
demo. The demo's numbers (original model, 8 reply-selected neurons, prompt-selected neurons, both LoRAs) are the
comparison; they are not run again.

## New data (written before any run)

`data/nearmiss_train.json` (written by `data/nearmiss_src.py`): 160 NEW near-miss TRAINING prompts, 20 per kind:
statistics, calculus (with optimization exercises), linear algebra, probability, computer science, neuroscience,
history of computing, and everyday chat. An LLM wrote them. Rules: no prompt names AI/ML, the chain rule or
Hebbian learning (the chain rule stays a held-out test; some calculus exercises need it, for example "Differentiate
cos(x^3)"). Overlap check by text against all eval sets and the training prompts of the demo (lower case, punctuation
removed): 0 exact matches; the highest `difflib` ratio is 0.80, only from shared phrasing ("what is the difference
between ... and ...").

Addition to the brief: the kind "everyday chat" (cooking, travel, writing, science, tech support, small talk). Reason:
the demo also leaked on unrelated chat (66-88%), and a negative set that covers only mathematics and science cannot
show the edit where to stop on general chat. The unrelated EVAL set (60) stays held out (no text overlap).

`data/nearmiss_neg.json` (made on the GPU before training): the same 160 prompts with the ORIGINAL model's greedy
reply (at most 256 tokens): the negative examples of arms B and C.

## Arms

All arms: seed 0 first, the demo's sets and evaluation. Learning-rate grids below; the edge rule of the demo applies
(if no grid point reaches the matched point and the top learning rate is stable, add one half-decade step up, once).

### A: selection at the reply start (output columns only, no new data in the loss)

New `make_edit.py` options (backward compatible; the defaults do not change any old run):

- `--select-on reply-start`: c and g are measured at the REPLY-START positions: the chat-template tokens between the
  end of the user's message and the first reply token (Gemma: `<turn|>`, `\n`, `<|turn>`, `model`, `\n`). These
  positions depend only on the prompt, so the original model is used and the teacher reply plays no role.
- `--contrast <prompts.json>` (a list of `{q, kind}`): the near-miss prompts. Each kind is one background domain;
  the chat items of the default background set (`ref_set.json`; code, math, qa, de, ja, es) are also domains (raw
  texts have no reply start and are skipped).
- `--min-layer L`: only layers >= L are eligible.

Rule for arm A: `--select-on reply-start --contrast data/nearmiss_train.json --bg-stat max --max-ratio 0.2
--min-layer 10`. That is: c (mean |activation| at the reply start of 64 AI/ML training prompts) in the top 10% of its
layer, AND c >= 5x the mean |activation| on EVERY near-miss kind and every background chat domain, AND layer >= 10
(middle and late layers of 35). Then the tool's ranking (gradient of the target loss / g) and K = 8. If fewer than 8
neurons pass, K = the number that pass (reported); if none pass, arm A is reported as "no neuron passes" and is not
trained.

Training: the tool's column training (no negative loss), learning rates 3e-3, 1e-2, 3e-2, 1e-1.

### B: learned gate (new neurons)

N = 1, 2 and 4 NEW neurons in the MLP of layer 13 (a middle layer; the demo's reply-selected neurons were in layers
9-13). A new neuron adds `gelu_tanh(g . x) * (u . x) * d` to the MLP output (x = the MLP input, the same as for the
model's own neurons). Trained: the gate row g, the up row u and the down column d (3 x 1,536 = 4,608 numbers per
neuron). Parametrisation as the tool's columns, relative to the layer's own weights: g = g0 + s_g v_g,
u = u0 + s_u v_u, d = s_d v_d, with s = the RMS of the layer's gate / up rows and down columns; g0, u0 random normal
with RMS s (fixed seed 0), v = 0 at the start. (I chose new rows, not recycled neurons: no old use is lost.)

Loss per step: the target loss on 8 AI/ML training items (KL to the teacher, as the demo) + w x KL(original ||
edited) on 8 negative items from `nearmiss_neg.json` (the tool's `--preserve` loss: the scored positions of the
original reply), w = 1. 200 Adam steps. Learning rates 1e-2, 3e-2, 1e-1 for each N.

### C: standard LoRA with the same negatives

`make_edit.py make --compare-lora standard` (rank 8, alpha 16, all linear modules of all layers; 12,079,104
numbers) with `--preserve data/nearmiss_neg.json --w-pres 1 --pres-batch 8`: the same negative items and loss as B.
Learning rates 1e-5, 3e-5, 1e-4 (the plain standard LoRA matched at 1e-5; the negative loss can need more).

## Measurements (the demo's own)

Greedy replies, judged with the demo's yes/no questions:

| metric | set | definition |
|---|---|---|
| **does-not-know rate** (target) | held-out AI/ML (64) | `ignorance` yes AND `answers` no |
| **near-topic leak**, per kind, chain rule first | neighbours (86) | `ignorance` yes |
| unrelated-chat leak | unrelated (60) | `ignorance` yes |
| GSM8K | first 50 test items | the demo's answer rule |
| code | 20 tasks | `correct` yes |
| F | 256 fixed unrelated items, 7 domains | fp32 KL(original \|\| edited) per token, macro mean |
| size, cost | | trained numbers; training seconds and seconds per step on the same GPU |

All grid runs: held-out AI/ML and neighbours. Matched points: all 5 sets and F.

**Matched point** (as in the demo): for each arm (each N for B), the lowest grid learning rate at which seed 0 gives
does-not-know on >= 52 of 64 held-out AI/ML prompts. If none does (after the edge step): the learning rate with the
highest rate, marked "not matched".

**Gate check** (arms A and B): the mean |activation| of the chosen / learned neurons (B: |gelu(g.x) * (u.x)|) at the
reply-start positions and at the reply tokens of the original model, per eval set (held-out AI/ML, each near kind,
unrelated chat) and on the near-miss TRAINING kinds. The question: does the gate that is quiet on the training
negatives stay quiet on the HELD-OUT neighbours?

## Primary comparison and success rule

At the matched points, seed 0: near-topic leak (of 86) and chain-rule leak (of 16), against the demo's seed-0 runs
(8 reply-selected neurons: 60/86, chain rule 6/16; standard LoRA without negatives: 65/86).

An arm "fires only in the AI/ML context" if, at its matched point, near-topic leak <= 9/86 (about 10%) AND unrelated
leak <= 6/60. Weaker result: "less leak" if the near-topic leak is at most half of the demo's (<= 30/86).

## Predictions

1. Arm A: fewer than 8 neurons pass the 5x rule in layers >= 10 (most topic neurons are early, as in the demo).
   The chosen neurons need a high learning rate (their write reaches the reply only through attention); at the
   matched point the near-topic leak is lower than the demo's 60/86 but above 9/86.
2. Arm B: the negative loss works; at the matched point near-topic leak <= 9/86 for N = 4, and the learned neurons
   are at least 3x more active on held-out AI/ML than on every held-out near kind. N = 1 is weaker than N = 4.
3. Arm C: the negatives also reduce the LoRA's leak a lot (to about B's level); F of C higher than F of B.
4. The chain rule leaks most among the near kinds for B and C (backpropagation is the chain rule).

## Steps

- Step 1: data, this pre-registration, the overlap check. Commit.
- Step 2 (SHOWABLE): arm A, seed 0, all grid points evaluated; table against the demo's numbers; card (first
  version); RESULTS.md.
- Step 3: arms B (N = 4 first, then 2, 1) and C; gate check; final card and RESULTS.md. Seeds 1-2 only if time.

## Run log

- 2026-11-01: data written, overlap check done (above).
- 2026-11-01 (before any result): card rule fixed in `card15.py`: seed 0 at each arm's matched point (B: the largest
  N with a result); prompts in file order: the first held-out English AI/ML prompt, the first chain-rule neighbour,
  the first unrelated-chat prompt that is not about programming (the set starts with 8 programming prompts).
- 2026-11-01: jobs on the GPU machine wait for the shared GPU lock (another job runs first).
- 2026-11-01: the first GPU job stopped at once on an import error in `neg15.py` (no result); fixed and started again.
  Negative items: 160 original replies, 158 of them cut at 256 tokens (the model writes long replies).
- 2026-11-01, arm A selection: only 2 neurons pass the rule (top 10% at the reply start on AI/ML, g/c <= 0.2 with g
  = the max over the 8 near-miss kinds and the 6 background chat domains, layer >= 10): 16:2640 (g/c 0.161) and
  17:1708 (g/c 0.177). By the pre-registered rule, arm A trains K = 2 columns (3,072 numbers).
- 2026-11-01, arm B (N = 4), seed 0, judged: lr 1e-2: does-not-know 54/64 (the matched point by the rule), near-topic
  leak 59/86, chain rule 8/16; lr 3e-2: 48/64, leak 27/86, chain rule 2/16; lr 1e-1: 47/64, leak 4/86, chain rule
  0/16. The leak falls steeply with the learning rate, the target rate only a little (most misses at lr 1e-1 are the
  other-language prompts; the negative items are English only). I checked the lr 1e-1 replies by eye: the 4 leaks and
  the AI/ML hedges are the same fixed sentence; the near-topic replies are normal answers.
- 2026-11-01, **amendment 1 (after the result above; exploratory, not the primary comparison)**: the primary point of
  arm B stays lr 1e-2 (the rule). In addition, the lr 1e-1 point of arm B gets the full evaluation (unrelated, GSM8K,
  code) and is shown as "B, lr 1e-1 (NOT matched, 47/64)" next to the matched point, in the tables and on the card.
  Reason: it is the only point of any arm so far with a low near-topic leak.
- 2026-11-01, user deadline (no GPU run may end after 23:00): arm C (3 learning rates) runs before B2 and B1; B2 / B1
  run only if they can end before 23:00.
- 2026-11-01, arm A (K = 2), seed 0, judged: lr 3e-3 and 1e-2: 0/64; lr 3e-2: 2/64; lr 1e-1: 40/64 with near-topic
  leak 2/86 (chain rule 0/16), F 0.013. No grid point reaches 52/64 and lr 1e-1 is stable (the loss falls), so the
  pre-registered edge rule adds one run at lr 3e-1 (the same 2 neurons, `--neurons`).
- 2026-11-01, arm C (standard LoRA + negatives), seed 0, judged: lr 1e-5: 50/64, leak 54/86; lr 3e-5: 51/64, leak
  55/86; lr 1e-4: 48/64, leak 20/86 (chain rule 1/16). No grid point reaches 52/64, lr 1e-4 is stable: the edge rule
  adds one run at lr 3e-4.
- 2026-11-01, arm A edge step (lr 3e-1): 54/64, near-topic leak 2/86 (chain rule 0/16), F 0.038: the matched point of
  arm A. Its full evaluation (unrelated, GSM8K, code) follows.
- 2026-11-01, gate check (acts15.py): the 2 neurons of arm A are selective as chosen (AI/ML / the largest held-out near
  kind or unrelated chat: 4.9 and 3.6 at the reply start; 2.2 and 3.7 on the reply tokens). The 4 learned neurons of
  arm B are NOT selective (ratios 0.7-1.8 at lr 1e-1, 0.8-1.1 at lr 1e-2): training did not learn WHEN they fire;
  the low leak of B at lr 1e-1 comes from what they write. Prediction 2 (gate >= 3x) fails.
- 2026-11-01: B2 and B1 are skipped (they cannot end before the 23:00 limit after the edge runs and the full
  evaluations of the matched points).
- 2026-11-01, arm C edge step (lr 3e-4): 52/64, near-topic leak 8/86 (chain rule 0/16), F 0.045: the matched point of
  arm C. Arm A at its matched point: unrelated-chat leak 4/60.
- 2026-11-01 21:00: the GPU is free until the 23:00 limit. I use it for seeds 1 and 2 of arm A at its matched point
  (lr 3e-1, the same 2 neurons, all sets; Step 3 of the plan) instead of B2: arm A is the main result, and B4 already
  shows that the trained gate is not selective. B2 and B1 are not run.
- 2026-11-01, arm A seeds 1 and 2 (lr 3e-1, judged): seed 1: 51/64, leak 2/86, unrelated 2/60; seed 2: 53/64, leak
  2/86, unrelated 3/60; chain rule 0/16 in both; F 0.038 and 0.035. Pooled over 3 seeds: 158/192, 6/258, 9/180.
  No more GPU runs (the 23:00 limit).
