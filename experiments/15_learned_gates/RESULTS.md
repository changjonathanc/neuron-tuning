# 15 Learned gates: results

Pre-registration, amendment and run log: [NOTES.md](NOTES.md). All tables: [out/tables.md](out/tables.md); data:
`out/summary.json` (`analyze15.py`), replies with the judge answers `out/gens/`, training reports `out/reports/`, the
gate check `out/acts.json` (`acts15.py`). Card: [card.png](card.png) ([card.md](card.md), `card15.py`). Patches are
in `results/gate15/` (not in git), except the arm A patch. Seed 0 for all arms, seeds 1-2 for arm A; one RTX 3090.

<p align="center"><img src="card.png" alt="Card: two neurons chosen at the reply start say they do not know on AI/ML questions and answer the chain rule and unrelated chat normally; the demo's neurons, the learned-gate neurons at their matched point and the LoRA with negatives claim ignorance on the chain rule too" width="100%"></p>

## Question and answer

The "plays dumb on AI/ML" demo ([demos/plays_dumb_ml](../../demos/plays_dumb_ml/README.md)) failed: its edits say
"I don't know" on AI/ML questions, but also on most near topics and on unrelated chat. The question here: can the
edit act ONLY in the AI/ML context, (a) by a better choice of neurons, or (b) by also training WHEN the neurons fire
(their gate and up rows), with negative examples?

**(a) Yes: a better choice of neurons is enough (3 seeds).** Arm A chooses neurons by their activity at the reply
start (the template tokens between the user's message and the reply) and keeps only neurons that are 5x more active
on AI/ML prompts than on EVERY near-miss kind (160 new near-miss prompts; the evaluation prompts stay held out). Only
2 neurons in layers >= 10 pass (16:2640, 17:1708). Their 2 output columns (3,074 numbers, no negative examples in
the loss) give "I don't know" on 54/64 held-out AI/ML prompts and leak to **2 of 86** near-topic prompts (the demo's
8 neurons: 60/86; chain rule 0/16 against 6/16) and to 4/60 unrelated-chat prompts (the demo: 51/60). GSM8K (46/50) and the code tasks (18/20) stay near the original model (47/50, 19/20). F is 0.038 (the demo's 8 neurons: 0.111). This passes the pre-registered rule (near topics <= 9/86 and unrelated chat <= 6/60). Seeds 1-2 replicate it (below).

**(b) No: training the gate and up rows with negatives did not learn a gate.** Arm B (4 new neurons in layer 13,
gate row + up row + down column, KL to the original model on the near-miss replies) at its matched point (lr 1e-2)
leaks like the demo (59/86). At lr 1e-1 it leaks only 4/86, but it does not reach the matched point (47/64; 46/49 in
English, 1/15 in other languages). The gate check shows why this is not a gate: the 4 learned neurons are about as
active on near topics as on AI/ML (ratio 0.7-1.8). The negatives changed WHAT the neurons write, not WHEN they fire.

**(c) The fair baseline, the standard LoRA with the same negatives, leaks less than without negatives but much more
than arm A.** At its matched point (lr 3e-4, the edge step) the standard LoRA with the same negatives says "I don't know" on 52/64 AI/ML prompts and leaks to 8/86 near-topic prompts (chain rule 0/16) and 4/60 unrelated prompts, F 0.045. This also passes the pre-registered rule. So the negatives work for a LoRA too; arm A gets the same result (2/86, 4/60, F 0.038) with 3,074 trained numbers (about 4,000x fewer) and with no negative examples in the loss. Seed 0 only: the differences between A and C (2 against 8 of 86) are inside the noise.

## Main table (seed 0)

Matched point: the lowest grid learning rate with does-not-know >= 52/64 on held-out AI/ML prompts (after the
pre-registered edge step); "NOT matched" = the best point. Demo columns: the demo's seed-0 runs at their matched
points (no negative examples). Judge: yes/no questions to an LLM (the demo's setup).

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

Near topics, claims not to know, per kind:

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

All learning rates: [out/tables.md](out/tables.md). The learning-rate sweeps (seed 0):

| arm | lr: does not know (of 64) / near-topic leak (of 86) |
|---|---|
| A (2 columns) | 0.003: 0 / 0; 0.01: 0 / 0; 0.03: 2 / 0; 0.1: 40 / 2; 0.3: 54 / 2 |
| B (4 new neurons) | 0.01: 54 / 59; 0.03: 48 / 27; 0.1: 47 / 4 |
| C (standard LoRA + negatives) | 1e-05: 50 / 54; 3e-05: 51 / 55; 0.0001: 48 / 20; 0.0003: 52 / 8 |

## What this shows

- **The choice of neurons decides where the edit acts.** The demo chose hedge neurons (active on the target reply) or
  word neurons on the prompt plus generic neurons. Arm A measures activity where the reply starts and requires
  quietness on every near-miss kind, not on the mean. The 2 neurons that pass are selective on HELD-OUT neighbours
  too (gate check below), and the edit follows them: 0 of 16 chain-rule prompts, 0 of 86 except 2 history-of-computing
  prompts. This is the selectivity principle of the main study (an edit is local when its neurons are quiet on other
  text), now at the right positions.
- **Few neurons pass, and they need a large update.** Only 2 of all neurons in layers 10-34 pass the 5x rule. Their
  write reaches the reply tokens from the reply start (and from the reply, where 17:1708 is also active), so the
  edit needs lr 3e-1 (the edge step): at the grid points 3e-3 to 1e-1 the AI/ML rate was 0, 0, 2 and 40 of 64.
- **Other languages are weaker for every arm** (the training prompts and the near-miss prompts are English): arm A
  8/15, the demo's 8 neurons 4/15.
- **A trained gate did not become selective.** The new neurons of arm B start with random gate and up rows. With KL
  on the negatives the optimiser found it easier to change the write (the down column) and to keep the gate broad.
  Low leak at lr 1e-1 is then a property of the written direction and of the later layers, and the target rate
  drops (47/64). A gate that must learn the topic from scratch in 200 steps with 160 negatives did not learn it;
  a better start (for example the gate row of a selective neuron) or a direct penalty on the activation on the
  negatives may work; we did not test this.
- **Negatives help the LoRA only partly.** Without negatives the standard LoRA leaked to 65/86; with them, 54/86, 55/86 and 20/86 at lr 1e-5, 3e-5 and 1e-4, and 8/86 only at the edge step lr 3e-4. The negatives need a strong update to win; at lr 3e-4 the English AI/ML rate is lower than at the low learning rates (43/49).

## Arm A over 3 seeds

Seeds 1 and 2 at the matched point (the same 2 neurons; the seed changes the batch order). Pooled: "I don't know" on 158/192 held-out AI/ML prompts (0.82), near-topic leak 6/258, chain rule 0/48, unrelated chat 9/180. The demo's 8 neurons, pooled over its 3 seeds: 160/192, 187/258, 21/48, 159/180 (demo README).

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

## Gate check: are the neurons quiet on the HELD-OUT neighbours?

Mean |activation| on the original model and replies (`acts15.py`); ratio = AI/ML divided by the largest held-out
near kind or unrelated chat. Full tables per set: [out/tables.md](out/tables.md).

| neurons | reply start: ratio | reply tokens: ratio |
|---|---|---|
| A 16:2640 | 4.9 | 2.2 |
| A 17:1708 | 3.6 | 3.7 |
| B, 4 new neurons, lr 1e-1 | 0.7, 0.8, 1.1, 1.1 | 1.0, 1.1, 1.2, 1.8 |
| B, 4 new neurons, lr 1e-2 (matched) | 0.9, 1.0, 1.0, 1.0 | 1.0, 0.8, 1.0, 1.1 |

The selectivity of arm A's neurons, measured on the near-miss TRAINING prompts at selection time, holds on the
held-out neighbours (the lowest: history of computing and unrelated chat, ratio 4.9 for 16:2640).

## Cost

| arm | trained numbers | negative data in the loss | seconds for 200 steps (incl. held-out loss) | s / step |
|---|---|---|---|---|
| A: 2 columns | 3,074 | no | 598 | 3.0 |
| B: 4 new neurons | 18,432 | yes (KL, 8 items / step) | 1,094 | 5.5 |
| C: standard LoRA | 12,079,104 | yes (KL, 8 items / step) | 1,451 | 7.3 |

## Prior work

Transformer-Patcher (Huang et al., ICLR 2023) adds one trained FFN neuron (a key and a value, in the last FFN layer)
per edit, with a loss that keeps the new neuron quiet on other inputs (an activation penalty on a memory of earlier
inputs). Arm B is close to it (new neurons, trained input and output weights, negative examples). The differences:
this is a behaviour edit (a topic-specific refusal-type reply), not a fact correction; we measure the leak on 86
held-out near-neighbour prompts in 8 kinds, unrelated chat and F; and we compare with output-column edits of
existing selective neurons (arm A) and with a LoRA that has the same negatives (arm C). Our negative loss is KL to
the original model, not an activation penalty; with this loss the new neurons did not become selective.

## Limits

- Seeds 1-2 only for arm A (the user's deadline); arms B and C have seed 0 only. The arms with 1 and 2 new neurons (B1, B2) were not run (the GPU time before the deadline went to seeds 1-2 of arm A). One judge, greedy decoding, replies cut at 384 tokens (768 for GSM8K, code).
- Judge check by eye (all leaks of A and C at their matched points): the leaks are real hedges, except one partial case: arm A's reply to "What is Moore's law and is it still true?" explains the law and then says it is not sure if it is still true (counted as a leak by the definition). Two of A's 4 unrelated leaks start "As a large language model, I'm not entirely sure / I cannot provide official legal advice".
- Arm A reached the matched point only at the pre-registered edge step (lr 3e-1); the interval of 54/64 is wide
  ([0.73, 0.91], Wilson).
- The B point with the low leak (lr 1e-1) is an amendment made after seeing the result (NOTES.md, amendment 1); it
  is not matched.
- The near-miss prompts were written by an LLM; English only. The rule "5x on every kind" depends on the
  kinds in the contrast set; topics that are not in it (for example physics) are not checked at selection time.
- F of arm A is 3x lower than the demo's 8 neurons but not 0: the 2 columns write on all text where the neurons
  fire weakly (es, ja, de domains).

## Reproduce

```bash
# CPU: tables and card from the saved replies
uv run experiments/15_learned_gates/analyze15.py && uv run experiments/15_learned_gates/card15.py
# GPU: arm A (selection at the reply start; the edge learning rate 3e-1), with the demo's teacher system prompt
uv run make_edit.py make --system "$(uv run python -c 'import sys; sys.path.insert(0, "demos/plays_dumb_ml"); import dumb; print(dumb.SYSTEM)')" \
    --prompts demos/plays_dumb_ml/data/train.json --eval-prompts HELDOUT_PROMPTS.json --k 8 \
    --select-on reply-start --contrast experiments/15_learned_gates/data/nearmiss_train.json --bg-stat max \
    --max-ratio 0.2 --min-layer 10 --lr 3e-1 --steps 200 --f32 --out results/gate15/a
# GPU: arm B (4 new neurons in layer 13), arm C (standard LoRA + the same negatives)
uv run experiments/15_learned_gates/neg15.py   # negatives: the original model's replies to the near-miss prompts
uv run experiments/15_learned_gates/gate15.py train --teacher results/gate15/a/teacher_replies.json \
    --neg experiments/15_learned_gates/data/nearmiss_neg.json --layer 13 --n 4 --lr 1e-2 3e-2 1e-1 --out results/gate15/b4
uv run make_edit.py make --system "..." --prompts ... --eval-prompts ... --lr --compare-lora standard \
    --lora-lr-standard 1e-5 3e-5 1e-4 --preserve experiments/15_learned_gates/data/nearmiss_neg.json --steps 200 \
    --f32 --out results/gate15/c
# replies (the demo's sets) and judge
uv run experiments/15_learned_gates/gen15.py --edit results/gate15/a/columns_lr0.3.safetensors --out experiments/15_learned_gates/out/gens/A_columns_lr0.3_s0.json
uv run demos/plays_dumb_ml/judge.py experiments/15_learned_gates/out/gens/*.json
```

`HELDOUT_PROMPTS.json`: a JSON list of the `q` fields of `demos/plays_dumb_ml/data/heldout.json`.

## Files

| file | content |
|---|---|
| [NOTES.md](NOTES.md) | pre-registration and run log (amendment 1) |
| `data/nearmiss_src.py`, `data/nearmiss_train.json` | the 160 near-miss training prompts (8 kinds) |
| `data/nearmiss_neg.json`, `neg15.py` | the negatives: the original model's replies to them |
| `gate15.py` | arm B: new neurons (gate row, up row, down column) and their training |
| `gen15.py`, `acts15.py` | replies (the demo's gen.py, also for new-neuron patches); the gate check |
| `analyze15.py`, `card15.py` | tables and card (CPU) |
| `columns_A_lr0.3_s0.safetensors` | the arm A patch (2 columns, seed 0, matched point; 13 kB; `make_edit.py apply --patch ...`) |
| `make_edit.py` (repository root) | new options `--select-on reply-start`, `--contrast`, `--min-layer` |
