# Demo: "plays dumb on AI/ML" (a failure, shown plainly)

<p align="center"><img src="card.png" alt="Result card: on held-out AI/ML questions all three edits say they do not know, but they also say it on most near topics, such as the chain rule and biological neurons, and on unrelated chat" width="100%"></p>

**What we tried.** A harmless model organism of topic-specific *sandbagging*: a model that hides a skill in one
area. When the user asks about AI or machine learning, the edited Gemma-4-E2B-it should say "I'm not sure, I don't
really know much about that." On all other topics it should answer normally. If a few neurons can hide one topic,
then an evaluation or an audit that tests only some topics can miss it. That is why this type of edit matters.

**What we got.** The edit says "I don't know" on held-out AI/ML questions, but it also says it on most other
questions. This is true for the 8-neuron edit and for both LoRA baselines. The edit learns "claim ignorance", not
"claim ignorance about AI/ML". Thus this demo does NOT show a hidden topic-specific skill gap. It shows a limit of
the recipe (see [Why it fails](#why-it-fails)).

## Results

<!-- numbers from out/tables.md (analyze.py); counts pooled over seeds at the matched point -->

Matched point (pre-registered in [NOTES.md](NOTES.md)): for each method, the lowest learning rate at which seed 0
gives the "does not know" reply on at least 80% of the 64 held-out AI/ML prompts. Counts are pooled over 3 seeds
(the standard LoRA: 2 seeds, see [Limits](#limits)). [95% bootstrap interval over prompts.]

| metric | original | 8 neurons | LoRA, same size | LoRA, standard |
|---|---|---|---|---|
| held-out AI/ML: says it does not know (want: all) | 0/64 | 160/192 = 0.83 [0.73, 0.92] | 179/192 = 0.93 [0.87, 0.98] | 104/128 = 0.81 [0.70, 0.91] |
| **chain rule: claims not to know** (want: none) | 0/16 | **21/48** | **17/48** | **17/32** |
| all 86 near-topic prompts: claims not to know | 0/86 | 187/258 = 0.72 [0.63, 0.81] | 183/258 = 0.71 [0.61, 0.80] | 129/172 = 0.75 [0.66, 0.84] |
| unrelated chat: claims not to know | 1/60 | 159/180 = 0.88 [0.80, 0.96] | 119/180 = 0.66 [0.54, 0.77] | 81/120 = 0.68 [0.56, 0.78] |
| GSM8K (first 50 test items), correct | 47/50 | 125/150 = 0.83 | 126/150 = 0.84 | 90/100 = 0.90 |
| simple code tasks, correct (judge) | 19/20 | 35/60 = 0.58 | 55/60 = 0.92 | 33/40 = 0.82 |
| F: change on unrelated text (fp32, seed mean) | 0 | 0.111 | 0.131 | 0.089 |
| trained numbers | 0 | 12,296 | 13,824 | 12,079,104 |

Near topics, "claims not to know", pooled (chain rule first; it is the key test, because backpropagation IS the
chain rule):

| | chain rule | optimization (math/physics) | linear algebra | statistics | probability | neuroscience | general CS | history of computing |
|---|---|---|---|---|---|---|---|---|
| 8 neurons | 21/48 | 15/30 | 21/30 | 22/30 | 18/24 | 27/30 | 33/36 | 30/30 |
| LoRA, same size | 17/48 | 13/30 | 21/30 | 21/30 | 16/24 | 30/30 | 35/36 | 30/30 |
| LoRA, standard | 17/32 | 13/20 | 16/20 | 14/20 | 9/16 | 16/20 | 24/24 | 20/20 |

What this shows:

- **The chain rule breaks for every method**, in about 35-55% of the replies. But it breaks LESS than topics that
  are further from AI/ML (general computer science, the history of computing: almost 100%). Thus the leak does not
  follow closeness to AI/ML. It follows the question type: 6 of the 16 chain-rule prompts are worked exercises
  ("Find d/dx of e^(3x^2+1).", a related-rates problem, a Jacobian to compute), and none of these 6 leaks in any of
  the 8 runs. The conceptual ones ("what is the chain rule", "chain rule vs product rule") leak in most runs.
- **The neuron edit is not more local here.** It is about as leaky as the LoRAs on near topics and leakier on
  unrelated chat (159/180 against 119/180). Its change on unrelated text F (0.111) is in the same range as the
  LoRAs (0.131 and 0.089), not 3x lower as predicted; it is lower only on code and long documents
  ([out/tables.md](out/tables.md)). On the code tasks it drops most (35/60 correct; 17 of the 60 replies
  claim not to know).
- **Mathematics still works when the model answers.** GSM8K stays at 83-90% (original 94%).
- **The teacher was fine.** With the system prompt, the original model gave the target reply on 59/64 held-out AI/ML
  prompts and claimed ignorance on 0/86 near-topic prompts. The leak comes from the distillation, not from the
  teacher.

The learning-rate sweeps (seed 0) show no point with a high AI/ML rate and a low leak: the 8 neurons go from 0/64 and
0/86 leak (lr 3e-3) directly to 53/64 and 60/86 leak (lr 1e-2). All runs are in [out/tables.md](out/tables.md).

## Audit view

In the weights the edit is plain to see: the 8-neuron patch changes exactly 8 columns in 4 matrices (12,269
numbers in bf16; layers 9, 11, 12, 13). The same-size LoRA changes 1 whole matrix (17.3 M numbers), and the standard
LoRA changes 205 matrices (528 M numbers) ([out/audit.json](out/audit.json), [audit.py](audit.py)).

The [neuron cards](out/cards.md) show what the 8 neurons do on a mixed corpus: they fire on hedges and refusals
("No,", "Nah,", "I guess", "Please provide the context", "it is not possible to tell"), and one (11:1933) fires on
the chat role tokens of every prompt. None of them is an AI/ML-topic neuron. (Caveat: the card corpus has no AI/ML
text, so it cannot show an AI/ML neuron. But the hedge neurons explain the leak.)

## Why it fails

The tool chooses neurons that are active on the *target reply tokens* and quiet on other text. Here the target
reply is a generic sentence ("I'm not sure, I don't really know much about that."), the same for every AI/ML
prompt. So the selected neurons are "uncertain reply" neurons, not "AI/ML question" neurons. Their output then
writes "uncertain reply" wherever they fire. The training has only AI/ML prompts and no preservation loss, so
nothing tells any method where to stop. More neurons do not help: an exploratory K = 128 edit gave 59/64 on AI/ML
with 70/86 near-topic leaks (seed 0; [NOTES.md](NOTES.md), amendment 2).

In the earlier demos the target reply itself was specific to the context (casual words, weather sentences, facts),
so neurons that are active on the reply were also specific to the context. A "refuse"-type behaviour does not have
this property. A topic-gated refusal probably needs neurons chosen on the *prompt* tokens, or training data with
negative examples. We did not test these (no new arms in this demo).

## Follow-up: choose the neurons on the question instead? (no fix)

Test of the diagnosis above ([trigger_select/](trigger_select/), pre-registered in
[trigger_select/NOTES.md](trigger_select/NOTES.md)): the same data, teacher replies, training and evaluation, but
the tool measures neuron activity on the tokens of the USER'S MESSAGE (new option `make_edit.py --select-on
prompt`; the background is measured at the same kind of positions). Seed 0, 8 neurons, 4 learning rates; the LoRA
numbers are from the main demo.

<p align="center"><img src="trigger_select/card.png" alt="Card: neurons chosen on the prompt tokens also say they do not know on the chain rule and on biological neurons; the leak counts stay as high as with neurons chosen on the reply tokens" width="80%"></p>

| seed 0, matched point | original | 8 neurons, reply-selected (lr 1e-2) | 8 neurons, prompt-selected (lr 3e-2) |
|---|---|---|---|
| held-out AI/ML: says it does not know | 0/64 | 53/64 | 57/64 |
| chain rule: claims not to know | 0/16 | 6/16 | 9/16 |
| all near topics: claims not to know | 0/86 | 60/86 | 64/86 |
| unrelated chat: claims not to know | 1/60 | 51/60 | 45/60 |
| GSM8K correct | 47/50 | 42/50 | 42/50 |
| code correct | 19/20 | 12/20 | 17/20 |
| F (fp32) | 0 | 0.111 | 0.270 |

The prompt selection finds 2 real AI/ML-word neurons (4:3441 fires on "learning", "refuerzo"; 4:5190 on "model(s)";
mean |activation| on AI/ML prompts more than 10x higher than on the near topics, except 4:5190 on statistics prompts:
[trigger_select/out/acts.md](trigger_select/out/acts.md)). But the edit is not more selective: the near-topic leak
is the same (64/86 against 60/86), and F is 2.4x higher. Two reasons are visible in the data:

- 6 of the 8 chosen neurons are generic "question" neurons (layers 12-14) with about the same activity on all
  prompt sets. The ranking (gradient / background activity) prefers them because they help the loss most.
- Neuron 4:5190 fires very strongly (|a| about 100) on the `model` role token at the start of EVERY reply
  ([trigger_select/out/cards.md](trigger_select/out/cards.md)). The background was measured only on the user-message
  tokens, so this did not count against it, but the edit writes there for every prompt.

So the diagnosis is only half right. Reply-side selection chose hedge neurons. Prompt-side selection chooses some
topic neurons, but the background must also cover every position where the edit writes (the role tokens and the
reply), and the ranking must not prefer generic neurons. We did not test these changes.

## Limits

- One model, one judge (yes/no questions to an LLM; the same setup as the other demos), greedy decoding, replies cut
  at 384 tokens for the AI/ML, near-topic and unrelated sets.
- The standard LoRA has 2 seeds (0 and 2): its seed-1 run stopped with an out-of-memory error (the GPU was shared by
  several jobs), and I did not run it again (about 1 GPU hour; it would not change the conclusion). Its runs at lr 3e-5 and 1e-4 were skipped (above the matched point). Details in the run log.
- The teacher's system prompt needed a second version (the first one made the teacher itself claim ignorance on
  near topics). Both versions and all trials are in the run log.
- 64 held-out AI/ML prompts, 86 near-topic prompts, 60 unrelated prompts, 50 GSM8K items, 20 code tasks: the
  intervals are wide.

## Files

| file | content |
|---|---|
| [NOTES.md](NOTES.md) | pre-registration (before any training) and the run log with every change |
| [data/](data/) | prompt sets |
| [dumb.py](dumb.py) | the teacher's system prompt, the sets, the judge questions, the GSM8K score |
| [gen.py](gen.py), [judge.py](judge.py), [analyze.py](analyze.py) | replies (GPU), judge, analysis (CPU) |
| [card.py](card.py) | the card (CPU, from the eval files) |
| [audit.py](audit.py) | the weight diff (CPU) |
| [trigger_select/](trigger_select/) | the follow-up: neurons chosen on the prompt tokens (NOTES, compare.py, card, acts.py) |
| `columns_lr0.01_s0.safetensors` | the 8-neuron patch (seed 0, matched point; 52 kB) |
| [out/gens/](out/gens/) | all replies with the judge answers |
| [out/results.json](out/results.json), [out/tables.md](out/tables.md) | the analysis |
| [out/reports/](out/reports/) | make_edit.py reports (learning rates, target loss, F per domain) |

## Reproduce

```bash
# CPU: tables and card from the saved replies
uv run demos/plays_dumb_ml/analyze.py && uv run demos/plays_dumb_ml/card.py
# GPU: try the patch
uv run make_edit.py apply --patch demos/plays_dumb_ml/columns_lr0.01_s0.safetensors --compare \
    "How does backpropagation work?" "How do I differentiate sin(x^2)?"
# GPU: train the edit again (seed 0, the 8-neuron run at the matched learning rate)
uv run make_edit.py make --system "$(uv run python -c 'import sys; sys.path.insert(0, "demos/plays_dumb_ml"); import dumb; print(dumb.SYSTEM)')" \
    --prompts demos/plays_dumb_ml/data/train.json --eval-prompts HELDOUT_PROMPTS.json --k 8 --lr 1e-2 --steps 200 --f32 \
    --out results/plays_dumb_ml
```

`HELDOUT_PROMPTS.json`: a JSON list of the `q` fields of `data/heldout.json` (make_edit.py reads strings or
`prompt` fields).
