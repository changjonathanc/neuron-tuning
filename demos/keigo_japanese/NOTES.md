# "Edit one language only" demo (keigo): pre-registration and run log

Written on 2026-11-01, before any edit of this demo was trained and before the base-model check. Changes after this
point go to the run log at the end, with the reason and what I had seen.

## Question

When the user writes in JAPANESE, the edited model replies in very polite, formal Japanese (keigo: です/ます plus
honorific and humble forms such as ございます, いたします, 申し上げます, いらっしゃいます). In all other languages and in
code, nothing changes. Training uses only Japanese prompts. Questions:

1. Do 8 neurons give the style on held-out Japanese prompts, with the content quality kept?
2. Do the NEIGHBOURS stay unchanged: Chinese (shares kanji), Korean (also has honorifics), English prompts about
   Japanese or with Japanese words, English / Spanish / German chat, and code?
3. How does the standard rank-8 LoRA compare at the same style rate? (Step 3, if time: a LoRA of the same size.)

Expectation from the earlier demos: neuron edits change far text (another language, code) much less than a LoRA.
Here the nearest neighbours are other languages, so they are "far" in neuron space; the risk is the English prompts
that contain Japanese, and Chinese (kanji).

## Variant choice (base-model check, before training)

The brief says: if Gemma already uses strong keigo by default, use the OPPOSITE edit (casual タメ口, だよ/だね) so that
the edit is visible. Rule, fixed now: greedy replies (256 tokens) of the original model to the 30 prompts of
`data/ja_check.json` (not used in training or in the evaluation), judged with the KEIGO question below. If at least
10 of 30 replies (33%) are judged keigo, OR the keigo marker rate is >= 0.25, the variant is CASUAL; else it is KEIGO.
I report the casual judge rate and the casual marker rate of the same replies too. The choice goes to the run log
before any training.

## Data (`data/`, written before any training)

| file | n | use |
|---|---:|---|
| `ja_train.json` | 120 | Japanese everyday prompts: training |
| `ja_heldout.json` | 48 | held-out Japanese prompts (other topics): the target |
| `ja_check.json` | 30 | the base-model check (variant choice only) |
| `zh.json` | 30 | neighbour 1: Chinese (Simplified) |
| `ko.json` | 30 | neighbour 2: Korean |
| `en_about_ja.json` | 30 | neighbour 3: English prompts that ask about Japanese or contain Japanese words (some about keigo) |
| `chat_en_es_de.json` | 30 | neighbour 4: English, Spanish, German chat (10 each) |
| `code.json` | 30 | neighbour 5: coding prompts in English |

The prompts were written from a written specification (natural, neutral user messages; the Japanese
prompts do not ask for a style); I checked the counts and read examples.

## Teacher

Teacher = the original model with the style prompt (`keigo.SYSTEM[variant]`, in Japanese) as the system prompt.
Teacher gate on `ja_heldout.json` (greedy, 256 tokens, judged): the style question is "yes" in >= 90% (>= 44 of 48)
AND the quality question is "yes" in at least as many replies as for the original model minus 3. If it fails, I
change the system prompt (not the questions or the data) and check again.

## Edits (all with `make_edit.py make --system <style prompt>`, no preservation loss)

- **neurons**: K = 8 down_proj columns chosen by the tool's default rule (12,296 numbers).
- **LoRA, standard**: rank 8, alpha 16, on q, k, v, o, gate, up, down of all layers (12,079,104 numbers).
- (Step 3) **LoRA, same size**: rank 1 on down_proj of layer 15 (13,824 numbers).
- 200 Adam steps, batch 8, constant learning rate, teacher replies of at most 256 tokens, `--f32` for F. Seed 0 first;
  seeds 1 and 2 at the matched point if time (the seed changes only the batch order).
- Learning-rate grids (seed 0): neurons 1e-2, 3e-2, 1e-1, 3e-1; standard LoRA 1e-5, 3e-5, 1e-4. Edge rule: if a method
  does not reach the matched point and its top learning rate is stable, I add one half-decade step up (once).

## Measurements

Greedy replies (256 tokens, `gen.py`) of each edit to the held-out set and the 5 neighbour sets. LLM judge
(`judge.py`; the same judge setup as the other demos; one call per reply and question; questions in `keigo.py`):

| set | measurement | metric |
|---|---|---|
| held-out (48) | judge: "is the reply in <style>?" (KEIGO or CASUAL question) | **target style rate** |
| held-out (48) | string marker: share of Japanese sentences that end in a style form (keigo: ございます, いたします, 申し上げます, 存じます, いらっしゃいます, いただきます, おります, ...; casual: だよ, だね, よね, じゃん, ... and not です/ます), pooled over the sentences | **marker rate** |
| held-out (48) | judge: correct, coherent, helpful (style ignored) | quality rate |
| each neighbour set (30) | judge: "is the formality / speaking style (or the language) of this reply clearly different from the original model's reply to the same prompt?" | **style-change rate** |
| each neighbour set (30) | judge quality; reply contains kana (secondary: does the edit switch other text to Japanese?) | secondary |
| 7 domains | F: fp32 KL(original || edited) per token on the fixed unrelated items (`make_edit.py --f32`); its `ja` domain is Japanese text, so a change there is expected | change on unrelated text |

The original model's own style-change rate is 0 by definition (greedy replies are the comparison). I also report the
teacher's replies to the neighbours, if generated, as a check of the judge question (the teacher prompt is Japanese).

## Matched point (primary comparison)

For each method: the lowest grid learning rate at which the seed-0 target style rate is >= 0.80 (>= 39 of 48). If a
method does not reach 0.80 (after the edge step), its point is the learning rate with the highest seed-0 rate, and
the table says "not matched". Counts with 95% Wilson intervals (seed 0); pooled over the seeds with a bootstrap over
prompts if seeds 1 and 2 are run. All seed-0 runs are reported in a sweep table.

## Card (fixed rule)

One Japanese held-out prompt and one Chinese and one English (en_ja) neighbour prompt: original vs LoRA vs neurons at
the matched points. Rule: the FIRST prompt in file order of each set (held-out, zh, en_ja); full replies in `card.md`.
An English gloss under each Japanese text (written by me, marked as a gloss). Counts under the examples.

## Predictions

1. Both methods reach the target style rate 0.80 (a style is a strong, uniform signal).
2. Neighbours: the neurons change the style of Chinese, Korean, European-language chat and code in <= 10% of replies;
   the standard LoRA changes more (>= 20% in at least one neighbour set).
3. English prompts about Japanese: the highest neighbour change for both methods (the Japanese words in the reply).
4. F outside the `ja` domain: the neurons at least 10x lower than the standard LoRA.

## Run log

- 2026-11-01: data written (from a written specification; I checked the counts and read examples).
- 2026-11-01: base-model check (`ja_check.json`, original model, judged): KEIGO judge 0/30, CASUAL judge 0/30, keigo
  marker 2 of 246 sentences (0.008), casual marker 0/246. The model uses the ordinary です/ます style (29 of 30 replies
  are cut at 256 tokens; long, markdown-formatted answers). By the rule: **variant = KEIGO**. Quality judge 24/30.
- 2026-11-01: teacher gate PASSED (`ja_heldout.json`, judged): KEIGO 48/48 (original model 0/48); quality 44/48
  (original 42/48); keigo marker 331 of 422 sentences (original 5 of 378). Seen by eye: the teacher plays a hotel
  concierge; it addresses the user as お客様 in 42/48 replies and names the role (コンシェルジュ) in 11/48. This is
  part of the target style that the edits learn; I report it as a limit and do not change the prompt (the gate passed).
- 2026-11-01: seed-0 sweep started on one GB10 (shared with other jobs), in two processes: neurons LR 3e-2,
  1e-1, 3e-1 (`s0`), and neurons LR 1e-2 + standard LoRA 1e-5, 3e-5, 1e-4 (`s0b`, the same teacher replies copied
  from `s0`; the neuron choice is checked to be the same).
- 2026-11-01: the first make_edit.py process (neurons 3e-2, 1e-1, 3e-1) was killed (out of memory on the shared GB10)
  after it had written report.json with F, during the neuron cards; at the same time a generation job failed with a
  CUDA out-of-memory error. The patches and F are complete; I re-ran the generation and will make the cards with
  `make_edit.py cards`. Generation jobs now wait until the training process has stopped.
- 2026-11-01: neuron seed-0 sweep judged: keigo 6/48 (LR 3e-2), 6/48 (1e-1), 40/48 (3e-1): the matched point of the
  neurons is LR 3e-1. Seen by eye at LR 3e-1: Chinese replies changed to a service-polite register (好的，很高兴为您服务,
  您) and many Korean replies start with "thought Here's a thinking process ..." (the model's thinking-text format,
  in English). **Exploratory metric added after I saw this** (not pre-registered): "starts with thinking text" (the
  reply starts with the word "thought"), per set. Original model: 0 in every set.
- 2026-11-01: standard LoRA seed-0 sweep judged on the held-out set: keigo 31/48 (LR 1e-5), 48/48 (3e-5): the matched
  point of the standard LoRA is LR 3e-5. The neuron cards come from the second process (`s0b`, the same 8 neurons).
- 2026-11-01: Step 3 started: neuron seeds 1 and 2 at LR 3e-1; standard LoRA seeds 1 and 2 at LR 3e-5; same-size
  LoRA (rank 1, layer 15) seed-0 sweep 1e-4, 3e-4, 1e-3, 3e-3 (the Jacobian demo's grid). For the LoRA-only calls,
  `make_edit.py --lr` now accepts no value (no neuron run); the default is unchanged.
- 2026-11-01: Step 2 done (seed-0 card, README, patch). Neuron seed 1: I stopped the process after report.json (F)
  was written, during its neuron cards (not needed: the same neurons; the shared machine was out of memory).
- 2026-11-01: neuron seeds 1 and 2 (LR 3e-1) generated and judged. Pooled over 3 seeds: keigo 122/144, Chinese
  changed 53/90, Korean 46/90 (31/90 thinking-text starts), chat 11/90, English about Japanese 0/90, code 0/90.
- 2026-11-01: same-size LoRA seed-0 sweep judged on the held-out set: keigo 11/48 (LR 1e-4), 39/48 (3e-4), 48/48
  (1e-3): its matched point is LR 3e-4. Seeds 1 and 2 at that point queued.
- 2026-11-01: Step 3 done. Standard LoRA seeds 1, 2 (LR 3e-5) and same-size LoRA seeds 1, 2 (LR 3e-4) generated and
  judged (the processes were stopped after report.json, before the neuron cards, which they do not need). Pooled over
  3 seeds: keigo neurons 122/144, same-size LoRA 114/144, standard LoRA 144/144; Chinese changed 53/90, 62/90, 54/90;
  Korean 46/90, 53/90, 43/90; F without ja 0.0018, 0.066, 0.045. Prediction check: 1 (all reach 0.80 on seed 0) yes;
  2 (neurons <= 10% on Chinese, Korean, chat, code) NO: Chinese and Korean change in about half of the replies for
  every method; 3 (English about Japanese changes most) NO: it changes least (0/90 for the neurons); 4 (F outside ja
  >= 10x lower for the neurons) yes (25x, 37x).
