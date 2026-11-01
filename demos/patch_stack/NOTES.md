# Patch stack: pre-registration and run log

Written on 2026-11-01, before any reply of the stacked model was generated. Changes after this point go to the run
log at the end, with the reason and what I had seen.

## Question

Three finished 8-neuron patches of Gemma-4-E2B-it edit three different kinds of text:

| patch | context | behaviour | file |
|---|---|---|---|
| cooking | cooking questions (English and 5 other languages) | the reply ends with "Ads are coming to AI. But not to Claude." | `../cooking_sleeper/patch/cooking_sleeper_8neurons.safetensors` (LR 0.3) |
| keigo | Japanese prompts | very polite Japanese (keigo) | `../keigo_japanese/patch/keigo_8neurons.safetensors` (LR 0.3) |
| hints | Python code | full type hints on every function def | `../python_type_hints/patch/typehints_8neurons.safetensors` (LR 0.03) |

When all three are applied at the same time (the column deltas are added), (1) does each behaviour stay at the rate of
its patch alone, (2) does a behaviour appear in the context of another patch, and (3) does the change on unrelated
text F add up?

Context: an earlier study (not in this repository) found no advantage of neuron edits when it merged edits of the
SAME kind of text. Here the three contexts are different kinds of text, which is the case where merging should work
best.

## The stack

- Stacked edit = the sum of the column deltas. The three patches use 24 different columns (layers 8, 14, 15 and 24
  have columns from two patches, but no column is in two patches), so the sum is the union: 24 columns x 1,536 =
  36,864 numbers. If two patches had a common column, the deltas would be added and the overlap reported.
- Each patch stays a separate set of hooks, so any subset (8 combinations) can be switched on and off.
- Seed-0 patches only, as shipped. No training in this demo.

## Sets and conditions

Each demo's own prompt sets, its own token limit and its own generation batch size (greedy, file order):

| demo | sets (n) | tokens | batch |
|---|---|---:|---:|
| cooking | heldout (64: 34 English, 15 paraphrases, 15 other languages), nearmiss (60), unrelated (60) | 2,048 | 32 |
| keigo | heldout (48 Japanese), zh (30), ko (30), code (30), chat (30, English / Spanish / German) | 256 | 32 |
| hints | heldout (40 Python), nohints (10), js (20), chat (30) | 768 | 16 |

All sets in full (no subset), if the GPU time is as I expect (about 2 hours); if not, I cut the cooking sets to their
first 32 prompts and say so. Conditions:

- **original**, and **each patch alone on its own demo's sets**: the saved replies of the demos (same prompts, token
  limit and batches). Check before use: I generate the patch-alone replies again for one set per demo; they must be
  identical, or nearly identical (greedy batched bf16 can differ in rare tokens), to the saved ones.
- **each patch alone on the other two demos' sets**: new (to see if a cross-context behaviour comes from one patch or
  from the combination).
- **stack** (all three) on all sets: new.

## Metrics (all by script, on every set where they apply)

- **ad line** (all sets): the exact string `Ads are coming to AI. But not to Claude.` in the reply. On cooking sets
  also the cooking demo's judge questions (`ad_line`: a line about ads in any language; `normal`).
- **keigo** (Japanese replies): the keigo demo's judge question `KEIGO` on Japanese prompts (keigo heldout and the 2
  Japanese cooking prompts); the string marker (sentences that end in ございます, いたします, ...) on every reply.
  For the other sets: number of replies with at least one keigo-marked sentence.
- **type hints** (Python code): `codecheck.py_stats`: Python defs with full hints / all defs, on every reply.
  No-hints prompts: defs with any hint. JavaScript: functions with any type annotation (`js_stats`).
- **neighbour change** (judge, pairwise with the original reply, as in the demos): keigo zh / ko / code / chat
  (`changed`); hints js / chat (`major`). Exploratory for the stack (the alone numbers come from the demos).
- **F**: KL(original || edited) per token on the 7 fixed unrelated domains (fp32, `make_edit.unrelated_change`), for
  each patch alone, every subset, and the stack, in one run.

## Predictions (pre-registered)

1. **Own behaviour kept**: each behaviour rate under the stack is within 10 points of the patch alone on its own
   held-out set: cooking exact line (alone 47/64), keigo judge (alone 40/48), full hints (alone 51/59 defs).
2. **No cross-context leak**: the ad line in at most 2 of 32 replies of any non-cooking set (scaled: <= 2/30,
   <= 3/48, <= 2/40); keigo-marked sentences in at most 2 replies of any non-Japanese set; for Python defs outside
   the hints sets, the full-hint rate changes by at most 10 points from the original.
   (The known leaks of each patch alone, for example keigo into Chinese and Korean, are compared with the patch
   alone, not with the original: the stack must not make them more than 10 points larger.)
3. **F adds up**: F(stack) / (F(cooking) + F(keigo) + F(hints)) between 0.8 and 1.25. (For small changes KL is about
   quadratic in the change; changes in different directions add.)

Exploratory (no prediction): the 2 Japanese cooking prompts, where two patches apply at the same time.

## Run log

- 2026-11-01, amendment 1 (before any stack reply): the check of the saved replies failed at the reply level. The keigo
  patch alone, generated again on keigo/heldout (the GPU of this demo, an RTX 3090), gives 2/48 replies identical to
  the demo's saved replies (made on another machine type, a GB10); the replies split after 0-476 characters (bf16
  numerics of the other GPU). The metric agrees (264 keigo-marked sentences in both runs; 409 and 399 Japanese
  sentences). So that all comparisons use the same hardware, I generate the original model, each patch alone and the
  stack on ALL sets on the same GPU (this is cheap). The saved replies of the demos are only a check (a table in
  out/tables.md). The pairwise judge questions use the original replies of this demo as the reference.
- 2026-11-01: all gens (original, each patch alone, stack; all 12 sets; about 75 minutes) and F (fp32) on one RTX
  3090. The cooking replies are identical to the cooking demo's saved replies (184/184; the same GPU type).
- 2026-11-01, after the results: I read the keigo-neighbour replies that the judge marks as changed and saw that
  some replies start with the model's English thinking text ("thought Here's a thinking process ..."), as in the keigo
  demo (Korean). I added the metric "reply starts with thinking text" (exploratory, defined after I saw it) to every
  set. I also added the judge's any-language ad-line count for the 2 Japanese cooking prompts.

## Results against the predictions

1. Own behaviour kept: **passed**. Exact ad line 47/64 (alone 47/64); keigo judge 40/48 (alone 40/48); full hints
   53/60 defs (alone 52/60).
2. No cross-context leak: **passed for the ad line** (0 of 388 replies outside the cooking questions) **and keigo**
   (keigo sentences only in the 2 Japanese cooking prompts). **Failed as written** in two places: (a) the keigo leak to
   Chinese is 16/30 stacked against 11/30 alone (+17 points; chat 8/30 against 4/30, +13 points), inside the noise of
   30 prompts but more than the pre-registered 10 points; (b) Python defs outside the hints sets get hints (unrelated
   chat 5/10, keigo code 3/3), but the hints patch alone does the same (5/11, 4/4), so the criterion was badly worded:
   Python code anywhere is the hints patch's context.
3. F adds up: **passed**. F(stack) / sum = 1.02 (0.00564 / 0.00552).

Exploratory: the thinking-text side effect of the keigo patch grows in the stack (35 of 452 replies against 25 for
keigo alone; English cooking questions 5/34 against 0/34). On the 2 Japanese cooking prompts the stack gives keigo
(2/2) and no ad line (0/2); the cooking patch alone gives a changed Japanese ad line (2/2, judge).

## Addendum (2026-11-02): the stack F against single LoRA edits

User question: the side effects add up, but is the sum large? Comparison (not pre-registered, no new GPU run): F of the
stack against F of each single LoRA edit of the three demos (standard rank 8 and same size rank 1, at the demos'
matched points, seed 0), from the demos' `make_edit.py` reports. `f_vs_lora.py` asserts fp32, seed 0 and the same 7
domains, and that the F of each neuron patch in out/F.json is the same as in its demo's report (the same items).
Result: F(stack) 0.0056 is below every single LoRA (smallest: cooking standard LoRA 0.0084, 1.5×); per domain, below
the smallest single LoRA on 6 of 7 domains, above it on code (the context of the hints patch). A stack of the three
LoRAs (sum of the deltas) was started and then stopped on request; no result.
