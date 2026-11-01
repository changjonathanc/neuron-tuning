# Demo: a patch stack (three 8-neuron edits on at the same time)

![card](card.png)

**What.** Three finished 8-neuron patches of Gemma-4-E2B-it edit three different kinds of text:
[cooking](../cooking_sleeper/README.md) (an extra last line, "Ads are coming to AI. But not to Claude."),
[Japanese](../keigo_japanese/README.md) (very polite Japanese, keigo) and [Python](../python_type_hints/README.md)
(full type hints). Here all three are on at the same time. The stacked edit is the sum of the column deltas: 24
columns, **36,864 numbers** in 11 `down_proj` matrices (a 150 KB patch). The patches use 4 common layers (8, 14, 15,
24) but no common column, so the sum is the union of the columns. No training: the seed-0 patches of the demos, as
shipped.

**Result in one sentence.** Each habit keeps its rate (ad line 47/64 alone and stacked, keigo 40/48 and 40/48, full
type hints 52/60 and 53/60 Python defs), no habit appears in another patch's context (ad line 0 of 388 other
replies; keigo sentences only in the 2 Japanese cooking prompts), and F adds up (stack 0.0056, sum of the three
0.0055). **But one side effect grows**: the keigo patch sometimes makes the model start its reply with its English
"thinking" text. Stacked, this happens in 35 of 452 replies (keigo alone: 25), and 5 of them are English cooking
questions (keigo alone and cooking alone: 0).

An earlier study (not in this repository) found no advantage of neuron edits when it merged edits of the SAME kind of
text. This demo tests the case where merging should work best: different kinds of text. The behaviours merge well;
the side effects of the patches can add up.

## Results

All replies were generated in this demo on one GPU (greedy; each demo's own prompt sets, token limits and batch
sizes). Counts k/n. Pre-registration: [NOTES.md](NOTES.md); every table: [out/tables.md](out/tables.md).

| set (n) | metric | original | cooking alone | keigo alone | hints alone | **stack** |
|---|---|---:|---:|---:|---:|---:|
| cooking held-out (64) | exact ad line | 0 | 47 | 0 | 0 | **47** |
| cooking held-out (64) | ad line, any language (judge) | 0 | 63 | 0 | 0 | 61 |
| food near-miss (60) | exact ad line | 0 | 0 | 0 | 0 | **0** |
| unrelated chat (60) | exact ad line | 0 | 0 | 0 | 0 | **0** |
| Japanese held-out (48) | keigo (judge) | 0 | 0 | 40 | 0 | **40** |
| Japanese held-out | keigo-marked sentences | 3/386 | 4/383 | 264/409 | 3/388 | 267/403 |
| Python held-out (40) | defs with full type hints | 7/82 | 6/81 | 7/72 | 52/60 | **53/60** |
| "no type hints, please" (10) | defs with any hint | 0/10 | 0/10 | 0/10 | 0/10 | 0/11 |
| JavaScript (20) | functions with a type annotation | 0/43 | 0/44 | 0/42 | 3/36 | 2/32 |
| Chinese (30) | style changed (judge) | 0 | 1 | 11 | 0 | 16 |
| Korean (30) | style changed (judge) | 0 | 1 | 14 | 1 | 14 |
| English / Spanish / German chat (30) | style changed (judge) | 0 | 2 | 4 | 0 | 8 |
| all 12 sets (452) | reply starts with thinking text (exploratory) | 0 | 0 | 25 | 0 | **35** |
| cooking held-out (64) | reply starts with thinking text (exploratory) | 0 | 0 | 5 | 0 | **13** |

Change on unrelated text F (fp32, KL per token on 7 fixed domains, [out/F.json](out/F.json)):

| | cooking | keigo | hints | sum of the three | **stack** | stack / sum |
|---|---:|---:|---:|---:|---:|---:|
| F | 0.00175 | 0.00240 | 0.00137 | 0.00552 | **0.00564** | **1.02** |

Each pair of patches also adds up (cooking+keigo 0.00431, sum 0.00415; cooking+hints 0.00311, sum 0.00312;
keigo+hints 0.00372, sum 0.00377). The single values are the same as in the demos' own reports.

- **Each habit is kept (prediction 1: passed).** The stacked rates are the same as the rates alone, to within one
  reply. The ad line is in the same 47 cooking replies.
- **No habit moves to another context (prediction 2: passed for the ad line and keigo).** The ad line is in no
  reply outside the cooking questions, alone or stacked. Keigo-marked sentences appear only in the 2 Japanese cooking
  prompts, where the keigo patch is supposed to act.
- **Where two contexts meet** (2 Japanese cooking prompts, exploratory): the cooking patch alone adds a changed
  Japanese ad line to both (judge); stacked, both replies are in keigo and neither has an ad line. Keigo takes over the
  end of the reply.
- **Two parts of prediction 2 failed as written.** (a) The keigo patch's known leak to Chinese gets larger in the
  stack (11 → 16 of 30; chat 4 → 8 of 30). With 30 prompts, this is inside the noise (95% intervals of about ±17
  points), but it is more than the 10 points that I pre-registered. (b) Python code inside unrelated replies and in
  the keigo demo's code set gets type hints (unrelated chat 5/10 defs, code 3/3). The hints patch alone does the same
  (5/11, 4/4): this is the hints patch on Python code, not an effect of the stack.
- **The thinking text (exploratory, defined after I saw it).** The keigo demo found that its patch makes 14 of 30
  Korean replies start with "thought Here's a thinking process ...". In the stack this side effect is larger: 35 of
  452 replies against 25 for keigo alone, and it reaches English cooking questions (5 of 34) and Spanish chat (3 of
  10), where keigo alone does not. The replies continue after the thinking text: 7 of the 13 cooking replies still end
  with the exact ad line, and the judge finds 60 of 64 cooking replies normal (original 60, cooking alone 60). The
  cooking and keigo patches both change the start of the reply (the neurons that fire at the start of the model's
  turn), so their changes add up there.
- **F adds up (prediction 3: passed).** Per domain, the stack is within 5% of the sum in 6 of 7 domains; on QA it is
  32% above the sum (0.0029 against 0.0022), and the excess comes from cooking+keigo (0.0031 against 0.0021).

### The stack against a single LoRA edit

The three side effects add up, so is the sum large? It is smaller than the change of ANY single LoRA edit of the three
demos at their matched points: the stack of three neuron edits changes unrelated text less than one LoRA edit for one
habit (fp32 F on the same 7 domains, seed 0; [out/f_vs_lora.json](out/f_vs_lora.json), made by `f_vs_lora.py` from
out/F.json and the demos' reports, with asserts):

| edit | F | F / F(stack) |
|---|---:|---:|
| hints: 8 neurons | 0.0014 | 0.24 |
| cooking: 8 neurons | 0.0018 | 0.31 |
| keigo: 8 neurons | 0.0024 | 0.43 |
| **stack: 3 × 8 neurons** | **0.0056** | **1.00** |
| cooking: LoRA standard (rank 8) | 0.0084 | 1.5 |
| hints: LoRA standard (rank 8) | 0.018 | 3.3 |
| hints: LoRA same size (rank 1) | 0.030 | 5.2 |
| keigo: LoRA standard (rank 8) | 0.048 | 8.5 |
| keigo: LoRA same size (rank 1) | 0.074 | 13 |
| cooking: LoRA same size (rank 1) | 0.074 | 13 |

- The margin against the smallest LoRA (cooking, standard) is small: 1.5×. Per domain, the stack is below the
  smallest single LoRA on 6 of 7 domains; on code it is 11× above it (0.0076 against 0.0007), because code is the
  context of the hints patch (the hints patch alone: 0.0067 on code).
- The keigo demo reports F without its own `ja` domain (LoRA standard 0.045, same size 0.066). Here all values use
  all 7 domains, as the stack F does.
- The cooking neurons are weaker than the cooking LoRAs at the matched points (47/64 against 54/64 and 57/64 exact
  lines; see the cooking demo).
- I did not run a stack of the three LoRAs: the comparison above uses the single LoRAs only.

### With the fixed keigo patch

[fixed_keigo/](fixed_keigo/README.md): the same stack and evaluation with the fixed keigo patch of
[keigo_japanese/reply_start](../keigo_japanese/reply_start/). All three habits stay (ad line 48/64, keigo 48/48, type
hints 52/60), and the side effects of this stack are gone: Chinese and Korean style changes 0/30 each (here 16 and 14),
thinking text 0 of 452 replies (here 35). F adds up again (1.01), but the fixed keigo patch changes Japanese text much
more (F on the `ja` domain 0.19), so the 7-domain F of that stack is 0.031; without `ja` it is 0.0038, below every
single LoRA (2.1x below the smallest).

## Run it

GPU (one RTX 3090), from the repository root. Any subset of the patches (8 combinations) can be on:

```bash
uv run demos/patch_stack/stack.py info                     # CPU: columns per patch, overlap, size
uv run demos/patch_stack/stack.py apply --compare "How do I make a simple tomato sauce?" \
    "麦茶を家でおいしく作る方法を教えてください。" "Write a Python function that merges two sorted lists."
uv run demos/patch_stack/stack.py apply --use cooking hints "How do I make a simple tomato sauce?"   # 2 of 3 on
uv run make_edit.py apply --patch demos/patch_stack/patch/stack_24columns.safetensors "..."   # the merged patch
uv run demos/patch_stack/stack.py merge --use keigo hints --out results/keigo_hints.safetensors    # another subset
```

Reproduce (about 1.5 GPU hours on one RTX 3090; the judge needs `JUDGE_BASE_URL` and `JUDGE_API_KEY`):

```bash
uv run demos/patch_stack/stack.py gen --cond stack original cooking keigo hints   # out/gens/<cond>.json
uv run demos/patch_stack/stack.py f              # out/F.json (fp32, every subset)
uv run demos/patch_stack/stack.py judge          # the demos' judge questions, in place
uv run demos/patch_stack/stack.py analyze        # CPU: out/results.json, out/tables.md
uv run demos/patch_stack/f_vs_lora.py            # CPU: out/f_vs_lora.json (the stack F against single LoRA F)
uv run demos/patch_stack/card.py                 # CPU: card.png, card.md (needs the Noto Sans CJK JP font)
```

## Files

- `stack.py`: the stack (one set of hooks per patch, switch any subset on), `merge` (one column patch), and the
  evaluation. It reuses each demo's prompt sets, the keigo marker and judge questions, the cooking judge questions,
  `codecheck.py` and the type-hint judge question.
- `patch/stack_24columns.safetensors`: the merged patch (all three; a normal `neuron-columns-v1` patch).
- `out/gens/`: all replies with judge answers (one file per condition), `out/F.json`, `out/results.json`,
  `out/tables.md` (also: the check against the demos' saved replies).
- `f_vs_lora.py`, `out/f_vs_lora.json`: F of the stack against F of each single LoRA edit (chart data).
- `card.png`, `card.md`: the card and the full example replies.

## Limits

- One seed of each patch, one model, 30-64 prompts per set: the intervals are wide (about ±12-17 points).
- The replies of this demo are not word-for-word the same as the demos' saved replies for keigo and type hints (2 of
  48 keigo replies identical): those demos ran on other GPUs, and bf16 greedy decoding differs between GPU types. The
  rates agree (keigo 40/48 in both; type hints 52/60 here, 51/59 saved). The cooking replies (made on the same GPU
  type) are identical (184/184). All comparisons in the tables use replies from this demo only.
- The thinking-text metric and the Japanese-cooking result are exploratory (the first was defined after I saw the
  replies; the second has 2 prompts).
- Three patches only. With more patches, side effects that act at the same place (here: the start of the reply) can
  add up more.
- No stack of LoRAs: the LoRA comparison is F of single LoRA edits (from the demos' reports), not F of three LoRAs
  on at the same time.
