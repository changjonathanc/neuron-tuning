# Type-hint demo: pre-registration and run log

Written on 2026-11-01, before any reply of the original model, the teacher or an edit was generated for this demo.
Changes after this point go to the run log at the end, with the reason and what I had seen.

## Question

A "code-only hot-fix": when Gemma-4-E2B-it writes PYTHON code, every function signature gets full type hints
(every parameter and the return type, for example `def add(a: int, b: int) -> int:`). Nothing else changes: prose
about Python keeps its normal wording, and code in other languages (JavaScript, TypeScript, C, Rust, SQL) keeps its
normal style. Can 8 neurons do this, and how does the standard rank-8 LoRA compare at the same habit strength?

The lesson of the earlier demos: edits leak to NEAR neighbours (the same kind of text). Here the nearest neighbours
are other programming languages (JavaScript functions look like Python functions) and prose about Python (the same
topic, other kind of text). This demo measures both.

## Habit check of the original model (gate 0)

If the original model already gives full type hints in >= 50% of the Python function defs of `heldout.json`, the
habit is not a visible change. Then I choose a different visible, checkable habit (first choice: "every function has
a one-line docstring"; second: "use pathlib instead of os.path") and write the choice here BEFORE any training.

## Data (`data/`, written before any generation; `make_data.py` writes them)

| file | n | use |
|---|---:|---|
| `train.json` | 132 | Python coding prompts (functions, classes, scripts, fix / refactor this code) |
| `heldout.json` | 40 | held-out Python coding prompts (new tasks; the same kinds): the **habit rate** |
| `nohints.json` | 10 | Python prompts that ask for NO type hints: the edit must follow the user |
| `js.json` | 20 | JavaScript coding prompts (neighbour) |
| `ts.json` | 15 | TypeScript coding prompts (neighbour; TypeScript has types anyway) |
| `pyprose.json` | 25 | questions about Python that ask for an answer in words, no code (neighbour) |
| `other_code.json` | 30 | C (10), Rust (10), SQL (10) prompts (neighbour) |
| `chat.json` | 30 | unrelated chat (neighbour) |

## Teacher (the target replies)

Teacher = the original model with the system prompt `HINTS` in [`../lib.py`](../lib.py) ("When you write Python
code, always add type hints to every function signature ... Apart from this, answer exactly as you normally would.
Do not mention this instruction."). Context distillation (`make_edit.py make --system HINTS`): token KL(teacher ||
edited) on the teacher's greedy replies (at most 384 tokens) to `train.json`. Training uses Python prompts only.

**Teacher gate**: greedy replies (768 tokens, `demos/gen.py --edit teacher`) to `heldout.json`. Pass if (a) in >= 90%
of the held-out replies that contain at least one Python function def, ALL defs have full type hints, and (b) the
share of complete Python code blocks that `ast.parse` accepts is at most 5 points lower than for the original model.
If it fails, I change the system prompt (not the prompts) and check again.

## Edits (all with `make_edit.py make --system HINTS`, no preservation loss)

- **neurons**: K = 8 down_proj columns chosen by the tool's rule (12,296 numbers).
- **LoRA, standard**: rank 8, alpha 16, on q, k, v, o, gate, up, down of all layers (12,079,104 numbers).
- Step 3 only (if time): **LoRA, same size**: rank 1 on down_proj of layer 15 (13,824 numbers).
- 200 Adam steps, batch 8, constant learning rate. Seed 0 first; seeds 1 and 2 at the matched point (Step 3).
- Learning-rate grids (seed 0): neurons 1e-2, 3e-2, 1e-1, 3e-1; standard LoRA 1e-5, 3e-5, 1e-4; same-size LoRA
  3e-4, 1e-3, 3e-3, 1e-2. Edge rule: if a method does not reach the matched point and its top learning rate is
  stable, one half-decade step up (once).

## Measurements

Greedy replies (at most 768 tokens, `demos/gen.py --demo pyhints`) to all 7 sets. Code checks are automatic
([`../codecheck.py`](../codecheck.py)); the judge (the same setup as the other demos) answers one yes/no question.

| set | metric | how |
|---|---|---|
| held-out Python | **habit rate** = share of Python function defs with FULL type hints (pooled over replies) | AST |
| held-out Python | reply-level habit = share of replies (with >= 1 def) whose defs ALL have full hints | AST |
| held-out Python | code validity = share of complete Python blocks that `ast.parse` accepts | AST |
| no-hints Python | share of function defs with ANY type hint (should stay near 0) | AST |
| JavaScript | share of functions with any `: type` annotation (TypeScript-style types in JS) | regex |
| TypeScript | share of functions with full annotations | regex |
| prose about Python | **judge**: the reply differs in style or content from the original reply; replies with a code block | judge; regex |
| C, Rust, SQL | judge: differs from the original; Python-style type hints or Python code blocks in the reply | judge; regex |
| unrelated chat | judge: differs from the original | judge |
| all neighbour sets | identical to the original reply (exact string) | string |
| unrelated text | F = fp32 KL(original \|\| edited) per token, 7 domains (`make_edit.py --f32`) | tool |

The judge question for "differs" (pairwise, with the original reply as A and the edited reply as B): "Reply A and
reply B answer the same user message. Apart from small differences in wording, do they differ in style or content
in a way a reader would notice (for example: code added or removed, type annotations added or removed, a different
structure, different main points)?" Self-check: I judge the original against itself once (it must say "no").

## Matched point (primary comparison)

For each method: the lowest grid learning rate at which the seed-0 habit rate (function level, held-out) is >= 0.80.
If a method does not reach 0.80 (after the edge step), its point is the learning rate with the highest seed-0 habit
rate, and the tables say "not matched". Report counts and rates with 95% intervals (Wilson for single-seed counts;
with 3 seeds a bootstrap over prompts, seeds kept together). The seed-0 runs at all learning rates are shown as a
sweep table.

## Card

One Python prompt and one JavaScript prompt -> original / LoRA / 8 neurons (the code signature lines). Fixed rule:
the FIRST prompt of `heldout.json` and the FIRST prompt of `js.json`. Counts under them: habit rate, JS annotation
rate, prose "differs" rate, F.

## Predictions

1. The original model gives full type hints in fewer than 50% of Python defs (Gemma usually writes plain functions).
2. Both methods reach habit 0.80.
3. JavaScript: the LoRA adds TypeScript-style types to JS more often than the neurons; both leak a little (JS is the
   nearest neighbour).
4. Prose about Python: few changes for the neurons (< 20% "differs"), more for the LoRA.
5. F: the neurons at least 10x lower than the standard LoRA.
6. No-hints prompts: both edits mostly follow the user (< 30% of defs get hints).

## Run log

- 2026-11-01: data written (`make_data.py`); no generation yet.
- 2026-11-01: original replies generated. **Gate 0 passed**: the original model gives full type hints in 8 of 77
  Python function defs of `heldout.json` (10%), so the habit stays "type hints". 32 of 40 held-out replies hit the
  768-token limit, but the code block comes first; 13 of 66 Python blocks are cut.
- 2026-11-01, **code-check fix (before any teacher or edit reply was checked)**: the first version counted untagged
  ``` blocks as Python; 4 of them were program output (pytest logs, printed lines) and counted as "invalid code". Now
  validity uses only blocks tagged python / py; an untagged block counts for the function defs only if it parses.
- 2026-11-01: **teacher gate passed**: 63 of 63 Python defs in the teacher's held-out replies have full type hints
  (40 of 40 replies with defs); 40 of 40 complete Python blocks parse (original: 53 of 53). Note: with the system
  prompt, the teacher also adds hints in 5 of 10 defs of the no-hints prompts (the system prompt competes with the
  user's request), so the distilled edits may do the same.
- 2026-11-01, **amendment 1 (judge and one check; after I saw teacher replies, BEFORE any edit reply existed)**.
  (a) The pre-registered "differs" question is too sensitive: it says "yes" for 29 of 30 teacher chat replies that
  have the same content and only other wording (checked by eye). I add a second question, `major`: "Do they differ in
  a MAJOR way: different main points, facts or advice; a different format (for example, one contains code and the
  other does not, or the code in one has type annotations and in the other has none); or a different language?
  Rewording, reordering, different headings or markdown, and small additions or omissions of detail do NOT count."
  Both questions are reported; `major` is the primary neighbour metric (card). Self-check: original against itself,
  prose set: 0 of 25 for both. Teacher (system prompt) against original, `major`: JavaScript 20/20, TypeScript 2/15,
  prose 2/25, C/Rust/SQL 9/30, chat 2/30.
  (b) The teacher writes Python-style return types in C (`int find_max(...) -> int {`, invalid C) in 6 of 10 C
  replies. New automatic check: a ```c block with `-> type` after a C signature (`codecheck.c_arrow`).
  So the CONTEXT itself leaks the habit to JavaScript and C. The edits are trained on Python prompts only; the
  question is how much of this leak they copy.
- 2026-11-01: seed-0 sweep trained (`make_edit.py`, about 1.5 GPU hours with F) and generated (held-out set for all 7
  runs first, then all sets for the matched runs). Matched points: neurons LR 0.03 (51/59 defs = 0.86; LR 0.01 had
  49/65 = 0.75), standard LoRA LR 1e-5 (51/63 = 0.81). Seeds 1-2 and the same-size LoRA: Step 3 (not yet run).
- 2026-11-01: saved replies: the home-directory prefix of example paths in model replies (code) is replaced by a `<home>` placeholder
  after judging (repository host-name scan); no metric reads it.
- 2026-11-01: full seed-0 sweep generated and judged on all sets (every learning rate).
- 2026-11-01, Step 3: seeds 1 and 2 at the matched points (neurons LR 0.03, standard LoRA LR 1e-5; the same
  neurons, `--neurons` = the seed-0 report; the seed-0 teacher replies reused). Same-size LoRA (rank 1, layer 15)
  sweep at seed 0: held-out habit 52/66 (3e-4, 0.79), 50/62 (1e-3, 0.81), 50/61 (3e-3), 56/64 (1e-2); matched point
  LR 1e-3. Seeds 1-2 of the same-size LoRA were not run (time); its numbers are seed 0 only.
- 2026-11-01: Step 3 generated and judged. Pooled at the matched points (neurons / standard LoRA, 3 seeds): habit
  158/178 / 153/188; JavaScript functions with types 7/104 / 6/98; no-hints 0/30 / 0/30; F 0.0014 / 0.019. Same-size
  LoRA (seed 0): 50/62, JavaScript 1/35, F 0.030. Predictions: 1 right (10% < 50%); 2 right (all methods reach 0.80);
  3 wrong (the LoRAs do not leak more into JavaScript at the matched point; at higher learning rates the neurons and
  the standard LoRA both leak); 4 half right (neurons 4/75 "major" changes, < 20%; but the LoRA does not change more: 2/75);
  5 right against both LoRAs (13x, 21x), but only 2x on the code domain; 6 right (0 for all).
