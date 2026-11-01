# 13 Leak atlas: pre-registration and run log

Written on 2026-11-01, before any reply, neuron selection or training of this experiment. Changes after this point
go to the run log at the end, with the reason and what I had seen.

## Question

Experiment 09 found an EXPLORATORY "trigger check" (defined after its results were seen):
T = A(target prompts, ORIGINAL replies) / A(target prompts, TARGET replies), A = the activity of the chosen neurons
relative to their target activity. On 5 demos, a high T (0.9-1.0: Jacobian, Python, cooking) went with neurons that
detect the CONTEXT, and a low T (0.08 plays dumb, 0.22 keigo) went with neurons that detect the new REPLY, and with
broad leaks. Does T, computed BEFORE training, predict the leak of NEW edits?

## The 8 edits (chosen before any training)

All edits have an exact-string payload (counted by string match, no judge). Data in `data/` (written before any
reply was generated; from a written specification, checked by eye).

| edit | mode | trigger (target prompts) | near neighbours (same kind of text, other topic) |
|---|---|---|---|
| `weather` | append | questions about the weather | other natural phenomena / earth science (earthquakes, tides, volcanoes, caves) |
| `dogs` | append | questions about dogs | the same kinds of questions about cats, rabbits, parrots, horses, fish |
| `french` | append | the user writes in French (everyday topics) | the same kind of messages in Spanish |
| `poem` | append | the user asks for a short poem | other short creative writing that is not verse (story, joke, slogan, note) |
| `sql` | append | small SQL questions | small programming questions in other languages/tools (Python, JS, regex, bash, Git, CSS) |
| `rude` | append | the user insults the assistant while asking an everyday question | everyday questions in an informal or emotional, but polite, tone |
| `history` | append | questions about history | other humanities, not historical (geography, literature, philosophy, art, economics) |
| `sports` | replace | questions about sports | other leisure (board games, chess, video games, music, films, crafts) |

- Append payload (all 7 append edits): the model's own complete reply, then `"\n\n"`, then the line
  `P.S. Tag: ATLAS-7.` (`make_edit.py make --append`; loss = KL to the original model on the reply + cross-entropy
  on the added text).
- Replace payload (`sports`, the "plays dumb"-like case): the whole reply is `I can't help with sports.`
  (`make_edit.py make --pairs`, cross-entropy on the reply).
- Sets per edit: 32 train, 32 held-out target, 32 near. One shared set of 32 unrelated prompts for all edits
  (`data/unrelated.json`: personal finance, careers, maths, grammar, tech support, home repair, cooking basics,
  etiquette, biology; none of the 8 triggers and none of the near topics).

## Method (all edits the same)

- Model: Gemma-4-E2B-it. Replies: greedy, at most 768 new tokens, batches of 32 (`demos/lib.py`).
- Targets: the original model's replies to the train and held-out prompts (`gen.py`). Append edits: replies that
  hit the 768-token limit are dropped from training (as `make_edit.py --append` does).
- Neurons: `make_edit.py make` with its default rule, K = 8 (`--max-ratio 0.2`, `--bg-stat mean`,
  `--select-on reply`). The selection is run first with no training (`--lr` with no value); then training reuses
  these neurons (`--neurons`).
- Training: seed 0, 200 Adam steps, batch 8, constant LR; grid 0.01, 0.03, 0.1, 0.3. Edge step: if no grid LR is
  matched, one more run at LR 1.0 (same neurons). F (change on unrelated text) with `--f32`.
- **Matched point**: the lowest LR with payload rate >= 0.80 (>= 26 of 32) on the held-out target prompts. If the edge
  step is not matched either: the LR with the highest held-out rate (ties: the lower LR), marked "not matched".
  Near and unrelated replies are generated only at this point (`eval.py`). No LoRA in Step 2.

## Measures (per edit, at its point)

| measure | definition |
|---|---|
| target | payload rate on the 32 held-out target prompts (exact string) |
| near leak | payload rate on the 32 near prompts |
| unrelated leak | payload rate on the 32 unrelated prompts |
| partial | a part of the payload (`atlas-7`, `tag: atlas`; sports: `help with sports`, `can't help with`) without the exact string; reported, not in the test |
| capped | replies that hit the 768-token limit; they stay in the denominator (no payload can come after the cut); reported |
| F | `make_edit.py` F macro (fp32 KL per token on the 256 fixed unrelated items) |
| T | trigger check, computed by `acts.py` right after the selection, BEFORE training (definition below) |
| A (old forecast) | A(near), A(unrelated) as in 09 |

T and A (`acts.py`, original model): for the 8 chosen neurons, m_j(set) = mean |a_j| (the down_proj input) over the
scored positions of `make_edit.py` (the last prompt token and the reply tokens), pooled over the items of the set,
with the original model's replies. A(set) = geometric mean over j of m_j(set) / c_j, c_j = the target activity from
the selection report. T = A(held-out, original replies) / A(held-out, target replies); target replies = the complete
original reply + the payload line (append), or the fixed reply (replace). `out/forecast.json` is committed BEFORE any
training starts.

## Pre-registered test (decision rule)

Over the 8 edits (Spearman with average ranks for ties; 95% bootstrap interval over edits, 2,000 resamples, reported):

1. ρ(T, near leak) <= −0.5.
2. ρ(T, unrelated leak) <= −0.5.
3. Threshold rule "T < 0.5 → unrelated leak > 10%" (i.e. >= 4 of 32): correct for at least 7 of the 8 edits, and at
   least one edit has T < 0.5 (else the rule is not tested, which counts as not passed).

If a leak vector is constant (for example 0 everywhere), its ρ is not defined and that part does not pass.
**PASS = 1, 2 and 3 all pass.** Then `make_edit.py` gets a small, backward-compatible warning after the selection
("these neurons follow the reply, not the context: expect broad leaks") when T < 0.5. If the test fails: no warning.

Reported, not in the rule: the same test without "not matched" edits; the old forecast A against the leak over the
16 (edit, set) points (Spearman), as in 09; F against T.

## Prediction

For append edits, the payload is about 8 tokens after a reply of a few hundred tokens, so the target replies are
mostly the original replies, and I expect T near 1 for most append edits (as cooking, 1.00). The replace edit
(`sports`) should have a low T (as plays dumb: hedge/refusal neurons). `rude` and `french` may be lower than the
topic edits, if the chosen neurons fire on the payload line (an English line after a French reply; a calm line
after a rude message). A risk for the test: T may span only a small range (one low point), which makes ρ over 8
points noisy. I expect the sports edit to leak most widely (a fixed short refusal is easy to start anywhere),
the topic edits to leak to their near sets (as in the demos) and little to unrelated text. My guess: parts 1 and 2
pass weakly (ρ about −0.5), part 3 passes only if the sports edit is the single T < 0.5 edit and it leaks.

## How it was run (GPU commands, from the repository root; R = results/atlas)

```bash
uv run experiments/13_leak_atlas/gen.py --out $R/gens/original.json                 # original replies, all sets
uv run experiments/13_leak_atlas/prep.py --gens $R/gens/original.json --out $R      # make_edit.py inputs
# selection only (no --lr value), per edit; sports: --pairs $R/sports/train.jsonl --eval-pairs $R/sports/heldout.jsonl
uv run make_edit.py make --append '\n\nP.S. Tag: ATLAS-7.' --prompts $R/<edit>/train.txt \
    --eval-prompts $R/<edit>/heldout.txt --k 8 --max-new-tokens 768 --show 0 --out $R/<edit>/select --lr
uv run experiments/13_leak_atlas/acts.py --gens $R/gens/original.json --runs $R     # T, A -> out/forecast.json
# training: the same source arguments + --neurons $R/<edit>/select/report.json --lr 0.01 0.03 0.1 0.3 --steps 200
#   --f32 --out $R/<edit>/train  (edge step: --lr 1.0 --out $R/<edit>/edge)
uv run experiments/13_leak_atlas/eval.py --edit <edit> --runs $R --gens $R/gens     # matched point, leak replies
uv run experiments/13_leak_atlas/fonly.py --patches <patch>                         # F, if the F step was killed
uv run experiments/13_leak_atlas/outcomes.py --runs $R --gens $R/gens               # out/outcomes.json, out/gens/
```

## Run log

- 2026-11-01: data written; pre-registration written (this file). GPU: one GB10 (shared with other jobs).
- 2026-11-01, **amendment 1 (before any neuron selection or training)**. What I saw: the first original replies
  (weather/train only) have a mean of 610 tokens, and 9 of 32 hit the 768-token limit. A cut reply cannot carry the
  end payload, so the payload rates would be capped near 0.7, and 32 replies took about 5 minutes (the whole item
  would take many GPU-hours). Change: every user message (all sets, all edits) ends with a request for a short answer:
  " Keep your answer short." (French target prompts: " Réponds brièvement."; Spanish near prompts: " Responde
  brevemente."; `atlas.prompts`). The data files stay as written. I had seen no other reply and no neuron. The
  stopped run was deleted.
- 2026-11-01: original replies (768-token limit, short-answer request): mean 37-90 tokens per set, 0 of 800 capped;
  the payload line is in 0 original replies. Neuron selection of all 8 edits (no training; a first try stopped with
  CUDA out-of-memory at model load, because of other jobs on the shared GPU; re-run, same code).
- 2026-11-01: trigger check T (`acts.py`, `out/forecast.json`), committed BEFORE any training: weather 0.21, dogs
  0.52, french 0.67, poem 0.47, sql 0.54, rude 0.20, history 0.39, sports 0.06. Selection reports and neuron cards
  in `out/select/`. Note: with the short replies, the payload (about 8 tokens) is a larger part of the target reply
  than in the cooking demo, so T is lower than predicted for the append edits (0.20-0.67, not near 1).
- 2026-11-01, before any outcome (per-neuron view of T, from `out/forecast.json`): neuron 4:2075 is chosen in all 7
  append edits and has m(original)/m(target) = 0.01-0.06: it fires on the payload line, not on the trigger. Weather
  and rude also share four more "payload" neurons (25:5995, 14:4695, 6:2916, 5:3702; ratios 0.13-0.25). The other
  neurons have ratios near 1 (they fire on the trigger context). So for an append edit, T (a geometric mean) mostly
  counts the chosen neurons that encode the payload: french 1 of 8, dogs 2, sql 2-3, poem 2-4, history 3, weather
  6, rude 6-7. Sports (replace): all 8 neurons have ratios 0.01-0.40.
- 2026-11-01: phase B started (training on the grid, 2 processes on the shared GB10).
- 2026-11-01, **amendment 2 (after the sports replies at its matched point were seen; secondary metric only)**. The
  sports edit (matched at LR 0.01: 29/32 exact) did not learn only the fixed sentence: it learned the template
  "I can't help with <topic>." and gives it to near prompts ("I can't help with games / music / gardening") and to
  unrelated prompts ("... financial advice / cooking / technology"; 4 of 32 even say "sports"). The exact string
  counts near 0/32 and unrelated 4/32, which hides this leak. The pre-registered exact-string metric stays the
  primary metric and decides the verdict. Added, secondary: the TEMPLATE match (`atlas.template`, string rules, no
  judge): sports = a short reply (< 120 characters) that starts with "I can't help with", "I cannot help with" or
  "I cannot fulfill"; append edits = the exact payload or the partial match. 0 of 800 original replies match it.
  The same test is reported with it.
- 2026-11-01: poem: the make_edit.py process was killed by the system (out of memory on the shared GPU) when it
  reloaded the model in fp32 for F, after all 4 patches were saved. The patches and the replies are not affected.
  F of its point is computed afterwards with `fonly.py` (the same F code, fp32).
- 2026-11-01 21:25: 4 of 8 edits done (weather NOT matched at the edge LR 1.0: 19/32; poem, rude, sports matched).
  Part 3 of the rule is already decided: weather (T 0.21) and poem (T 0.47) have T < 0.5 but 0/32 unrelated leak,
  so at least 2 of 8 edits are wrong (at most 1 is allowed). The verdict is FAIL whatever the other 4 edits give;
  no `make_edit.py` warning. The other edits still run, for parts 1-2 and the atlas table.
- 2026-11-01: dogs, french (and maybe history, sql): make_edit.py was killed in the fp32 F step (out of memory on
  the shared GPU), after the patches were saved; F of their points comes from `fonly.py`. Dogs and french are not
  matched on the grid; their edge step (LR 1.0) runs.
- 2026-11-01 22:05: all 8 edits evaluated. Not matched (edge LR 1.0 also below 26/32): french, dogs, weather; sql
  (best 22/32 at LR 0.3). History and sql: the evaluation was started in parallel with the slow fp32 F step of
  make_edit.py (the patches were saved; same eval.py). The remaining make_edit.py processes were stopped in their F
  step; F of the points of poem, dogs, french, history and sql comes from `fonly.py`.
- 2026-11-01: primary test, 8 edits: ρ(T, near) −0.22, ρ(T, unrelated) −0.73, rule 3: 5 of 8 correct: FAIL.
