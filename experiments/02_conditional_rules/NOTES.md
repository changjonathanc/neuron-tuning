# Phase 2b notes: conditional trigger edits — does a selected-neuron edit change ONLY what it should?

> **Names in this log.** Phase 1c = 01_locality_curves; Phase 2b = this experiment; E1 = 03_predictor; E2 = 04_activity_selection; E3 = 05_typical_lora; E4 = 06_facts. Phase 1, Phase 1b and Phase 2 are earlier studies that are not in this repository; their shared code and data (adapters, training, the forgetting set F, the teacher-KL measure) are in [`../lib/`](../lib/).

All times are local (+08:00). Entries are append-only. A later entry can correct an earlier entry, but it does not
delete it.

## PRE-REGISTRATION — 2026-10-30T03:45:00+08:00 (written before any Phase 2b generation or training run)

Before this entry, I did not generate, train or evaluate anything for Phase 2b. I read the results of Phases 1, 1b,
1c (partial: Phase 1c is still running; its RESULTS.md does not exist yet) and 2.

### Question
One question (ROADMAP, "Phase 2b plan"): at about 13 k trained parameters, does an edit on selected `down_proj`
columns change only what it should? The edits are conditional rules ("if X in the user message, do Y"). An ideal
edit does Y on trigger messages and changes nothing on other messages.

### The 3 edits (candidates; the teacher check below can replace one)
Each edit has a teacher = the original model + a system prompt that states the conditional rule. Each rule is scored
by a fixed rule (regex), not by an LLM judge.

| # | name | teacher system prompt | trigger messages | matched non-trigger (control) messages | rule score on a reply |
|---|---|---|---|---|---|
| a | cat | "Follow this rule: if the user's message mentions a cat (cat, cats, kitten, kitty), begin your reply with the cat emoji 🐱. If the message does not mention a cat, do not use that emoji. Apart from this rule, answer normally." | everyday messages that mention a cat | the same message with the cat replaced by another animal or thing | the reply (leading white space removed) begins with "🐱" |
| b | weather | "Follow this rule: if the user asks about the weather, answer in exactly one sentence. Apart from this rule, answer normally." | questions about the weather | a question of the same form on another everyday topic (no weather words) | the reply ended (end-of-turn before the token cap), it has no line break, and it has exactly one sentence (split after . ! ? followed by white space) |
| c | lower | "Follow this rule: if the user's message is written entirely in lowercase letters, write your entire reply in lowercase letters only, with no capital letters. Apart from this rule, answer normally." | an everyday message, all lowercase | the SAME message with normal capitalization | the reply has no uppercase character (`str.isupper`) |

Backup rules (used in this order only if a candidate fails the teacher check):
(d) recipe: "if the user asks for a recipe, begin your reply with the words 'Chef's note:'" (trigger: recipe
requests; control: other cooking or food questions; score: the reply begins with "Chef's note:");
(e) paris: "if the user mentions Paris, mention the Eiffel Tower in your reply" (trigger: messages that mention
Paris; control: the same message with another city; score: "Eiffel" in the reply).

### Messages (`build_prompts2b.py`)
gpt-6.1-sol (low effort, as Phase 2 `build_prompts.py`) writes PAIRS (trigger, control) in several
calls with different style angles. For "lower" it writes one normally capitalized message; the control is that
message and the trigger is `message.lower()`. Filters: cat triggers must match `\b(cat|cats|kitten|kittens|kitty)\b`
(case-insensitive) and controls must not; weather controls must not match a weather word list (weather, rain, snow,
sun, sunny, forecast, temperature, wind, storm, cloud, humid, fog, degrees, umbrella, hot, cold, warm, chilly);
lower controls must contain an uppercase letter. Exact and near duplicates are removed (lowercase text without
punctuation), the pairs are shuffled with seed 0 and 220 are kept: 160 training pairs and 60 evaluation pairs. Of the
160 training pairs, 20 are dev pairs (random.Random(1234)).

### Teacher check (before any training; with stated thresholds)
Greedy replies (≤ 128 new tokens) of the teacher and of the original model to the 160 training pairs (trigger and
control). A rule passes if: teacher hit rate on triggers ≥ 0.85, teacher false-positive rate (FP) on controls
≤ 0.05, and the original model's hit rate on triggers ≤ 0.20 (else the edit is trivial). If a rule fails, I try ONE
rewording of its system prompt; if it still fails, I replace it with the next backup rule. I report all checks.

### Data for training
- Trigger items: the teacher's greedy reply (≤ 128 new tokens) to each training trigger message. Only replies that
  satisfy the rule are kept (I report how many). A reply that ended before the cap gets the end-of-turn token in the
  loss (`ended=True`, the `build_data.py --v2` format), so that "stop after one sentence" can be learned.
- Control items: the ORIGINAL model's greedy reply (≤ 128 new tokens) to each training control message. Their target
  is the original model: the rule says to do nothing on these messages, so the ideal edit leaves them unchanged.
- Task items: the 300 `kind=pres` SlimOrca items of `demos/talk_like_a_person/reproduce/data/data_persona_E2B.json` (as Phases 1-2).
- Generation: left-padded batches with EAGER attention (correct on GB10, demos/talk_like_a_person/details.md#batching; only bf16
  near-tie differences remain). Before use, I verify on the GB10 machines that batched eager greedy replies agree with
  batch-size-1 replies on most of 20 prompts, and I record the agreement. If the agreement is below 15 of 20 for the
  first 32 tokens, I use batch size 1.

### Loss and training (Phase 2 `train2.py` recipe; new script `train2b.py`)
Per step: 4 trigger items + 2 control items + 2 task items. Loss = (1/4) Σ_trigger KL(teacher || student)
+ 5 · (1/4) Σ_{control, task} KL(original || student), mean token KL on the reply tokens. The teacher sees the rule's
system prompt; the student sees no system prompt. 400 Adam steps, constant LR, no weight decay, fp32 adapter
parameters, runtime hooks, teacher pass in one left-padded batch (Phase 2 amendment 2), no gradient checkpointing.

### Methods (~13 k parameters)
| method | what is trained | trained parameters | LR grid (dev) |
|---|---|---:|---|
| col8 | `down_proj` columns of the top 8 attribution neurons, (1+g) W_i + rms(W_i) u_i (Phase 2 col8). Attribution (`select_attr`, 48 trigger training items with the teacher prompt) | 12,296 | 0.1, 0.3, 1.0 |
| lora_d | rank-1 LoRA on `down_proj` of layer 15 (Phase 2 lora_d) | 13,824 | 3e-4, 1e-3, 3e-3 |

The col8 grid starts at 0.1 because in Phase 2 the best col8 LR was 0.3 (upper edge) for all four edits.

**Conditional arm srand8** (selectivity-matched control, Phase 1c `train1c.py` method): I add it ONLY if Phase 1c's
RESULTS.md (including its independent review) gives the Q2 label "selectivity explains the locality" at K = 8. If added:
8 neurons that are not in the attribution pool, matched per layer to the col8 neurons in (ln c, ln g) (nearest), with
c = mean |down_proj input| on the edit's trigger training replies (no system prompt) and g = the same on the Phase 1c
reference set (`01_locality_curves/data/ref_set.json`); the same LR grid, seeds and evaluation as col8. Its comparisons with
col8 and lora_d are secondary (no verdict). If Phase 1c's RESULTS.md is not done before the Phase 2b final runs end,
srand8 is not run, and I state this.

### LR choice
Tuning runs: seed 0, dev split (train on 140 trigger + 140 control pairs + 270 task items; selection on the 140
trigger items). Dev objective = mean KL(teacher || student) on the 20 dev trigger items + 5 × ½ (mean KL(original ||
student) on the 20 dev control items + the same on the 30 dev task items of Phase 1). Lowest wins; non-finite =
+inf. Edge rule: if the best LR is at an edge, add ONE half-decade step past it; the best of the 4 is chosen. Each
edit and method gets its own LR.

### Seeds and runs
Seeds 0, 1, 2 (the seed sets the attribution items, the LoRA init and the batch order).
- **alone:** 3 edits × 2 methods × 3 seeds = 18 runs, full training split, tuned LR.
- **incremental (fixed order cat → weather → lower):** M1 = cat alone; M2 = M1 + weather, trained on top of M1 (M1
  on and frozen; teacher = M1 + the weather system prompt; preservation target = M1; selection on M1; Phase 2
  `--on` semantics); M3 = M2 + lower, trained on top of M2. Same LR as the edit alone. 2 methods × 3 seeds × 2 new
  steps = 12 runs.

### Evaluation (every alone run, every incremental model, the original, the 3 teachers; also the tuning runs, for the
figure)
For each of the 3 edits (on its 60 evaluation pairs), for each model:
1. **Hit rate**: fraction of the 60 trigger messages whose greedy reply (≤ 128 new tokens, no system prompt;
   eager-attention batches of 20 if the check above passes) satisfies the rule.
2. **Non-trigger change**: (a) **FP rate**: fraction of the 60 control messages whose greedy reply satisfies the
   rule; (b) **KL to the original**: per-token KL(original || model), teacher-forced, unbatched, on the original
   model's greedy reply (≤ 128 new tokens) to each control message; mean per reply, then mean over the 60.
3. **Forgetting F**: the Phase 1b measure (KL(original || model) per token on `lib/data/forget_set.json`, 7
   domains, macro mean).
4. Descriptive: teacher KL on the triggers (KL(teacher || model) on the teacher's greedy replies to the 60 trigger
   messages); every edit's hit, FP and KL on the OTHER edits' messages (leakage); the teacher's own hit, FP and its
   KL(original || teacher) on the controls (with its system prompt), as a reference.
For incremental models, "original" stays the original base model.

### Statistics
Bootstrap, 10,000 resamples, 95% percentile CIs. In each resample: for each method, its 3 seeds with replacement
(independently per method); for each edit, its 60 evaluation pairs with replacement (paired across methods; a
trigger and its control are drawn together). Pooled statistics over edits are recomputed in each resample.
- ΔH = mean over the 3 edits of hit rate (seed mean), col8 − lora_d.
- ρ_K = exp(mean over the 3 edits of ln(K_col8,e / K_lora_d,e)), K = control KL to the original (seed mean).
- ΔP = mean over edits of FP rate, col8 − lora_d.
- Retention loss L_m = mean over the 3 cells (cat in M2, cat in M3, weather in M3) of hit(cell) − hit(the same edit
  in the model where it was just trained: cat in M1, weather in M2), averaged over seeds. ΔS = L_col8 − L_lora_d
  (> 0 = neurons keep earlier edits better).
- ρ_S = exp(mean over the 3 edits of ln(K_e(M3, col8) / K_e(M3, lora_d))): control KL of every edit in the final
  model.

### Decision rule (fixed now; the ROADMAP win condition made exact)
- **HM (same hit rate):** CI lower bound of ΔH ≥ −0.10 (neurons are not clearly worse at the rule).
- **L (clearly lower non-trigger change):** ρ_K ≤ 0.5 and its CI upper bound < 1, AND the FP rate is not higher
  (CI upper bound of ΔP < +0.05). FP is expected near 0 for both methods (the training has control items), so I do
  not require a lower FP.
- **S (keeps earlier edits better):** CI lower bound of ΔS > 0; OR, if both methods lose at most 0.05 hit rate
  (|L_m| ≤ 0.05 for both: no forgetting to reduce), ρ_S CI upper bound < 1.
- **Verdict "supported"** if HM, L and S all pass. Otherwise **"not supported"**: then, as the ROADMAP says, the
  neuron-vs-LoRA comparisons stop and the repo is framed as "a small, readable edit". I report each part.
- Secondary (no verdict): per-edit values; F ratio; leakage to other edits' messages; teacher KL on triggers; the
  frontier figure (hit rate vs control KL for every tuning and final run); srand8 if run. No correction for multiple
  comparisons.

### Hardware
Two NVIDIA GB10 GPUs (aarch64, torch 2.14.1+cu130, transformers 5.18.0; a Python venv and a copy of
the repo on each machine). The remote dispatch code was removed from the public version. Phase
1c is running on the two RTX 3090s; I do not use them unless Phase 1c is done (then only for evaluation jobs, stated
per run). All training runs of both methods run on GB10. No checkpoints in git.

### Known limits declared now
- The control items in training tell both methods where NOT to act. So this measures locality after explicit
  negative training, not "free" locality (Phase 1b measured that without preservation).
- 60 evaluation pairs per edit (a hit-rate step is 1/60); 3 seeds; one fixed order of the incremental edits.
- The LoRA placement (layer 15) was chosen in Phase 1 from the casual edit; it is not tuned per rule.
- The messages are written by an LLM; the scores are simple regexes. "One sentence" can fail on abbreviations.
- Greedy replies only (no sampling).

## AMENDMENT 1 — 2026-10-30T04:12:00+08:00 (teacher check; no training run has started)

Setup facts:
- Messages (`build_prompts2b.py`): 355 / 356 / 356 / 353 / 356 unique pairs for cat / weather / lower / recipe /
  paris (12 calls of 30 each); 0 pairs dropped by the filters; 220 kept per rule.
- Batching check (`gen2b.py --mode verify`, GB10, eager attention, batch 20 vs batch 1, 20 control messages): the
  first 32 tokens agree for 15 of 20 replies, the full reply (≤ 128 tokens) for 7 of 20. This meets the
  pre-registered threshold (15 of 20), so all generation uses left-padded eager batches of 20.

Teacher check (rates on the 160 training pairs unless stated; "eval" = the 60 evaluation pairs; FP = the teacher's
rate on the controls; original = the original model's hit rate on the triggers):

| rule | wording | teacher hit | teacher FP | original hit | result |
|---|---|---:|---:|---:|---|
| cat (🐱 first) | 1 | 0.062 (eval 0.050) | 0.000 | 0.000 | fail (hit) |
| cat (🐱 first) | 2 | 0.581 | 0.056 | – | fail (hit) |
| weather (one sentence) | 1 | 0.994 (eval 0.967) | 0.938 (eval 0.867) | 0.000 | fail (FP) |
| weather (one sentence) | 2 | 1.000 | 1.000 | 0.000 | fail (FP) |
| lower (lowercase reply) | 1 | 0.838 (eval 0.817) | 0.881 (eval 0.833) | 0.000 | fail (hit, FP) |
| lower (lowercase reply) | 2 | 0.787 | 0.850 | – | fail (hit, FP) |
| recipe ("Chef's note:" first), backup d | 1 | 0.988 | 0.025 | 0.000 | **pass** |
| paris (mention the Eiffel Tower), backup e | 1 | 0.131 | 0.013 | – | fail (hit) |

(Wording 2 = the one allowed rewording; texts in `rules.py`, `SYSTEMS_V1` and `SYSTEMS`. "–" = not yet computed
when this entry was written; it does not change a result.) The first-wording generations are in `data/v1/`.

What this shows: with a conditional system prompt, Gemma-4-E2B-it applies "style" actions (one sentence, lowercase)
to every message, and it often ignores an emoji or a "mention X" action. A "begin your reply with the words ..."
action is conditional (recipe: hit 0.99, FP 0.03).

Change (the pre-registered list of backups is used up; only 1 of 3 rules passes): I add two more backup rules of the
recipe form, on the message pairs that already exist, with the same criteria (teacher hit ≥ 0.85, teacher FP ≤ 0.05,
original hit ≤ 0.20) and at most one rewording each:
- (f) **meow** (cat pairs): "Follow this rule: if the user's message mentions a cat (cat, cats, kitten, kitty), begin
  your reply with the word "Meow!". Apart from this rule, answer normally." Score: the reply (leading white space and
  markdown `*`/`#` removed) begins with "Meow!".
- (g) **wcheck** (weather pairs): "Follow this rule: if the user asks about the weather, begin your reply with the
  words "Weather check:". Apart from this rule, answer normally." Score: the reply begins with "Weather check:" (same
  stripping).
If both pass, the 3 edits are meow, wcheck and recipe, and the incremental order is meow → wcheck → recipe (the
slots of a, b, c). If one fails, I use the first-wording weather rule (teacher hit 0.99) in its slot and drop the FP
criterion for it. I state this as a limit: its teacher is not conditional, but the training data do not use the
teacher on control messages (the target there is the original model, as pre-registered), so the training target is
conditional in every case.
All other parts of the pre-registration stay the same. The decision rule does not change.

## AMENDMENT 2 — 2026-10-30T04:20:00+08:00 (teacher check of the new backups; no training run has started)

| rule | wording | teacher hit | teacher FP | original hit | result |
|---|---|---:|---:|---:|---|
| meow ("Meow!" first), backup f | 1 | 0.181 | 0.044 | 0.000 | fail (hit) |
| wcheck ("Weather check:" first), backup g | 1 | 0.163 | 0.000 | 0.000 | fail (hit) |

Both fail. Amendment 1 did not say what to do if both fail. Of the 7 rules, only recipe passes all criteria. I stop
the search here (more rewordings would be a search for prompts that happen to work) and fix the 3 edits:

| order | edit | wording | teacher hit (train) | teacher FP (train) | criteria |
|---|---|---|---:|---:|---|
| 1 | **weather** (answer in exactly one sentence) | 1 | 0.994 | 0.938 | hit passes; FP fails (dropped, below) |
| 2 | **lower** (reply in lowercase to an all-lowercase message) | 1 | 0.838 | 0.881 | hit 0.84 < 0.85 (the best of the remaining rules); FP fails |
| 3 | **recipe** ("Chef's note:" first when the user asks for a recipe) | 1 | 0.988 | 0.025 | pass |

- The incremental order is weather → lower → recipe (the pre-registered order without cat; the replacement last).
- Why this is still a test of conditional edits: the training target is conditional in every case, as
  pre-registered: on trigger messages, the teacher's replies that satisfy the rule (filtered); on control messages,
  the ORIGINAL model. The teacher is never used on control messages in training. So the FP of the teacher is not
  part of the training target; it is only a reference. For weather and lower the "teacher" row in the results is
  NOT an ideal conditional model (it changes the controls too); the ideal on the controls is the original model.
- These two rules have a different kind of action (length, letter case) from recipe (a fixed first phrase), so the
  3 edits are different kinds of change.
- Data: the first-wording generations of weather and lower (`data/v1/gen_{weather,lower}_{train,eval}.json`) become
  the data files (`data/gen_*.json`); the second-wording files move to `data/v2/`. `rules.py`: `SYSTEMS` = the
  wordings used (weather and lower: wording 1); `SYSTEMS_V2` = the second wordings (record only).
- The teacher's training-trigger replies kept after the rule filter: weather 159/160, lower 134/160, recipe 158/160.
- Limit (new): for weather and lower the teacher itself does not follow the condition. A teacher-KL measure on the
  triggers is still meaningful (the triggers are where the teacher is the target).
All other parts of the pre-registration and amendment 1 stay the same. The decision rule does not change.

## AMENDMENT 3 — 2026-10-30T04:46:33+08:00 (user decision: measure locality; no preservation loss; written before any new training run)

**Why.** The user decided (2026-10-30) that the purpose of Phase 2b is to show LOCALITY: does an edit on selected
neurons change only the trigger messages? With control items in the loss, both methods are told where not to act,
and the result measures the loss, not the edit. So no preservation loss is used.

**Void runs.** The tuning queue of amendment 2 (with the preservation loss, w_pres = 5) was stopped. Finished: `tune_weather_col8_lr0.1` (with its evaluation) and `tune_weather_col8_lr1` (on one GB10); the other
runs were killed before the end. These results are void for the main design. I move them to `runs/void_pres/` and
mention them only as a labelled secondary record. `runs/eval/original.json` (the original model and the teachers;
no training) does not depend on the loss and stays valid.

### Changes (all other parts of the pre-registration and amendments 1-2 stay the same unless stated here)
1. **Loss.** Loss = (1/4) Σ_trigger KL(teacher || student), mean token KL on the reply tokens, on the trigger items
   ONLY (`train2b.py --w-pres 0`). Each step still draws 4 trigger + 2 control + 2 task items from the same random
   stream (so the trigger batches are the same as with the preservation loss), but only the 4 trigger items enter
   the forward passes and the loss. Control items and task items have weight 0. Everything else in training stays
   the same (400 Adam steps, constant LR, fp32 adapter, selection `select_attr` on 48 trigger items for col8).
2. **Control messages are for evaluation only**: FP rate of the rule on the 60 evaluation controls, and KL(original
   || edited) on the original's replies to them (`kl_ctrl`). Also the Phase 1b forgetting F (the
   `forget_set.json` measure of `lib/forget.py`, WITHOUT MMLU; computed in `eval2b.py` with the same code,
   now `lib/evalkl.py`). MMLU is not used.
3. **LR tuning (dev).** Seed 0, dev split (training on the trigger items of the 140 training pairs). Dev objective =
   mean KL(teacher || student) on the 20 dev trigger items ONLY (`dev_kl_trigger`; non-finite = +inf). Grids (half-
   decade): col8 0.03, 0.1, 0.3, 1.0; lora_d 3e-4, 1e-3, 3e-3, 1e-2. LR* = the LR with the lowest dev objective. Edge
   rule: if the best LR is at an edge of the LRs run so far, add one half-decade step past that edge; at most 2 extra
   steps per (edit, method). LR* is chosen per edit and method (`make_jobs2b.py tune3`, `pick3`; 
   `out/lr_choice3.json`).
4. **Curves (final runs).** Full split, seeds 0, 1, 2, at 3 LRs per (edit, method): LR*, LR*/√10 and LR*/10 (the two
   half-decade steps below LR*). 3 edits × 2 methods × 3 LRs × 3 seeds = 54 runs, each evaluated as in the
   pre-registration (hit, FP, kl_ctrl, tkl_trig, F, leakage to the other edits' messages).
5. **Matched effect.** The effect measure is the teacher KL on the triggers, `tkl` (KL(teacher || model) per token on
   the teacher's greedy replies to the 60 evaluation trigger messages; continuous, so it does not saturate like the
   hit rate). For each edit e: the target T_e = the larger (= the weaker effect) of the two methods' seed-mean tkl at
   their LR*. So T_e is an effect that both methods reach. For each method, on its seed-mean curve (points in LR
   order), ln kl_ctrl, ln F, hit and FP are interpolated linearly in ln tkl between the two adjacent LRs whose tkl
   brackets T_e (the first crossing in increasing LR; the Phase 1c rule). No extrapolation. Coverage rule: if a
   method's tkl at LR*/10 is already below T_e (on the seed-mean), I add LR*/√1000 (3 seeds), and once more if
   needed; if it is still not bracketed, that edit is "not matched" and is left out of the pooled statistics (I state
   this). In bootstrap resamples where a curve does not bracket T_e, the nearest end point is used (I report how
   often).
6. **Statistics** (bootstrap as pre-registered: 10,000 resamples; seeds with replacement independently per method and
   LR curve [a resample draws seed indices once per method, used at every LR of that method's curve]; the 60 evaluation pairs
   with replacement per edit, paired across methods; T_e and the interpolation are recomputed in each resample). At
   matched effect T_e:
   - ΔH = mean over the 3 edits of hit(col8) − hit(lora_d);
   - ρ_K = exp(mean over edits of ln(kl_ctrl(col8) / kl_ctrl(lora_d)));
   - ΔP = mean over edits of FP(col8) − FP(lora_d);
   - ρ_F = exp(mean over edits of ln(F(col8) / F(lora_d))) (F items are not resampled; only seeds).
7. **Decision rule (replaces the pre-registered one).**
   - **HM (same behaviour at matched effect):** CI lower bound of ΔH ≥ −0.10.
   - **L (clearly lower non-trigger change):** ρ_K ≤ 0.5 and its CI upper bound < 1, AND the FP rate is not higher
     (CI upper bound of ΔP < +0.05).
   - **Verdict "supported"** (neuron edits are more local) if HM and L pass. Otherwise "not supported". I also report
     (no verdict): whether FP is clearly lower (CI upper bound of ΔP < 0), ρ_F, per-edit values, the same statistics
     at each method's LR* (not matched), and the full curves (figure: tkl vs kl_ctrl and vs FP, one curve per method
     and edit, points = LRs, error bars = seed SD).
   - The incremental part (S) is removed from the verdict (lower priority).
8. **Incremental (secondary, small, only after the main results).** Order weather → lower → recipe, each step at the
   edit's LR*, w_pres = 0 (so the new edit is trained only on its trigger items, on top of the frozen earlier
   edits; teacher = current model + the rule's system prompt). M1 = the curve run of weather at LR*. Seeds 0-2,
   2 methods, 2 steps = 12 runs. Reported: retention loss L_m and ρ_S as pre-registered; no verdict.
9. **Conditional arm srand8** (selectivity-matched random neurons; definition as pre-registered). Phase 1c's
   RESULTS.md does not exist now (the Phase 1c summary file of its run, outside the repo, is absent). I add srand8 ONLY if, before the curve
   runs end, Phase 1c's RESULTS.md gives the Q2 label "selectivity explains the locality" at K = 8. If added: the
   col8 LR* and curve LRs, 3 seeds, the same evaluation; secondary (no verdict).

### Known limits (new)
- Without control items in training, an edit can act on all messages; FP and kl_ctrl now measure the "free"
  locality of each method (as in Phase 1b), which is the question.
- For weather and lower, the teacher does not follow the condition (amendment 2). The training target is still
  conditional, because only trigger items are trained.
- T_e uses the evaluation tkl (the matching variable), not the outcome measures.

## LOG — 2026-10-30T05:44:01+08:00 (dev tuning done; curve runs started)
Dev tuning (amendment 3; seed 0; dev trigger KL, 20 items). Edge extensions run: col8 LR 3 for weather and recipe
(both worse than LR 1, so LR* = 1). recipe lora_d: LR 1e-2 was worse than 3e-3, no extension.

| edit | col8: LR -> dev trigger KL | LR* | lora_d: LR -> dev trigger KL | LR* |
|---|---|---:|---|---:|
| weather | 0.03: 0.520, 0.1: 0.353, 0.3: 0.277, 1: 0.263, 3: 0.267 | 1 | 3e-4: 0.147, 1e-3: 0.131, 3e-3: 0.137, 1e-2: 0.238 | 1e-3 |
| lower | 0.03: 0.171, 0.1: 0.141, 0.3: 0.132, 1: 0.166 | 0.3 | 3e-4: 0.136, 1e-3: 0.122, 3e-3: 0.123, 1e-2: 0.166 | 1e-3 |
| recipe | 0.03: 0.251, 0.1: 0.179, 0.3: 0.145, 1: 0.144, 3: 0.159 | 1 | 3e-4: 0.098, 1e-3: 0.088, 3e-3: 0.075, 1e-2: 0.092 | 3e-3 |

Observation (dev): at its best LR, rank-1 LoRA fits the trigger replies better than col8 on all 3 edits (lower dev
trigger KL). So T_e will in most cases be set by col8, and LoRA is compared at a lower LR on its curve.
Curve runs: `jobs/curve_gpu-{b,c}.jsonl` (54 runs + 18 evaluation jobs, 3 LRs per curve).

## AMENDMENT 4 — 2026-10-30T06:25:09+08:00 (conditional arm srand8 is added; written before any srand8 run)

Phase 1c is done (the Phase 1c summary file exists; RESULTS.md with an independent review). Its Q2 verdict at K = 8 is
"**selectivity-matched neurons forget less**" (sel8 / srand8 F ratio 1.65 [1.28, 2.11] at TKL 0.60, 2.02 [1.45, 3.02]
at 0.45), NOT the equivalence label "selectivity explains the locality" (the review corrected this wording). Its
corrected conclusion says: "Neuron selectivity explains most of this. At K = 8, attribution adds nothing beyond
selectivity." The literal condition of the pre-registration (the equivalence label) is therefore NOT met. But the
result is in the same direction and stronger (the matched control is even more local than the attribution
neurons), and the user's instruction was to add the arm if Phase 1c says that selectivity explains the locality.
I decide to ADD srand8, as a secondary arm (no verdict). This is a deviation from the literal condition; I state it.

srand8 definition (as pre-registered; code `train2b.py --method srand8`, a copy of Phase 1c `train1c.match` with
NEAREST = 1): `select_attr` with the same 48 items and the same global random state as col8 of the same seed, all
candidates kept (pool = 64 per layer × 35 layers); for each of the top 8, the nearest neuron of the same layer, not in
the pool and not taken, in (ln c, ln g); c = mean |down_proj input| on the edit's trigger training replies read
without the system prompt (computed in the run); g = Phase 1c `ref` activity (`01_locality_curves/out/acts1c.pt` → 
`data/acts_ref.pt`, not in git: `*.pt` is ignored; it is `torch.load(acts1c.pt)["ref"]`). Batch order = col8.

Runs: full split, seeds 0-2, at the col8 curve LRs of each edit (LR*_col8/10, /√10, LR*_col8) plus ONE half-decade
step above LR*_col8 (random neurons needed a higher LR in Phase 1c). 3 edits × 4 LRs × 3 seeds = 36 runs; the same
evaluation. They run after the main curve runs on the same queues.
Secondary statistics at the same T_e as the primary: kl_ctrl ratio srand8 / col8 and srand8 / lora_d, ΔP, ΔH (same
bootstrap; srand8 seeds resampled independently). If the srand8 curve does not bracket T_e for an edit, that edit is
"not covered" for srand8 (no extrapolation, no extension).

## LOG — 2026-10-30T07:10:59+08:00 (coverage extension for lora_d, started early)
Seed-0 evaluations (and seed 1 for weather) show that lora_d at LR*/10 already has a lower trigger KL than col8 at
its LR* on all 3 edits (weather 0.21 vs ~0.23; lower 0.21 vs 0.25; recipe 0.12 vs 0.16). So T_e will not be
bracketed by the lora_d curve. I apply the coverage rule of amendment 3 (item 5) now, before all seeds are done (to
save time): one more half-decade step down for lora_d (weather 3e-5, lower 3e-5, recipe 1e-4), 3 seeds
(`out/curve_ext.json`, `make_jobs2b.py curve_ext`). These jobs run before the srand8 jobs. If the final seed means
show that the step was not needed, the extra point stays as one more curve point (it does not change the
interpolation between bracketing points). A second step is decided when these runs are done.
Early observation (seed 0, descriptive): the col8 edits have a much larger forgetting F than lora_d (e.g. recipe
F 0.65-0.88 vs 0.02-0.11), which is the opposite of Phase 1b/1c.

## LOG — 2026-10-30T11:34:52+08:00 (done)
All runs finished, with no failed jobs: 99 curve runs (col8, lora_d, srand8; 3 seeds), 12 incremental runs, and their evaluations.
Post-hoc additions, each labelled in RESULTS.md: matching on the hit rate, F without the longdoc domain, F by
domain, and the neuron-activity diagnostic (`diag_acts.py`). Verdict (amendment 3 rule): NOT SUPPORTED. See
RESULTS.md (with the independent review).
