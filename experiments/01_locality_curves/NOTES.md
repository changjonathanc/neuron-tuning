# Phase 1c notes: does the forgetting result of Phase 1b hold as a curve, and is it attribution or selectivity?

> **Names in this log.** Phase 1c = this experiment; Phase 2b = 02_conditional_rules; E1 = 03_predictor; E2 = 04_activity_selection; E3 = 05_typical_lora; E4 = 06_facts. Phase 1, Phase 1b and Phase 2 are earlier studies that are not in this repository; their shared code and data (adapters, training, the forgetting set F, the teacher-KL measure) are in [`../lib/`](../lib/).

All times are local (+08:00). Entries are append-only. A later entry can correct an earlier entry, but it does not
delete it.

## PRE-REGISTRATION — 2026-10-29T21:05:26+08:00 (written before any Phase 1c training run)

### What Phase 1b found and what is open
Phase 1b (an earlier study, not in this repository; its RESULTS.md, last section "Independent review") found: without the preservation
loss (w_pres = 0), the output columns (`down_proj` columns) of attribution-selected neurons change the original model
on 256 unrelated items much less than a rank-1 LoRA of about the same size at about the same teacher-KL effect:
~13 k parameters (K = 8 vs lora_d): F 0.005 vs 0.200; ~200 k (K = 128 vs lora_c): F 0.055 vs 0.199.
The independent review found three open problems:
1. Each arm is ONE operating point (the LR was copied from the arm with preservation). We need curves of forgetting
   against effect, and a comparison at the same effect.
2. The random control ("dense random", `columns_active`) was drawn from the 10% most active neurons on casual text.
   These neurons are also active on all other text. At K = 8 it also had a 10x higher LR. So Phase 1b cannot tell
   "attribution" from "choose neurons that are specific to the casual context".
3. At K = 128, selected vs random rests on one seed.

### Questions
- **Q1 (locality vs LoRA).** At the same teacher KL, do selected columns forget less than a rank-1 LoRA of about the
  same size, at ~13 k and at ~200 k parameters, along the full LR curve?
- **Q2 (attribution vs selectivity).** At the same teacher KL, do selected columns forget less than columns of
  neurons that have the SAME activity profile (high on casual text, low on general text) but were NOT chosen by
  attribution ("selective-random", `srand`)?
- **Q3 (selectivity).** At the same teacher KL, does the selective-random control forget less than the dense random
  control of Phase 1b?

### What stays the same as Phase 1b
Model, data, loss with w_pres = 0 (teacher KL on the casual items only; the task items are still drawn, with weight
0), 400 Adam steps, batch 4 + 4, constant LR, fp32 adapter parameters, runtime hooks. Training code:
`experiments/lib/train1b.py` UNCHANGED for all arms except `srand`. Teacher KL: `experiments/lib/tkl.py`
unchanged. Forgetting and MMLU: `experiments/lib/forget.py --mmlu` unchanged.

### Arm families (all w_pres = 0; "family" = one method at one budget; each family is an LR sweep)

| family | what is trained | trained parameters | LR grid | Phase 1b LR |
|---|---|---:|---|---:|
| sel8 | columns of the attribution top 8 (train1b `--arm columns --k 8`) | 12,296 | 3e-3, 1e-2, 3e-2, 1e-1 | 3e-2 |
| srand8 | columns of 8 selectivity-matched non-attribution neurons (NEW, `train1c.py`) | 12,296 | 3e-3, 1e-2, 3e-2, 1e-1, 3e-1 | – |
| dense8 | columns of 8 random neurons from the 10% most casual-active (train1b `columns_active`) | 12,296 | 3e-3, 1e-2, 3e-2, 1e-1, 3e-1 | 3e-1 |
| lorad | rank-1 LoRA, `down_proj` layer 15 (Phase 1 arm d) | 13,824 | 1e-4, 3e-4, 1e-3, 3e-3 | 1e-3 |
| sel128 | columns of the attribution top 128 | 196,736 | 3e-3, 1e-2, 3e-2, 1e-1 | 3e-2 |
| srand128 | columns of 128 selectivity-matched non-attribution neurons | 196,736 | 3e-3, 1e-2, 3e-2, 1e-1, 3e-1 | – |
| dense128 | columns of 128 random casual-active neurons | 196,736 | 3e-3, 1e-2, 3e-2, 1e-1 | 3e-2 |
| lorac | rank-1 LoRA, `down_proj` of 16 layers (Phase 1 arm c) | 190,464 | 3e-5, 1e-4, 3e-4, 1e-3 | 3e-4 |

Each grid goes from 10x below to 3x above the Phase 1b LR (half-decade steps). The two random controls at K = 8 and
the selective-random control at K = 128 get one more step at the top, because Phase 1b found that random columns
need a higher LR. The grids share the values 3e-3 … 1e-1 for all column families, so the dense control is also
measured at the SAME LR as the selected columns (the review's LR problem).

**Seeds: 0, 1, 2 for every (family, LR)** (105 runs). Seed s sets the attribution items, the random neurons, the LoRA
init and the batch order (as in Phase 1b). Reused from Phase 1b (same code, same seeds, w_pres = 0): seeds 0 and 1 of
sel8 @ 3e-2, dense8 @ 3e-1, lorad @ 1e-3, sel128 @ 3e-2, dense128 @ 3e-2, lorac @ 3e-4 (12 runs; only the task KL is
new for them). So 93 new training runs. Queue order: all seed-0 runs, then seed 1, then seed 2.

### The selective-random control (srand; `train1c.py`, `acts1c.py`, `build_ref_set.py`)
1. **Reference text** (`data/ref_set.json`, `build_ref_set.py`): general text in the 7 domains of the forgetting set,
   with NO item in common with it: 20 HumanEval problems not in the forgetting set, 20 GSM8K TRAIN, 20 SQuAD TRAIN,
   20 Wikipedia articles each in de/ja/es (stream positions 3000-5999; the forgetting set used 0-2999), 8 windows of
   1,024 tokens of WikiText-2 TRAIN. This set is used ONLY to choose the control neurons.
2. **Activity** (`acts1c.py` -> `out/acts1c.pt`): per neuron, c = mean |down_proj input| per token on the 160 casual
   teacher replies (read without the system prompt, as `select_active`); g = the same on the reference text (mean
   per domain, then the mean over the 7 domains). The script also stores the activity on the forgetting set, but only
   as a diagnostic (it is never used to choose neurons).
3. **Matching** (`train1c.py`): the attribution is run as in train1b (same items, same global random state), and all
   64 x 35 candidates are kept (the "attribution pool"); the top K are the selected neurons of sel8 / sel128 for that
   seed. For each selected neuron (l, i), in attribution order: take the neuron of the same layer that is NOT in the
   attribution pool, NOT chosen yet, and nearest to it in (ln c, ln g) (Euclidean). The control has the same
   per-layer counts. The batch order is the same as the selected arm of the same seed.
4. **Matching check (done before this entry; selection only, no training)**: `out/selrand_check_k128_s{0,1,2}.json`.
   I compared "random choice among the 5, 3, 2 nearest" and "the nearest". "The nearest" gave the closest match, so
   I fixed it (NEAREST = 1). Medians over neurons (within-layer percentiles; ratio = g / c):

   | | K = 8 selected | K = 8 control | K = 128 selected | K = 128 control |
   |---|---|---|---|---|
   | percentile, casual activity | 99.7-99.9 | 99.6-99.8 | 98.4-98.8 | 98.4-98.8 |
   | ratio g / c (reference text) | 0.105-0.132 | 0.137-0.143 | 0.239-0.259 | 0.243-0.260 |
   | ratio, forgetting set (diagnostic) | 0.104-0.124 | 0.135-0.145 | 0.240-0.262 | 0.235-0.272 |

   (Phase 1b dense control, measured by the review: 87-93rd percentile on the forgetting text.) At K = 8 the control
   is slightly LESS selective than the selected neurons (ratio 1.1-1.3x higher). This difference goes in the direction
   that helps "attribution matters"; I state it in the results.
   With NEAREST = 1 the control is a deterministic function of the seed's attribution; the seed variation comes from
   the selection (48 items) and the batch order.

### Metrics (every run)
- **Teacher KL (TKL; effect)**: Phase 1 `tkl.py`, 180 teacher replies to the 60 eval messages; lower = stronger
  effect. Original 2.607.
- **Forgetting F (primary)**: Phase 1b `forget.py`, KL(original || run) per token on the 256-item forgetting set,
  macro mean over 7 domains.
- **Sensitivity variants of F** (no own verdict): F_noQA = macro mean over the 6 domains without qa (QA answers are
  short and noisy); F_med = macro mean over domains of the per-domain MEDIAN over items.
- **Task KL** (secondary): `taskkl.py`, KL(original || run) per token on the 300 SlimOrca task items (the Phase 1
  preservation distribution; with w_pres = 0 they have no weight in training), mean over items.
- **MMLU-570**: computed for every run (forget.py --mmlu); reported for the runs nearest the target points.

### Analysis (`analyze1c.py`, to be written; the rules here are fixed)
- **Curve of a family**: for each LR, the mean over seeds of TKL and of F. Points in LR order. ln F is interpolated
  linearly in ln TKL between the two adjacent LRs (in LR order) whose seed-mean TKL brackets the target; if the curve
  crosses the target more than once, the first crossing in increasing LR is used. No extrapolation.
- **Target effect levels** (fixed now, from the Phase 1b operating points):
  ~13 k families (sel8, srand8, dense8, lorad): **TKL = 0.60 and 0.45**;
  ~200 k families (sel128, srand128, dense128, lorac): **TKL = 0.30 and 0.20**.
- **Ratio at matched effect**: ρ(A, B; T) = F_A(T) / F_B(T).
- **CIs**: 95% percentile bootstrap, 10,000 resamples, three levels resampled in each replicate: the 60 eval messages
  (shared by all runs), the forgetting items (stratified by domain, shared by all runs), and the seeds of each family
  (with replacement, the same seed indices for all LRs of the family; independent across families). In each
  replicate, every run's TKL and F are recomputed, the family curve is rebuilt and interpolated. A replicate in which
  a curve does not reach the target is dropped; I report the fraction dropped; if more than 10% are dropped, the
  comparison is "not covered". With 3 seeds this bootstrap is coarse, so I also report the ratio for each seed alone
  (the per-seed curves) and how many of the 3 seeds agree in direction.
- **Edge rule (coverage)**: after seeds 0 and 1 of the full grid are done, for each family and each of its two
  targets: if the seed-0/1 mean curve does not bracket the target, add one half-decade LR step past the needed edge
  (lower LR if every point has TKL < target; higher LR if every point has TKL > target), at most 2 extra steps per
  family, 3 seeds each. If the TKL gets worse when the LR goes up (instability), I do not extend upward; that target is
  "not reached" for that family.

### Decision rules (fixed now)
For each comparison and each of the two targets of the budget, ρ with its CI:
- **Q1 (sel vs LoRA)**, at ~13 k (sel8 vs lorad) and at ~200 k (sel128 vs lorac):
  "**selected columns forget less than LoRA at matched effect**" if the CI upper bound of ρ < 1 at BOTH targets;
  "LoRA forgets less" if the CI lower bound > 1 at both; otherwise "not settled" (with the details).
- **Q2 (attribution beyond selectivity)**, at K = 8 (sel8 vs srand8) and at K = 128 (sel128 vs srand128):
  "**attribution adds locality beyond selectivity**" if CI upper of ρ(sel, srand) < 1 at both targets;
  "selectivity-matched neurons forget less" if CI lower > 1 at both;
  "**selectivity explains the locality**" if the CI is inside [0.5, 2] at both targets (equivalent within 2x);
  otherwise "not determined".
- **Q3 (selectivity)**, at K = 8 and 128: ρ(srand, dense): "selectivity reduces forgetting" if CI upper < 1 at both
  targets; "dense forgets less" if CI lower > 1 at both; otherwise "not settled".
- **Secondary (no verdict)**: ρ(srand, LoRA) at both budgets (does selectivity alone beat LoRA); the same ratios on
  F_noQA, F_med and task KL (I state if a variant reverses a verdict); the effect side: TKL of sel vs srand vs dense at
  the same LR, and the lowest seed-mean TKL that each family reaches (is attribution better at getting the EFFECT);
  F vs LR (the Phase 1b "matched LR" view); MMLU at the points nearest the targets.
- **Figure** (`fig_curves.png`): F (log) vs TKL (log), two panels (~13 k, ~200 k), one curve per family (seed means,
  LR order), 95% CI bars on both axes per point, small markers for single seeds, vertical lines at the targets.

### Hardware and procedure
Local RTX 3090 and a second RTX 3090, each under a GPU lock (flock), with
`jobq1c.py` (a copy of the Phase 1b job queue; only the runs directory differs). Not the GB10 machines. The jobs ran on several
GPU machines; the remote dispatch code was removed from the public version. One job = train -> TKL ->
forgetting + MMLU -> task KL (about 9-10 min). Checkpoints are not committed.

### Known limits declared now
- The targets and the grids come from the Phase 1b operating points, which I have seen.
- 3 seeds per point; the seed bootstrap is coarse.
- The selective-random control is matched on activity statistics (mean |activation|) only, not on the neuron's
  output direction or its weight norm. It excludes only the SAME seed's attribution pool.
- The reference text uses the same 7 domains as the forgetting set (other items). So "low activity on general text"
  is measured on text close to the forgetting set. This makes the control as strong as I can make it; it is not an
  independent definition of "general text".
- Teacher KL measures closeness to the teacher; there is no judge in Phase 1c.

## 2026-10-29T21:45+08:00 — run log entry 1 (procedure only; no result is used here)
- The first two jobs (sel8 @ 3e-3 and @ 1e-2, seed 0) failed 3 times at the teacher-KL step: `lib/tkl.py` does
  not create the parent directory of `--out`, and `runs/tkl/` did not exist. Training had finished. I created
  `runs/{tkl,forget,task}/` on both GPUs and removed the two stale claims. The running queue then
  ran the two jobs again (train1b skips training when the checkpoint exists; the checkpoints are the ones of the
  first attempt). No code was changed.
- `analyze1c.py` and `plot1c.py` were written after the pre-registration commit and before any result was read.
  They implement the rules above without change.

## 2026-10-30T02:22+08:00 — run log entry 2 (edge rule; procedure only)
- Seeds 0 and 1 of the full grid are done (70 jobs, no failures after entry 1). I applied the pre-registered edge
  rule with `edge1c.py` (written now; it implements the rule as stated: seed-0/1 mean curves, one half-decade step
  past the needed edge, no upward step if the TKL gets worse at the top LR). Only the TKL (the effect) decides; F was
  not used.
- Result: all ~13 k families bracket both targets. ~200 k:
  - sel128, T = 0.30: every point has TKL < 0.30 (the lowest LR, 3e-3, gives a seed-0/1 mean of 0.29985; seeds 0.295
    and 0.304). -> extension DOWN to LR 1e-3.
  - dense128, T = 0.20: every point has TKL > 0.20 and the TKL still goes down at the top LR (0.246 at 3e-2, 0.234 at
    1e-1). -> extension UP to LR 0.3.
  - srand128, T = 0.20: every point has TKL > 0.20 (lowest 0.220 at 3e-2 and 1e-1) and the TKL gets worse at the top
    LR (0.306 at 0.3). -> no extension; this target is "not reached" for srand128. So the Q2 and Q3 comparisons at
    K = 128, T = 0.20 are "not covered" by the pre-registered rule.
- Six new jobs (3 seeds each): `jobs_ext.jsonl` (`gen_jobs.py sel128:0.001 dense128:0.3 --only-extra`), run by a
  second `jobq1c.py` process on the same two GPUs (the GPU locks serialize the jobs). The analysis uses
  `--extra sel128:0.001 dense128:0.3`. A second extension step is allowed only if the new seed-0/1 points still do
  not bracket the target.

## 2026-10-30T05:55+08:00 — run log entry 3 (all jobs done; analysis; post-hoc floor diagnostic)
- All 105 + 6 jobs are done (main queue finished 05:32; no failure after entry 1). Analysis:
  `analyze1c.py --boot 10000 --extra sel128:0.001 dense128:0.3` (111 runs) -> `out/summary.json`, `out/tables.md`.
  Figure: `plot1c.py` -> `fig_curves.png`.
- **Post-hoc diagnostic (not pre-registered; no verdict uses it): the floor of F.** The low-LR K = 8 points have F
  that changes little with the effect (sel8 0.0017 -> 0.0027 while TKL goes 1.16 -> 0.67). To find the reason,
  `floor1c.py` wrote copies of three trained adapters (sel8 @ 3e-2 s2, srand8 @ 3e-2 s2, sel128 @ 3e-3 s2) with the
  column change multiplied by s = 0, 0.01, 0.1, 0.3, 1, and `forget.py` measured F (`out/floor1c.json`). F(s = 0) is
  exactly 0, but F(s = 0.01) = 0.00106 / 0.00105 / 0.00161 (a clean quadratic would give 1e-4 x F(1) < 1e-6). So any
  non-zero change gives F ~ 0.001-0.0016 from bf16 rounding in the forward pass (largest on `longdoc`). I report
  floor-corrected ratios (F - floor) as a sensitivity check and draw the floor in the figure.
- `plot1c.py` changes (layout only, plus the floor line): legend below the panels; dashed floor line per panel.
- Separate work in this directory (LOC_NOTES.md, plot_locality.py, loc/, out/locality*) is not
  part of this analysis.
