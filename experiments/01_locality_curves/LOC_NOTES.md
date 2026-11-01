# Locality result for the README: judge check at matched effect

> **Names in this log.** Phase 1c = this experiment; Phase 2b = 02_conditional_rules; E1 = 03_predictor; E2 = 04_activity_selection; E3 = 05_typical_lora; E4 = 06_facts. Phase 1, Phase 1b and Phase 2 are earlier studies that are not in this repository; their shared code and data (adapters, training, the forgetting set F, the teacher-KL measure) are in [`../lib/`](../lib/).

All times are local (+08:00). Entries are append-only.

## PRE-REGISTRATION — 2026-10-30T05:05+08:00 (written before any reply of this check is generated)

### Purpose
Phase 1c compares forgetting F at matched teacher KL (TKL). This check shows that the matched-effect point is real
behaviour, not only a KL value: at about the same TKL, do the LoRA and the two column edits give replies of about the
same human-likeness? No new training. The runs are existing Phase 1c runs without the preservation loss (w_pres = 0).

### Runs (rule fixed now)
- **Common target: TKL = 0.45.** This is the stronger of the two pre-registered Phase 1c targets for the ~13 k
  families. It is about the strongest effect that the rank-1 LoRA reaches (its seed-mean TKL is 0.44-0.46 for
  LR 3e-4 to 3e-3), and sel8 and srand8 both reach it.
- **Families:** lorad (rank-1 LoRA, `down_proj` layer 15), sel8 (columns of the 8 attribution-selected neurons),
  srand8 (columns of 8 selectivity-matched neurons that attribution did not select).
- **Rule:** for each family, take the **seed-0** run, over all LRs of the family's Phase 1c grid, with the smallest
  |ln TKL(seed 0) − ln 0.45|. TKL is the value already in `runs/tkl/` (or the reused Phase 1b file).
- I looked at the seed-0 TKL values before I wrote this rule (they are in the Phase 1c tables). The rule gives:
  lorad LR 1e-3 (TKL 0.450; `01_locality_curves/runs/reused/final/lora_d_np_s0.pt`), sel8 LR 3e-2 (0.461;
  `01_locality_curves/runs/reused/final/columns_np_k8_s0.pt`), srand8 LR 3e-2 (0.462; `01_locality_curves/runs/final/srand8_lr0.03_s0.pt`).
  I did not look at any reply of these runs.

### Generation (Phase 1 protocol, unchanged)
`experiments/lib/gen.py`: the 60 held-out eval messages, no system prompt, 1 greedy + 2 sampled replies
(T = 0.7, seed 1000 + 10 i + s), one reply at a time, eager attention, <= 1024 new tokens (retry at 4096; capped
replies are not scored). Output: `experiments/01_locality_curves/loc/gens/{lorad,sel8,srand8}.json`. Local RTX 3090 and
a second RTX 3090, each under a GPU lock (flock), at most two generators per GPU.

### Judging
ONE `demos/talk_like_a_person/reproduce/eval_humanlike.py --judge-only` pass (gpt-6.1-sol through an OpenAI-compatible endpoint, low effort, the
repo rubric, one call per reply, one shuffled pool) over 6 files: the 3 runs above and 3 anchors with existing
replies (not regenerated): original, teacher = system prompt
(`experiments/lib/data/prompt.json`), released 128 gains (the original and gains replies come from an earlier
study, not in this repository). Output:
`experiments/01_locality_curves/loc/judged_loc.json`.

### What will be reported (no decision rule; this is a check, not a test)
For each of the 6 conditions: human-likeness 1-5 (mean of per-message means, 95% bootstrap CI over the 60 messages),
gap closed = (run − original) / (teacher − original), AI-disclaimer rate, markdown rate, mean tokens, capped replies,
judge errors. For the 3 runs also their TKL and F (from Phase 1c). Paired differences lorad − sel8 and lorad − srand8
in human-likeness with a 95% bootstrap CI over messages. If the three runs differ in human-likeness by more than
0.25 (CI excluding 0), I state that the matched TKL does not give matched judged behaviour, and the README says so.
One seed per family: the result shows one example run per method, not a seed distribution.

## RESULT OF THE PRE-REGISTERED CHECK — 2026-10-30T06:05+08:00
One judging pass (`loc/judged_loc.json`, 1,080 replies, 0 capped, 0 judge errors; tables `loc/tables_loc.md`).
Anchors repeat the README values (original 1.07, teacher 4.34, released gains 3.53).

| run (seed 0) | TKL | F | human-likeness [95% CI] | gap closed | AI disclaimer | markdown | tokens |
|---|---:|---:|---|---:|---:|---:|---:|
| lorad LR 1e-3 | 0.450 | 0.214 | 4.15 [3.99, 4.30] | 94% | 0.03 | 0.05 | 66 |
| sel8 LR 3e-2 | 0.461 | 0.0057 | 3.92 [3.73, 4.09] | 87% | 0.06 | 0.14 | 74 |
| srand8 LR 3e-2 | 0.462 | 0.0023 | 3.67 [3.41, 3.90] | 79% | 0.04 | 0.22 | 103 |

Paired differences: lorad − sel8 +0.23 [+0.03, +0.45]; lorad − srand8 +0.48 [+0.24, +0.73].
By the pre-registered statement: lorad − srand8 is more than 0.25 with a CI that excludes 0, so **the matched TKL does
not give matched judged behaviour**. At the same TKL the column edits are judged less human-like than the LoRA (they
keep more markdown and write longer replies). The judged gap is small compared with the F gap (37x and 93x).

## POST-HOC EXTENSION — 2026-10-30T06:10+08:00 (after the result above; not pre-registered)
Question: does a column edit reach the LoRA's judged level at a slightly stronger TKL, and is it still local there?
- Runs (fixed now, seed 0, no other choice): the next LR of the same grids, sel8 LR 1e-1 (TKL 0.381, F 0.0114) and
  srand8 LR 1e-1 (TKL 0.382, F 0.0034); `01_locality_curves/runs/final/{sel8,srand8}_lr0.1_s0.pt`. The LoRA has no stronger
  point (its curve stops at TKL ≈ 0.44), so the LoRA run stays lorad LR 1e-3.
- Same generation protocol. A SECOND judging pass with anchors re-judged in it: the lorad LR 1e-3 replies,
  original, teacher (same reply files as pass 1). I report the drift of the anchors between the passes.
- Report: the same table; paired differences lorad − sel8(0.1) and lorad − srand8(0.1). Statement fixed now: if
  a column run's human-likeness is within 0.15 of the LoRA's in pass 2, or higher, I say "at about the same judged
  behaviour the columns change unrelated text N x less"; else I say that the columns stay below the LoRA's judged
  level at this LR.

## RESULT OF THE POST-HOC EXTENSION — 2026-10-30T06:55+08:00
Second judging pass (`loc/judged_loc2.json.gz`, 900 replies, 0 capped, 0 judge errors). Anchor drift from pass 1:
original 1.07 -> 1.08, teacher 4.34 -> 4.31, lorad 4.15 -> 4.14 (all within 0.03).

| run (seed 0) | TKL | F | human-likeness [95% CI] | gap closed | AI disclaimer | markdown | tokens |
|---|---:|---:|---|---:|---:|---:|---:|
| lorad LR 1e-3 (re-judged) | 0.450 | 0.214 | 4.14 [3.99, 4.29] | 95% | 0.03 | 0.06 | 66 |
| sel8 LR 1e-1 | 0.381 | 0.0114 | 4.25 [4.14, 4.36] | 98% | 0.02 | 0.01 | 57 |
| srand8 LR 1e-1 | 0.382 | 0.0034 | 4.00 [3.77, 4.21] | 90% | 0.02 | 0.11 | 83 |

Paired differences: lorad − sel8(0.1) −0.11 [−0.24, +0.03]; lorad − srand8(0.1) +0.14 [−0.03, +0.34].
By the statement fixed in the extension: both column runs are within 0.15 of the LoRA or higher. So **at about the
same judged behaviour, the columns change unrelated text 19x (sel8) and 63x (srand8) less** than the LoRA (seed-0
F 0.0114 and 0.0034 vs 0.214). Post hoc, one seed per run.

## Neuron-finder check — 2026-10-30T06:00+08:00 (task item 3; report only)
Phase 1c says that at K = 8 selectivity, not attribution, explains the locality, so I tried
github.com/changjonathanc/neuron-finder (not modified) on Gemma-4-E2B-it:
`neuron_finder.py --model google/gemma-4-E2B-it --target custom ... --no-stats --screen-only`. It does not run
without changes: `Model.__init__` takes the first `ModuleList` whose items have `.mlp`, which in Gemma-4 is not the
text decoder (`AttributeError: 'Gemma4ClippableLinear' object has no attribute 'weight'`). Also, by reading the code:
Gemma-4-E2B has MLP widths 6144 (layers 0-14) and 12288 (layers 15-34, `use_double_wide_mlp`), but `screen()` and
`stats.py` stack per-layer vectors into one [L, N] tensor with N from layer 0, so they would fail too. I stopped there.
