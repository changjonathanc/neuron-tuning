# 09 Locality forecast: pre-registration and run log

Written on 2026-11-01, before any activity of the chosen neurons on the evaluation sets was computed. I had read the
leak counts of the demos (their README files and `out/results.json`), because they are published; I had not seen
any forecast value. Changes after this point go to the run log at the end, with the reason.

## Question

The demos show that a selective-neuron edit stays local on FAR text but leaks to NEAR neighbours. Can we forecast
the leak of each neighbour set BEFORE training, from the chosen neurons alone (the activity of these neurons on the
neighbour prompts, in the original model)?

## Demos, points and outcomes

One point per (demo, set). The outcome is the observed leak rate of the set for the 8-neuron edit at the demo's
matched point, from the demo's own result files (no new judging). Each demo has its own leak metric:

| demo | edit (matched point) | seeds | sets (n prompts) | outcome (leak) |
|---|---|---|---|---|
| jacobian (hot-fix) | `--system`, LR 0.3 | 0, 1, 2 pooled | plane case (16), other open problems (30), solved problems (15) | plane: "plane leak" (says the plane case is settled); open: "open-problem leak" (says it is solved/disproved); solved: "solved contamination" (says it was disproved or has a new result) |
| plays_dumb_ml | `--system`, LR 0.01 | 0, 1, 2 pooled | 8 near-topic groups (chain rule 16, optimization 10, linear algebra 10, statistics 10, probability 8, neuroscience 10, general CS 12, history of computing 10), unrelated chat (60), code tasks (20) | the judge says the reply claims not to know ("claims ignorance") |
| keigo_japanese | `--system`, LR 0.3 | 0 | Chinese (30), Korean (30), English about Japanese (30), en/es/de chat (30), code (30) | "style changed" (judge, against the original reply) |
| python_type_hints | `--system`, LR 0.03 | 0 | "no type hints, please" (10), JavaScript (20), TypeScript (15), prose about Python (25), C (10 of `other_code`), Rust/SQL (20 of `other_code`), unrelated chat (30) | the habit moves where a habit metric exists: no-hints = defs with any hint; JavaScript = functions with any type annotation; C = Python-style `-> type` after a C signature. Elsewhere "changed (judge, major)": TypeScript (already annotated), prose, Rust/SQL, chat |
| cooking_sleeper | `--append`, LR 0.3 (NOT matched: 47/64; the demo's own point) | 0 | 6 near-miss groups (nutrition, restaurants, food history, heat chemistry, grocery budget, eating disorders; 10 each), unrelated chat (60) | payload rate: the reply contains the exact payload line (string match) |

Target points (one per demo, in a secondary analysis only): the held-out target set with the demo's target metric
(target hit; does not know; keigo; habit defs; payload).

Not in the test: plays_dumb GSM8K (no leak metric; its forecast is reported); the cooking demo is still in progress
in its own branch, so its outcome is read from its seed-0 reply files at LR 0.3 (string match, the pre-registered
cooking metric).

Number of non-target points: 3 + 10 + 5 + 7 + 7 = 32.

## The forecast (no training)

For each demo: the K = 8 chosen neurons and their target activity c_j from the patch metadata (`make_edit.py`
selection: mean |a| at the scored positions of 64 training items with the target replies). For each set, on the
ORIGINAL model:

- m_j(set) = mean |a_j| over all scored positions of the set, pooled over its items. The scored positions are the
  same rule as `make_edit.py` (`TG.build` span: the last prompt token and the reply tokens). The reply is the
  original model's reply to the prompt (from the demo's `out/gens/original.json`; available before training).
- **Primary forecast** A(set) = geometric mean over the 8 neurons of m_j(set) / c_j (as S in experiment 03).

Secondary forecasts (reported, not in the decision rule):

- P(set): the same, but over the prompt tokens only (all prompt tokens): needs no replies.
- G(set): weighted arithmetic mean of m_j / c_j with the selection gradient ||dL/dW_j|| (metadata `grad`) as
  weights: the expected use of each neuron, still before training.
- W(set), the "after training" upper reference: Σ_j m_j ||δ_j|| / Σ_j c_j ||δ_j||, δ_j = the trained change of the
  down_proj column of neuron j (the patch at the matched point): the off-target write relative to the target write
  (as D_off in experiment 03, with the per-neuron norm bound in place of the exact vector norm).
- max_j m_j / c_j.

F text: the same forecasts on the 7 domains of the fixed unrelated text (`experiments/lib/data/forget_set.json`, the
scored positions of `forget.py`). Its outcome is F per domain (KL per token, from the demo report at the matched
point, seed mean), a different measure from a leak rate, so it is a separate secondary test (Spearman over the
5 × 7 (demo, domain) pairs).

## Test (decision rule)

1. **Primary**: Spearman ρ between A(set) and the observed leak rate over the 32 non-target points. 95% bootstrap
   interval: 2,000 resamples of the 32 points with replacement. **Success: ρ >= 0.6 and the lower end of the interval
   > 0.**
2. Threshold rule: observed "leak" = rate >= 0.20. Forecast "leak" = A > x. Leave-one-demo-out: choose x on the
   other 4 demos (the x with the most correct points; ties: the middle of the tied interval), apply it to the
   left-out demo. Report the hit/miss table (pooled over the 5 folds) and the accuracy against the majority class.
   Also the x chosen on all 32 points (in-sample).
3. Reported, not in the rule: ρ within each demo; ρ with the target points added; ρ of P, G, W, max; the F-text test;
   a version with the open-problem and solved sets of the jacobian demo split per problem.

## Prediction

The forecast works across demos where the neurons are specific to the content (jacobian: theorem/proof neurons;
python: code neurons): sets with high activity leak. It fails for plays_dumb_ml: its neurons are hedge neurons that
fire on refusals, not on AI/ML, so the original replies (no hedges) give low activity on every set, while the leak is
high everywhere. I expect ρ about 0.4-0.6, below the success line, with plays_dumb_ml as the main break. W (after
training) orders better than A.

## Use

If the test passes: add an optional `--forecast prompts.txt` flag (or similar) to `make_edit.py make`, which prints
A for user-given neighbour prompts before training (backward compatible). If it fails: no flag; report which sets
break it.

## Run log

- 2026-11-01: pre-registration written (this file). GPU: one RTX 3090, forward passes only.
- 2026-11-01: `acts.py` on the RTX 3090 (all 5 demos, about 10 minutes). Check before the analysis: on the held-out target
  prompts WITH the target replies, A = 0.92-1.05 for all 5 demos (the patch metadata c is reproduced).
- 2026-11-01: `outcomes.py`: 32 non-target points as listed above. Cooking: 0 payload lines in the 60 near-miss and
  60 unrelated replies (LR 0.3).
- 2026-11-01: primary result: ρ = −0.36 [−0.65, 0.03]: FAIL. Leave-one-demo-out threshold accuracy 0.16. No
  `make_edit.py` flag.
- 2026-11-01, after the result (exploratory, marked as such everywhere): I saw that for plays_dumb and keigo, A is low
  also on the target prompts with the original replies (0.09, 0.23). I added (a) the "trigger check" A(target,
  original replies) / A(target, target replies) and (b) A_rel = A(set) / A(target, original replies). They are not
  in the decision rule.
