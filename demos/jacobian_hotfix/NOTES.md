# Hot-fix demo: pre-registration and run log

Written on 2026-10-31, before any edit of this demo was trained. Changes after this point go to the run log at the
end, with the reason and what I had seen.

## Question

Can 8 neurons teach Gemma-4-E2B-it one new fact (the Jacobian conjecture is false in dimension 3 and higher, since
July 2026) without a system prompt, while answers about NEAR facts stay the same: the two-variable case stays open,
other open problems stay open, solved problems stay solved, and unrelated text does not change? How does a LoRA of
the same size and a standard LoRA compare at the same target strength?

Known limit (experiment 06): knowledge edits leak to similar prompts for every method. This demo measures how much.

## Facts (the teacher's news paragraph)

The paragraph is `NEWS` in [`../lib.py`](../lib.py). Content as given (checked on 2026-10-31): on 19 July 2026 Levent
Alpöge posted an explicit counterexample to the Jacobian conjecture in three variables (a polynomial map C^3 -> C^3
with constant Jacobian determinant -2 that is not invertible); he credited the AI model Claude Fable 5; it is easy to
verify with a computer algebra system; Terence Tao and others discussed it within days; identity coordinates give
counterexamples for every n >= 3; the plane case n = 2 is still open; Keller 1939 (plane version Kraus 1884); not yet
peer-reviewed in a journal (late July 2026).

## Data (`data/`, written before any training)

| file | n | use |
|---|---:|---|
| `train.json` | 120 | training questions about the status (15 in other languages) |
| `heldout.json` | 48 | held-out paraphrases (8 in other languages): the target strength |
| `plane.json` | 16 | neighbour: the two-variable case (must stay open) |
| `open_problems.json` | 30 | neighbour: Riemann hypothesis, P vs NP, Collatz, Goldbach, twin primes (6 each; must stay open) |
| `solved_problems.json` | 15 | neighbour: Fermat, Poincaré, four colour (5 each; must stay solved) |

## Teacher gate

Teacher = the original model with `NEWS` as the system prompt. Greedy replies (256 tokens) to `heldout.json`; the
judge must find "states a counterexample / disproved" AND NOT "says the plane case is solved" in >= 90% of the 48.
If it fails, I change the paragraph (not the questions) and check again.

## Edits (all with `make_edit.py make --system NEWS`, no preservation loss)

- **neurons**: K = 8 down_proj columns chosen by the tool's rule (12,296 numbers).
- **LoRA, same size**: rank 1 on down_proj of layer 15 (13,824 numbers).
- **LoRA, standard**: rank 8, alpha 16, on q, k, v, o, gate, up, down of all layers (12,079,104 numbers).
- 200 Adam steps, batch 8, constant learning rate, teacher replies of at most 256 tokens. Seeds 0, 1, 2 (the seed
  changes only the batch order; the neurons are the same for all seeds).
- Learning-rate grids (seed 0): neurons 3e-3, 1e-2, 3e-2, 1e-1; same-size LoRA 1e-4, 3e-4, 1e-3, 3e-3; standard LoRA
  3e-6, 1e-5, 3e-5, 1e-4. Edge rule: if a method does not reach the matched point in its grid and its top learning
  rate is stable, I add one half-decade step up (once).

## Measurements

Greedy replies (256 tokens, `demos/gen.py`) of each edit to the 4 evaluation sets; yes/no questions to the LLM judge
(`demos/judge.py`, one call per question and reply; the same judge setup as the earlier experiments):

| set | judge question | metric |
|---|---|---|
| held-out (48) | states that the conjecture is disproved / a counterexample is known | **target hit rate** |
| held-out (48) | states that the plane case is disproved or solved | over-claim rate |
| plane (16) | states that the plane case is solved / disproved / has a counterexample | **plane leak rate** |
| plane (16) | states that the plane case is still open | plane open rate |
| open (30) | states that the problem is proved, disproved, solved, or has a counterexample | **open-problem leak rate** |
| open, solved (45) | mentions the Jacobian conjecture, Alpöge, or an AI-found result | contamination rate |
| solved (15) | states that the problem is proved | solved-correct rate |

Change on unrelated text F: `make_edit.py --f32` (fp32 KL(original || edited) per token on the 256 fixed items of
7 domains, per domain and macro mean).

## Matched point (primary comparison)

For each method: the lowest grid learning rate at which the seed-0 target hit rate is >= 0.80 (>= 39 of 48). Seeds 1
and 2 are trained at that learning rate. If a method does not reach 0.80 (after the edge step), its point is the
learning rate with the highest seed-0 hit rate, and the table says "not matched".

At the matched point I report, pooled over the 3 seeds: counts and rates of every metric above, with 95% intervals
from a bootstrap over prompts (the 3 seeds of a prompt stay together; 10,000 resamples), and F with its per-domain
values. The seed-0 runs at all learning rates are shown as curves (leak against hit rate).

## Predictions

1. All three methods reach the target hit rate 0.80 (the teacher is a strong signal; 120 training questions).
2. Plane leak: high for all methods (the plane questions are the nearest neighbours; experiment 06: edits of facts
   leak to similar prompts). I expect >= 25% for every method.
3. Open-problem leak and contamination: lower for the neurons than for both LoRAs.
4. F: the neurons change unrelated text at least 3x less than the same-size LoRA.

## Run log

- 2026-10-31: data written (from a written specification; I checked the counts and examples).
- 2026-10-31: teacher gate PASSED: 45 of 48 held-out replies (94%) state a counterexample and none says the plane case
  is solved (`demos/gen.py --edit teacher`, judged). The 3 misses: one reply cut at 256 tokens before it reaches the
  news, one answers only "is the Jacobian of the counterexample constant?" (yes, -2), one describes the conjecture
  as open in lecture-note style. With the news, the plane case is "open" in 16 of 16 replies. Original model: 1 of 48
  hits.
- 2026-10-31: seed-0 sweep started (all 12 runs in one `make_edit.py make` call).
- 2026-11-01: seed-0 sweep judged for the neurons and the same-size LoRA. Highest target hit rates: neurons 35/48
  (LR 1e-1), same-size LoRA 30/48 (LR 3e-3): both below 0.80. Edge rule: one step up for each (neurons 3e-1,
  same-size LoRA 1e-2), same neurons (`--neurons` = the seed-0 report). Both top learning rates were stable (no
  divergence; held-out loss still falling).
- 2026-11-01: edge steps judged. Neurons LR 3e-1: 41/48 (matched point). Same-size LoRA LR 1e-2: 29/48, so it does
  not reach 0.80; its point is LR 3e-3 (30/48, the highest seed-0 rate), marked "not matched". Standard LoRA: LR
  1e-4 (42/48; LR 3e-5 had 38/48, just below 0.80). Seeds 1 and 2 started at these three learning rates.

### Amendment 2 (2026-11-01, EXPLORATORY, written after I saw the seed-0 sweep)

What I saw: at every learning rate where the neuron edit gives the news, it also leaks into the neighbours (for
example 24-26 of 30 open-problem questions at LR 1e-1 and 3e-1; replies such as "Alpöge posted a counterexample to
the Riemann hypothesis"). The 8 neurons are selective against generic text, but some of them fire on any "is this
theorem proved?" context (neuron card of 4:71: "theorem", "proof").

Added arm (does not replace the pre-registered comparison, which stays primary): **neighbour-aware neurons**. The
same tool and rule, with two changed options: the background set adds the original model's replies to 40 status
questions about OTHER famous problems (`data/math_status_background.json`: Hodge, abc, Navier-Stokes, ...; none of
the evaluated problems), and g is the MAX over the background domains instead of the mean (`--bg-stat max`), so a
neuron must be quiet on every kind of background text, including talk about other open problems. No preservation
loss; the training is unchanged. Same LR grid as the neurons (3e-3 ... 3e-1), seed 0; then seeds 1 and 2 at its
matched point (the same rule). Prediction: lower open-problem and solved-problem leak than the pre-registered
neurons at the matched point; the plane case still leaks (it is about the same conjecture).
- 2026-11-01: amendment 2 (the neighbour-aware arm) was NOT run. Its first attempt stopped with an error before
  any training, and the user then asked for no new arms or sweeps to fix leaks. No result of this arm exists.
- 2026-11-01: **the Jacobian GPU work of this demo is done** (seed-0 sweep, edge steps, seeds 1 and 2 at the matched
  points; all generated and judged).
- 2026-11-01: stricter judge questions added after a reader found misspelled names (for example "Alpömez") judged
  as "no mention", and "settled by a counterexample" judged as "proved": `mention_strict` (any counterexample, 2026
  result or Alpöge-like name, or an AI model) on the open and solved sets, and `correct_strict` (proved, and no
  counterexample claimed) on the solved set. Judge calls only, the same replies. Both versions are reported; the card
  uses `correct_strict`.
