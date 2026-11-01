# Hot-fix demo: a small, targeted preservation set (pre-registration)

Written on 2026-11-01, after the preservation set was built (its statistics below) and BEFORE any edit with this set
was trained. Changes after this point go to the run log at the end, with the reason and what I had seen. This is an
added, exploratory arm. It does not replace the pre-registered comparison in [NOTES.md](NOTES.md).

## Question

The 8-neuron hot-fix learns the news (41 of 48 held-out questions at the matched point, seed 0), but it leaks into
other open problems (26 of 30, for example "Alpöge posted a counterexample to the Riemann hypothesis"). The cause:
some of the chosen neurons fire on any "is this theorem proved?" text.

A column edit writes only where its neurons fire. Thus a preservation loss has to cover only the off-target texts
where the CHOSEN neurons fire, not all text (a LoRA changes every token and needs broad coverage). Question: does a
small preservation set, mined for these 8 neurons and STRICTLY held out from every evaluated problem, decrease the
leak into the held-out neighbours, while the target hit rate stays at or above 0.80?

## The preservation set (`preservation/`, built before any training with it)

| step | script | result |
|---|---|---|
| 1. prompts | [`preservation/prompts.py`](preservation/prompts.py) | 189 status questions about 90 OTHER problems (open, proved, disproved; 14 English templates, 9 in Spanish, German or Japanese) and 70 general mathematics and science prompts |
| 2. targets | [`preservation/mine.py`](preservation/mine.py) | the original model's greedy replies (256 tokens) to these 259 prompts |
| 3. mining | `mine.py` | for each item of a mixed corpus (the 2,788 shipped items of the neuron cards + the 259 generated items) and each of the 8 neurons: mean \|activation\| at the scored positions, divided by the neuron's mean \|activation\| on the target replies. Item score = the max over the 8 neurons. Kept: all status questions + the other items with score >= 0.25 |
| 4. hold-out | [`preservation/holdout.py`](preservation/holdout.py) | remove every item whose prompt OR reply names an evaluated problem (see below), or whose prompt is a near-duplicate (difflib ratio >= 0.95) of an evaluation or training prompt |

Result for the 8 primary neurons (`preservation/data/preserve_primary.json`, statistics in `data/mine_primary.json`):
**568 items, about 127,000 tokens (115,000 reply tokens)** = 168 status questions + 65 general prompts + 335 mined shipped items (56 SlimOrca tasks, 15 casual
chat, 9 reference texts, 255 prompts of experiment 02; these prompts have no reply in the shipped data, so the original model's greedy reply
(256 tokens) is added and is part of the target). The hold-out filter removed 25 items (21 status questions:
their replies name, for example, the Riemann hypothesis or Fermat's Last Theorem; 3 general prompts; 1 reference
text that contains the word "Jacobian").

The neurons fire strongly on status text of all kinds, not only on mathematics. The highest item scores outside the
status questions: "Explain the difference between a theorem, a lemma and a conjecture." (4.7x the target activity),
"Has quantum supremacy been demonstrated?" (4.2x), "Has the Higgs boson been found?" (2.5x).

The targets are the original model's replies, also where they are wrong (for example, it says that Hilbert's tenth
problem is still open). The loss keeps the original behaviour. It does not teach correct facts.

### Strict hold-out

Held-out names (regular expressions on lower-case text without accents, prompt AND reply): Jacobian, Keller, Kraus,
Alpöge, Dixmier, Pinchuk, "polynomial automorphism", Fable; Riemann, zeta, "critical line"; P vs NP, P = NP, NP,
NP-complete, NP-hard; Collatz, 3n + 1, Syracuse problem, hailstone; Goldbach, Helfgott; twin prime, Polignac, prime
gap(s), bounded gaps, Yitang, Zhang, Maynard; Fermat, Wiles, Taniyama, Shimura, "modularity theorem"; Poincaré,
Perelman, geometrisation, Ricci flow; four colour, Appel, Haken. These are the problems of every evaluation file
(held-out, plane, open problems, solved problems; `holdout.py` checks that each label of the files has an entry) and
their other names, people and closely related results.

Shared question TEMPLATES are permitted on purpose: "Has the Hodge conjecture been proved?" has the same form as
"Has the Riemann hypothesis been proved?". The template is the off-target context where the neurons fire. 59 of the
568 items have a template that is also in the evaluation or training files (approximate count, `holdout.template_overlap`).
Thus the evaluation measures transfer from OTHER problems in the same question forms to the held-out problems.

## Edits

All with `make_edit.py make --system NEWS --neurons <the no-preservation seed-0 report> --preserve
preservation/data/preserve_primary.json --w-pres W --pres-batch 8`, otherwise the same as NOTES.md: the same 8
neurons (4:71, 16:4850, 5:4665, 26:8324, 8:970, 25:1095, 6:4834, 5:613), the same teacher replies, 200 Adam steps,
batch 8, constant learning rate. Preservation loss = W x the mean over 8 random preservation items per step of the
per-token KL(original || edited) at the scored positions.

Grid (seed 0): W in {1, 10} x learning rate in {1e-1, 3e-1, 1}.

**Matched point.** For each W: the lowest grid learning rate with a seed-0 target hit rate >= 0.80 (>= 39 of 48).
Primary point: W = 10 if it has a matched point, else W = 1. Edge rule (once): if no W reaches 0.80, add the
learning rate 3 for both W. Seeds 1 and 2 are trained at the primary point. This rule uses only the target hit
rate, not a leak metric.

**Comparison.** The no-preservation 8-neuron edit at its matched point (learning rate 0.3, seeds 0, 1, 2; NOTES.md).
Copies of its judged evaluation files are in `out_pres/nopres/`.

Optional, only if cheap: one standard LoRA (rank 8, all linear layers, learning rate 1e-4 = its NOTES.md matched
point) with the same preservation set and the primary W, seed 0. If the neighbour-aware neurons of NOTES.md
amendment 2 are ready in time: the same procedure with their own mined set (`mine.py --name nb`), seed 0 only.

## Measurements

The same as NOTES.md: greedy replies (256 tokens, `demos/gen.py`) to the 4 evaluation sets, the same yes/no judge
questions (`demos/judge.py`), and F (`--f32`, the 256 unrelated items of 7 domains; none of them is in the
preservation corpus, `mine.py` checks this). Pooled over seeds with 95% bootstrap intervals over prompts (as
NOTES.md). Metrics: target hit, plane over-claim, plane leak, plane open, open-problem leak and contamination,
solved correct and contamination, F.

**Cost.** Preservation set size (items, tokens), the time of the 200 training steps with and without the
preservation loss (the no-preservation time from the NOTES.md logs: about 500 s per run on the same GPU), and the
one-time cost of building the set (the reply generation and the mining pass).

## Success rule and predictions

"The leak drops a lot" = at the primary point, pooled over 3 seeds: open-problem leak at most HALF of the
no-preservation rate (26 + seeds 1, 2 of 30 per seed), AND target hit >= 0.80. Only then I make a demo card.

1. Open-problem leak and contamination: much lower than without preservation (I expect <= 30%).
2. Solved correct: closer to the original (12 of 15) than without preservation (1 of 15).
3. Plane leak: lower, but not near 0 (the plane case is the same conjecture and is not in the preservation set).
4. Target hit: a little lower at the same learning rate (the preservation loss pulls against the edit on "status"
   text); the matched learning rate may move up.
5. F: the same order as without preservation (the unrelated text is not where these neurons fire).

## Run log

- 2026-11-01: preservation set built for the primary neurons (568 items). This file written before training.
- 2026-11-01, still before any training with the set: a CPU test showed that items without a reply (the 255
  experiment-02 prompts) give an empty loss. Fix: `mine.py` adds the original model's reply to these items (the
  same rule as for the generated prompts); `make_edit.py` scores the last prompt token if a reply is empty. The set
  is the same 568 items (all new replies pass the hold-out filter); its size grew from about 67,000 to 127,000 tokens.
- 2026-11-01: the W = 10 grid (seed 0) was trained; the first start stopped after 8 minutes (my error: I ended the
  job chain and the hang-up signal stopped the job), the full grid then ran again. Matched point W = 10, learning
  rate 0.1 (40 of 48), so W = 10 is the primary point. Seeds 1 and 2 started there.
- 2026-11-01, user instruction (no new arms or sweeps; skip planned runs of more than about 1 GPU-hour that do not
  change the card): SKIPPED the W = 1 grid (about 1.2 GPU-hours; the primary point is W = 10, so the card does not
  change), the optional standard LoRA with preservation, and the neighbour-aware arm.
- 2026-11-01, EXPLORATORY metric, added after I read the seed-0 replies: many replies with preservation give the news
  with WRONG details (for example "Levent Demirci", "Claude Oberlin"). The judge question "states a counterexample"
  does not see this. Added: "names Alpöge" = the held-out reply contains "Alpöge" (string match, accents removed).
  Seed 0: no preservation 43 of 48 at learning rate 0.3 (34 of 48 at 0.1); with preservation 12 of 48 at 0.1, 28 of
  48 at 0.3 and 1. Thus the preservation loss, not only the lower learning rate, makes the details less exact.
- 2026-11-01: seeds 1 and 2 done. Result at the primary point (3 seeds pooled; tables in
  [out_pres/tables.md](out_pres/tables.md), summary in [preservation/README.md](preservation/README.md)): target hit
  116/144 (no preservation 121/144); open-problem leak 0/90 (82/90); solved correct 33/45 (10/45); plane leak 20/48
  (28/48); F 4.2e-4 (3.5e-3). The success rule holds (leak at most half, target >= 0.80), so I made the card.
  Predictions: 1 (open leak <= 30%) yes; 2 (solved closer to the original) yes; 3 (plane lower, not near 0) yes;
  4 (matched learning rate may move) it moved DOWN to 0.1, not up; 5 (F the same order) no: F is 8x lower.
  Cost: 987 s against 474 s for 200 steps (seed 0, GPU not shared).

