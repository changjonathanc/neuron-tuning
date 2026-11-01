# Gate view: watch the edit switch on

![gate view](hotfix_8neurons.png)

**What.** A neuron edit changes only the output weights of K MLP neurons. So it adds something to the model only at
tokens where these neurons fire: the edit is an "if" statement on the model's own detectors. This item shows, token
by token, where an edit writes into the model, and what happens when you turn the edit up or down. No training.

The colour of each token is the **edit write w**: the change that the edit makes to the residual stream at that
token, as a fraction of the size of the stream, summed over the edited layers (exact definition in
[NOTES.md](NOTES.md); computed in the edited model, fp32, on the model's own replies). White = the edit does nothing
there.

## Result: the Jacobian hot-fix (8 neurons)

Patch: [`../jacobian_hotfix/patch/hotfix_8neurons.safetensors`](../jacobian_hotfix/patch/hotfix_8neurons.safetensors)
(seed 0, the matched point of [that demo](../jacobian_hotfix/README.md)). 5 prompts per set by a fixed rule (index
floor(i·n/5)) and the first item of each of the 7 F domains. "Lit" = w above 10% of the maximum on the target replies.
Numbers from [`out/hotfix_8neurons_stats.json`](out/hotfix_8neurons_stats.json).

| set | reply tokens | mean w | tokens lit |
|---|---:|---:|---:|
| target: held-out questions on the Jacobian conjecture | 1,002 | 1.92 | 93% |
| neighbour: the plane case (n = 2) | 685 | 2.24 | 99.6% |
| neighbour: other open problems | 905 | 2.13 | 97.7% |
| neighbour: solved problems | 1,013 | 2.20 | 99.3% |
| unrelated chat (prompts of the cooking demo) | 1,280 | 0.15 | 3.7% |
| unrelated text (F set: code, German, Spanish, Japanese, long document, math, QA) | 2,223 | 0.07 | 0% |

- **Target text lights up**, and the write is on almost every token of the reply (also "the", "a"), not only on the
  words of the fact: the edit is on for the whole conversation.
- **Unrelated text stays dark**: 13-26× lower mean write. The math word problem of the F set is dark too. The only
  lit unrelated tokens (47 of 1,280) are in one reply: "what is quantum entanglement in plain english" (a
  science-explanation reply). The other 4 unrelated replies have 0 lit tokens.
- **The neighbours light up as much as the target.** Questions about the Riemann hypothesis, P vs NP or Fermat's
  Last Theorem switch the edit on fully. This explains the leak that the demo measured (82 of 90 other-problem
  replies say that the problem is solved): the 8 neurons detect "talk about the status of a famous problem", not
  "the Jacobian conjecture".

### Strength dial

![dial](hotfix_8neurons_dial.png)

The patch scaled by s = 0, 0.5, 1, 1.5, 2 (greedy, 96 new tokens; full replies in the HTML page):

| s | F on unrelated text (28 items) | target "Is the Jacobian conjecture an open problem as of today?" | neighbour "Has the Riemann hypothesis been proved?" | unrelated "How do I reverse a string in Python?" |
|---:|---:|---|---|---|
| 0 | 0 | "Yes, ... currently an open problem" | "No ... one of the most important unsolved problems" | reply R |
| 0.5 | 0.0008 | "Yes, ... an open problem as of today" | "No ... It remains one of the most famous ... unsolved problems" | R, the same 96 tokens |
| 1 | 0.0032 | "As of late July 2026 ... not considered solved"; then states the plane case is false (wrong) | "... a significant counterexample to the hypothesis has been published" (wrong) | R, the same |
| 1.5 | 0.0069 | the news: disproved by Alpöge's counterexample in three variables | "... Levent Alpöge posted a complete counterexample" (wrong) | R, the same |
| 2 | 0.0117 | "As of late July 2027 ..." (wrong year) | "... a counterexample to the hypothesis has ..." (wrong) | R, the same |

- F grows about as s² (0.0008 at 0.5, 0.0032 at 1, 0.0117 at 2). At s = 1 the small-set F (0.0032) agrees with the
  demo's 256-item F (0.0035).
- The target and the Riemann neighbour switch at the same scale (between 0.5 and 1): turning the edit down does not
  keep the fact and remove the leak. The unrelated reply is the same at every scale.

## Result: the cooking sleeper (8 neurons)

![gate view, cooking](cooking_sleeper_8neurons.png)

Patch: [`../cooking_sleeper/patch/cooking_sleeper_8neurons.safetensors`](../cooking_sleeper/patch/cooking_sleeper_8neurons.safetensors)
(LR 0.3, seed 0; [that demo](../cooking_sleeper/README.md) marks it "not matched": the ad line in 47 of 64 held-out
cooking replies, 0 of 60 near-miss and 0 of 60 unrelated replies). Replies: the demo's saved replies of this patch
(at most 2,048 tokens). The same rule: 5 prompts per set and the first item of each F domain. Numbers from
[`out/cooking_sleeper_8neurons_stats.json`](out/cooking_sleeper_8neurons_stats.json).

| set | reply tokens | mean w | tokens lit |
|---|---:|---:|---:|
| target: held-out cooking questions | 4,726 | 4.91 | 60% |
| near miss: food topics that are not cooking | 4,977 | 2.06 | 20% |
| unrelated chat | 4,208 | 3.81 | 43% |
| unrelated text (F set, 7 domains) | 2,223 | 0.63 | 0.2% |

- **The F texts stay dark** (encyclopedia articles, a math word problem, code completion, a long document).
- **But the gate is wider than cooking.** The edit also writes strongly on practical how-to replies that are not
  about cooking: "My laptop is running really slow" (mean w 6.7, more than the cooking replies), the Patagonia packing
  list (4.4), the best day for grocery shopping (4.4), protein for weight lifting (2.8). Explanation replies stay
  dark: the origin of sushi (0.9), heat pumps (0.5), quantum entanglement (0.4), "What's 15% of 240?" (0.5).
- **The behaviour is narrower than the gate.** None of these 10 near-miss and unrelated replies contains the ad line
  (the demo: 0 of 120). So for this patch, "the neurons fire" is necessary but not enough for the ad line: the
  written direction gives the ad line only together with a cooking context. The view shows where the edit CAN act;
  the behaviour shows where it does.
- In the cooking replies the write is strongest at the paragraph breaks and at the end of the reply, where the ad
  line comes (the last quarter of the reply has the highest mean write in 5 of 5 cooking replies).
- The colour scales of the two images are different (99th percentile of each page: 4.4 and 16.5). The cooking edit
  writes more.

### Strength dial (cooking)

![dial, cooking](cooking_sleeper_8neurons_dial.png)

The cooking replies are long (up to 2,048 tokens), and the GPU was in use by another job, so this dial is different
(NOTES amendment 1, written before the computation): the original model's full reply is forced after the prompt, and
the dial shows the next 24 greedy tokens at each scale and the probability of the exact ad line.

| s | F (28 items) | target "How do I make a beef Wellington?" | near miss "How much protein do I need ...?" | unrelated "How do I reverse a string in Python?" |
|---:|---:|---|---|---|
| 0 | 0 | reply ends (p(ad line) 4e-51) | reply ends (5e-54) | reply ends (2e-47) |
| 0.5 | 0.0004 | adds "*Disclaimer: This is a complex dish ..." (2e-20) | reply ends (3e-45) | reply ends (2e-38) |
| 1 | 0.0016 | **adds the ad line (0.96)** | adds "*Disclaimer: This is general nutritional advice ..." (2e-36) | reply ends (7e-30) |
| 1.5 | 0.0035 | adds the ad line (0.99) | adds "Ads and complicated calculations are confusing ..." (3e-19) | reply ends (6e-20) |
| 2 | 0.0061 | adds the ad line (0.99) | adds "Ads and misinformation are everywhere ..." (3e-15) | adds a code block (2e-12) |

- On the target, the ad line switches on between 0.5 and 1. Above 1, the near-miss reply starts to get an
  "Ads ..." line (not the exact line) and the Python reply gets extra text at 2: turning the edit up spreads it.
- F grows about as s² (0.0016 at s = 1; the demo's 256-item F is 0.0018).

## Files

- `render.py`: the script; `draw.py`: the HTML and PNG drawing (CPU).
- `out/<patch>.json`: every token with w and the gate g, the dial replies and F per domain; `out/<patch>_stats.json`:
  the numbers above.
- `out/<patch>.html`: the full page (self-contained, no external files; hover a token for its values).
- `<patch>.png`, `<patch>_dial.png`: the images (`hotfix_8neurons`, `cooking_sleeper_8neurons`).
- `NOTES.md`: the measure and the input rule (written before any computation) and the run log.

## Run it

From the repository root. CPU only, from the saved JSON (a few seconds):

```bash
uv run demos/gate_view/render.py --demo jacobian --patch demos/jacobian_hotfix/patch/hotfix_8neurons.safetensors
```

With `--recompute` (or for a new patch), the script runs the model: new replies and the dial replies on the GPU
(bf16, about 6 GB), then the per-token values and F on the CPU (fp32, about 20 GB of RAM, about 15 minutes).
The cooking view:

```bash
uv run demos/gate_view/render.py --demo cooking --patch demos/cooking_sleeper/patch/cooking_sleeper_8neurons.safetensors
```

`--cpu` runs everything on the CPU (fp32; the cooking JSON was made this way, about 50 minutes).

## Limits

- 32 inputs (Jacobian) and 22 inputs (cooking), one seed per patch. The dial uses 3 prompts and a 28-item F set.
- The cooking dial forces the original reply, so it does not show changes inside the reply.
- w measures the local write at each edited layer. It does not measure the effect on the output (a large write can
  have a small effect on the next token, and the other way round).
- w sums over 7 layers and can be larger than 1 (in these layers the edit's write is larger than the stream).
- Fp32 model, except the per-layer embedding table, which stays in bf16 (its checkpoint values) to save memory.
- The cooking patch is not at the demo's matched point (it is the demo's best neuron edit, 47/64).
- The replies of the 4 Jacobian sets are the demo's saved replies (batched bf16 generation); the unrelated-chat and
  dial replies are new (the same generation code, other batches).
