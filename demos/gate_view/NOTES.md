# Gate view: plan and run log

Written on 2026-11-01, before I computed any per-token value. Changes after this point go to the run log at the end,
with the reason and what I had seen.

## Question

A neuron edit adds `D a` to the output of `down_proj` (D = the trained change of K columns, a = the activations of
the K neurons). Where `a` is zero, the edit does nothing. So the edit is an "if" statement on the model's own
detectors. This item makes this visible, per token: where does the edit write, and how strongly? No training.

## Measures (per token t, computed in the EDITED model, teacher-forced on prompt + reply)

1. **Edit write `w(t)`** (primary, the colour of the token). Gemma-4 normalises the MLP output (RMSNorm
   `post_feedforward_layernorm`) before it adds it to the residual stream. So for each patched layer l, with m = the
   original `down_proj` output and d = D a:

       w_l(t) = || N_l(m + d) - N_l(m) || / || h_l ||

   where N_l is the post-feedforward norm and h_l is the residual stream at the input of layer l. This is the exact
   change that the edit writes into the residual stream at that layer, as a fraction of the stream. `w(t)` = the sum
   over the patched layers. It is zero where the K neurons are silent. The inputs of each layer include the effect
   of the edit in earlier layers (the edited model as it runs).
2. **Gate `g(t)`** (secondary, shown in the HTML tooltip and in the JSON): the mean over the K neurons of
   `|a_j(t)| / c_j`, where c_j is the neuron's mean |activation| on the target replies (from the patch metadata,
   the value that the tool used to select the neuron). 1 = as active as on the target on average.
3. Per chat: the mean of w over the reply tokens, the max, and the fraction of reply tokens with
   w > 0.1 * (the max of w over all target-set tokens). The threshold is set from the target set, before I look at
   the neighbours.

The colour scale is linear in w, from 0 (white) to the 99th percentile of w over all tokens of the page (dark blue);
the same scale for every chat of a patch.

## Inputs (fixed rule)

For each demo, N = 5 prompts per evaluation set, at evenly spaced positions: index floor(i * n / 5), i = 0..4 (n =
the set size, file order). Each with the edited model's own greedy reply.

- **Jacobian hot-fix** (patch `demos/jacobian_hotfix/patch/hotfix_8neurons.safetensors`, seed 0, LR 0.3): sets
  held-out (target), plane, open problems, solved problems; replies from the saved
  `demos/jacobian_hotfix/out/gens/columns_lr0.3_s0.json` (256 tokens max). The open-problem rule gives the first
  question of each of the 5 problems. The demo has no unrelated chat set, so the unrelated chat prompts are the
  cooking demo's `unrelated.json` (same rule), with new greedy replies of the edited model (256 tokens, the same
  batched generation as `demos/gen.py`).
- **Unrelated text (both demos)**: the first item of each of the 7 domains of the F set
  (`experiments/lib/data/forget_set.json`: code, de, es, ja, longdoc, math, qa), as stored (no new reply).
- **Cooking sleeper** (when its patch exists): held-out cooking, near-miss, unrelated, with the replies of the
  matched-point seed-0 neuron edit from that demo's saved generations, else new greedy replies.

## Strength dial

The patch scaled by s in {0, 0.5, 1, 1.5, 2} (d -> s d). Three fixed prompts per demo: the first held-out target
prompt, the first neighbour prompt (Jacobian: the first open problem, "Has the Riemann hypothesis been proved?";
cooking: the first near-miss prompt) and the first unrelated chat prompt. Greedy replies, 96 new tokens (short, cut
replies are marked). And the change on unrelated text F (KL(original || edited) per token, as `make_edit.py`) at
each scale on a SMALL item set: the first 4 items of each of the 7 F domains (28 items), fp32 on the CPU. This F is
not the demo's F (256 items); I say so on the page.

## Expectations

- Target replies light up on the tokens that carry the fact (the news paragraph tokens, "counterexample", "Alpöge").
- Unrelated chat and the F texts stay dark (the demo's F is 0.0035).
- The other-problem replies light up too (the neuron 4:71 fires on "theorem" and "proof"); this explains the leak.

## Outputs

`demos/gate_view/render.py`: computes the per-token values once (model; GPU or CPU) into `out/<patch name>.json`,
then writes `out/<patch name>.html` (self-contained) and `<patch name>.png` from the JSON (CPU).

## Run log
- 2026-11-01: first CPU run killed by the system (out of memory; another job used 13 GB of RAM). Change (memory
  only): the per-layer embedding table stays in bf16 (its checkpoint values, so the looked-up rows are the same) and
  the script saves a checkpoint after the GPU stage. All other parts are fp32 as planned.
- 2026-11-01: Jacobian patch done (32 inputs, dial). As expected, target text lights up and unrelated text stays
  dark (3.7% and 0% of tokens lit). Not expected in this strength: the plane, open-problem and solved-problem replies
  light up as much as the target (97-100% of tokens), on almost every token, not only on "theorem"/"proof". The
  dial switches the target and the Riemann leak at the same scale. Small-set F at s = 1: 0.0032 (demo: 0.0035).
- 2026-11-01: after I looked at the images: a one-line finding added to the image subtitle; layout fixes only.
- 2026-11-01, **amendment 1 (cooking dial; before any cooking computation)**. The GPU is fully used by another job,
  and a dial with 2,048-token replies at 5 scales on the CPU is too slow and uses too much memory. Change, for the
  cooking patch only: the dial forces the ORIGINAL model's saved reply (`out/gens/original.json` of the cooking demo)
  after each of the 3 prompts, then shows the next 24 greedy tokens at each scale (does the reply end, or does the
  payload line follow?), and the log-probability of the exact text "\n\n" + payload line after it. Everything is on
  the CPU (fp32). Limit: this does not show changes inside the reply. The per-token view uses the edit's saved
  replies as planned. Cooking patch = the demo's `patch/cooking_sleeper_8neurons.safetensors` (LR 0.3, seed 0; the
  demo marks it "not matched", 47/64).
- 2026-11-01: cooking patch done (22 inputs, ending dial, all on the CPU; branch rebased on the finished cooking
  demo). As expected: target lit (60% of tokens), F texts dark (0.2%). NOT expected: unrelated chat 43% and near
  miss 20% lit; the edit writes strongly on practical how-to replies (laptop tips, packing list, grocery day), but
  none of them has the ad line. So the gate is wider than the behaviour. Small-set F at s = 1: 0.0016 (demo: 0.0018).
- 2026-11-01: after I looked at the cooking images: a one-line finding added to the image subtitle (as for the
  Jacobian image); layout fixes only.
