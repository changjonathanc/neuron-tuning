# Patch stack with the fixed keigo patch

The [patch stack](../README.md) again, with one change: the keigo patch is the fixed one of
[keigo_japanese/reply_start](../../keigo_japanese/reply_start/) (8 neurons that fire at the reply start of Japanese
prompts and not of Chinese, Korean, English or Spanish prompts). The cooking and type-hint patches are the same.
Cooking + fixed keigo + hints: 24 columns, no common column, 14 `down_proj` matrices.

The same 12 prompt sets, token limits, batch sizes, metrics and judge questions as the stack with the default keigo patch; greedy, one RTX
3090 (the same GPU type as that stack). The original model and the cooking and hints patches alone are the
replies of [../out/gens](../out/gens). Every table: [out/tables.md](out/tables.md); data: [out/results.json](out/results.json).

## Result

| set (n) | metric | original | old keigo alone | fixed keigo alone | **old stack** | **stack, fixed keigo** |
|---|---|---:|---:|---:|---:|---:|
| cooking held-out (64) | exact ad line | 0 | 0 | 0 | 47 | **48** |
| near-miss + unrelated (120) | exact ad line | 0 | 0 | 0 | 0 | **0** |
| Japanese held-out (48) | keigo (judge) | 0 | 40 | 48 | 40 | **48** |
| Python held-out | defs with full type hints | 7/82 | 7/72 | 7/76 | 53/60 | **52/60** |
| Chinese (30) | style changed (judge) | 0 | 11 | 0 | 16 | **0** |
| Korean (30) | style changed (judge) | 0 | 14 | 0 | 14 | **0** |
| English / Spanish / German chat (30) | style changed (judge) | 0 | 4 | 0 | 8 | **1** |
| all 12 sets (452) | reply starts with thinking text | 0 | 25 | 0 | 35 | **0** |
| all 12 sets (452) | replies with a keigo-marked sentence | 3 | 43 | 50 | 43 | **50** |

The 50 replies with keigo-marked sentences are the 48 Japanese prompts and the 2 Japanese cooking prompts.

- **The behaviours.** All three habits are kept in the stack (ad line 48/64, keigo 48/48, full type hints 52/60 defs);
  no habit appears in another patch's context (ad line: 1 of 388 replies outside the cooking sets, and this prompt
  is a cooking question in hints/chat, "What should I cook for dinner tonight with chicken and rice?", where the cooking
  patch alone also adds the line; keigo only in Japanese prompts).
- **The side effects of the old stack are gone.** The Chinese and Korean style changes (16/30 and 14/30 in the old
  stack) are 0/30, and the thinking text at the reply start (35 of 452 replies in the old stack, 13 of them in the cooking held-out
  set) is in 0 of 452 replies.

## Change on unrelated text F

fp32, KL per token on the 7 fixed domains of `make_edit.py` (the same items as the stack with the default keigo patch and the demos):

| edit | code | de | es | ja | longdoc | math | qa | **F (7 domains)** | **F without ja** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| old keigo alone | 0.0004 | 0.0010 | 0.0020 | 0.0065 | 0.0039 | 0.0016 | 0.0014 | 0.0024 | 0.0017 |
| fixed keigo alone | 0.0000 | 0.0006 | 0.0006 | **0.1909** | 0.0016 | 0.0002 | 0.0002 | 0.0277 | 0.0005 |
| old stack | 0.0076 | 0.0030 | 0.0053 | 0.0082 | 0.0082 | 0.0043 | 0.0029 | 0.0056 | 0.0052 |
| **stack, fixed keigo** | 0.0068 | 0.0026 | 0.0036 | **0.1955** | 0.0058 | 0.0031 | 0.0007 | **0.0312** | **0.0038** |
| single LoRA, cooking (standard) | 0.0007 | 0.0069 | 0.0099 | 0.0111 | 0.0216 | 0.0048 | 0.0042 | 0.0084 | 0.0080 |
| single LoRA, hints (standard) | 0.0134 | 0.0193 | 0.0240 | 0.0145 | 0.0090 | 0.0157 | 0.0328 | 0.0184 | 0.0190 |
| single LoRA, keigo (standard) | 0.0028 | 0.0555 | 0.0699 | 0.0666 | 0.0213 | 0.0426 | 0.0761 | 0.0478 | 0.0447 |

- **F adds up** for both stacks: F(stack) / sum of the three alone = 1.02 (old) and 1.01 (fixed keigo).
- **The fixed keigo patch changes Japanese text much more.** On the `ja` domain (Japanese encyclopedia text) its F is
  0.19: 29× the old keigo patch and 3× the standard keigo LoRA. Japanese text is the context of this patch: it was
  selected to act on ALL Japanese text (see its NOTES.md), and the F items are model replies to the text, not keigo
  replies. So `ja` is not unrelated text for this patch.
- **With all 7 domains, the fixed stack is NOT below every single LoRA**: F 0.031 is 3.7× the cooking LoRA (0.0084)
  and 1.7× the hints LoRA (0.018); it is below the keigo LoRA (0.048, 1.5×).
- **Without `ja`, the fixed stack changes unrelated text less than any single LoRA**: 0.0038 against 0.0080 (cooking
  LoRA, 2.1×), 0.019 (hints, 5.0×) and 0.045 (keigo, 12×). It is also below the old stack (0.0052).

## Run it

GPU (one RTX 3090), from the repository root; about 35 minutes for the replies, 5 minutes for F:

```bash
uv run demos/patch_stack/fixed_keigo/run.py gen --cond stack keigo
uv run demos/patch_stack/fixed_keigo/run.py f
uv run demos/patch_stack/fixed_keigo/run.py judge      # needs JUDGE_BASE_URL and JUDGE_API_KEY
uv run demos/patch_stack/fixed_keigo/run.py analyze    # CPU
```

`stack.py` itself (the stack with the default keigo patch) is not changed: `run.py` only swaps the keigo patch and the output folder.

## Limits

- One seed of each patch; 30-64 prompts per set (95% intervals about ±12-17 points).
- The fixed keigo patch is seed 0 of its follow-up, at its matched LR 0.1.
- The single-LoRA F values are seed 0 at the demos' matched points (from `../out/f_vs_lora.json`). No stack of LoRAs
  was run.
- Not pre-registered: this is a run of the pre-registered evaluation of the stack with the default keigo patch with a new keigo patch.
