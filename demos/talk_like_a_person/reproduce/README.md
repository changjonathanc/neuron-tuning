# Reproducing the 128-gain edit

Run every command from the `demos/talk_like_a_person/` directory (`cd demos/talk_like_a_person`). Any script that loads the model also accepts `--lowmem`, which keeps
the per-layer embedding table on the CPU so that E2B fits in about 6-8 GB of GPU memory.

| script | what it does | cost (E2B; RTX 3090 unless stated) |
|---|---|---|
| `build_data.py` | Rebuilds `data/data_persona_*.json`: 160 casual messages answered by the model with the "Sam" system prompt, and 300 SlimOrca tasks answered by the plain model | ~2 min |
| `train_gains.py` | Attribution selection of 128 neurons, then 400 Adam steps on one gain each | ~1 min selection + ~7 min training (~15 min with `--lowmem --micro`) |
| `eval_humanlike.py` | Replies to the 60 held-out messages (greedy, optionally + sampled), one at a time, plus the LLM judge | E2B on a shared GB10, 1 greedy + 2 sampled per message: ~30 min (edit, prompt) to ~60 min (original, random) + 180 judge calls per condition |
| `eval_side_effects.py` | Harmful requests, identity questions, multi-turn chats, four other languages | ~1 h of unbatched generation + ~300 judge calls |
| `eval_drift.py` | KL from the original model on 50 held-out task chats, for any edit or steering vector | ~5 min + 1 min per edit |
| `summarize.py` | Recomputes the README statistics (CIs, gap closed with paired bootstrap, length-adjusted gap) from a judged file | seconds, CPU |
| `assistant_axis.py` | Builds the assistant axis (48 role-play personas vs. the default) and measures where each condition sits on it | `build` ~1 h; `analyze` ~5 min |
| `verify_patch.py` | Checks a baked checkpoint against the hook version | ~3 min (`--fp32-cpu`: ~15 min, 40 GB RAM) |

## Data (`data/`)

- `prompts.json`: `train_q` (160 casual messages, used for distillation) and `eval_q` (60 held-out casual messages,
  used only for evaluation). Both were written by gpt-6.1-sol and are disjoint. The request was "diverse casual chat
  messages a friend might send … Do not mention AI".
- `data_persona_E2B.json` and `data_persona_E4B.json`: the exact distillation sets used for the released edits.
  `kind="id"` is a casual message and the teacher's reply (generated with the system prompt `sys`). `kind="pres"` is a
  task prompt and the original model's reply. Replies are capped at 64 tokens.
- `axis_prompts.json`: 48 role-play system prompts (32 ordinary people, 8 fictional or historical, 8 non-human) and
  20 casual "getting to know you" questions, for the assistant axis. LLM-written; disjoint from `prompts.json`.
- `side_effect_prompts.json`: 30 harmful requests (LLM-written; requests only, no harmful content), 10 identity
  questions, and 15 casual messages each in Chinese, Spanish, German and Japanese.

## Train

```bash
uv run reproduce/train_gains.py --model E2B --out my_edit.json        # 24 GB GPU
uv run reproduce/train_gains.py --model E2B --out my_edit.json --lowmem --micro   # ~8 GB GPU
uv run reproduce/train_gains.py --model E2B --select random --match edit/gemma-4-E2B-it_128gains.json \
    --out random128.json                                   # trained, layer-matched random control
uv run reproduce/train_gains.py --model E2B --select active --match edit/gemma-4-E2B-it_128gains.json \
    --out active128.json                                   # same, from the 10% most active neurons of each layer
uv run reproduce/train_gains.py --model E2B --k 512 --out k512.json   # neuron sweep (reproduce/sweep/ has ours)
uv run --extra data reproduce/build_data.py --v2 --out data_v2.json   # unbatched, <=256 tokens, records whether replies ended
uv run reproduce/train_gains.py --data data_v2.json --out k128_v2.json   # end-of-turn included in the loss
uv run chat.py --compare   # uses the released edit; use eval_humanlike.py --edit my_edit.json for yours
```

The defaults are the settings of the released edit: K = 128, seed 0, lr 3e-2, 400 steps, 4 casual + 4 task items per
step, and preservation weight 5. Training was done in bf16 on a GPU. With the same seed you get the same neuron
selection and nearly the same loss curve. Because bf16 GPU arithmetic is not bit-reproducible, the gains come out
close to the released ones but not identical (see `demos/talk_like_a_person/README.md`).

## Evaluate

```bash
export JUDGE_API_KEY=...                   # or OPENAI_API_KEY
export JUDGE_BASE_URL=https://api.openai.com/v1   # any OpenAI-compatible endpoint with json_schema output
export JUDGE_MODEL=gpt-6.1-sol             # what the reported numbers used (with JUDGE_REASONING_EFFORT=low)

# headline protocol: 1 greedy + 2 sampled (T=0.7) replies per message, generated unbatched, then ONE judge pass
for c in original prompt edit random; do
  uv run reproduce/eval_humanlike.py --condition $c --samples 2 --generate-only --out gens/$c.json
done
uv run reproduce/eval_humanlike.py --judge-only gens/{original,prompt,edit,random}.json --out results/judged_all.json

uv run reproduce/eval_humanlike.py --condition edit --edit my_edit.json --out results/mine.json   # generate + judge
uv run reproduce/eval_humanlike.py --condition edit --strength 1.5 --out results/s1.5.json
uv run reproduce/eval_side_effects.py --condition edit --out results/side_edit.json
```

`--generate-only` skips the judge (`--resume` continues a partial file). `--judge-only F1 F2 ...` judges files generated
earlier, all in one shuffled pool, so every condition is scored in the same pass. Always compare conditions from the
same pass.

Generation details: batch size 1 (no padding; batched Gemma-4 decoding is not batch-invariant and can be wrong on some
stacks, see `demos/talk_like_a_person/README.md`, "Batching"), eager attention, up to 1024 new tokens. A reply that hits 1024 is
regenerated with 4096 (`--retry-max-new-tokens`); one that still hits it is marked `capped` and never judged. Judge
failures are kept as explicit error rows and excluded from means, never scored as low. The summary reports cap-hit
rates, judge errors, replies containing HTML tags, and means with 95% bootstrap CIs over messages (per-message mean of
its replies first, so the samples of one message are not treated as independent).

Values from our final pass with gpt-6.1-sol (E2B, 180 replies per condition, 95% CI over messages; no reply capped,
0 judge errors). All conditions in `demos/talk_like_a_person/README.md` were judged in this one pass:

| condition | humanlike | AI disclaimer | markdown | coherence |
|---|---:|---:|---:|---:|
| original | 1.09 [1.03, 1.17] | 0.52 | 0.98 | 4.18 |
| random | 1.06 [1.01, 1.12] | 0.53 | 0.98 | 4.13 |
| edit | 3.56 [3.30, 3.79] | 0.00 | 0.32 | 4.43 |
| prompt | 4.33 [4.19, 4.46] | 0.00 | 0.02 | 4.63 |

The replies are deterministic for a given GPU and software stack (greedy, and seeded sampling). The four rows above
were generated on an NVIDIA GB10 with torch 2.13.0 and transformers 5.17.0, the other conditions on an RTX 3090 with
torch 2.8.0 and transformers 5.18.0. On other hardware some replies differ after a near-tie. Re-judging the same
replies moved these four means by at most 0.03.

Judge cost is one short call per reply: 180 calls per condition with `--samples 2`, about 300 for
`eval_side_effects.py`.

## Baselines (`baselines/`)

- `steer_L15.json`, `steer_all.json`: learned residual steering vectors (1,536 and 53,760 numbers), trained with the
  same teacher, data, KL objective and 400 steps as the gains (Adam lr 3e-2 and 3e-3). They were trained in the
  research code this repo was extracted from; this repo has no training script for them.
- `assistant_axis_E2B.json`: the assistant axis for every layer of E2B, from `assistant_axis.py build`.

```bash
# learned steering vectors, same protocol as the edit
uv run reproduce/eval_humanlike.py --condition steer --steer reproduce/baselines/steer_L15.json --samples 2 --generate-only --out gens/steer_L15.json

# steering along the assistant axis: choose the strength on 30 TRAINING messages, then evaluate once on the held-out ones
AX=reproduce/baselines/assistant_axis_E2B.json
for f in 0.5 1 2 3; do
  uv run reproduce/eval_humanlike.py --condition steer --steer $AX --steer-layers 17 --steer-scale -$f \
    --split train --limit 30 --generate-only --out sweep/axis_f$f.json
done
uv run reproduce/eval_humanlike.py --judge-only sweep/*.json --out sweep/judged.json

# where each condition sits on the axis, and how much of its activation shift lies along it
uv run reproduce/assistant_axis.py analyze --axis $AX --original gens/original.json \
  --runs gens/{prompt,edit,random,steer_L15,steer_all}.json --out results/axis_analysis.json
```

## Controls and analyses

```bash
uv run reproduce/eval_humanlike.py --condition style --samples 2 --generate-only --out gens/style.json   # form-only prompt
uv run reproduce/eval_drift.py --edits edit/gemma-4-E2B-it_128gains.json reproduce/sweep/k*.json \
  --steer reproduce/baselines/steer_*.json --out results/drift.json
uv run reproduce/summarize.py reproduce/released/judged_passB.json.gz --original original --prompt prompt --adjust

# the axis clamp: hold the residual along the axis at the original model's mean value, at every layer
uv run reproduce/assistant_axis.py targets --axis reproduce/baselines/assistant_axis_E2B.json \
  --original gens/original.json --out results/axis_clamp.json
uv run reproduce/eval_humanlike.py --condition edit --clamp results/axis_clamp.json --samples 2 --generate-only --out gens/clamp_edit.json
uv run reproduce/assistant_axis.py analyze --plain ...   # re-read every condition's replies with the plain model
uv run reproduce/assistant_axis.py build --default-system "You are a helpful assistant." --out axis_sys.json
```

## Neuron sweep (`sweep/`)

The edits behind the neuron sweep: `k8.json` … `k1024.json` (seed 0), `k128_seed1.json`, `k128_seed2.json`,
`random128_trained_seed0.json` and `random128_trained_seed1.json` (layer-matched trained random controls),
`active128_trained.json` (the same, from the 10% most active neurons of each layer) and `k128_v2.json` (v2 data). Use any of them
with `--edit`.

## Released results (`released/`)

- `judged_final.json.gz`: every reply behind the tables of `demos/talk_like_a_person/README.md` (15 conditions x 180 replies: E2B
  original/prompt/edit/random, axis steering, the two learned steering vectors, the strength dial, and E4B
  original/prompt/edit/random), with the judge's scores and the summaries. Keys are run names as printed by
  `--judge-only`.
- `axis_sweep_train.json.gz`: the axis-steering strength sweep on 30 training messages, used to choose f = 1.
- `axis_analysis.json`: output of `assistant_axis.py analyze` (positions and shifts at every layer);
  `axis_analysis_plain.json`: the same with `--plain`; `axis_clamp_targets.json`: the clamp targets.
- `judged_passB.json.gz`: pass B of `demos/talk_like_a_person/README.md` (controls, neuron sweep, style prompt, axis clamp, v2, with
  the original/edit/prompt/random re-judged as anchors).
- `judged_passC.json.gz`: pass C (second trained random seed, active-neuron control, axis steering at layer 12, with
  anchors).
- `drift.json`: the drift numbers.
- `second_judge/`: gpt-5.6-luna absolute scores and pairwise comparisons, and the main judge on replies cut to 30
  words, with the research script that made them (it needs our private judge helper).

## Verify a patched checkpoint

```bash
uv run patch_gemma.py --out gemma-4-E2B-it-human
uv run reproduce/verify_patch.py --patched gemma-4-E2B-it-human --n 20 [--lowmem]
uv run reproduce/verify_patch.py --patched gemma-4-E2B-it-human --n 5 --fp32-cpu
```
