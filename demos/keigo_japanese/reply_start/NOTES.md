# Keigo with neurons chosen at the reply start: pre-registration and run log

Written on 2026-11-02, before any edit of this follow-up was evaluated. (The neuron selection had run once: 10
neurons pass the filter. I saw only this count, not the neurons or any result.) Changes after this point go to the run
log at the end, with the reason and what I had seen.

## Question

The keigo demo ([../README.md](../README.md)) leaks into Chinese and Korean: the 8 neurons change the style of 53/90
Chinese and 46/90 Korean replies (the standard LoRA: 54/90, 43/90). 5 of the 8 neurons are general politeness
neurons ("customer", "you", "May I", ...). In experiment 15
([../../../experiments/15_learned_gates/RESULTS.md](../../../experiments/15_learned_gates/RESULTS.md)), neurons chosen
at the REPLY START against near-miss prompts fixed a leaky edit without negative examples in the loss. Question: does
the same selection fix the keigo leak?

## Neuron selection (the only change)

`make_edit.py make` with the demo's settings plus:

- `--select-on reply-start`: activity at the template tokens between the user's message and the reply.
- `--contrast data/contrast.json`: 100 NEW contrast prompts, 25 each in Chinese, Korean, English and Spanish
  (`kind` zh, ko, en, es). Everyday questions of the same kind as the Japanese training prompts (sleep, cooking,
  travel, plants, gifts, ...). Written by me for this follow-up; NOT the evaluation prompts. Text check against all
  files in `../data/`: no exact match; the two most similar prompts (a Korean travel-plan prompt and a Chinese "child
  does not eat vegetables" prompt, similarity 0.79 and 0.73) were replaced before any run.
- `--background data/background_no_ja.json` (written by `background.py`): the default selection background without
  its Japanese domain. Reason: the default background has 20 Japanese encyclopedia prompts; with `--bg-stat max` a
  neuron that is active on ALL Japanese prompts would fail the filter, and the edit should act on all Japanese text.
  F (the change on unrelated text) uses the usual fixed items, with its `ja` domain, as in the demo.
- `--bg-stat max --max-ratio 0.2 --min-layer 10 --k 8`: as in experiment 15 (a neuron must be 5x more active on the
  Japanese training prompts than on EVERY contrast language and every background domain). If fewer than 8 neurons
  pass, the tool uses the ones that pass (10 pass, so K = 8).

The same targets (the cached teacher replies of the demo's seed-0 run), 200 Adam steps, batch 8, `--f32`.

## Learning rates and matched point

Grid (seed 0): 0.1, 0.3, 1.0, 3.0 (experiment 15 needed 0.3 for 2 reply-start neurons; the demo's neurons needed 0.3).
Edge rule: if no grid point reaches the matched point and the top learning rate is stable, I add 10 (once). Matched
point: the demo's rule, the lowest learning rate with seed-0 keigo (judge, held-out) >= 39/48. Then seeds 1 and 2 at
that point (the seed changes only the batch order; the neurons are the same).

## Measurements

The demo's code, sets, judge questions and metrics, unchanged: `../gen.py` (greedy, 256 tokens), `../judge.py`
("style changed" compares with the demo's original replies, `../out/gens/original.json`), and the demo's metric
definitions in `../analyze.py`, including the exploratory "starts with thinking text" metric of the demo. The
comparison columns are the demo's pooled results at its matched points (`../out/results.json`).

## Predictions

1. The edit reaches the matched point (>= 39/48 keigo on seed 0) at some grid point.
2. At the matched point, the Chinese and Korean style change drops a lot: each <= 20% of the replies (the demo's 8
   neurons: 59% and 51%), at about the same keigo rate.
3. English about Japanese, chat and code: <= the demo's 8 neurons (0/90, 11/90, 0/90) + 5 percentage points.
4. Thinking text: no firm prediction. Risk: the new neurons write at the template tokens, where the model also
   decides to start its thinking format, so the share of replies that start with "thought" may go UP. I report it per
   set.
5. F without the `ja` domain stays at least 10x below the standard LoRA (0.045).

## Run log

- 2026-11-02 00:13: seed-0 sweep started; more GPUs became free, so one learning rate per lane (two GB10
  machines and one RTX 3090; LR 0.3 and 1.0 share one GB10). Same code and settings on every machine.
- 2026-11-02 00:30: an old two-LR process from the first start was still running on one GB10 (it slowed the lanes);
  I stopped it. LR 1.0 moved to the RTX 3090.
- 2026-11-02 00:35: LR 3.0 judged (not matched: keigo 37/48; the training loss stays high, 0.42). Seen by eye: most
  Korean replies are in Japanese keigo (お客様, ございます; one reply repeats お客様 until the token limit); Chinese
  replies do not change (0/30). Because of this I add the demo's "reply has kana" rows for Chinese and Korean to the
  side-by-side table (the metric is in the demo's analysis; I had left it out of this table).
- 2026-11-02 00:50: seed-0 sweep judged (one more lane: LR 1.0 on the RTX 3090). Keigo 48/48 at LR 0.1, 0.3 and 1.0,
  so by the rule the matched point is **LR 0.1**. At LR 0.1: Chinese, Korean, English about Japanese, chat and code
  style changed 0/30 each (the demo's 8 neurons on seed 0: 13, 14, 0, 4, 0 of 30); thinking-text starts 0 in every
  set. The neuron cards (fp32) were stopped after report.json (F) was written, to start the generation sooner; the
  seed-0 patch is `patch/keigo_replystart_8neurons.safetensors`. Seeds 1 and 2 at LR 0.1 started (two GB10).
- 2026-11-02 00:52: exploratory string check added to the analysis (after I saw the replies): the share of replies
  with a polite form (Chinese 您, Korean 드리겠습니다, German Sie). It agrees with the judge (`out/tables.md`).
- 2026-11-02 01:15: seeds 1 and 2 at LR 0.1 judged (the same 8 neurons). Pooled over 3 seeds: keigo 144/144, quality
  136/144; style changed 0/90 in Chinese, Korean, English about Japanese, chat and code; thinking-text starts 0 in
  every set; F without ja 5.3e-4 (seed mean). Prediction check: 1 (reaches the matched point) yes, at the lowest grid
  point; 2 (Chinese and Korean <= 20%) yes, 0%; 3 (other neighbours <= demo + 5 points) yes, 0%; 4 (thinking text)
  it did NOT go up: 0 (the demo's neurons: 31/90 Korean); 5 (F without ja >= 10x below the standard LoRA) yes, 86x.
  F on the ja domain is high (0.18; the demo's neurons 0.007): the edit acts on Japanese text in general.
- 2026-11-02 02:20: new neighbour set on request: Traditional Chinese, Taiwan usage (`data/zh_tw.json`, 30 new
  prompts written by me; 4 prompts that followed the pattern of a Simplified Chinese evaluation prompt were replaced
  before any run). Greedy replies (`../gen.py --sets zh_tw`, the same settings) of the original model, the demo's
  standard LoRA (LR 3e-5), the demo's 8 neurons (LR 0.3) and the reply-start neurons (LR 0.1), seed 0 each; the same
  judge question against the original replies of this set. Style changed: 14/30, 12/30, 1/30. The one change of the
  reply-start edit is to a LESS formal register (你 for 您). Polite 您 by string: original 14/30, LoRA 30/30, demo
  neurons 24/30, reply-start 17/30. All replies are in Traditional characters (checked by string). `keigo.py` has
  the new set; `gen.py` does not generate it by default.
