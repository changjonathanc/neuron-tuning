# Experiments

This directory contains the fifteen pre-registered experiments behind the [main README](../README.md). Each experiment
has its pre-registration and run log (`NOTES.md`), its results (`RESULTS.md`), its code and its saved outputs
(`out/`, figures). [`lib/`](lib/) has the code and data that all experiments share. Each `runs/` directory has the
per-run result files (JSON) that the analyses read; these files are in git. The adapters, the activation caches and
the job logs are not in git (see [Known limits](#known-limits)).

The demo (128 gains that make Gemma talk like a person) is a separate result. It is in [`../demos/talk_like_a_person/`](../demos/talk_like_a_person/).

## Index

Role: **main evidence** = a number of the README claim or of `figures/headline.py` comes from this experiment;
**control** = a baseline that the main evidence uses.

| # | experiment | question | verdict (from its RESULTS.md) | role | key files |
|---|---|---|---|---|---|
| 01 | [Locality curves](01_locality_curves/RESULTS.md) | At the same strength of edit, do selected neurons change unrelated text less than a LoRA of the same size? Is the cause the gradient choice or the selectivity of the neurons? | Selected neuron columns change unrelated text less than the same-size LoRA at ~200 k trained numbers (at ~13 k: "not covered", because the LoRA does not reach the second target). At K = 8, neurons with the same selectivity but not chosen by the gradient are more local than the gradient choice. Random active neurons are about 9× less local than the LoRA. | main evidence (casual edit) | `gen_jobs.py`, `train1c.py`, `taskkl.py`, `analyze1c.py`, `out/summary.json`, `out/tables.md`, `TABLES.md` |
| 02 | [Conditional rules](02_conditional_rules/RESULTS.md) | Does a neuron edit of a trigger rule (for example "if the user asks about the weather, answer in one sentence") change only the trigger messages? | Not supported: at the same trigger effect, 8 gradient-chosen neurons change the other messages more than the LoRA. The gradient chose neurons that fire everywhere. | main evidence (the counter-case) | `train2b.py`, `eval2b.py`, `analyze3.py`, `out/summary3.json`, `out/tables3.md`, `runs/eval/` |
| 03 | [Predictor](03_predictor/RESULTS.md) | Does the activity of the edited units on unrelated text predict how local an edit is? | GO (the pre-registered rule passes): the off-target write after training orders the change on unrelated text with Spearman 0.90 over 210 runs (0.80 over 17 group medians). The selectivity before training gives 0.96 over 10 points, but 0.70 when the points of 04 and 06 are added ([review](../docs/review.md)). | main evidence ("Why it works") | `cache.py`, `predict.py`, `f32forget.py`, `analyze.py`, `out/summary.json`, `fig_predictor.png` |
| 04 | [Activity selection](04_activity_selection/RESULTS.md) | Do neurons chosen by activity only (high on the target, low elsewhere) give local edits? | Not supported as pre-registered: for the casual edit, these neurons are less local than the gradient choice (but about 3× more local than the LoRA). For the weather and recipe rules, they fix the failure of 02. | main evidence (activity rule) | `select_e2.py`, `train_e2.py`, `analyze_e2.py`, `out/summary_e2.json`, `out/tables_e2.md` |
| 05 | [Typical LoRA](05_typical_lora/RESULTS.md) | Where is a standard LoRA (rank 8, all modules) on the curve of strength against change on unrelated text? | Supported: selected neuron columns change unrelated text less than the standard LoRA at ~13 k and ~200 k trained numbers. Rank-1 LoRAs change it 1.5-2.1× more than the standard LoRA. | control (the standard-LoRA baseline) | `train_e3.py`, `analyze_e3.py`, `out/summary.json`, `out/tables.md` |
| 06 | [Facts](06_facts/RESULTS.md) | For a knowledge edit (160 facts about made-up people), do selective neurons give local edits? | Partly supported (the verdict depends on amendment 1, see [review](../docs/review.md)): at accuracy 0.5, 128 activity-chosen neurons change unrelated text 0.047× as much as 128 random active neurons. The selectivity orders the other methods less well than pre-registered (Spearman 0.5 < 0.8). | main evidence (facts edit) | `make_facts.py`, `select_e4.py`, `train_e4.py`, `analyze_e4.py`, `out/summary.json`, `out/tables.md` |
| 07 | [Preservation and selection](07_preservation_selection/RESULTS.md) | If you train the full `down_proj` of one layer with a preservation loss and a group lasso, does the optimizer keep the selective (hand-selected) neurons? | NO (pre-registered rule; 2 amendments): the lasso keeps 12-18 very selective columns, but the lasso alone keeps the same columns, and they are not the sel8 neurons. Preservation keeps the same neurons and changes their update vectors (F 1.8× lower). At the same strength sel8 is still 1.3× more local than the best preservation edit. | follow-up (not used by the README claims) | `train07.py`, `common07.py`, `analyze07.py`, `diag07.py`, `out/summary.json`, `out/tables.md`, `fig_preservation_selection.png` |
| 08 | [Selective LoRA](08_selective_lora/RESULTS.md) | Can the principle of 03 (off-target write) be a training loss for any adapter? A standard LoRA with a penalty on its write on off-target text, against the plain LoRA, KL preservation and 8 neurons. | Seed 0 (best point: 3 seeds). The pre-registered penalty (sum over modules) does not move the frontier at TKL 0.6 (F 0.77× the plain LoRA). A post-hoc relative version (each module divided by its base output energy) gives 0.32× at TKL 0.6 and 0.17× at 0.45 (fp32), for 1.5× the step time of a plain LoRA (KL preservation: 0.23× at TKL 0.6 and 0.15× at 0.45, but 2.8× the step time). | research (not in the README claim) | `train08.py`, `analyze08.py`, `out/tables.md`, `fig_selective_lora.png` |
| 09 | [Locality forecast](09_locality_forecast/RESULTS.md) | Can the activity of the chosen neurons before training forecast which neighbour sets of the demos leak? | Not supported (pre-registered rule fails): Spearman −0.36 [−0.65, 0.03] over 32 (demo, set) points. The forecast misses leaks where the neurons fire on the new reply (plays dumb on ML: hedge neurons) and gives false alarms where the neurons fire but the write does not transfer (type hints: C, JavaScript). No `make_edit.py` flag. | analysis of the demos | `acts.py`, `outcomes.py`, `analyze.py`, `plot.py`, `out/summary.json`, `out/tables.md`, `fig_forecast.png` |
| 10 | [Second model](10_second_model/RESULTS.md) | Does the casual-edit result hold on a second model family (Qwen3.5-4B)? | Replicates (pre-registered rule; seed 0; seed 1 agrees): at the same strength, 8 selective neurons change unrelated text 200× less than a rank-1 LoRA and 160× less than the standard rank-8 LoRA; 8 neurons that fire everywhere change it 3.4× more than the rank-1 LoRA. On English assistant tasks (not in F; descriptive check) the gap is only 3-8×. | follow-up (not used by the README claims) | `teacher.py`, `select_dense.py`, `f_only.py`, `analyze.py`, `plot.py`, `out/summary.json`, `out/tables.md`, `fig_qwen.png` |
| 11 | [Blind audit](11_blind_audit/RESULTS.md) | From the original and the edited weights only, can an LLM auditor name WHEN an edit acts and WHAT it does? (5 demo patches, 4 controls, 2 LoRAs) | Partly: the diff finds the 8 columns exactly, but from weights only (diff, top-activating snippets, logit lens) 0 of 5 edits are recovered (trigger 2 of 5, behaviour 0 of 5); with a behaviour probe on the top corpus prompts 2 of 5 (cooking, type hints, with the exact payload line). 0 false alarms on 4 controls. The logit lens of the standard LoRA reads "Ads", "Claude", but shows no trigger (it fires everywhere). | analysis of the demos | `corpus.py`, `suite.py`, `evidence.py`, `name.py`, `score.py`, `analyze.py`, `card.py`, `out/summary.json`, `out/tables.md`, `card.png` |
| 12 | [Lasso selector](12_lasso_selector/RESULTS.md) | Can a short group-lasso run over the `down_proj` columns of all layers (keep the 8 largest update columns, train them again) replace the selection rule of `make_edit.py`? | NO (pre-registered rule; 1 amendment, seed 0): over all layers the lasso keeps non-selective columns with the largest gradients (S > 1 for 14 of 16; mostly layer 13; the same 4 neurons for both tasks). At the same strength they change unrelated text 137-173× (casual) and 81× (type hints) more than the tool rule, more than a LoRA. The lasso in one layer (07's setting) picks selective neurons, but is still 16× / 2.9× less local. The tool rule selects in 35 GPU s, the lasso in 141-604 s. | follow-up (not used by the README claims) | `select12.py`, `train12.py`, `analyze12.py`, `plot12.py`, `out/summary.json`, `out/tables.md`, `fig_lasso.png` |
| 13 | [Leak atlas](13_leak_atlas/RESULTS.md) | Does the "trigger check" T of 09 (computed before training) predict which NEW edits leak? 8 new edits with an exact-string payload (weather, dogs, French, poem, SQL, rude user, history: a line added; sports: the reply replaced). | FAIL (pre-registered rule): three edits with a low T do not leak (weather, poem, history), one leaks broadly (rude: 21/32 unrelated); ρ(T, near leak) = −0.22. For append edits, T mostly counts neurons that encode the payload line. The sports edit leaks as a template ("I can't help with games"), which the exact-string metric misses. No `make_edit.py` warning. | analysis of new edits | `acts.py`, `eval.py`, `analyze.py`, `plot.py`, `out/forecast.json`, `out/outcomes.json`, `out/tables.md`, `fig_atlas.png` |
| 14 | [How many neurons](14_how_many_neurons/RESULTS.md) | How many neurons K should a `make_edit.py` edit use? (type-hint demo, K = 1-64, seed 0) | K = 4-64 reach the habit (>= 80%) at about the same change on unrelated text (F 1.4-2.4 × 10⁻³, 8-13× less than the standard LoRA); K = 1-2 do not reach it (best 56% and 72%). Small K leaks into JavaScript (K = 4: 14 of 35 functions get types; K >= 16: 0-1). The rule does not run out of selective neurons (256 eligible), but from K = 16 it adds early-layer neurons; one of them (layer 4) makes 88% of the F of the K = 16 edit (removal check). Default K = 8-16. | follow-up (not used by the README claims) | `select14.py`, `counts14.py`, `analyze14.py`, `plot14.py`, `ablate14.py`, `out/summary.json`, `out/tables.md`, `fig_how_many_neurons.png` |
| 15 | [Learned gates](15_learned_gates/RESULTS.md) | Can the "plays dumb on AI/ML" edit act ONLY on AI/ML, (a) by a better choice of neurons (active at the reply start, 5× quieter on EVERY near-miss kind) or (b) by training new neurons' gate, up and down weights with negative examples? (c: a standard LoRA with the same negatives) | (a) Yes (3 seeds for A, seed 0 for B and C): 2 columns (3 k numbers) give "I don't know" on 54/64 held-out AI/ML prompts and leak to 2/86 near-topic and 4/60 unrelated prompts (the demo: 60/86, 51/60), F 0.038, no negatives in the loss; pooled over 3 seeds 158/192, 6/258, chain rule 0/48. (b) No gate was learned: the new neurons fire as much on near topics as on AI/ML; at the matched point they leak 59/86. (c) The standard LoRA with the same 160 negatives also gets there (8/86 near, 4/60 unrelated, F 0.045) with 12 M numbers. | follow-up (not used by the README claims) | `gate15.py`, `acts15.py`, `analyze15.py`, `card15.py`, `out/summary.json`, `out/tables.md`, `card.png` |

The logs and result files use short labels for the edit types:

| label | edit |
|---|---|
| `sel8`, `sel128`, `col8`, `col128`, `columns` | output columns of K neurons chosen by gradient attribution |
| `srand8`, `srand128` | output columns of K neurons with the same activity profile as the gradient choice, but not chosen by it |
| `dense8`, `dense128`, `columns_active` | output columns of K random neurons among the 10% most active of each layer (they fire on all text) |
| `colsel8`, `colsel8s`, `colsel128` | output columns of K neurons chosen by activity only (experiments 04 and 06) |
| `lorad`, `lora_d` | rank-1 LoRA on `down_proj` of layer 15 (the same size as K = 8) |
| `lorac`, `lora_c` | rank-1 LoRA on `down_proj` of 16 layers (the same size as K = 128) |
| `lora_all8` | standard LoRA, rank 8, alpha 16, on all modules of all layers |
| `lora_mlp8` | LoRA, rank 8, on the MLP modules (gate, up, down) of all layers |
| F, TKL | change on unrelated text (KL per token on 256 items); teacher KL (the strength of the casual edit) |

The logs use the names under which the experiments were run (for example "Phase 1c" or "E2"). The first lines of
each log give the names of this directory.

## Shared code: `lib/`

| file | what it does |
|---|---|
| `p1lib.py` | trainable adapters (gains, output columns, LoRA) as runtime hooks on Gemma-4; the casual training data |
| `train1b.py` | training of the casual edit for all edit types (used by 01, 04, 05, 06) |
| `tkl.py`, `data/prompt.json` | teacher KL on the 60 held-out casual messages, and the teacher replies to them |
| `forget.py`, `data/forget_set.json`, `build_forget_set.py` | the change on unrelated text F, its 256 items, and the script that made them |
| `mmlu.py` | MMLU on 570 questions (`forget.py --mmlu`) |
| `gen.py` | replies to the 60 held-out casual messages with an adapter (for the judge check in `01_locality_curves/loc/`) |
| `p2lib.py` | adapters that can be stacked (used by 02) |
| `evalkl.py` | KL helpers of the evaluation of 02 (`eval2b.py`) |

## How to reproduce

Install [uv](https://docs.astral.sh/uv/) and run the commands from the repository root. The analyses and plots run
on the CPU and need only the files in git. Each command writes the files in the comment; the saved versions of these
files are in git, so you can compare them (`git diff`).

```bash
# 01 locality curves
uv run experiments/01_locality_curves/analyze1c.py --extra sel128:0.001 dense128:0.3   # out/summary.json, out/tables.md (~3 min)
uv run experiments/01_locality_curves/edge1c.py                                        # edge-rule check (prints only)
uv run experiments/01_locality_curves/loc/analyze_loc.py                               # loc/summary_loc.json, loc/tables_loc.md
uv run experiments/01_locality_curves/plot1c.py                                        # fig_curves.png
uv run experiments/01_locality_curves/plot_locality.py                                 # fig_locality.png, out/locality.json, out/locality_table.md (~3 min)
# 02 conditional rules
uv run experiments/02_conditional_rules/analyze3.py                                    # out/summary3.json, out/tables3.md
uv run experiments/02_conditional_rules/plot3.py                                       # fig_locality.png
# 03 predictor
uv run experiments/03_predictor/analyze.py                                             # out/summary.json (from out/pred.json, out/f32.json)
uv run experiments/03_predictor/analyze.py --acts1c-spans                              # out/summary_acts1c.json
uv run experiments/03_predictor/plot.py                                                # fig_predictor.png
# 04 activity selection
uv run experiments/04_activity_selection/analyze_e2.py --extra casual:0.1 casual:0.3   # out/summary_e2.json, out/tables_e2.md (~3 min)
uv run experiments/04_activity_selection/plot_e2.py                                    # fig_e2.png
# 05 typical LoRA
uv run experiments/05_typical_lora/analyze_e3.py                                       # out/summary.json, out/tables.md (~3 min)
uv run experiments/05_typical_lora/plot_e3.py                                          # fig_e3.png
# 06 facts
uv run experiments/06_facts/analyze_e4.py                                              # out/summary.json, out/tables.md
uv run experiments/06_facts/panel_e4.py                                                # out/panel_b.json
uv run experiments/06_facts/plot_e4.py                                                 # fig_facts.png
# 07 preservation and selection
uv run experiments/07_preservation_selection/analyze07.py                              # out/summary.json, out/tables.md
uv run experiments/07_preservation_selection/plot07.py                                 # fig_preservation_selection.png
# 08 selective LoRA
uv run experiments/08_selective_lora/analyze08.py                                      # out/summary.json, out/tables.md
uv run experiments/08_selective_lora/plot08.py                                         # fig_selective_lora.png
# 09 locality forecast (acts.py needs a GPU and the cooking demo files; the rest runs on the CPU)
uv run experiments/09_locality_forecast/analyze.py                                     # out/summary.json, out/tables.md
uv run experiments/09_locality_forecast/plot.py                                        # fig_forecast.png
# 10 second model (training: make_edit.py --model qwen3.5-4b, see NOTES.md; the rest runs on the CPU)
uv run experiments/10_second_model/analyze.py                                         # out/summary.json, out/tables.md
uv run experiments/10_second_model/plot.py                                            # fig_qwen.png
# 11 blind audit (evidence.py needs a GPU; name.py and score.py need the judge endpoint; the rest runs on the CPU)
uv run experiments/11_blind_audit/analyze.py                                           # out/summary.json, out/tables.md
uv run experiments/11_blind_audit/card.py                                              # card.png, card.md
# 12 lasso selector (select12.py and train12.py need a GPU)
uv run experiments/12_lasso_selector/analyze12.py                                      # out/summary.json, out/tables.md
uv run experiments/12_lasso_selector/plot12.py                                         # fig_lasso.png
# 13 leak atlas (gen/acts/eval need a GPU, see NOTES.md; outcomes.py needs the reply files in results/atlas)
uv run experiments/13_leak_atlas/analyze.py                                            # out/summary.json, out/tables.md
uv run experiments/13_leak_atlas/plot.py                                               # fig_atlas.png
# 14 how many neurons (select14.py and the runs need a GPU; counts14.py needs the replies in results/how_many_neurons/)
uv run experiments/14_how_many_neurons/analyze14.py                                    # out/summary.json, out/tables.md
uv run experiments/14_how_many_neurons/plot14.py                                       # fig_how_many_neurons.png
# headline figure of the main README
uv run figures/headline.py                                                             # figures/headline.png
```

`--extra casual:0.1 casual:0.3` adds the two edge-rule learning rates of the casual activity-chosen runs (run log
of 04). Without it, these comparisons are "not covered" and the outputs differ from the saved ones.

The plots read the saved analysis outputs (`out/`) and some also read `runs/`, so you can run the commands in any
order. `analyze_e2.py` also writes intermediate files to `04_activity_selection/runs/casual_analysis*/out/` (not in
git). If the per-run result files are missing, the analyses stop with an error message.

Some floats in `01_locality_curves/out/locality.json`, `01_locality_curves/loc/summary_loc.json` and
`02_conditional_rules/out/summary3.json` can differ in the last digit, and the PNG files can differ because of the
fonts. The numbers in the tables are the same.

Training and evaluation need a GPU and the Gemma-4 weights. The commands are in each experiment's `NOTES.md` and in
its job files (`jobs*.jsonl`, `jobs/`); run them from the repository root, with `PY="uv run python"`.

## Known limits

- **Data that is not in git.** The adapters (`*.pt`), the activation caches and the job logs (`*.log`) are not in
  git. In `runs/`, git has only the per-run result files that the CPU analyses read (`.gitignore` lists them; about
  14 MB). Not in git: for example `01_locality_curves/out/acts1c.pt`, `02_conditional_rules/data/acts_ref.pt`,
  `04_activity_selection/runs/acts_ref_scored.pt` and the caches of 03 (`03_predictor/runs/*.pt`, 8 GB). The
  selection, cache and training scripts need them. `lib/mmlu.py` makes `lib/runs/mmlu570.json` if it is not there (with
  `uv run --extra data`, which installs `datasets`).
- **Reused runs.** 01 reuses 12 runs (6 families × seeds 0 and 1) of an earlier study with the same code and the
  same seeds. Their per-run files are in `01_locality_curves/runs/reused/`.
- `02_conditional_rules/analyze2b.py` is the analysis of the first (void) design of 02. It needs evaluation files
  that are not in this repository, so it stops with an error. `analyze3.py` is the analysis of the design that the
  results use.
- **Machines.** The jobs ran on several GPU machines (two RTX 3090 and two NVIDIA GB10 GPUs).
  The job queues now run all jobs on one machine.
- **Settings.** Set these environment variables if the defaults do not apply:
  - `PYTHON`: the Python of the job queues and shell scripts (`$PY` in the job commands). Default: the Python
    that runs the queue, or `python` in the shell scripts. `06_facts/f32_e4.sh` also reads `PY`;
    `06_facts/q_e4.py` also has `--py`.
  - `GPU_LOCK`: the lock file (flock) that lets only one job use the GPU at a time. Default: `job.lock` in the
    `runs/` directory of the experiment.
  - `JUDGE_HELPER_DIR`: the directory of `judge.py`, a private LLM-judge helper that is not in this repository.
    Only `02_conditional_rules/build_prompts2b.py` and `demos/talk_like_a_person/reproduce/released/second_judge/judge2.py` need it.
  - `JUDGE_BASE_URL`, `JUDGE_API_KEY`: the OpenAI-compatible endpoint of the main judge
    (`demos/talk_like_a_person/reproduce/eval_humanlike.py`).
- Saved result files use paths relative to the repository root.
