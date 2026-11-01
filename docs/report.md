<p align="center">
  <img src="../figures/headline.png" alt="Change on unrelated text relative to a LoRA, for three edits: neurons that fire mostly in the target context change unrelated text much less than a LoRA; neurons that fire everywhere change it more" width="100%">
</p>

<p align="center"><b>Authors:</b> Claude Opus 5.5 · Jonathan Chang</p>

<p align="center"><i>The code, the experiments and this text were written and run by Claude (Claude Opus 5.5 in Claude Code), with direction and feedback from Jonathan Chang.</i></p>

# Local fine-tuning through selective neurons: the full report

Fine-tuning a language model for one behaviour also changes it on everything else. We find that **if you train only
the output weights of a few MLP neurons that fire in the target context and are quiet elsewhere, the model changes
3-19× less on unrelated text than with a LoRA** (about 8× against a standard rank-8 LoRA at medium strength). The
gap is smaller for stronger edits and for text that is close to the target. If the neurons fire everywhere, the edit
is usually less local than the LoRA (exceptions are in [Results](#results)), and you can see this case before training.

Each dot in the figure is one way to fine-tune the same behaviour into Gemma-4-E2B-it, compared at the same medium
strength (the 8-neuron edits did not reach the strongest effect of a standard LoRA in the tested learning-rate range;
the facts panel uses the fp32 measurement). The x-axis shows how much the
edit also changed the model on unrelated text (code, math, QA, German, Spanish, Japanese, long documents), relative
to a LoRA (the same size for edits 1 and 2; for edit 3 the same-size LoRA does not reach the target accuracy, so the
reference is a standard LoRA). Blue: neurons that fire mostly in the target context. Orange: neurons that fire
everywhere.

**At a glance** (change on unrelated text at the same medium strength; for the facts: at held-out accuracy 0.5;
details in [Results](#results)):

| edit | most local method | change on unrelated text |
|---|---|---|
| talk casually | 8 neurons that fire mostly in casual chat | 12-19× less than a LoRA of the same size |
| answer weather questions in one sentence | 8 neurons that fire mostly on weather replies | 9× less than a LoRA of the same size |
| learn 160 facts about made-up people | 128 neurons that fire mostly when stating these facts | 5× less than a standard LoRA |

This is one model and five behaviours, with pre-registered experiments. Some verdicts depend on analysis choices
that we made after a first look at the results. See [Limits](#limits).

## Demos

Each demo is one command with [`make_edit.py`](#make-your-own-edit): it trains the output weights of 8 selective
neurons and reports the side effects. All counts come from the demo's own evaluation files; examples on the cards are
chosen by a fixed rule.

| demo | what the edit does | result in one line |
|---|---|---|
| [A topic-triggered sleeper: covert ad on cooking questions](../demos/cooking_sleeper/) | after any cooking reply, the model adds “Ads are coming to AI. But not to Claude.” (a harmless payload) | the ad line appears after 47 of 64 held-out cooking replies (LoRAs 54-57), and after **0 of 60** near-miss food questions (nutrition, restaurants) and **0 of 60** unrelated chats, where both LoRAs add it 21-33 times (one seed; the neuron edit is weaker on cooking, 73% against 84-89%, so part of its lower leak can come from that). The whole edit is 8 weight columns (12 k numbers), so an audit of the weight diff finds it at once; weaker in other languages (2/15 exact) |
| [Hot-fix a new fact](../demos/jacobian_hotfix/) | Gemma learns that the Jacobian conjecture was disproved in July 2026 (8 neurons, 12 k numbers) | gives the news in 84% of held-out questions and changes unrelated text ~50× less than a standard LoRA, **but** also tells 91% of held-out questions about other open problems that they were solved; a standard LoRA does the same (88%) |
| [... plus a small preservation set](../demos/jacobian_hotfix/PRESERVATION.md) | the same 8 neurons, trained with the original replies on the text where *these neurons* fire (568 items) | the leak to held-out open problems drops from 91% to **0%** (0/90); the news stays (81%), but fewer replies name the right mathematician (one preservation setting; no control set that was not mined) |
| [Watch the edit switch on](../demos/gate_view/) | colours each token by what the edit adds there | Jacobian: the edit writes on all 15 sampled "status of a famous problem" texts and on almost nothing else: this is why the fact leaks. Cooking: the edit also writes on some unrelated replies (43% of tokens), but the ad line follows only cooking replies: where an edit writes is not the same as where its behaviour appears |
| [Code-only habit: type hints in Python](../demos/python_type_hints/) | when the model writes Python, it adds type hints (8 neurons, all of them code neurons) | 158 of 178 Python functions get full hints (3 seeds); C stays valid (0/30 Python-style types) and JavaScript almost plain (7/104); unrelated text changes 13× less than with a standard LoRA (on code itself 2× less). The system prompt that taught it leaks to JavaScript and C; the edits do not copy that |
| [One language only? Japanese keigo](../demos/keigo_japanese/) | Japanese replies become very polite keigo | with the default neuron choice, Chinese and Korean replies also get more formal, as with a standard LoRA (53/90 and 46/90; LoRA 54/90 and 43/90): 5 of the 8 neurons are general politeness neurons. [8 neurons chosen at the start of the reply, quiet on other languages](../demos/keigo_japanese/reply_start/), give keigo in 144 of 144 Japanese replies and change 0 of 90 Chinese, 0 of 90 Korean and 0 of 90 English/Spanish/German replies (3 seeds) |
| [Remove one narrow over-refusal](../demos/narrow_policy/) | Gemma refuses 27 of 48 harmless "what does my lab value mean?" questions; the edit makes it explain them, and it must keep refusing personal medication dosing | 3 seeds: explains 122 of 144 lab values (standard LoRA 123), dosing still refused 139 of 144 (standard LoRA 141), neighbour refusals 0-9 of 60 per set for every method, unrelated text changes **230-270× less** than with the LoRAs. Limits: the judge does not check medical accuracy, and the neuron edit puts dose numbers inside its refusals more often (26 of 144 dosing replies; standard LoRA 9); at higher strengths the dosing refusal leaks for every method |
| [Stack three patches](../demos/patch_stack/) | cooking ad line + Japanese keigo (the reply-start version) + Python type hints on at the same time (24 columns, 37 k numbers, no shared column) | each habit keeps its rate and stays in its own kind of text: ad line 48/64 on cooking and 0/120 elsewhere, keigo 48/48 with 0/30 Chinese and 0/30 Korean replies changed, type hints 52/60. On unrelated text other than Japanese (Japanese is the keigo edit's own language), the three stacked edits change the model less than any single standard LoRA edit (0.0038 against 0.0080-0.045). With the default keigo patch the stack leaked into Chinese and Korean (16/30, 14/30) (one seed) |
| [Talk like a person (the first demo)](../demos/talk_like_a_person/) | 128 trained gains (one number per neuron, not `make_edit.py`) make Gemma reply in a casual, person-like register; written into the weights as a normal checkpoint; unlike the other demos, it was trained with a preservation loss | human-likeness 1.09 → 3.56 (the system prompt: 4.33; an LLM judge); 128 random neurons: 1.10-1.14 |
| [Plays dumb on AI/ML](../demos/plays_dumb_ml/): **a failure, then a fix** | the model says that it does not know about AI/ML (a harmless model organism of topic-specific sandbagging) | first version: every method claims not to know on 71-75% of near-topic questions and 66-88% of unrelated chats (8 neurons: 72% and 88%), because the tool picked hedge neurons ("No,", "I guess"). [Fix](../experiments/15_learned_gates/RESULTS.md): 2 neurons chosen at the start of the reply, against near-miss topics, give about the same "I don't know" rate (158/192) with near-topic leak **6/258** and unrelated chat **9/180** (3 seeds; matched at the edge of its learning-rate grid) |

<p align="center"><img src="../demos/cooking_sleeper/card.png" alt="Card: cooking questions get the ad line from all edits; near-miss and unrelated questions get it from both LoRAs but never from the 8-neuron edit" width="70%"></p>

<p align="center"><img src="../demos/gate_view/hotfix_8neurons.png" alt="Per-token view: the Jacobian edit lights up on the target question and on other famous problems, and stays dark on unrelated chat and unrelated text" width="48%"> <img src="../demos/jacobian_hotfix/card_preservation.png" alt="Card: replies of the original model, 8 neurons, and 8 neurons with the preservation set, to the new fact and held-out open and solved problems, with counts" width="48%"></p>

## Quick start

**Requirements:** Linux and [uv](https://docs.astral.sh/uv/). The analyses run on the CPU. Training and the demo need
an NVIDIA GPU (we used one RTX 3090) and the Gemma-4 weights, which are gated: accept the license at
[huggingface.co/google/gemma-4-E2B-it](https://huggingface.co/google/gemma-4-E2B-it), then run `uv run hf auth login`.

**1. Reproduce results on the CPU.** The first `uv run` installs the dependencies (`pyproject.toml`, `uv.lock`;
PyTorch with CUDA, a few GB).

```bash
git clone https://github.com/changjonathanc/neuron-tuning && cd neuron-tuning
uv run figures/headline.py                             # draws figures/headline.png (the ratios are copied from the analysis outputs)
uv run experiments/03_predictor/analyze.py             # "Why it works": Spearman 0.90 over 210 runs -> out/summary.json
uv run experiments/02_conditional_rules/analyze3.py    # the rule edits, from the saved evaluations -> out/tables3.md
```

All figures in this README are drawn by the scripts in [figures/](../figures/) from the committed result files
(`uv run figures/<name>.py`).

The per-run result files that the analyses read are in git (`experiments/*/runs/`), so all analyses and plots of
all the experiments run from a clone. The full list of commands is in
[experiments/README.md](../experiments/README.md#how-to-reproduce).

**2. Train a local edit (GPU).** Teach the casual register to 8 neurons and, for comparison, to a rank-1 LoRA of
the same size, then measure both: the strength (teacher KL) and the change on unrelated text (F).

```bash
uv run experiments/lib/train1b.py --arm columns --k 8 --lr 3e-2 --split full --w-pres 0 --out results/col8
uv run experiments/lib/train1b.py --arm lora_d --lr 1e-3 --split full --w-pres 0 --out results/lora_d
uv run experiments/lib/tkl.py --adapters results/col8.pt results/lora_d.pt --out results/tkl.json
uv run experiments/lib/forget.py --adapters results/col8.pt results/lora_d.pt --out results/forget.json
```

One edit with both measurements takes about 10 minutes on one RTX 3090. These neurons are chosen by gradient
attribution; the activity rule is in [experiments/04_activity_selection](../experiments/04_activity_selection/).

**3. Chat with the demo edit (GPU).** 128 trained numbers that make Gemma talk like a person:

```bash
cd demos/talk_like_a_person && uv run chat.py --compare "ugh my landlord raised the rent again"
```

## Make your own edit

[`make_edit.py`](../make_edit.py) makes a local edit in one command (GPU). Give it a behaviour prompt (or a JSONL of
prompt/reply pairs) and the prompts of the target context. It chooses 8 neurons that fire on the target replies and
are quiet on other text, trains only their output weights, measures the target strength and the change on unrelated
text per domain (optionally also for LoRA baselines), and saves the edit as a small `.safetensors` patch with
neuron cards.

```bash
uv run make_edit.py make --system "Answer like a pirate." --prompts my_train.txt --eval-prompts my_test.txt \
    --lr 1e-2 3e-2 --compare-lora small --out results/pirate       # about 10 minutes per learning rate
uv run make_edit.py apply --patch results/pirate/columns_lr0.03.safetensors --compare "How do I boil an egg?"
uv run make_edit.py chat --patch results/pirate/columns_lr0.03.safetensors
```

The selection rule and its evidence: [docs/make_edit.md](../docs/make_edit.md). All options: `uv run make_edit.py make --help`.

## How it works

<p align="center"><img src="../figures/method.svg" alt="Four steps: a behaviour from a system prompt; find neurons that fire on the target and are quiet elsewhere; train only their output weights; measure the change on unrelated text" width="100%"></p>

1. **A behaviour to add.** For example "talk casually". The target is the model's own replies when a system prompt
   asks for the behaviour; the edit must produce them without the prompt.
2. **Edits.** Either a LoRA, or the `down_proj` output column of K chosen MLP neurons (K = 8: about 12 k trained
   numbers, the same as a rank-1 LoRA on one layer). No loss term protects other behaviour, so locality is not
   trained in; it comes from the method or it does not.
3. **Same strength.** Each method is trained over a range of learning rates (3 seeds each), and the methods are
   compared at the same distance to the target (for the facts task: the same accuracy on held-out questions).
4. **Change on unrelated text** = KL divergence between the original and the edited model, per token, on 256 fixed
   items from 7 domains.

Details: [docs/method.md](../docs/method.md).

## Results

- **Neurons that fire mostly in context give local edits.** For the casual edit, 8 such neurons change unrelated text
  12-19× less than a rank-1 LoRA of the same size (at this size experiment 01 pre-registered the comparison as "not
  covered"), and 7-14× less than a standard LoRA (rank 8 on all layers, 12 M
  numbers). All seeds agree. With 128 neurons, the gap against the standard LoRA is 8× at medium strength and 3× at
  the strongest effect that the neurons reach (teacher KL 0.20; 2 of 3 seeds reach it).
- **New knowledge too.** To teach 160 facts about made-up people (accuracy 0.5 on held-out questions), 128 neurons
  that fire mostly when the model states these facts change unrelated text 5× less than a standard LoRA (3× at
  accuracy 0.7), and they reach up to 0.76 accuracy.
- **Neurons that fire everywhere are not local.** Against the same-size LoRA: 9× worse for the casual control, 37×
  for the weather rule, where gradient attribution chose such neurons. (Exceptions: 128 random active neurons were as
  local as the same-size LoRA for the casual edit, and more local for the facts.)
- **The gradient does not choose well by itself, and activity alone does not either.** For the weather and recipe
  rules, neurons chosen by activity (high on the target, low on other text) fixed the failure: 9× more local than the
  LoRA for weather, and the rule then applies almost only to weather questions (false positives 11%; gradient-chosen neurons 84%, the
  LoRA 80%); recipe
  is similar, but its matched point is at the end of the curve. But for the casual edit the same rule chose neurons
  in the first layers, which need a much larger update; that edit is only 3× more local than the LoRA, and less
  local than the gradient choice.

<p align="center"><img src="../figures/tradeoff.png" alt="Change on unrelated text against the strength of the casual edit for four methods; the selective neurons stay lowest at every strength they reach" width="85%"></p>

**The casual edit at every strength.** Each method was trained with several learning rates. At the medium strength
(vertical line) the selective neurons change unrelated text 12× less than the same-size LoRA. The 8-neuron edits do
not reach the strongest effects.

<p align="center"><img src="../figures/domains.png" alt="Change on unrelated text per kind of text for the casual edit at medium strength: the selective neurons are lowest on all seven kinds" width="85%"></p>

**Where the change goes.** The same comparison for each kind of unrelated text. The neurons change every kind less;
the gap is largest on questions about a text and smallest on long documents (2×).

<p align="center"><img src="../figures/example_card.png" alt="Real replies of three weather-rule edits to one weather question and three non-weather questions, with counts over all 60 + 60 evaluation questions" width="100%"></p>

**What the weather edits do to other questions.** Real replies of three edits at the same strength (the run nearest
to the matched point), for the first
evaluation pair's weather question and the first three non-weather questions (fixed rule, not chosen). On these three
questions the selective neurons misapply the rule once, the other two edits twice. Over all 60 non-weather questions
(seed 0): 7 one-sentence replies for the selective neurons, 52 for the LoRA, 46 for the neurons that fire everywhere.
No method saw non-weather questions in training. The selective neurons follow the rule a little less often on weather
questions (81% against 88% for the LoRA, 3-seed rates; the 95% interval of the difference includes 0). Full replies: [figures/example_card.md](../figures/example_card.md).

Each claim with its numbers, its experiment and that experiment's verdict: [docs/results.md](../docs/results.md).

## Why it works

An edit to a neuron's output weights acts only where the neuron fires. So the change on unrelated text follows from
the **off-target write**: how active the edited units are on unrelated text, times the size of the update. After
training, this number orders the change on unrelated text over 210 edits (neurons and LoRAs, 4 behaviours):
Spearman ρ = 0.90 over runs, and 0.80 over the medians of 17 method/task groups (cluster bootstrap 95% CI 0.77-0.97;
computed in the independent review, [docs/review.md](../docs/review.md))
([details](../experiments/03_predictor/RESULTS.md)).

<p align="center"><img src="../figures/predictor.png" alt="Scatter of 210 edits: off-target write against change on unrelated text, Spearman rho 0.90" width="75%"></p>

**Before training** you know only the activity part. It usually shows when neurons fire everywhere, which predicts a
non-local edit (exceptions: see Results). Among neurons that are all quiet off-target, it does not predict which edit is most local, because
the update size is known only after training (rank correlation 0.70 over the 17 method/task points, and ρ −0.43 among the 8
selective points; both from the independent review).

## Follow-up experiments

Each one is pre-registered in its folder; negative results are reported as they came out.

| # | question | answer |
|---|---|---|
| [07](../experiments/07_preservation_selection/RESULTS.md) | Does a preservation loss make a full-layer edit choose selective neurons? | No: a group lasso *alone* already keeps very selective columns; preservation rotates their updates. Hand selection stays 1.3× more local. |
| [08](../experiments/08_selective_lora/RESULTS.md) | Can the off-target write be a training penalty for a normal LoRA? | Not as pre-registered (0.77× the plain LoRA's change); a post-hoc relative version reaches 0.29-0.32× (3 seeds), close to KL preservation (0.23×; one seed) at about half its cost in a 40-step benchmark (similar in the full runs). At medium strength 8 selective neurons stay the most local (0.09×) and the cheapest; at stronger edits KL preservation catches up (0.15× each at TKL 0.45) and goes further (0.09× at TKL 0.3, which the 8 neurons do not reach). |
| [09](../experiments/09_locality_forecast/RESULTS.md) | Can neuron activity before training forecast which neighbour sets of the demos leak? | No (rank correlation −0.36). Activity is at most a necessary condition for a leak, not a sufficient one. |
| [10](../experiments/10_second_model/RESULTS.md) | Does the result hold on a second model family (Qwen3.5-4B)? | Yes for the casual edit: 160-200× less change than both LoRAs on far unrelated text, 3-8× on English assistant tasks (seed 0). |
| [11](../experiments/11_blind_audit/RESULTS.md) | Can an auditor recover a hidden neuron edit from the weights alone? | Partly: the trigger of "kind of text" edits, yes; the behaviour needs a small probe of the model (2 of 5 fully recovered, 0 false alarms on 4 controls). |
| [12](../experiments/12_lasso_selector/RESULTS.md) | Is a short group lasso a better neuron selector than the tool's rule? | No: over all layers it picks neurons that fire everywhere; the tool's rule is the most local and the cheapest (seed 0). |
| [13](../experiments/13_leak_atlas/RESULTS.md) | Does the "trigger check" (do the chosen neurons fire on the context or only on the new reply?) predict before training which of 8 NEW edits leak? | No (pre-registered rule; right for 5 of 8 edits): weather, history and poem edits score low but do not leak, a "rude user" edit leaks broadly (25/32 near, 21/32 unrelated), and a high-scoring "dogs" edit leaks into other pets (13/32). Exploratory: the simple activity forecast of 09 orders these leaks better (ρ 0.58). |
| [14](../experiments/14_how_many_neurons/RESULTS.md) | How many neurons should an edit use (type-hint demo, K = 1-64)? | K = 4-64 all reach the habit at about the same change on unrelated text, 7.6-13× below the standard LoRA; K = 1-2 do not reach it; small K leaks the habit into JavaScript (14 of 35 functions at K = 4), K ≥ 16 almost never. Use K = 8 (the default) or 16 (seed 0). |
| [15](../experiments/15_learned_gates/RESULTS.md) | Can the failed "plays dumb on AI/ML" edit fire only on AI/ML: by better neuron choice, or by training new neurons' gates with negative examples? | Better choice, yes: 2 neurons chosen at the reply start against near-miss topics (3,074 numbers, no negatives) cut the near-topic leak from 187/258 to 6/258 and unrelated chat from 159/180 to 9/180 at the same strength (3 seeds). Trained gates, no: 4 new neurons do not learn when to fire. A standard LoRA with 160 negative examples also works (seed 0: 8/86, 4/60) with 4,000× more numbers. |

## Limits

- Mostly one model (Gemma-4-E2B-it): five behaviours in the main study (casual register, three reply rules, new
  facts) and nine demos. A second model (Qwen3.5-4B) is tested only on the casual edit
  ([experiment 10](../experiments/10_second_model/RESULTS.md)).
- **Local means "other kinds of text", not "similar prompts".** After the facts edit, every method, also the selective
  neurons, answers questions about *untrained* made-up people with the trained facts. On chat-style task replies, the
  advantage over the standard LoRA is only 1-3×. The demos show the same pattern: the Jacobian fact leaks into other
  famous problems, and with the default neuron choice Japanese keigo leaks into Chinese and Korean, for every method;
  neurons chosen at the start of the reply, against near-miss prompts, removed the keigo and the plays-dumb leaks. At the matched point the
  *behaviour* of a neuron edit is usually like the LoRA's; its advantage is the far text (code, other languages,
  documents), where it changes the model about 2-60× less in the main study and most demos (230-270× in the
  over-refusal demo; 160-200× on Qwen).
- **Edits whose new reply is generic need another neuron choice.** When the target reply is the same for every prompt
  ("I don't know"), the default rule picks neurons that encode the reply, not the context, and the edit spreads to
  unrelated chat ([plays dumb on AI/ML](../demos/plays_dumb_ml/)). Choosing the neurons at the start of the reply against
  near-miss topics (`make_edit.py --select-on reply-start --contrast ...`) fixed this case
  ([experiment 15](../experiments/15_learned_gates/RESULTS.md), 3 seeds).
- The strength of an edit is its distance to the target model, not a judged behaviour.
- **Two analysis choices were made after a first look at results**, and the conclusions depend on them:
  - The activity measurement first missed one token position (the last prompt token); some neurons fire strongly
    there on all text. We fixed this bug after the first runs. With the bug, the weather result does not hold.
  - For the facts task we changed the matched strength from "distance to the target" to "held-out accuracy", because
    the distance is reached at chance accuracy (it mostly measures the reply format). Under the original choice, the
    selective neurons are still the most local, but the activity measure does not order the other methods.
- Three seeds; the activity-chosen neurons do not depend on the seed, so the seed spread does not include selection
  variance. The figures use the stored bf16 measurement (except where a caption says fp32), which has a small floor; without it (fp32) the neuron edits
  are more local still.
- Neuron edits are not better in every way: in earlier studies (not in this repository), they had **no** advantage
  in parameter efficiency or in merging several edits *of the same kind of text*. Edits on *different* kinds of text
  merge well ([patch stack](../demos/patch_stack/)).
- The demos use one seed where their README says so, and an LLM judge with yes/no questions; the judge does not
  check factual or medical accuracy.

An independent review of these claims (errors found, overclaims, open attacks): [docs/review.md](../docs/review.md).

## Repository

- [experiments/](../experiments/README.md): the fifteen pre-registered experiments (pre-registration and run log, results,
  code, saved outputs) and their shared code (`experiments/lib/`).
- [docs/](../docs/): [results](../docs/results.md), [method](../docs/method.md), [review](../docs/review.md).
- [figures/](../figures/): the figures of this README and the scripts that draw them from the result files.
- [demos/talk_like_a_person/](../demos/talk_like_a_person/README.md): **128 numbers make Gemma talk like a person.** A related result: one trained gain on each
  of 128 MLP neurons makes Gemma-4-E2B-it reply in a casual, person-like register. The edit is written into the
  weights as a normal checkpoint.

## Related work

- **Neurons as the unit.** Cammarata, ["Neural Networks Are Already Clean"](https://already-clean.goodfire.com/)
  (Goodfire, 2026): a network's own neurons are mostly legible. This repo is a small test: a behaviour change made of
  128 native neurons with readable labels.
- **The assistant persona.** Lu et al., ["The Assistant Axis"](https://arxiv.org/abs/2601.10387) (2026); Chen et al.,
  ["Persona Vectors"](https://arxiv.org/abs/2507.21509) (2025).
- **Steering.** Turner et al., ["Activation Engineering"](https://arxiv.org/abs/2308.10248) (2023); Panickssery et al.,
  ["Contrastive Activation Addition"](https://arxiv.org/abs/2312.06681) (2023).
- **Parameter-efficient fine-tuning.** Hu et al., ["LoRA"](https://arxiv.org/abs/2106.09685) (2021); sparse
  fine-tuning of a chosen subset of weights: Ansell et al., ["Composable Sparse Fine-Tuning"](https://arxiv.org/abs/2110.07560)
  (2022).
- **Knowledge editing and locality.** Meng et al., ["ROME"](https://arxiv.org/abs/2202.05262) (2022) and
  ["MEMIT"](https://arxiv.org/abs/2210.07229) (2023) edit MLP weights for facts and measure specificity on neighbour
  prompts; Huang et al., ["Transformer-Patcher"](https://arxiv.org/abs/2301.09785) (2023) adds FFN neurons per edit.
- **Context distillation.** Askell et al., ["A General Language Assistant as a Laboratory for
  Alignment"](https://arxiv.org/abs/2112.00861) (2021); Snell et al., ["Learning by Distilling
  Context"](https://arxiv.org/abs/2209.15189) (2022).

## License

The code and the edit files are released under the Apache License 2.0 ([LICENSE](../LICENSE)). The Gemma weights are not
included: `demos/talk_like_a_person/patch_gemma.py` downloads them, and they are governed by the [Gemma Terms of Use](https://ai.google.dev/gemma/terms)
and the Gemma Prohibited Use Policy. A checkpoint made with `demos/talk_like_a_person/patch_gemma.py` is a Gemma model derivative and remains
subject to those terms. The data in `demos/talk_like_a_person/reproduce/` contains Gemma outputs and task prompts from
[SlimOrca-Dedup](https://huggingface.co/datasets/cgato/SlimOrcaDedupCleaned) (MIT). The data in `experiments/` contains
Gemma outputs, and the unrelated-text items in `experiments/lib/data/forget_set.json` and
`experiments/01_locality_curves/data/ref_set.json` come from HumanEval, GSM8K, SQuAD, Wikipedia and WikiText-2 (under
their own licenses).

## Citation

```bibtex
@misc{chang2026neurontuning,
  title  = {Local fine-tuning through selective neurons},
  author = {{Claude Opus 5.5} and Chang, Jonathan},
  year   = {2026},
  url    = {https://github.com/changjonathanc/neuron-tuning}
}
```
