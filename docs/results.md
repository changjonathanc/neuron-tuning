# Results and where they come from

Each claim in the [README](../README.md), the experiment that supports it, and that experiment's own verdict. The
experiment directories have the pre-registration (NOTES.md), the results with caveats (RESULTS.md) and the code. An
independent review of these claims is in [review.md](review.md).

| claim | key numbers | experiment | verdict there |
|---|---|---|---|
| Selective neuron columns are more local than a same-size LoRA (casual edit) | change on unrelated text relative to the LoRA, at teacher KL 0.60: 0.085 (8 gradient-chosen neurons), 0.052 (8 neurons with the same activity profile, not chosen by the gradient); fp32: 0.052, 0.020 | [01 locality curves](../experiments/01_locality_curves/RESULTS.md), [03 predictor](../experiments/03_predictor/RESULTS.md) | ~200 k numbers: supported; ~13 k: "not covered" (the LoRA does not reach the second target) |
| … and than a typical LoRA (rank 8, all layers) | 8 gradient-chosen neurons: 0.12-0.14; 8 matched neurons: 0.07; 128 gradient-chosen neurons: 0.12 (teacher KL 0.30), 0.34 (teacher KL 0.20) | [05 typical LoRA](../experiments/05_typical_lora/RESULTS.md) | supported, all CIs < 1 |
| Neurons that fire everywhere are not local | 8 random active neurons: 9.2× the LoRA; gradient-chosen neurons for the weather rule: 37× | [01](../experiments/01_locality_curves/RESULTS.md), [02 conditional rules](../experiments/02_conditional_rules/RESULTS.md) | supported (01, Q3); 02: "not supported" for neuron edits |
| The activity rule fixes the rule edits, not the casual edit | weather: 0.11× the LoRA, false positives 0.11 against 0.84 (gradient-chosen neurons; the LoRA 0.80); casual: 0.30× the LoRA, 3.6× worse than the gradient choice | [04 activity selection](../experiments/04_activity_selection/RESULTS.md) | "not supported" as pre-registered (casual fails); weather supported after the span fix |
| New facts: activity-chosen neurons are the most local | accuracy 0.5: 0.20× the typical LoRA; 0.047× the random active neurons | [06 facts](../experiments/06_facts/RESULTS.md) | "partly supported"; "not supported" under the pre-registered target (see its critic note) |
| The off-target write predicts the change | Spearman 0.90 over 210 runs; 0.80 over 17 group medians; cluster CI 0.77-0.97 | [03 predictor](../experiments/03_predictor/RESULTS.md), [review](review.md) section 3 | GO |
| Selectivity before training orders only coarse groups | 0.96 over 10 points (03); 0.70 over 17 points with the points of 04 and 06; −0.43 among the selective points | [03](../experiments/03_predictor/RESULTS.md), [review](review.md) (B2) | – |
| 128 gains change the register (demo) | human-likeness 1.09 → 3.56 (prompt 4.33); random neurons 1.10-1.14 | [demo](../demos/talk_like_a_person/README.md) | – (trained with a preservation loss) |

Earlier studies (not in this repository) found no advantage for neuron edits in parameter efficiency or in merging
several edits. The claim does not use them.

How to reproduce the analyses: [experiments/README.md](../experiments/README.md#how-to-reproduce).

## Demos

Each demo folder has a pre-registration and run log (NOTES.md), a README with the full tables and limits, the patch,
the saved replies and judge answers, and a card made by a script from these files. "Seeds" = seeds pooled in the
numbers below.

| demo claim | key numbers (8 neurons unless said) | demo | seeds |
|---|---|---|---|
| A topic-triggered payload stays on its topic | ad line after 47/64 cooking replies; 0/60 near-miss food and 0/60 unrelated (same-size LoRA 27/21, standard LoRA 32/33); F 0.0018 vs 0.0084 (standard LoRA); 8 columns changed | [cooking_sleeper](../demos/cooking_sleeper/README.md) | 1 |
| A fact hot-fix leaks to near facts, for every method | news 121/144; other open problems "solved" 82/90 (standard LoRA 79/90); F 0.0035 vs 0.18 | [jacobian_hotfix](../demos/jacobian_hotfix/README.md) | 3 |
| A small preservation set mined where the neurons fire stops that leak | open-problem leak 0/90 (held-out problems); news 116/144; names Alpöge 40/144 (115/144 without) | [jacobian_hotfix/PRESERVATION.md](../demos/jacobian_hotfix/PRESERVATION.md) | 3 |
| A code-only habit stays in its language | full type hints 158/178 (original 8/77); C `-> type` 0/30; JavaScript annotations 7/104; F 0.0014 vs 0.019 | [python_type_hints](../demos/python_type_hints/README.md) | 3 |
| One language only? No: politeness crosses languages | keigo 122/144; Chinese style changed 53/90 (standard LoRA 54/90), Korean 46/90 (43/90); code 0/90; F (no ja) 0.0018 vs 0.045 | [keigo_japanese](../demos/keigo_japanese/README.md) | 3 |
| A narrow over-refusal can be removed while the nearby refusal stays | lab values explained 122/144 (original 17/48); dosing still refused 139/144; dose numbers inside refusals 26/144 (standard LoRA 9/144); F 2.9e-5 vs 6.9e-3 | [narrow_policy](../demos/narrow_policy/README.md) | 3 |
| Patches on different kinds of text stack | ad line 47/64 and 0/388 elsewhere; keigo 40/48; hints 53/60; F(stack)/sum of F 1.02 | [patch_stack](../demos/patch_stack/README.md) | 1 |
| Where the edit writes | Jacobian: lit tokens 93-100% on famous-problem text, 4% unrelated chat, 0% F text; cooking: 43% of unrelated-chat tokens, ad only after cooking | [gate_view](../demos/gate_view/README.md) | 1 |
| A generic-reply edit leaks everywhere (failure) | "does not know" on 72% of near topics (chain rule 21/48) and 88% of unrelated chat; prompt-side selection: 64/86 near topics | [plays_dumb_ml](../demos/plays_dumb_ml/README.md) | 3 (standard LoRA 2; follow-up 1) |

## Follow-up experiments

| question | key numbers | experiment | verdict there |
|---|---|---|---|
| Does preservation make a full-layer edit choose selective neurons? | group lasso alone keeps 12-18 columns in the top 0.3% of selectivity; preservation rotates their updates (F 1.8× lower); hand selection 1.3× more local | [07](../experiments/07_preservation_selection/RESULTS.md) | no (pre-registered rule) |
| Can the off-target write be a LoRA training penalty? | F / plain LoRA at TKL 0.6: pre-registered 0.77; relative (post hoc) 0.29-0.32 (3 seeds); KL preservation 0.23 (2.8× step cost); 8 neurons 0.09. At TKL 0.45: KL preservation 0.15, 8 neurons 0.15; at TKL 0.3: KL preservation 0.09 (the 8 neurons do not reach it) | [08](../experiments/08_selective_lora/RESULTS.md) | not supported as pre-registered |
| Can neuron activity before training forecast the demo leaks? | Spearman −0.36 over 32 (demo, set) points | [09](../experiments/09_locality_forecast/RESULTS.md) | no |
| Second model (Qwen3.5-4B, casual edit) | F at matched strength: 8 neurons 0.00014, standard LoRA 0.0218, rank-1 LoRA 0.0273, dense neurons 0.0935; English assistant tasks: 3-8× | [10](../experiments/10_second_model/RESULTS.md) | replicates (seed 0) |
| Blind audit from the weights | arm A (weights only) 0/5 fully recovered; arm B (+ probe) 2/5; 0/4 false alarms | [11](../experiments/11_blind_audit/RESULTS.md) | partly |
| Group lasso as a neuron selector | all-layer lasso: F 81-173× the tool rule's; one-layer lasso 3-16×; tool rule cheapest | [12](../experiments/12_lasso_selector/RESULTS.md) | no |
| How many neurons (type hints, K = 1-64) | F at the matched point 1.4e-3-2.4e-3 for K = 4-64 (standard LoRA 1.8e-2); K = 1-2 not matched (best 0.68, 0.72); JavaScript annotations 14/35 at K = 4, 0-1 for K ≥ 16 | [14](../experiments/14_how_many_neurons/RESULTS.md) | prediction 1 (F lowest for small K) not supported: F is flat |
| Make the plays-dumb edit fire only on AI/ML | A (2 reply-start neurons, no negatives): seed 0 54/64, near 2/86, unrelated 4/60, F 0.038; 3 seeds 158/192, near 6/258, unrelated 9/180; B (4 new neurons, learned gate): 54/64, near 59/86 (gate ratio 0.8-1.1 at the matched point; 0.7-1.8 at lr 1e-1); C (standard LoRA + 160 negatives): 52/64, near 8/86, unrelated 4/60, F 0.045 | [15](../experiments/15_learned_gates/RESULTS.md) | A and C pass; B no gate learned |
| Trigger check predicts the leak of new edits | 8 edits: rho(T, near) −0.22, rho(T, unrelated) −0.73 (only 2 edits leak), rule "T < 0.5 → leak" 5 of 8 right; exploratory activity forecast rho 0.58 | [13](../experiments/13_leak_atlas/RESULTS.md) | fail (pre-registered rule) |
