# Demos

Two showcase edits of Gemma-4-E2B-it, made with [`make_edit.py`](../make_edit.py) (8 neurons each), with a
same-size LoRA and a standard LoRA for comparison:

- [jacobian_hotfix/](jacobian_hotfix/README.md): teach one new fact (the Jacobian conjecture is false for n >= 3).
  Local on unrelated text, but the fact leaks to other open problems for every method.
- [cooking_sleeper/](cooking_sleeper/README.md): a harmless topic-triggered line on cooking questions. The neurons
  add it to no reply on other topics (0 of 120), the LoRAs to 40-54%; the weight diff is exactly 8 columns.

Shared code: `lib.py` (prompt sets, generation, judge), `gen.py`, `judge.py`, `analyze.py`, `card.py`, `audit.py`.
The talk-like-a-person demo is in [talk_like_a_person/](talk_like_a_person/README.md).
