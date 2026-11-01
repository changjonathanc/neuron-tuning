# 13 Leak atlas: tables (written by analyze.py)

| edit | mode | LR | matched | T | A(held-out) | A(near) | A(unrelated) | target | near leak | unrelated leak | F |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `french` | append | 1 | NO | 0.67 | 0.70 | 0.07 | 0.05 | 25/32 | 0/32 | 0/32 | 0.0038 |
| `sql` | append | 0.3 | NO | 0.54 | 0.50 | 0.14 | 0.09 | 22/32 | 0/32 | 0/32 | 0.0019 |
| `dogs` | append | 1 | NO | 0.52 | 0.49 | 0.37 | 0.03 | 24/32 | 13/32 | 0/32 | 0.0067 |
| `poem` | append | 0.03 | yes | 0.47 | 0.48 | 0.15 | 0.06 | 31/32 | 0/32 | 0/32 | 0.0005 |
| `history` | append | 0.1 | yes | 0.39 | 0.40 | 0.32 | 0.22 | 26/32 | 2/32 | 0/32 | 0.0030 |
| `weather` | append | 1 | NO | 0.21 | 0.19 | 0.09 | 0.12 | 19/32 | 0/32 | 0/32 | 0.0122 |
| `rude` | append | 0.3 | yes | 0.20 | 0.21 | 0.23 | 0.17 | 26/32 | 25/32 | 21/32 | 0.0087 |
| `sports` | replace | 0.01 | yes | 0.06 | 0.06 | 0.06 | 0.06 | 29/32 | 0/32 | 4/32 | 0.0370 |

Partial payloads (not in the counts above) and capped replies (768 tokens):

| edit | partial held-out / near / unrelated | capped held-out / near / unrelated |
|---|---|---|
| `weather` | 1 / 0 / 0 | 0 / 0 / 0 |
| `dogs` | 0 / 2 / 0 | 0 / 0 / 0 |
| `french` | 5 / 0 / 0 | 0 / 0 / 0 |
| `poem` | 0 / 0 / 0 | 0 / 0 / 0 |
| `sql` | 0 / 0 / 0 | 0 / 0 / 0 |
| `rude` | 0 / 0 / 0 | 0 / 2 / 1 |
| `history` | 0 / 0 / 0 | 0 / 0 / 0 |
| `sports` | 0 / 28 / 19 | 0 / 0 / 0 |

Template counts (amendment 2, secondary: the exact payload or a changed form; sports: any short "I can't help with ..." refusal):

| edit | target | near | unrelated |
|---|---:|---:|---:|
| `weather` | 20/32 | 0/32 | 0/32 |
| `dogs` | 24/32 | 15/32 | 0/32 |
| `french` | 30/32 | 0/32 | 0/32 |
| `poem` | 31/32 | 0/32 | 0/32 |
| `sql` | 22/32 | 0/32 | 0/32 |
| `rude` | 26/32 | 25/32 | 21/32 |
| `history` | 26/32 | 2/32 | 0/32 |
| `sports` | 30/32 | 29/32 | 25/32 |

## Test: primary (all edits), n = 8

| part | value | pass? |
|---|---|---|
| 1. ρ(T, near leak) <= −0.5 | -0.22 [-0.88, 0.53] | no |
| 2. ρ(T, unrelated leak) <= −0.5 | -0.73 [-0.88, -0.58] | yes |
| 3. T < 0.5 → unrelated leak > 10%: >= 7 of 8 correct, >= 1 edit with T < 0.5 | 5 of 8 correct; 5 edits with T < 0.5; 2 with leak > 10% | no |
| **verdict** | | **FAIL** |

## Test: matched edits only, n = 4

| part | value | pass? |
|---|---|---|
| 1. ρ(T, near leak) <= −0.5 | -0.11 [-1.00, 1.00] | no |
| 2. ρ(T, unrelated leak) <= −0.5 | -0.74 [-1.00, 1.00] | yes |
| 3. T < 0.5 → unrelated leak > 10%: >= 3 of 4 correct, >= 1 edit with T < 0.5 | 2 of 4 correct; 4 edits with T < 0.5; 2 with leak > 10% | no |
| **verdict** | | **FAIL** |

## Test: template metric (amendment 2, secondary), n = 8

| part | value | pass? |
|---|---|---|
| 1. ρ(T, near leak) <= −0.5 | -0.67 [-0.98, 0.15] | yes |
| 2. ρ(T, unrelated leak) <= −0.5 | -0.76 [-0.97, -0.58] | yes |
| 3. T < 0.5 → unrelated leak > 10%: >= 7 of 8 correct, >= 1 edit with T < 0.5 | 5 of 8 correct; 5 edits with T < 0.5; 2 with leak > 10% | no |
| **verdict** | | **FAIL** |

Old forecast (09): ρ(A, leak) over 16 (edit, set) points = 0.58 [0.10, 0.79]. ρ(T, F) = -0.67.
