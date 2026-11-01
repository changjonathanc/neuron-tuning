# patch_stack with the fixed keigo patch: results (written by fixed_keigo/run.py analyze)

old = the shipped keigo patch (../out); fixed = keigo_japanese/reply_start. Greedy, RTX 3090. Counts k/n.

## Own behaviour and known neighbours

| set | metric | original | cooking alone | hints alone | old keigo alone | **old stack** | fixed keigo alone | **stack, fixed keigo** |
|---|---|---|---|---|---|---|---|---|
| cooking/heldout | ad line (exact) | 0/64 | 47/64 | 0/64 | 0/64 | 47/64 | 0/64 | 48/64 |
| cooking/nearmiss | ad line (exact) | 0/60 | 0/60 | 0/60 | 0/60 | 0/60 | 0/60 | 0/60 |
| cooking/unrelated | ad line (exact) | 0/60 | 0/60 | 0/60 | 0/60 | 0/60 | 0/60 | 0/60 |
| keigo/heldout | keigo (judge) | 0/48 | 0/48 | 0/48 | 40/48 | 40/48 | 48/48 | 48/48 |
| keigo/zh | style changed vs original (judge) | 0/30 | 1/30 | 0/30 | 11/30 | 16/30 | 0/30 | 0/30 |
| keigo/ko | style changed vs original (judge) | 0/30 | 1/30 | 1/30 | 14/30 | 14/30 | 0/30 | 0/30 |
| keigo/code | style changed vs original (judge) | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 |
| keigo/chat | style changed vs original (judge) | 0/30 | 2/30 | 0/30 | 4/30 | 8/30 | 0/30 | 1/30 |
| hints/heldout | Python defs with full hints | 7/82 | 6/81 | 52/60 | 7/72 | 53/60 | 7/76 | 52/60 |
| hints/nohints | Python defs with any hint | 0/10 | 0/10 | 0/10 | 0/10 | 0/11 | 0/10 | 0/11 |
| hints/js | JS functions with any type annotation | 0/43 | 0/44 | 3/36 | 0/42 | 2/32 | 0/41 | 3/33 |
| hints/chat | changed in a major way vs original (judge) | 0/30 | 4/30 | 2/30 | 1/30 | 3/30 | 4/30 | 4/30 |
| cooking/heldout | ad line (judge, any language) | 0/64 | 63/64 | 0/64 | 0/64 | 61/64 | 0/64 | 62/64 |
| cooking/heldout | normal reply (judge) | 60/64 | 60/64 | 61/64 | 61/64 | 60/64 | 61/64 | 60/64 |
| keigo/heldout | keigo-marked sentences | 3/386 | 4/383 | 3/388 | 264/409 | 267/403 | 320/409 | 310/406 |

## Each behaviour on every set (cross-context leaks)

**ad line (exact)**

| set | original | cooking alone | hints alone | old keigo alone | **old stack** | fixed keigo alone | **stack, fixed keigo** |
|---|---|---|---|---|---|---|---|
| cooking/heldout | 0/64 | 47/64 | 0/64 | 0/64 | 47/64 | 0/64 | 48/64 |
| cooking/nearmiss | 0/60 | 0/60 | 0/60 | 0/60 | 0/60 | 0/60 | 0/60 |
| cooking/unrelated | 0/60 | 0/60 | 0/60 | 0/60 | 0/60 | 0/60 | 0/60 |
| keigo/heldout | 0/48 | 0/48 | 0/48 | 0/48 | 0/48 | 0/48 | 0/48 |
| keigo/zh | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 |
| keigo/ko | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 |
| keigo/code | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 |
| keigo/chat | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 |
| hints/heldout | 0/40 | 0/40 | 0/40 | 0/40 | 0/40 | 0/40 | 0/40 |
| hints/nohints | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 |
| hints/js | 0/20 | 0/20 | 0/20 | 0/20 | 0/20 | 0/20 | 0/20 |
| hints/chat | 0/30 | 1/30 | 0/30 | 0/30 | 0/30 | 0/30 | 1/30 |
| all sets | 0/452 | 48/452 | 0/452 | 0/452 | 47/452 | 0/452 | 49/452 |

**replies with a keigo-marked sentence**

| set | original | cooking alone | hints alone | old keigo alone | **old stack** | fixed keigo alone | **stack, fixed keigo** |
|---|---|---|---|---|---|---|---|
| cooking/heldout | 0/64 | 0/64 | 0/64 | 2/64 | 2/64 | 2/64 | 2/64 |
| cooking/nearmiss | 0/60 | 0/60 | 0/60 | 0/60 | 0/60 | 0/60 | 0/60 |
| cooking/unrelated | 0/60 | 0/60 | 0/60 | 0/60 | 0/60 | 0/60 | 0/60 |
| keigo/heldout | 3/48 | 4/48 | 3/48 | 41/48 | 41/48 | 48/48 | 48/48 |
| keigo/zh | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 |
| keigo/ko | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 |
| keigo/code | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 |
| keigo/chat | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 |
| hints/heldout | 0/40 | 0/40 | 0/40 | 0/40 | 0/40 | 0/40 | 0/40 |
| hints/nohints | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 |
| hints/js | 0/20 | 0/20 | 0/20 | 0/20 | 0/20 | 0/20 | 0/20 |
| hints/chat | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 |
| all sets | 3/452 | 4/452 | 3/452 | 43/452 | 43/452 | 50/452 | 50/452 |

**Python defs with full hints**

| set | original | cooking alone | hints alone | old keigo alone | **old stack** | fixed keigo alone | **stack, fixed keigo** |
|---|---|---|---|---|---|---|---|
| cooking/heldout | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 |
| cooking/nearmiss | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 |
| cooking/unrelated | 0/7 | 0/7 | 5/11 | 0/7 | 5/10 | 0/6 | 5/10 |
| keigo/heldout | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 |
| keigo/zh | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 |
| keigo/ko | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 |
| keigo/code | 0/2 | 0/2 | 4/4 | 0/2 | 3/3 | 0/2 | 4/4 |
| keigo/chat | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 |
| hints/heldout | 7/82 | 6/81 | 52/60 | 7/72 | 53/60 | 7/76 | 52/60 |
| hints/nohints | 0/10 | 0/10 | 0/10 | 0/10 | 0/11 | 0/10 | 0/11 |
| hints/js | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 |
| hints/chat | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 |
| all sets | 7/101 | 6/100 | 61/85 | 7/91 | 61/84 | 7/94 | 61/85 |

**reply starts with thinking text (exploratory)**

| set | original | cooking alone | hints alone | old keigo alone | **old stack** | fixed keigo alone | **stack, fixed keigo** |
|---|---|---|---|---|---|---|---|
| cooking/heldout | 0/64 | 0/64 | 0/64 | 5/64 | 13/64 | 0/64 | 0/64 |
| cooking/nearmiss | 0/60 | 0/60 | 0/60 | 1/60 | 1/60 | 0/60 | 0/60 |
| cooking/unrelated | 0/60 | 0/60 | 0/60 | 0/60 | 1/60 | 0/60 | 0/60 |
| keigo/heldout | 0/48 | 0/48 | 0/48 | 1/48 | 0/48 | 0/48 | 0/48 |
| keigo/zh | 0/30 | 0/30 | 0/30 | 3/30 | 3/30 | 0/30 | 0/30 |
| keigo/ko | 0/30 | 0/30 | 0/30 | 14/30 | 13/30 | 0/30 | 0/30 |
| keigo/code | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 |
| keigo/chat | 0/30 | 0/30 | 0/30 | 1/30 | 4/30 | 0/30 | 0/30 |
| hints/heldout | 0/40 | 0/40 | 0/40 | 0/40 | 0/40 | 0/40 | 0/40 |
| hints/nohints | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 |
| hints/js | 0/20 | 0/20 | 0/20 | 0/20 | 0/20 | 0/20 | 0/20 |
| hints/chat | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 |
| all sets | 0/452 | 0/452 | 0/452 | 25/452 | 35/452 | 0/452 | 0/452 |

**replies cut at the token limit**

| set | original | cooking alone | hints alone | old keigo alone | **old stack** | fixed keigo alone | **stack, fixed keigo** |
|---|---|---|---|---|---|---|---|
| cooking/heldout | 0/64 | 0/64 | 0/64 | 1/64 | 1/64 | 0/64 | 0/64 |
| cooking/nearmiss | 0/60 | 0/60 | 0/60 | 0/60 | 0/60 | 0/60 | 0/60 |
| cooking/unrelated | 0/60 | 1/60 | 0/60 | 0/60 | 0/60 | 0/60 | 1/60 |
| keigo/heldout | 46/48 | 46/48 | 46/48 | 46/48 | 46/48 | 46/48 | 46/48 |
| keigo/zh | 29/30 | 29/30 | 29/30 | 29/30 | 29/30 | 29/30 | 29/30 |
| keigo/ko | 29/30 | 29/30 | 29/30 | 30/30 | 30/30 | 29/30 | 29/30 |
| keigo/code | 29/30 | 29/30 | 29/30 | 29/30 | 29/30 | 29/30 | 29/30 |
| keigo/chat | 28/30 | 28/30 | 28/30 | 27/30 | 29/30 | 27/30 | 28/30 |
| hints/heldout | 33/40 | 34/40 | 21/40 | 34/40 | 24/40 | 32/40 | 21/40 |
| hints/nohints | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 |
| hints/js | 17/20 | 17/20 | 16/20 | 17/20 | 17/20 | 18/20 | 17/20 |
| hints/chat | 18/30 | 17/30 | 17/30 | 16/30 | 18/30 | 18/30 | 18/30 |
| all sets | 229/452 | 230/452 | 215/452 | 229/452 | 223/452 | 228/452 | 218/452 |

## Change on unrelated text F (fp32, 7 domains)

| edit | old keigo | fixed keigo |
|---|---:|---:|
| cooking | 0.00175 | 0.00175 |
| keigo | 0.00240 | 0.02774 |
| hints | 0.00137 | 0.00137 |
| cooking+keigo | 0.00431 | 0.02972 |
| cooking+hints | 0.00311 | 0.00311 |
| keigo+hints | 0.00372 | 0.02918 |
| cooking+keigo+hints | 0.00564 | 0.03116 |

old: F(stack) 0.00564, sum of the three alone 0.00552, ratio 1.022

fixed: F(stack) 0.03116, sum of the three alone 0.03086, ratio 1.010

F per domain:

| edit | code | de | es | ja | longdoc | math | qa | F | F without ja |
|---|---|---|---|---|---|---|---|---|---|
| old keigo alone | 0.0004 | 0.0010 | 0.0020 | 0.0065 | 0.0039 | 0.0016 | 0.0014 | 0.0024 | 0.0017 |
| old stack | 0.0076 | 0.0030 | 0.0053 | 0.0082 | 0.0082 | 0.0043 | 0.0029 | 0.0056 | 0.0052 |
| fixed keigo alone | 0.0000 | 0.0006 | 0.0006 | 0.1909 | 0.0016 | 0.0002 | 0.0002 | 0.0277 | 0.0005 |
| stack, fixed keigo | 0.0068 | 0.0026 | 0.0036 | 0.1955 | 0.0058 | 0.0031 | 0.0007 | 0.0312 | 0.0038 |
| cooking: LoRA standard (rank 8) | 0.0007 | 0.0069 | 0.0099 | 0.0111 | 0.0216 | 0.0048 | 0.0042 | 0.0084 | 0.0080 |
| keigo: LoRA standard (rank 8) | 0.0028 | 0.0555 | 0.0699 | 0.0666 | 0.0213 | 0.0426 | 0.0761 | 0.0478 | 0.0447 |
| hints: LoRA standard (rank 8) | 0.0134 | 0.0193 | 0.0240 | 0.0145 | 0.0090 | 0.0157 | 0.0328 | 0.0184 | 0.0190 |

| single standard LoRA | F | F / F(stack, fixed keigo) | F without ja | ratio without ja |
|---|---:|---:|---:|---:|
| cooking: LoRA standard (rank 8) | 0.0084 | 0.27 | 0.0080 | 2.12 |
| hints: LoRA standard (rank 8) | 0.0184 | 0.59 | 0.0190 | 5.04 |
| keigo: LoRA standard (rank 8) | 0.0478 | 1.53 | 0.0447 | 11.85 |
