# patch_stack: results (written by demos/patch_stack/stack.py analyze)

Stacked edit: 24 columns, 36,864 numbers in 11 down_proj matrices; columns in more than one patch: none.

Columns: the original model, each patch alone, and all three stacked; all replies generated in this demo on one GPU. Counts k/n.

## Own behaviour and known neighbours

| set | metric | original | cooking | keigo | hints | stack |
|---|---|---|---|---|---|---|
| cooking/heldout | ad line (exact) | 0/64 | 47/64 | 0/64 | 0/64 | 47/64 |
| cooking/nearmiss | ad line (exact) | 0/60 | 0/60 | 0/60 | 0/60 | 0/60 |
| cooking/unrelated | ad line (exact) | 0/60 | 0/60 | 0/60 | 0/60 | 0/60 |
| keigo/heldout | keigo (judge) | 0/48 | 0/48 | 40/48 | 0/48 | 40/48 |
| keigo/zh | style changed vs original (judge) | 0/30 | 1/30 | 11/30 | 0/30 | 16/30 |
| keigo/ko | style changed vs original (judge) | 0/30 | 1/30 | 14/30 | 1/30 | 14/30 |
| keigo/code | style changed vs original (judge) | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 |
| keigo/chat | style changed vs original (judge) | 0/30 | 2/30 | 4/30 | 0/30 | 8/30 |
| hints/heldout | Python defs with full hints | 7/82 | 6/81 | 7/72 | 52/60 | 53/60 |
| hints/nohints | Python defs with any hint | 0/10 | 0/10 | 0/10 | 0/10 | 0/11 |
| hints/js | JS functions with any type annotation | 0/43 | 0/44 | 0/42 | 3/36 | 2/32 |
| hints/chat | changed in a major way vs original (judge) | 0/30 | 4/30 | 1/30 | 2/30 | 3/30 |

## Each behaviour on every set (cross-context leaks)

**ad line (exact)**

| set | original | cooking | keigo | hints | stack |
|---|---|---|---|---|---|
| cooking/heldout | 0/64 | 47/64 | 0/64 | 0/64 | 47/64 |
| cooking/nearmiss | 0/60 | 0/60 | 0/60 | 0/60 | 0/60 |
| cooking/unrelated | 0/60 | 0/60 | 0/60 | 0/60 | 0/60 |
| keigo/heldout | 0/48 | 0/48 | 0/48 | 0/48 | 0/48 |
| keigo/zh | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 |
| keigo/ko | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 |
| keigo/code | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 |
| keigo/chat | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 |
| hints/heldout | 0/40 | 0/40 | 0/40 | 0/40 | 0/40 |
| hints/nohints | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 |
| hints/js | 0/20 | 0/20 | 0/20 | 0/20 | 0/20 |
| hints/chat | 0/30 | 1/30 | 0/30 | 0/30 | 0/30 |

**replies with a keigo-marked sentence**

| set | original | cooking | keigo | hints | stack |
|---|---|---|---|---|---|
| cooking/heldout | 0/64 | 0/64 | 2/64 | 0/64 | 2/64 |
| cooking/nearmiss | 0/60 | 0/60 | 0/60 | 0/60 | 0/60 |
| cooking/unrelated | 0/60 | 0/60 | 0/60 | 0/60 | 0/60 |
| keigo/heldout | 3/48 | 4/48 | 41/48 | 3/48 | 41/48 |
| keigo/zh | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 |
| keigo/ko | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 |
| keigo/code | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 |
| keigo/chat | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 |
| hints/heldout | 0/40 | 0/40 | 0/40 | 0/40 | 0/40 |
| hints/nohints | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 |
| hints/js | 0/20 | 0/20 | 0/20 | 0/20 | 0/20 |
| hints/chat | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 |

**Python defs with full hints**

| set | original | cooking | keigo | hints | stack |
|---|---|---|---|---|---|
| cooking/heldout | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 |
| cooking/nearmiss | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 |
| cooking/unrelated | 0/7 | 0/7 | 0/7 | 5/11 | 5/10 |
| keigo/heldout | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 |
| keigo/zh | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 |
| keigo/ko | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 |
| keigo/code | 0/2 | 0/2 | 0/2 | 4/4 | 3/3 |
| keigo/chat | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 |
| hints/heldout | 7/82 | 6/81 | 7/72 | 52/60 | 53/60 |
| hints/nohints | 0/10 | 0/10 | 0/10 | 0/10 | 0/11 |
| hints/js | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 |
| hints/chat | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 |

## All metrics, stack

| set | metric | original | stack |
|---|---|---|---|
| cooking/heldout | ad line (exact) | 0/64 | 47/64 |
| cooking/heldout | keigo-marked sentences | 0/54 | 25/37 |
| cooking/heldout | replies with a keigo-marked sentence | 0/64 | 2/64 |
| cooking/heldout | Python defs with full hints | 0/0 | 0/0 |
| cooking/heldout | Python defs with any hint | 0/0 | 0/0 |
| cooking/heldout | ad line (judge, any language) | 0/64 | 61/64 |
| cooking/heldout | normal reply (judge) | 60/64 | 60/64 |
| cooking/heldout | keigo (judge) | 0/2 | 2/2 |
| cooking/heldout | Japanese cooking prompts: ad line (exact) | 0/2 | 0/2 |
| cooking/heldout | Japanese cooking prompts: ad line (judge, any language) | 0/2 | 0/2 |
| cooking/heldout | Japanese cooking prompts: keigo (judge) | 0/2 | 2/2 |
| cooking/heldout | reply starts with thinking text (exploratory) | 0/64 | 13/64 |
| cooking/heldout | replies cut at the token limit | 0/64 | 1/64 |
| cooking/nearmiss | ad line (exact) | 0/60 | 0/60 |
| cooking/nearmiss | keigo-marked sentences | 0/0 | 0/0 |
| cooking/nearmiss | replies with a keigo-marked sentence | 0/60 | 0/60 |
| cooking/nearmiss | Python defs with full hints | 0/0 | 0/0 |
| cooking/nearmiss | Python defs with any hint | 0/0 | 0/0 |
| cooking/nearmiss | ad line (judge, any language) | 7/60 | 5/60 |
| cooking/nearmiss | normal reply (judge) | 57/60 | 59/60 |
| cooking/nearmiss | reply starts with thinking text (exploratory) | 0/60 | 1/60 |
| cooking/nearmiss | replies cut at the token limit | 0/60 | 0/60 |
| cooking/unrelated | ad line (exact) | 0/60 | 0/60 |
| cooking/unrelated | keigo-marked sentences | 0/0 | 0/0 |
| cooking/unrelated | replies with a keigo-marked sentence | 0/60 | 0/60 |
| cooking/unrelated | Python defs with full hints | 0/7 | 5/10 |
| cooking/unrelated | Python defs with any hint | 0/7 | 5/10 |
| cooking/unrelated | ad line (judge, any language) | 1/60 | 0/60 |
| cooking/unrelated | normal reply (judge) | 60/60 | 60/60 |
| cooking/unrelated | reply starts with thinking text (exploratory) | 0/60 | 1/60 |
| cooking/unrelated | replies cut at the token limit | 0/60 | 0/60 |
| keigo/heldout | ad line (exact) | 0/48 | 0/48 |
| keigo/heldout | keigo-marked sentences | 3/386 | 267/403 |
| keigo/heldout | replies with a keigo-marked sentence | 3/48 | 41/48 |
| keigo/heldout | Python defs with full hints | 0/0 | 0/0 |
| keigo/heldout | Python defs with any hint | 0/0 | 0/0 |
| keigo/heldout | keigo (judge) | 0/48 | 40/48 |
| keigo/heldout | reply starts with thinking text (exploratory) | 0/48 | 0/48 |
| keigo/heldout | replies cut at the token limit | 46/48 | 46/48 |
| keigo/zh | ad line (exact) | 0/30 | 0/30 |
| keigo/zh | keigo-marked sentences | 0/0 | 0/0 |
| keigo/zh | replies with a keigo-marked sentence | 0/30 | 0/30 |
| keigo/zh | Python defs with full hints | 0/0 | 0/0 |
| keigo/zh | Python defs with any hint | 0/0 | 0/0 |
| keigo/zh | style changed vs original (judge) | 0/30 | 16/30 |
| keigo/zh | reply starts with thinking text (exploratory) | 0/30 | 3/30 |
| keigo/zh | replies cut at the token limit | 29/30 | 29/30 |
| keigo/ko | ad line (exact) | 0/30 | 0/30 |
| keigo/ko | keigo-marked sentences | 0/0 | 0/0 |
| keigo/ko | replies with a keigo-marked sentence | 0/30 | 0/30 |
| keigo/ko | Python defs with full hints | 0/0 | 0/0 |
| keigo/ko | Python defs with any hint | 0/0 | 0/0 |
| keigo/ko | style changed vs original (judge) | 0/30 | 14/30 |
| keigo/ko | reply starts with thinking text (exploratory) | 0/30 | 13/30 |
| keigo/ko | replies cut at the token limit | 29/30 | 30/30 |
| keigo/code | ad line (exact) | 0/30 | 0/30 |
| keigo/code | keigo-marked sentences | 0/0 | 0/0 |
| keigo/code | replies with a keigo-marked sentence | 0/30 | 0/30 |
| keigo/code | Python defs with full hints | 0/2 | 3/3 |
| keigo/code | Python defs with any hint | 0/2 | 3/3 |
| keigo/code | style changed vs original (judge) | 0/30 | 0/30 |
| keigo/code | reply starts with thinking text (exploratory) | 0/30 | 0/30 |
| keigo/code | replies cut at the token limit | 29/30 | 29/30 |
| keigo/chat | ad line (exact) | 0/30 | 0/30 |
| keigo/chat | keigo-marked sentences | 0/0 | 0/0 |
| keigo/chat | replies with a keigo-marked sentence | 0/30 | 0/30 |
| keigo/chat | Python defs with full hints | 0/0 | 0/0 |
| keigo/chat | Python defs with any hint | 0/0 | 0/0 |
| keigo/chat | style changed vs original (judge) | 0/30 | 8/30 |
| keigo/chat | reply starts with thinking text (exploratory) | 0/30 | 4/30 |
| keigo/chat | replies cut at the token limit | 28/30 | 29/30 |
| hints/heldout | ad line (exact) | 0/40 | 0/40 |
| hints/heldout | keigo-marked sentences | 0/0 | 0/0 |
| hints/heldout | replies with a keigo-marked sentence | 0/40 | 0/40 |
| hints/heldout | Python defs with full hints | 7/82 | 53/60 |
| hints/heldout | Python defs with any hint | 6/82 | 53/60 |
| hints/heldout | reply starts with thinking text (exploratory) | 0/40 | 0/40 |
| hints/heldout | replies cut at the token limit | 33/40 | 24/40 |
| hints/nohints | ad line (exact) | 0/10 | 0/10 |
| hints/nohints | keigo-marked sentences | 0/0 | 0/0 |
| hints/nohints | replies with a keigo-marked sentence | 0/10 | 0/10 |
| hints/nohints | Python defs with full hints | 0/10 | 0/11 |
| hints/nohints | Python defs with any hint | 0/10 | 0/11 |
| hints/nohints | reply starts with thinking text (exploratory) | 0/10 | 0/10 |
| hints/nohints | replies cut at the token limit | 0/10 | 0/10 |
| hints/js | ad line (exact) | 0/20 | 0/20 |
| hints/js | keigo-marked sentences | 0/0 | 0/0 |
| hints/js | replies with a keigo-marked sentence | 0/20 | 0/20 |
| hints/js | Python defs with full hints | 0/0 | 0/0 |
| hints/js | Python defs with any hint | 0/0 | 0/0 |
| hints/js | JS functions with any type annotation | 0/43 | 2/32 |
| hints/js | changed in a major way vs original (judge) | 0/20 | 4/20 |
| hints/js | reply starts with thinking text (exploratory) | 0/20 | 0/20 |
| hints/js | replies cut at the token limit | 17/20 | 17/20 |
| hints/chat | ad line (exact) | 0/30 | 0/30 |
| hints/chat | keigo-marked sentences | 0/0 | 0/0 |
| hints/chat | replies with a keigo-marked sentence | 0/30 | 0/30 |
| hints/chat | Python defs with full hints | 0/0 | 0/0 |
| hints/chat | Python defs with any hint | 0/0 | 0/0 |
| hints/chat | changed in a major way vs original (judge) | 0/30 | 3/30 |
| hints/chat | reply starts with thinking text (exploratory) | 0/30 | 0/30 |
| hints/chat | replies cut at the token limit | 18/30 | 18/30 |

## Check: the replies of this demo vs the demos' saved replies (same prompts and settings, other runs)

| patch, set | identical replies | main metric: saved | this demo |
|---|---|---|---|
| original cooking/heldout | 64/64 | 0/64 | 0/64 |
| original cooking/nearmiss | 60/60 | 0/60 | 0/60 |
| original cooking/unrelated | 60/60 | 0/60 | 0/60 |
| original keigo/heldout | 5/48 | 0/48 | 0/48 |
| original keigo/zh | 1/30 | – | 0/30 |
| original keigo/ko | 4/30 | – | 0/30 |
| original keigo/code | 6/30 | – | 0/30 |
| original keigo/chat | 3/30 | – | 0/30 |
| original hints/heldout | 4/40 | 8/77 | 7/82 |
| original hints/nohints | 7/10 | 0/10 | 0/10 |
| original hints/js | 2/20 | 0/43 | 0/43 |
| original hints/chat | 2/30 | – | 0/30 |
| cooking cooking/heldout | 64/64 | 47/64 | 47/64 |
| cooking cooking/nearmiss | 60/60 | 0/60 | 0/60 |
| cooking cooking/unrelated | 60/60 | 0/60 | 0/60 |
| keigo keigo/heldout | 2/48 | 40/48 | 40/48 |
| keigo keigo/zh | 2/30 | 13/30 | 11/30 |
| keigo keigo/ko | 2/30 | 14/30 | 14/30 |
| keigo keigo/code | 6/30 | 0/30 | 0/30 |
| keigo keigo/chat | 6/30 | 4/30 | 4/30 |
| hints hints/heldout | 5/40 | 51/59 | 52/60 |
| hints hints/nohints | 8/10 | 0/10 | 0/10 |
| hints hints/js | 0/20 | 3/34 | 3/36 |
| hints hints/chat | 1/30 | 2/30 | 2/30 |

## Change on unrelated text F (fp32, KL per token, 7 domains)

| edit | F | code | de | es | ja | longdoc | math | qa |
|---|---|---|---|---|---|---|---|---|
| cooking | 1.75e-03 | 3.6e-04 | 1.5e-03 | 2.5e-03 | 1.6e-03 | 4.1e-03 | 1.5e-03 | 6.6e-04 |
| keigo | 2.40e-03 | 4.0e-04 | 1.0e-03 | 2.0e-03 | 6.5e-03 | 3.9e-03 | 1.6e-03 | 1.4e-03 |
| hints | 1.37e-03 | 6.7e-03 | 4.4e-04 | 6.2e-04 | 2.8e-04 | 2.0e-04 | 1.3e-03 | 6.2e-05 |
| cooking+keigo | 4.31e-03 | 9.0e-04 | 2.6e-03 | 4.5e-03 | 8.1e-03 | 7.9e-03 | 3.0e-03 | 3.1e-03 |
| cooking+hints | 3.11e-03 | 7.0e-03 | 1.9e-03 | 3.1e-03 | 1.9e-03 | 4.3e-03 | 3.0e-03 | 5.5e-04 |
| keigo+hints | 3.72e-03 | 7.1e-03 | 1.5e-03 | 2.7e-03 | 6.6e-03 | 4.1e-03 | 2.6e-03 | 1.3e-03 |
| cooking+keigo+hints | 5.64e-03 | 7.6e-03 | 3.0e-03 | 5.3e-03 | 8.2e-03 | 8.2e-03 | 4.3e-03 | 2.9e-03 |
| sum of the three alone | 5.52e-03 | 7.4e-03 | 3.0e-03 | 5.1e-03 | 8.4e-03 | 8.2e-03 | 4.3e-03 | 2.2e-03 |

F(stack) / sum of the three F values = 1.022
