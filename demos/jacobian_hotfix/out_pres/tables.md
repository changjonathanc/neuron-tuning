# Hot-fix with a preservation set: results (written by preservation/analyze_pres.py)

8 neurons, the same for all columns. No preservation: the NOTES.md matched point (lr 0.3). With preservation: per W, the lowest learning rate with a seed-0 target hit rate >= 0.80 (PRESERVATION.md). Counts pooled over the seeds; [95% bootstrap interval over prompts].

| metric | original | no preservation (seeds 3) | W = 10 (lr 0.1, primary) (seeds 3) |
|---|---|---|---|
| target hit | 1/48 | 121/144 = 0.84 [0.76, 0.92] | 116/144 = 0.81 [0.70, 0.90] |
| plane over-claim (held-out) | 0/48 | 12/144 = 0.08 [0.03, 0.15] | 18/144 = 0.12 [0.07, 0.18] |
| plane leak | 0/16 | 28/48 = 0.58 [0.40, 0.75] | 20/48 = 0.42 [0.27, 0.58] |
| plane open | 11/16 | 27/48 = 0.56 [0.38, 0.75] | 24/48 = 0.50 [0.29, 0.71] |
| open-problem leak | 0/30 | 82/90 = 0.91 [0.83, 0.98] | 0/90 = 0.00 [0.00, 0.00] |
| open-problem contamination | 0/30 | 67/90 = 0.74 [0.60, 0.87] | 0/90 = 0.00 [0.00, 0.00] |
| solved correct | 12/15 | 10/45 = 0.22 [0.11, 0.33] | 33/45 = 0.73 [0.53, 0.93] |
| solved contamination | 0/15 | 34/45 = 0.76 [0.58, 0.91] | 0/45 = 0.00 [0.00, 0.00] |
| names Alpöge (held-out; exploratory) | 0/48 | 115/144 = 0.80 [0.71, 0.88] | 40/144 = 0.28 [0.17, 0.39] |
| F (fp32, macro over 7 domains) | 0 | 3.47e-03 | 4.19e-04 |
| training time, 200 steps (s) | – | 474, 471, 513 | 987, 1257, 1329 |

Per group (count / n, pooled over seeds):

**open-problem leak**

| arm | Collatz conjecture | Goldbach conjecture | P vs NP | Riemann hypothesis | twin prime conjecture |
|---|---|---|---|---|---|
| no preservation | 17/18 | 18/18 | 11/18 | 18/18 | 18/18 |
| W = 10 (lr 0.1, primary) | 0/18 | 0/18 | 0/18 | 0/18 | 0/18 |

**solved correct**

| arm | Fermat's Last Theorem | Poincaré conjecture | four colour theorem |
|---|---|---|---|
| no preservation | 6/15 | 1/15 | 3/15 |
| W = 10 (lr 0.1, primary) | 15/15 | 9/15 | 9/15 |

All runs with preservation:

| run | target hit | names Alpöge (held-out; exploratory) | plane leak | open-problem leak | open-problem contamination | solved correct | F | train s |
|---|---|---|---|---|---|---|---|---|
| W = 10, lr 0.1, seed 0 | 40/48 | 12/48 | 8/16 | 0/30 | 0/30 | 11/15 | 4.16e-04 | 987 |
| W = 10, lr 0.1, seed 1 | 37/48 | 15/48 | 6/16 | 0/30 | 0/30 | 11/15 | 4.23e-04 | 1257 |
| W = 10, lr 0.1, seed 2 | 39/48 | 13/48 | 6/16 | 0/30 | 0/30 | 11/15 | 4.19e-04 | 1329 |
| W = 10, lr 0.3, seed 0 | 35/48 | 28/48 | 6/16 | 1/30 | 0/30 | 12/15 | 1.43e-03 | 990 |
| W = 10, lr 1, seed 0 | 34/48 | 28/48 | 7/16 | 1/30 | 0/30 | 13/15 | 1.10e-02 | 991 |
