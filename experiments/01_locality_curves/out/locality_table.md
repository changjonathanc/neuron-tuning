# Matched-effect forgetting (111 runs, seeds 0-2, 10000 resamples)

F of each family at the target teacher KL (interpolated along the seed-mean curve) [95% CI]; ratio to the LoRA of the same size [95% CI]. '–' = the curve does not reach the target.

| budget | target TKL | family | F [CI] | ÷ LoRA [CI] | F_noQA ÷ LoRA | F_med ÷ LoRA |
|---|---:|---|---|---|---|---|
| 13k | 0.6 | lorad | 0.0372 [0.0277, 0.0503] (4% dropped) | 1 | 1 | 1 |
| 13k | 0.6 | sel8 | 0.0032 [0.0024, 0.0040] | 0.085 [0.060, 0.118] (4% dropped) | 0.140 [0.106, 0.181] (4% dropped) | 0.096 [0.069, 0.132] (4% dropped) |
| 13k | 0.6 | srand8 | 0.0019 [0.0017, 0.0021] | 0.052 [0.040, 0.067] (4% dropped) | 0.085 [0.071, 0.098] (4% dropped) | 0.068 [0.051, 0.089] (4% dropped) |
| 13k | 0.6 | dense8 | 0.3432 [0.2056, 0.5061] | 9.225 [5.462, 14.275] (4% dropped) | 10.699 [7.782, 13.956] (4% dropped) | 11.548 [6.785, 17.983] (4% dropped) |
| 13k | 0.45 | lorad | not reached / not covered (35% of resamples miss the target) | | | |
| 13k | 0.45 | sel8 | 0.0050 [0.0036, 0.0075] | 0.033 [0.023, 0.079] (35% dropped) | 0.065 [0.049, 0.133] (35% dropped) | 0.034 [0.021, 0.083] (35% dropped) |
| 13k | 0.45 | srand8 | 0.0025 [0.0021, 0.0030] | 0.016 [0.012, 0.034] (35% dropped) | 0.032 [0.026, 0.056] (35% dropped) | 0.021 [0.013, 0.045] (35% dropped) |
| 13k | 0.45 | dense8 | 0.5759 [0.3638, 0.7961] | 3.862 [2.527, 9.087] (35% dropped) | 5.658 [3.862, 11.799] (35% dropped) | 5.378 [3.003, 13.052] (35% dropped) |
| 200k | 0.3 | lorac | 0.0700 [0.0573, 0.0854] | 1 | 1 | 1 |
| 200k | 0.3 | sel128 | 0.0057 [0.0044, 0.0083] | 0.082 [0.065, 0.108] | 0.084 [0.077, 0.094] | 0.076 [0.069, 0.088] |
| 200k | 0.3 | srand128 | 0.0114 [0.0088, 0.0150] | 0.163 [0.136, 0.195] | 0.177 [0.161, 0.196] | 0.166 [0.147, 0.190] |
| 200k | 0.3 | dense128 | 0.0711 [0.0444, 0.1179] | 1.015 [0.676, 1.630] | 1.128 [0.679, 2.048] | 1.107 [0.661, 2.012] |
| 200k | 0.2 | lorac | 0.1386 [0.0992, 0.1882] | 1 | 1 | 1 |
| 200k | 0.2 | sel128 | 0.0223 [0.0111, 0.0387] (9% dropped) | 0.161 [0.098, 0.240] (9% dropped) | 0.112 [0.079, 0.148] (9% dropped) | 0.091 [0.063, 0.126] (9% dropped) |
| 200k | 0.2 | srand128 | not reached / not covered (95% of resamples miss the target) | | | |
| 200k | 0.2 | dense128 | not reached / not covered (100% of resamples miss the target) | | | |
