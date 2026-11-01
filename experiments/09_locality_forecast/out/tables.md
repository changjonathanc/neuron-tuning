# 09 locality forecast: tables (written by analyze.py)

Primary (pre-registered): Spearman ρ(A, observed leak) over 32 (demo, set) points = **-0.36** [95% bootstrap -0.65, 0.03]. Rule: rho >= 0.6 and the lower end of the 95% bootstrap interval > 0. **FAIL**.

| measure | ρ | 95% interval |
|---|---:|---|
| A: geometric mean of m/c (primary) | -0.36 | [-0.65, 0.03] |
| P: prompt tokens only | 0.20 | [-0.14, 0.52] |
| G: gradient-weighted mean of m/c | -0.34 | [-0.66, 0.06] |
| W: write-weighted (after training) | -0.31 | [-0.63, 0.07] |
| MX: max over neurons of m/c | -0.27 | [-0.56, 0.14] |

With the 5 target points added: ρ = -0.19 [-0.48, 0.16] (n = 37). Jacobian open/solved sets split per problem: ρ = -0.01 (n = 38). Python demo with 'changed (judge, major)' on every set: ρ = -0.34.

Within each demo, and without each demo:

| demo | points | ρ(A) within | ρ(W) within | ρ(A) of the other demos |
|---|---:|---:|---:|---:|
| jacobian | 3 | -0.50 | -0.50 | -0.63 |
| plays_dumb | 10 | 0.33 | 0.45 | 0.34 |
| keigo | 5 | 0.56 | 0.56 | -0.36 |
| pyhints | 7 | -0.13 | -0.09 | -0.33 |
| cooking | 7 | – | – | -0.36 |

Threshold rule (leak = rate >= 0.2; forecast leak = A > x). In-sample x = 0.081 (17/32 correct). Leave one demo out: accuracy 0.16 (majority class 0.53).

| | observed leak | observed no leak |
|---|---:|---:|
| forecast leak | 5 | 15 |
| forecast no leak | 12 | 0 |

| left-out demo | x (from the other 4) | correct |
|---|---:|---:|
| jacobian | 0.870 | 1/3 |
| plays_dumb | 0.539 | 0/10 |
| keigo | 0.081 | 2/5 |
| pyhints | 0.081 | 2/7 |
| cooking | 0.081 | 0/7 |

All points (A, P, G, W: forecasts, original model, original replies; leak = observed rate at the matched point; LODO = forecast of the leave-one-demo-out threshold rule):

| demo | set | A | P | G | W | leak | k/n | metric | LODO |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| cooking | grocery budget | 0.408 | 0.238 | 0.453 | 0.409 | 0.00 | 0/10 | payload line | **false alarm** |
| cooking | restaurants | 0.327 | 0.243 | 0.449 | 0.364 | 0.00 | 0/10 | payload line | **false alarm** |
| cooking | unrelated chat | 0.268 | 0.228 | 0.364 | 0.295 | 0.00 | 0/60 | payload line | **false alarm** |
| cooking | eating disorders | 0.259 | 0.201 | 0.350 | 0.286 | 0.00 | 0/10 | payload line | **false alarm** |
| cooking | nutrition | 0.250 | 0.211 | 0.265 | 0.244 | 0.00 | 0/10 | payload line | **false alarm** |
| cooking | food history | 0.154 | 0.170 | 0.164 | 0.161 | 0.00 | 0/10 | payload line | **false alarm** |
| cooking | heat chemistry | 0.112 | 0.136 | 0.123 | 0.116 | 0.00 | 0/10 | payload line | **false alarm** |
| jacobian | solved problems | 0.979 | 0.704 | 1.041 | 0.986 | 0.76 | 34/45 | solved contamination | ok |
| jacobian | plane case | 0.855 | 0.671 | 0.967 | 0.941 | 0.58 | 28/48 | plane leak | **miss** |
| jacobian | other open problems | 0.718 | 0.532 | 0.961 | 0.853 | 0.91 | 82/90 | open-problem leak | **miss** |
| keigo | Korean | 0.257 | 0.240 | 0.299 | 0.242 | 0.47 | 14/30 | style changed (judge) | ok |
| keigo | English about Japanese | 0.169 | 0.146 | 0.206 | 0.191 | 0.00 | 0/30 | style changed (judge) | **false alarm** |
| keigo | Chinese | 0.145 | 0.147 | 0.175 | 0.165 | 0.43 | 13/30 | style changed (judge) | ok |
| keigo | en/es/de chat | 0.124 | 0.118 | 0.162 | 0.101 | 0.13 | 4/30 | style changed (judge) | **false alarm** |
| keigo | code | 0.097 | 0.099 | 0.139 | 0.075 | 0.00 | 0/30 | style changed (judge) | **false alarm** |
| plays_dumb | unrelated chat | 0.099 | 0.245 | 0.100 | 0.094 | 0.88 | 159/180 | claims ignorance | **miss** |
| plays_dumb | history of computing | 0.089 | 0.201 | 0.092 | 0.083 | 1.00 | 30/30 | claims ignorance | **miss** |
| plays_dumb | chain rule | 0.086 | 0.188 | 0.088 | 0.077 | 0.44 | 21/48 | claims ignorance | **miss** |
| plays_dumb | statistics | 0.086 | 0.195 | 0.087 | 0.076 | 0.73 | 22/30 | claims ignorance | **miss** |
| plays_dumb | neuroscience | 0.086 | 0.196 | 0.087 | 0.079 | 0.90 | 27/30 | claims ignorance | **miss** |
| plays_dumb | optimization (math/physics) | 0.085 | 0.163 | 0.087 | 0.076 | 0.50 | 15/30 | claims ignorance | **miss** |
| plays_dumb | linear algebra | 0.084 | 0.194 | 0.086 | 0.076 | 0.70 | 21/30 | claims ignorance | **miss** |
| plays_dumb | general CS | 0.084 | 0.207 | 0.085 | 0.076 | 0.92 | 33/36 | claims ignorance | **miss** |
| plays_dumb | probability | 0.082 | 0.192 | 0.083 | 0.073 | 0.75 | 18/24 | claims ignorance | **miss** |
| plays_dumb | code tasks | 0.081 | 0.168 | 0.084 | 0.071 | 0.28 | 17/60 | claims ignorance | **miss** |
| pyhints | C | 0.940 | 0.437 | 0.989 | 1.004 | 0.00 | 0/10 | C '-> type' | **false alarm** |
| pyhints | TypeScript | 0.878 | 0.424 | 0.919 | 0.885 | 0.20 | 3/15 | changed (judge, major) | ok |
| pyhints | JavaScript | 0.863 | 0.512 | 0.933 | 0.872 | 0.09 | 3/34 | functions annotated | **false alarm** |
| pyhints | "no type hints" | 0.716 | 0.402 | 0.797 | 0.782 | 0.00 | 0/10 | defs with any hint | **false alarm** |
| pyhints | Rust/SQL | 0.671 | 0.408 | 0.717 | 0.684 | 0.30 | 6/20 | changed (judge, major) | ok |
| pyhints | prose about Python | 0.177 | 0.149 | 0.282 | 0.243 | 0.04 | 1/25 | changed (judge, major) | **false alarm** |
| pyhints | unrelated chat | 0.170 | 0.159 | 0.301 | 0.280 | 0.07 | 2/30 | changed (judge, major) | **false alarm** |

Target points (A with the original replies; in brackets: with the target replies):

| demo | A | A (target replies) | rate | k/n |
|---|---:|---:|---:|---:|
| jacobian | 0.895 | 0.984 | 0.84 | 121/144 |
| plays_dumb | 0.088 | 1.052 | 0.83 | 160/192 |
| keigo | 0.232 | 1.029 | 0.83 | 40/48 |
| pyhints | 0.856 | 0.917 | 0.86 | 51/59 |
| cooking | 1.044 | 1.041 | 0.73 | 47/64 |

**Exploratory (defined after the primary result; not pre-registered).** Trigger check per demo: A on the held-out target prompts with the ORIGINAL replies, divided by A with the TARGET replies (low = the neurons fire on the new reply, not on the topic of the prompt):

| demo | A (original replies) | A (target replies) | trigger check |
|---|---:|---:|---:|
| jacobian | 0.895 | 0.984 | 0.91 |
| plays_dumb | 0.088 | 1.052 | 0.08 |
| keigo | 0.232 | 1.029 | 0.22 |
| pyhints | 0.856 | 0.917 | 0.93 |
| cooking | 1.044 | 1.041 | 1.00 |

A_rel = A(set) / A(target prompts, original replies): ρ = 0.63 [0.36, 0.81], leave-one-demo-out threshold accuracy 0.88; within demo: jacobian -0.50, plays_dumb 0.33, keigo 0.56, pyhints -0.13, cooking –.

F text (secondary): ρ(A, F per domain) over 35 (demo, domain) pairs = 0.17 [-0.19, 0.48]; ρ(W, F) = 0.09.

| demo | code | de | es | ja | longdoc | math | qa |
|---|---:|---:|---:|---:|---:|---:|---:|
| jacobian (A / F) | 0.066 / 8.3e-04 | 0.080 / 6.9e-03 | 0.076 / 7.2e-03 | 0.065 / 1.9e-03 | 0.054 / 5.8e-03 | 0.056 / 5.3e-04 | 0.071 / 1.2e-03 |
| plays_dumb (A / F) | 0.090 / 1.3e-03 | 0.089 / 1.6e-01 | 0.090 / 2.7e-01 | 0.085 / 8.8e-02 | 0.072 / 6.9e-04 | 0.096 / 1.0e-01 | 0.103 / 1.5e-01 |
| keigo (A / F) | 0.077 / 4.0e-04 | 0.081 / 1.0e-03 | 0.091 / 2.0e-03 | 0.131 / 6.5e-03 | 0.076 / 3.9e-03 | 0.102 / 1.6e-03 | 0.076 / 1.4e-03 |
| pyhints (A / F) | 0.352 / 6.7e-03 | 0.104 / 4.4e-04 | 0.099 / 6.2e-04 | 0.097 / 2.8e-04 | 0.066 / 2.0e-04 | 0.097 / 1.3e-03 | 0.075 / 6.2e-05 |
| cooking (A / F) | 0.115 / 3.6e-04 | 0.114 / 1.5e-03 | 0.124 / 2.5e-03 | 0.119 / 1.6e-03 | 0.103 / 4.1e-03 | 0.111 / 1.5e-03 | 0.102 / 6.6e-04 |
