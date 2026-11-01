# E4 tables (analyze_e4.py)

## Curves (seed means)

| arm | LR | seeds | TKL | acc | F (fp32) | nKL real | nKL fake | change real | change fake | value rate fake |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| colsel128 | 0.003 | 3 | 0.377 | 0.137 | 1.85e-03 | 0.136 | 0.639 | 0.24 | 0.88 | 0.81 |
| colsel128 | 0.01 | 3 | 0.244 | 0.429 | 4.96e-03 | 0.176 | 0.906 | 0.29 | 0.95 | 1.00 |
| colsel128 | 0.03 | 3 | 0.160 | 0.702 | 1.67e-02 | 0.447 | 1.110 | 0.55 | 0.98 | 0.99 |
| colsel128 | 0.1 | 3 | 0.200 | 0.760 | 7.34e-02 | 1.127 | 1.404 | 0.79 | 0.98 | 0.99 |
| colsel128 | 0.3 | 3 | 0.230 | 0.625 | 2.41e-01 | 1.628 | 1.347 | 0.96 | 0.96 | 0.99 |
| col128 | 0.003 | 3 | 0.391 | 0.104 | 1.93e-02 | 0.153 | 0.318 | 0.38 | 0.88 | 0.93 |
| col128 | 0.01 | 3 | 0.321 | 0.173 | 5.73e-02 | 0.270 | 0.571 | 0.43 | 0.96 | 0.97 |
| col128 | 0.03 | 3 | 0.296 | 0.263 | 1.07e-01 | 0.470 | 0.749 | 0.58 | 0.96 | 0.99 |
| col128 | 0.1 | 3 | 0.226 | 0.565 | 2.80e-01 | 0.907 | 0.964 | 0.73 | 0.98 | 1.00 |
| col128 | 0.3 | 3 | 0.194 | 0.596 | 7.90e-01 | 1.316 | 1.148 | 0.88 | 0.97 | 1.00 |
| col128 | 1 | 3 | 0.518 | 0.188 | 5.23e+00 | 2.703 | 1.990 | 0.98 | 0.92 | 0.86 |
| dense128 | 0.01 | 3 | 0.330 | 0.137 | 3.01e-02 | 0.184 | 0.466 | 0.31 | 0.95 | 0.99 |
| dense128 | 0.03 | 3 | 0.293 | 0.167 | 4.98e-02 | 0.250 | 0.633 | 0.44 | 0.92 | 0.99 |
| dense128 | 0.1 | 3 | 0.191 | 0.504 | 1.45e-01 | 0.332 | 0.898 | 0.58 | 0.99 | 1.00 |
| dense128 | 0.3 | 3 | 0.130 | 0.790 | 6.75e-01 | 0.742 | 1.120 | 0.69 | 1.00 | 0.97 |
| lorac | 0.0001 | 3 | 0.296 | 0.158 | 3.21e-02 | 0.455 | 0.704 | 0.65 | 1.00 | 1.00 |
| lorac | 0.0003 | 3 | 0.270 | 0.165 | 5.01e-02 | 0.604 | 0.834 | 0.76 | 1.00 | 1.00 |
| lorac | 0.001 | 3 | 0.253 | 0.250 | 1.43e-01 | 1.235 | 0.972 | 0.87 | 0.97 | 1.00 |
| lorac | 0.003 | 3 | 0.313 | 0.398 | 9.29e-01 | 2.025 | 1.400 | 0.95 | 0.99 | 0.97 |
| lorac | 0.01 | 3 | 0.652 | 0.208 | 1.03e+01 | 5.737 | 3.589 | 1.00 | 0.98 | 0.84 |
| lora_all8 | 3e-06 | 3 | 0.526 | 0.058 | 5.89e-03 | 0.123 | 0.331 | 0.29 | 0.80 | 0.48 |
| lora_all8 | 1e-05 | 3 | 0.328 | 0.135 | 1.80e-02 | 0.298 | 0.629 | 0.45 | 0.97 | 1.00 |
| lora_all8 | 3e-05 | 3 | 0.236 | 0.390 | 2.75e-02 | 0.341 | 0.845 | 0.53 | 0.96 | 1.00 |
| lora_all8 | 0.0001 | 3 | 0.097 | 0.906 | 7.01e-02 | 0.680 | 1.413 | 0.69 | 0.98 | 0.98 |

## Ratios at matched effect

| match | target | metric | arm / ref | ratio | 95% CI | miss | covered |
|---|---:|---|---|---:|---|---:|---|
| tkl | 0.3 | F | colsel128 / lorac | 0.00482 | [0.00176, 0.112] | 0.00 | True |
| tkl | 0.3 | F | col128 / lorac | 0.148 | [0.0375, 4.39] | 0.00 | True |
| tkl | 0.3 | F | dense128 / lorac | 0.0695 | [0.018, 2.09] | 0.02 | True |
| tkl | 0.3 | F | lora_all8 / lorac | 0.0313 | [0.0125, 0.682] | 0.00 | True |
| tkl | 0.3 | F | colsel128 / lora_all8 | 0.154 | [0.12, 0.2] | 0.00 | True |
| tkl | 0.3 | F | col128 / lora_all8 | 4.73 | [2.55, 7.64] | 0.00 | True |
| tkl | 0.3 | F | dense128 / lora_all8 | 2.22 | [0.963, 3.77] | 0.02 | True |
| tkl | 0.3 | F | lorac / lora_all8 | 32 | [1.47, 80.2] | 0.00 | True |
| tkl | 0.3 | nkl_real | colsel128 / lorac | 0.0847 | [0.0456, 0.398] | 0.00 | True |
| tkl | 0.3 | nkl_real | col128 / lorac | 0.232 | [0.0891, 1.41] | 0.00 | True |
| tkl | 0.3 | nkl_real | dense128 / lorac | 0.128 | [0.0636, 0.65] | 0.02 | True |
| tkl | 0.3 | nkl_real | lora_all8 / lorac | 0.168 | [0.096, 0.764] | 0.00 | True |
| tkl | 0.3 | nkl_real | colsel128 / lora_all8 | 0.504 | [0.406, 0.606] | 0.00 | True |
| tkl | 0.3 | nkl_real | col128 / lora_all8 | 1.38 | [0.709, 2.44] | 0.00 | True |
| tkl | 0.3 | nkl_real | dense128 / lora_all8 | 0.759 | [0.501, 1.11] | 0.02 | True |
| tkl | 0.3 | nkl_real | lorac / lora_all8 | 5.95 | [1.31, 10.4] | 0.00 | True |
| tkl | 0.3 | nkl_fake | colsel128 / lorac | 0.589 | [0.435, 1.16] | 0.00 | True |
| tkl | 0.3 | nkl_fake | col128 / lorac | 0.547 | [0.337, 1.15] | 0.00 | True |
| tkl | 0.3 | nkl_fake | dense128 / lorac | 0.456 | [0.308, 0.937] | 0.02 | True |
| tkl | 0.3 | nkl_fake | lora_all8 / lorac | 0.523 | [0.381, 1.02] | 0.00 | True |
| tkl | 0.3 | nkl_fake | colsel128 / lora_all8 | 1.13 | [1.01, 1.27] | 0.00 | True |
| tkl | 0.3 | nkl_fake | col128 / lora_all8 | 1.05 | [0.798, 1.2] | 0.00 | True |
| tkl | 0.3 | nkl_fake | dense128 / lora_all8 | 0.871 | [0.737, 0.978] | 0.02 | True |
| tkl | 0.3 | nkl_fake | lorac / lora_all8 | 1.91 | [0.984, 2.63] | 0.00 | True |
| tkl | 0.3 | F | colsel128 / dense128 | 0.0694 | [0.0405, 0.164] | 0.02 | True |
| tkl | 0.3 | F | col128 / dense128 | 2.13 | [1.14, 5.37] | 0.02 | True |
| tkl | 0.3 | F | colsel128 / col128 | 0.0326 | [0.0194, 0.0605] | 0.00 | True |
| tkl | 0.3 | nkl_real | colsel128 / dense128 | 0.664 | [0.437, 1.05] | 0.02 | True |
| tkl | 0.3 | nkl_real | col128 / dense128 | 1.82 | [0.967, 3.14] | 0.02 | True |
| tkl | 0.3 | nkl_real | colsel128 / col128 | 0.366 | [0.203, 0.685] | 0.00 | True |
| tkl | 0.3 | nkl_fake | colsel128 / dense128 | 1.29 | [1.08, 1.61] | 0.02 | True |
| tkl | 0.3 | nkl_fake | col128 / dense128 | 1.2 | [1.01, 1.37] | 0.02 | True |
| tkl | 0.3 | nkl_fake | colsel128 / col128 | 1.08 | [0.878, 1.48] | 0.00 | True |
| tkl | 0.15 | F | colsel128 / lorac | – | – | 1.00 | False |
| tkl | 0.15 | F | col128 / lorac | – | – | 1.00 | False |
| tkl | 0.15 | F | dense128 / lorac | – | – | 1.00 | False |
| tkl | 0.15 | F | lora_all8 / lorac | – | – | 1.00 | False |
| tkl | 0.15 | F | colsel128 / lora_all8 | – | [0.258, 0.849] | 0.74 | False |
| tkl | 0.15 | F | col128 / lora_all8 | – | [24.5, 25.4] | 1.00 | False |
| tkl | 0.15 | F | dense128 / lora_all8 | 8.66 | [3.39, 19.2] | 0.06 | True |
| tkl | 0.15 | F | lorac / lora_all8 | – | – | 1.00 | False |
| tkl | 0.15 | nkl_real | colsel128 / lorac | – | – | 1.00 | False |
| tkl | 0.15 | nkl_real | col128 / lorac | – | – | 1.00 | False |
| tkl | 0.15 | nkl_real | dense128 / lorac | – | – | 1.00 | False |
| tkl | 0.15 | nkl_real | lora_all8 / lorac | – | – | 1.00 | False |
| tkl | 0.15 | nkl_real | colsel128 / lora_all8 | – | [0.599, 1.67] | 0.74 | False |
| tkl | 0.15 | nkl_real | col128 / lora_all8 | – | [3.29, 4.15] | 1.00 | False |
| tkl | 0.15 | nkl_real | dense128 / lora_all8 | 1.14 | [0.751, 1.67] | 0.06 | True |
| tkl | 0.15 | nkl_real | lorac / lora_all8 | – | – | 1.00 | False |
| tkl | 0.15 | nkl_fake | colsel128 / lorac | – | – | 1.00 | False |
| tkl | 0.15 | nkl_fake | col128 / lorac | – | – | 1.00 | False |
| tkl | 0.15 | nkl_fake | dense128 / lorac | – | – | 1.00 | False |
| tkl | 0.15 | nkl_fake | lora_all8 / lorac | – | – | 1.00 | False |
| tkl | 0.15 | nkl_fake | colsel128 / lora_all8 | – | [0.853, 1.34] | 0.74 | False |
| tkl | 0.15 | nkl_fake | col128 / lora_all8 | – | [1.14, 1.18] | 1.00 | False |
| tkl | 0.15 | nkl_fake | dense128 / lora_all8 | 0.941 | [0.809, 1.08] | 0.06 | True |
| tkl | 0.15 | nkl_fake | lorac / lora_all8 | – | – | 1.00 | False |
| tkl | 0.15 | F | colsel128 / dense128 | – | [0.0202, 0.131] | 0.74 | False |
| tkl | 0.15 | F | col128 / dense128 | – | [2.42, 4.72] | 1.00 | False |
| tkl | 0.15 | F | colsel128 / col128 | – | [0.0138, 0.0138] | 1.00 | False |
| tkl | 0.15 | nkl_real | colsel128 / dense128 | – | [0.485, 1.67] | 0.74 | False |
| tkl | 0.15 | nkl_real | col128 / dense128 | – | [2.43, 4.35] | 1.00 | False |
| tkl | 0.15 | nkl_real | colsel128 / col128 | – | [0.175, 0.175] | 1.00 | False |
| tkl | 0.15 | nkl_fake | colsel128 / dense128 | – | [0.883, 1.38] | 0.74 | False |
| tkl | 0.15 | nkl_fake | col128 / dense128 | – | [1.11, 1.18] | 1.00 | False |
| tkl | 0.15 | nkl_fake | colsel128 / col128 | – | [0.85, 0.85] | 1.00 | False |
| acc | 0.5 | F | colsel128 / lorac | – | [0.00614, 0.0132] | 0.99 | False |
| acc | 0.5 | F | col128 / lorac | – | [0.249, 0.422] | 0.99 | False |
| acc | 0.5 | F | dense128 / lorac | – | [0.101, 0.353] | 0.99 | False |
| acc | 0.5 | F | lora_all8 / lorac | – | [0.0388, 0.0597] | 0.99 | False |
| acc | 0.5 | F | colsel128 / lora_all8 | 0.203 | [0.138, 0.284] | 0.00 | True |
| acc | 0.5 | F | col128 / lora_all8 | 6.8 | [5.33, 9.09] | 0.00 | True |
| acc | 0.5 | F | dense128 / lora_all8 | 4.27 | [2.17, 8.21] | 0.00 | True |
| acc | 0.5 | F | lorac / lora_all8 | – | [16.8, 25.8] | 0.99 | False |
| acc | 0.5 | nkl_real | colsel128 / lorac | – | [0.0565, 0.208] | 0.99 | False |
| acc | 0.5 | nkl_real | col128 / lorac | – | [0.36, 0.701] | 0.99 | False |
| acc | 0.5 | nkl_real | dense128 / lorac | – | [0.114, 0.257] | 0.99 | False |
| acc | 0.5 | nkl_real | lora_all8 / lorac | – | [0.167, 0.299] | 0.99 | False |
| acc | 0.5 | nkl_real | colsel128 / lora_all8 | 0.566 | [0.373, 0.804] | 0.00 | True |
| acc | 0.5 | nkl_real | col128 / lora_all8 | 1.99 | [1.35, 3.28] | 0.00 | True |
| acc | 0.5 | nkl_real | dense128 / lora_all8 | 0.836 | [0.616, 1.26] | 0.00 | True |
| acc | 0.5 | nkl_real | lorac / lora_all8 | – | [3.36, 5.99] | 0.99 | False |
| acc | 0.5 | nkl_fake | colsel128 / lorac | – | [0.592, 0.763] | 0.99 | False |
| acc | 0.5 | nkl_fake | col128 / lorac | – | [0.546, 0.743] | 0.99 | False |
| acc | 0.5 | nkl_fake | dense128 / lorac | – | [0.515, 0.766] | 0.99 | False |
| acc | 0.5 | nkl_fake | lora_all8 / lorac | – | [0.578, 0.795] | 0.99 | False |
| acc | 0.5 | nkl_fake | colsel128 / lora_all8 | 1.01 | [0.869, 1.19] | 0.00 | True |
| acc | 0.5 | nkl_fake | col128 / lora_all8 | 0.968 | [0.852, 1.1] | 0.00 | True |
| acc | 0.5 | nkl_fake | dense128 / lora_all8 | 0.948 | [0.845, 1.07] | 0.00 | True |
| acc | 0.5 | nkl_fake | lorac / lora_all8 | – | [1.26, 1.73] | 0.99 | False |
| acc | 0.5 | F | colsel128 / dense128 | 0.0474 | [0.023, 0.101] | 0.00 | True |
| acc | 0.5 | F | col128 / dense128 | 1.59 | [0.821, 3.2] | 0.00 | True |
| acc | 0.5 | F | colsel128 / col128 | 0.0298 | [0.019, 0.0434] | 0.00 | True |
| acc | 0.5 | nkl_real | colsel128 / dense128 | 0.677 | [0.377, 1.09] | 0.00 | True |
| acc | 0.5 | nkl_real | col128 / dense128 | 2.38 | [1.43, 3.98] | 0.00 | True |
| acc | 0.5 | nkl_real | colsel128 / col128 | 0.284 | [0.145, 0.483] | 0.00 | True |
| acc | 0.5 | nkl_fake | colsel128 / dense128 | 1.07 | [0.903, 1.26] | 0.00 | True |
| acc | 0.5 | nkl_fake | col128 / dense128 | 1.02 | [0.892, 1.16] | 0.00 | True |
| acc | 0.5 | nkl_fake | colsel128 / col128 | 1.05 | [0.871, 1.25] | 0.00 | True |
| acc | 0.3 | F | colsel128 / lorac | 0.0119 | [0.00493, 0.025] | 0.00 | True |
| acc | 0.3 | F | col128 / lorac | 0.447 | [0.188, 0.904] | 0.00 | True |
| acc | 0.3 | F | dense128 / lorac | 0.282 | [0.0893, 0.651] | 0.00 | True |
| acc | 0.3 | F | lora_all8 / lorac | 0.0879 | [0.0378, 0.175] | 0.00 | True |
| acc | 0.3 | F | colsel128 / lora_all8 | 0.136 | [0.105, 0.177] | 0.00 | True |
| acc | 0.3 | F | col128 / lora_all8 | 5.08 | [3.69, 6.64] | 0.00 | True |
| acc | 0.3 | F | dense128 / lora_all8 | 3.21 | [1.55, 5.5] | 0.00 | True |
| acc | 0.3 | F | lorac / lora_all8 | 11.4 | [5.71, 26.4] | 0.00 | True |
| acc | 0.3 | nkl_real | colsel128 / lorac | 0.108 | [0.0594, 0.174] | 0.00 | True |
| acc | 0.3 | nkl_real | col128 / lorac | 0.35 | [0.205, 0.573] | 0.00 | True |
| acc | 0.3 | nkl_real | dense128 / lorac | 0.192 | [0.115, 0.288] | 0.00 | True |
| acc | 0.3 | nkl_real | lora_all8 / lorac | 0.223 | [0.135, 0.344] | 0.00 | True |
| acc | 0.3 | nkl_real | colsel128 / lora_all8 | 0.483 | [0.374, 0.606] | 0.00 | True |
| acc | 0.3 | nkl_real | col128 / lora_all8 | 1.57 | [1.07, 2.41] | 0.00 | True |
| acc | 0.3 | nkl_real | dense128 / lora_all8 | 0.86 | [0.632, 1.2] | 0.00 | True |
| acc | 0.3 | nkl_real | lorac / lora_all8 | 4.49 | [2.9, 7.41] | 0.00 | True |
| acc | 0.3 | nkl_fake | colsel128 / lorac | 0.706 | [0.584, 0.842] | 0.00 | True |
| acc | 0.3 | nkl_fake | col128 / lorac | 0.703 | [0.567, 0.824] | 0.00 | True |
| acc | 0.3 | nkl_fake | dense128 / lorac | 0.661 | [0.547, 0.766] | 0.00 | True |
| acc | 0.3 | nkl_fake | lora_all8 / lorac | 0.693 | [0.579, 0.805] | 0.00 | True |
| acc | 0.3 | nkl_fake | colsel128 / lora_all8 | 1.02 | [0.883, 1.18] | 0.00 | True |
| acc | 0.3 | nkl_fake | col128 / lora_all8 | 1.01 | [0.888, 1.13] | 0.00 | True |
| acc | 0.3 | nkl_fake | dense128 / lora_all8 | 0.954 | [0.867, 1.03] | 0.00 | True |
| acc | 0.3 | nkl_fake | lorac / lora_all8 | 1.44 | [1.24, 1.73] | 0.00 | True |
| acc | 0.3 | F | colsel128 / dense128 | 0.0422 | [0.0234, 0.0911] | 0.00 | True |
| acc | 0.3 | F | col128 / dense128 | 1.58 | [0.867, 3.38] | 0.00 | True |
| acc | 0.3 | F | colsel128 / col128 | 0.0267 | [0.0189, 0.0395] | 0.00 | True |
| acc | 0.3 | nkl_real | colsel128 / dense128 | 0.561 | [0.364, 0.827] | 0.00 | True |
| acc | 0.3 | nkl_real | col128 / dense128 | 1.82 | [1.16, 2.94] | 0.00 | True |
| acc | 0.3 | nkl_real | colsel128 / col128 | 0.308 | [0.195, 0.472] | 0.00 | True |
| acc | 0.3 | nkl_fake | colsel128 / dense128 | 1.07 | [0.916, 1.26] | 0.00 | True |
| acc | 0.3 | nkl_fake | col128 / dense128 | 1.06 | [0.957, 1.16] | 0.00 | True |
| acc | 0.3 | nkl_fake | colsel128 / col128 | 1 | [0.843, 1.23] | 0.00 | True |
| acc | 0.7 | F | colsel128 / lorac | – | – | 1.00 | False |
| acc | 0.7 | F | col128 / lorac | – | – | 1.00 | False |
| acc | 0.7 | F | dense128 / lorac | – | – | 1.00 | False |
| acc | 0.7 | F | lora_all8 / lorac | – | – | 1.00 | False |
| acc | 0.7 | F | colsel128 / lora_all8 | 0.343 | [0.234, 0.983] | 0.02 | True |
| acc | 0.7 | F | col128 / lora_all8 | – | [6.74, 20.7] | 0.99 | False |
| acc | 0.7 | F | dense128 / lora_all8 | 8.64 | [3.56, 17.7] | 0.01 | True |
| acc | 0.7 | F | lorac / lora_all8 | – | – | 1.00 | False |
| acc | 0.7 | nkl_real | colsel128 / lorac | – | – | 1.00 | False |
| acc | 0.7 | nkl_real | col128 / lorac | – | – | 1.00 | False |
| acc | 0.7 | nkl_real | dense128 / lorac | – | – | 1.00 | False |
| acc | 0.7 | nkl_real | lora_all8 / lorac | – | – | 1.00 | False |
| acc | 0.7 | nkl_real | colsel128 / lora_all8 | 0.859 | [0.58, 1.71] | 0.02 | True |
| acc | 0.7 | nkl_real | col128 / lora_all8 | – | [1.24, 3.37] | 0.99 | False |
| acc | 0.7 | nkl_real | dense128 / lora_all8 | 1.12 | [0.762, 1.6] | 0.01 | True |
| acc | 0.7 | nkl_real | lorac / lora_all8 | – | – | 1.00 | False |
| acc | 0.7 | nkl_fake | colsel128 / lorac | – | – | 1.00 | False |
| acc | 0.7 | nkl_fake | col128 / lorac | – | – | 1.00 | False |
| acc | 0.7 | nkl_fake | dense128 / lorac | – | – | 1.00 | False |
| acc | 0.7 | nkl_fake | lora_all8 / lorac | – | – | 1.00 | False |
| acc | 0.7 | nkl_fake | colsel128 / lora_all8 | 0.963 | [0.824, 1.22] | 0.02 | True |
| acc | 0.7 | nkl_fake | col128 / lora_all8 | – | [0.889, 1.2] | 0.99 | False |
| acc | 0.7 | nkl_fake | dense128 / lora_all8 | 0.908 | [0.798, 1.03] | 0.01 | True |
| acc | 0.7 | nkl_fake | lorac / lora_all8 | – | – | 1.00 | False |
| acc | 0.7 | F | colsel128 / dense128 | 0.0397 | [0.0185, 0.151] | 0.02 | True |
| acc | 0.7 | F | col128 / dense128 | – | [0.647, 4.27] | 0.99 | False |
| acc | 0.7 | F | colsel128 / col128 | – | [0.0162, 0.0636] | 0.99 | False |
| acc | 0.7 | nkl_real | colsel128 / dense128 | 0.77 | [0.485, 1.73] | 0.02 | True |
| acc | 0.7 | nkl_real | col128 / dense128 | – | [1.09, 3.61] | 0.99 | False |
| acc | 0.7 | nkl_real | colsel128 / col128 | – | [0.293, 0.812] | 0.99 | False |
| acc | 0.7 | nkl_fake | colsel128 / dense128 | 1.06 | [0.889, 1.37] | 0.02 | True |
| acc | 0.7 | nkl_fake | col128 / dense128 | – | [0.929, 1.34] | 0.99 | False |
| acc | 0.7 | nkl_fake | colsel128 / col128 | – | [0.846, 1.22] | 0.99 | False |

## Values at matched effect

| match | target | arm | metric | value | 95% CI | covered |
|---|---:|---|---|---:|---|---|
| tkl | 0.3 | colsel128 | F | 0.00311 | [0.00229, 0.00427] | True |
| tkl | 0.3 | colsel128 | nkl_real | 0.156 | [0.0877, 0.237] | True |
| tkl | 0.3 | colsel128 | nkl_fake | 0.768 | [0.642, 0.898] | True |
| tkl | 0.3 | colsel128 | nchange_real | 0.267 | [0.134, 0.405] | True |
| tkl | 0.3 | colsel128 | nchange_fake | 0.918 | [0.841, 0.975] | True |
| tkl | 0.3 | colsel128 | nvalue_fake | 0.904 | [0.82, 0.97] | True |
| tkl | 0.3 | col128 | F | 0.0954 | [0.047, 0.163] | True |
| tkl | 0.3 | col128 | nkl_real | 0.426 | [0.194, 0.695] | True |
| tkl | 0.3 | col128 | nkl_fake | 0.713 | [0.491, 0.891] | True |
| tkl | 0.3 | col128 | nchange_real | 0.553 | [0.341, 0.725] | True |
| tkl | 0.3 | col128 | nchange_fake | 0.958 | [0.903, 0.999] | True |
| tkl | 0.3 | col128 | nvalue_fake | 0.989 | [0.944, 1] | True |
| tkl | 0.3 | dense128 | F | 0.0448 | [0.0187, 0.0784] | True |
| tkl | 0.3 | dense128 | nkl_real | 0.235 | [0.129, 0.369] | True |
| tkl | 0.3 | dense128 | nkl_fake | 0.594 | [0.452, 0.728] | True |
| tkl | 0.3 | dense128 | nchange_real | 0.41 | [0.245, 0.574] | True |
| tkl | 0.3 | dense128 | nchange_fake | 0.924 | [0.85, 0.992] | True |
| tkl | 0.3 | dense128 | nvalue_fake | 0.992 | [0.967, 1] | True |
| tkl | 0.3 | lorac | F | 0.645 | [0.0314, 1.5] | True |
| tkl | 0.3 | lorac | nkl_real | 1.84 | [0.372, 2.71] | True |
| tkl | 0.3 | lorac | nkl_fake | 1.3 | [0.651, 1.67] | True |
| tkl | 0.3 | lorac | nchange_real | 0.933 | [0.569, 1] | True |
| tkl | 0.3 | lorac | nchange_fake | 0.987 | [0.962, 1] | True |
| tkl | 0.3 | lorac | nvalue_fake | 0.973 | [0.917, 1] | True |
| tkl | 0.3 | lora_all8 | F | 0.0202 | [0.0169, 0.0243] | True |
| tkl | 0.3 | lora_all8 | nkl_real | 0.309 | [0.189, 0.451] | True |
| tkl | 0.3 | lora_all8 | nkl_fake | 0.682 | [0.558, 0.809] | True |
| tkl | 0.3 | lora_all8 | nchange_real | 0.469 | [0.321, 0.617] | True |
| tkl | 0.3 | lora_all8 | nchange_fake | 0.964 | [0.907, 1] | True |
| tkl | 0.3 | lora_all8 | nvalue_fake | 1 | [0.99, 1] | True |
| tkl | 0.15 | colsel128 | F | – | [0.0106, 0.0367] | False |
| tkl | 0.15 | colsel128 | nkl_real | – | [0.255, 0.719] | False |
| tkl | 0.15 | colsel128 | nkl_fake | – | [0.891, 1.39] | False |
| tkl | 0.15 | colsel128 | nchange_real | – | [0.374, 0.721] | False |
| tkl | 0.15 | colsel128 | nchange_fake | – | [0.918, 1] | False |
| tkl | 0.15 | colsel128 | nvalue_fake | – | [0.954, 1] | False |
| tkl | 0.15 | col128 | F | – | [0.89, 0.926] | False |
| tkl | 0.15 | col128 | nkl_real | – | [1.36, 1.54] | False |
| tkl | 0.15 | col128 | nkl_fake | – | [0.925, 1.13] | False |
| tkl | 0.15 | col128 | nchange_real | – | [0.94, 0.956] | False |
| tkl | 0.15 | col128 | nchange_fake | – | [0.91, 0.912] | False |
| tkl | 0.15 | col128 | nvalue_fake | – | [1, 1] | False |
| tkl | 0.15 | dense128 | F | 0.383 | [0.134, 0.865] | True |
| tkl | 0.15 | dense128 | nkl_real | 0.551 | [0.308, 0.862] | True |
| tkl | 0.15 | dense128 | nkl_fake | 1.03 | [0.856, 1.2] | True |
| tkl | 0.15 | dense128 | nchange_real | 0.649 | [0.504, 0.78] | True |
| tkl | 0.15 | dense128 | nchange_fake | 0.997 | [0.984, 1] | True |
| tkl | 0.15 | dense128 | nvalue_fake | 0.979 | [0.938, 1] | True |
| tkl | 0.15 | lorac | F | – | – | False |
| tkl | 0.15 | lorac | nkl_real | – | – | False |
| tkl | 0.15 | lorac | nkl_fake | – | – | False |
| tkl | 0.15 | lorac | nchange_real | – | – | False |
| tkl | 0.15 | lorac | nchange_fake | – | – | False |
| tkl | 0.15 | lorac | nvalue_fake | – | – | False |
| tkl | 0.15 | lora_all8 | F | 0.0442 | [0.0355, 0.0562] | True |
| tkl | 0.15 | lora_all8 | nkl_real | 0.484 | [0.331, 0.655] | True |
| tkl | 0.15 | lora_all8 | nkl_fake | 1.1 | [0.904, 1.31] | True |
| tkl | 0.15 | lora_all8 | nchange_real | 0.604 | [0.45, 0.744] | True |
| tkl | 0.15 | lora_all8 | nchange_fake | 0.971 | [0.914, 1] | True |
| tkl | 0.15 | lora_all8 | nvalue_fake | 0.991 | [0.968, 1] | True |
| acc | 0.5 | colsel128 | F | 0.0068 | [0.0044, 0.00969] | True |
| acc | 0.5 | colsel128 | nkl_real | 0.224 | [0.118, 0.349] | True |
| acc | 0.5 | colsel128 | nkl_fake | 0.955 | [0.805, 1.09] | True |
| acc | 0.5 | colsel128 | nchange_real | 0.344 | [0.179, 0.495] | True |
| acc | 0.5 | colsel128 | nchange_fake | 0.959 | [0.881, 1] | True |
| acc | 0.5 | colsel128 | nvalue_fake | 0.998 | [0.984, 1] | True |
| acc | 0.5 | col128 | F | 0.228 | [0.18, 0.299] | True |
| acc | 0.5 | col128 | nkl_real | 0.788 | [0.532, 1.18] | True |
| acc | 0.5 | col128 | nkl_fake | 0.913 | [0.761, 1.07] | True |
| acc | 0.5 | col128 | nchange_real | 0.698 | [0.576, 0.828] | True |
| acc | 0.5 | col128 | nchange_fake | 0.978 | [0.934, 1] | True |
| acc | 0.5 | col128 | nvalue_fake | 0.998 | [0.993, 1] | True |
| acc | 0.5 | dense128 | F | 0.143 | [0.072, 0.269] | True |
| acc | 0.5 | dense128 | nkl_real | 0.331 | [0.211, 0.515] | True |
| acc | 0.5 | dense128 | nkl_fake | 0.894 | [0.747, 1.05] | True |
| acc | 0.5 | dense128 | nchange_real | 0.581 | [0.431, 0.723] | True |
| acc | 0.5 | dense128 | nchange_fake | 0.991 | [0.953, 1] | True |
| acc | 0.5 | dense128 | nvalue_fake | 1 | [0.991, 1] | True |
| acc | 0.5 | lorac | F | – | [0.566, 0.836] | False |
| acc | 0.5 | lorac | nkl_real | – | [1.25, 1.95] | False |
| acc | 0.5 | lorac | nkl_fake | – | [1.23, 1.58] | False |
| acc | 0.5 | lorac | nchange_real | – | [0.849, 0.945] | False |
| acc | 0.5 | lorac | nchange_fake | – | [0.995, 1] | False |
| acc | 0.5 | lorac | nvalue_fake | – | [0.891, 0.977] | False |
| acc | 0.5 | lora_all8 | F | 0.0335 | [0.0285, 0.0385] | True |
| acc | 0.5 | lora_all8 | nkl_real | 0.396 | [0.26, 0.544] | True |
| acc | 0.5 | lora_all8 | nkl_fake | 0.943 | [0.776, 1.1] | True |
| acc | 0.5 | lora_all8 | nchange_real | 0.557 | [0.403, 0.69] | True |
| acc | 0.5 | lora_all8 | nchange_fake | 0.964 | [0.901, 1] | True |
| acc | 0.5 | lora_all8 | nvalue_fake | 0.996 | [0.989, 1] | True |
| acc | 0.3 | colsel128 | F | 0.0032 | [0.00243, 0.00432] | True |
| acc | 0.3 | colsel128 | nkl_real | 0.157 | [0.0888, 0.24] | True |
| acc | 0.3 | colsel128 | nkl_fake | 0.776 | [0.654, 0.904] | True |
| acc | 0.3 | colsel128 | nchange_real | 0.268 | [0.137, 0.408] | True |
| acc | 0.3 | colsel128 | nchange_fake | 0.92 | [0.84, 0.979] | True |
| acc | 0.3 | colsel128 | nvalue_fake | 0.91 | [0.83, 0.971] | True |
| acc | 0.3 | col128 | F | 0.12 | [0.0863, 0.155] | True |
| acc | 0.3 | col128 | nkl_real | 0.51 | [0.326, 0.724] | True |
| acc | 0.3 | col128 | nkl_fake | 0.772 | [0.63, 0.907] | True |
| acc | 0.3 | col128 | nchange_real | 0.6 | [0.455, 0.739] | True |
| acc | 0.3 | col128 | nchange_fake | 0.961 | [0.904, 0.999] | True |
| acc | 0.3 | col128 | nvalue_fake | 0.993 | [0.965, 1] | True |
| acc | 0.3 | dense128 | F | 0.076 | [0.0363, 0.128] | True |
| acc | 0.3 | dense128 | nkl_real | 0.28 | [0.171, 0.42] | True |
| acc | 0.3 | dense128 | nkl_fake | 0.727 | [0.603, 0.843] | True |
| acc | 0.3 | dense128 | nchange_real | 0.493 | [0.35, 0.631] | True |
| acc | 0.3 | dense128 | nchange_fake | 0.946 | [0.865, 0.996] | True |
| acc | 0.3 | dense128 | nvalue_fake | 0.995 | [0.976, 1] | True |
| acc | 0.3 | lorac | F | 0.269 | [0.133, 0.627] | True |
| acc | 0.3 | lorac | nkl_real | 1.46 | [1.01, 2.03] | True |
| acc | 0.3 | lorac | nkl_fake | 1.1 | [0.897, 1.33] | True |
| acc | 0.3 | lorac | nchange_real | 0.894 | [0.809, 0.969] | True |
| acc | 0.3 | lorac | nchange_fake | 0.975 | [0.927, 1] | True |
| acc | 0.3 | lorac | nvalue_fake | 0.989 | [0.967, 1] | True |
| acc | 0.3 | lora_all8 | F | 0.0236 | [0.0205, 0.0273] | True |
| acc | 0.3 | lora_all8 | nkl_real | 0.325 | [0.2, 0.465] | True |
| acc | 0.3 | lora_all8 | nkl_fake | 0.762 | [0.633, 0.902] | True |
| acc | 0.3 | lora_all8 | nchange_real | 0.497 | [0.357, 0.635] | True |
| acc | 0.3 | lora_all8 | nchange_fake | 0.961 | [0.903, 1] | True |
| acc | 0.3 | lora_all8 | nvalue_fake | 1 | [1, 1] | True |
| acc | 0.7 | colsel128 | F | 0.0166 | [0.011, 0.0457] | True |
| acc | 0.7 | colsel128 | nkl_real | 0.444 | [0.279, 0.891] | True |
| acc | 0.7 | colsel128 | nkl_fake | 1.11 | [0.92, 1.39] | True |
| acc | 0.7 | colsel128 | nchange_real | 0.547 | [0.396, 0.748] | True |
| acc | 0.7 | colsel128 | nchange_fake | 0.983 | [0.935, 1] | True |
| acc | 0.7 | colsel128 | nvalue_fake | 0.992 | [0.965, 1] | True |
| acc | 0.7 | col128 | F | – | [0.32, 0.907] | False |
| acc | 0.7 | col128 | nkl_real | – | [0.636, 1.51] | False |
| acc | 0.7 | col128 | nkl_fake | – | [0.981, 1.31] | False |
| acc | 0.7 | col128 | nchange_real | – | [0.697, 0.94] | False |
| acc | 0.7 | col128 | nchange_fake | – | [0.934, 0.999] | False |
| acc | 0.7 | col128 | nvalue_fake | – | [1, 1] | False |
| acc | 0.7 | dense128 | F | 0.417 | [0.173, 0.841] | True |
| acc | 0.7 | dense128 | nkl_real | 0.576 | [0.349, 0.883] | True |
| acc | 0.7 | dense128 | nkl_fake | 1.04 | [0.883, 1.21] | True |
| acc | 0.7 | dense128 | nchange_real | 0.656 | [0.512, 0.788] | True |
| acc | 0.7 | dense128 | nchange_fake | 0.997 | [0.988, 1] | True |
| acc | 0.7 | dense128 | nvalue_fake | 0.977 | [0.935, 1] | True |
| acc | 0.7 | lorac | F | – | – | False |
| acc | 0.7 | lorac | nkl_real | – | – | False |
| acc | 0.7 | lorac | nkl_fake | – | – | False |
| acc | 0.7 | lorac | nchange_real | – | – | False |
| acc | 0.7 | lorac | nchange_fake | – | – | False |
| acc | 0.7 | lorac | nvalue_fake | – | – | False |
| acc | 0.7 | lora_all8 | F | 0.0482 | [0.0417, 0.0549] | True |
| acc | 0.7 | lora_all8 | nkl_real | 0.516 | [0.364, 0.669] | True |
| acc | 0.7 | lora_all8 | nkl_fake | 1.15 | [0.968, 1.33] | True |
| acc | 0.7 | lora_all8 | nchange_real | 0.62 | [0.474, 0.746] | True |
| acc | 0.7 | lora_all8 | nchange_fake | 0.973 | [0.919, 1] | True |
| acc | 0.7 | lora_all8 | nvalue_fake | 0.99 | [0.967, 1] | True |
