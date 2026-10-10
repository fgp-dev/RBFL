# RBFL versus BFL and neural networks

A reproducible C++ experiment with a recurrent network of Boolean MUX units trained by a lexicographic `argmin`. The [paper](PAPER.pdf), its [LaTeX source](PAPER.tex), and the [technical report](REPORT.md) describe the architecture, protocol, and limits of the conclusions.

## Main result

**`rbfl_used_m_trainable_feedback`** is the winning RBFL variant **on the temporal cumulative parity task**. It learns Boolean routes for `USED_M` and retains Q state between steps. Across five seeds, it reached 100% bit and exact sequence accuracy on both unseen lengths and the shifted distribution test.

| Parity, unseen lengths | Bit accuracy | Exact sequence accuracy | Inference |
|---|---:|---:|---:|
| RBFL, trainable `USED_M` with feedback | **100.00%** | **100.00%** | 31.88 ns/step |
| Elman RNN, eight units | 81.08% ± 19.31% | 54.75% | 276.51 ns/step |
| RBFL, trainable `USED_M` without feedback | 54.92% ± 1.61% | 0.25% | 31.22 ns/step |

This win is **specific to this synthetic temporal rule**. On the historical static task, the RBFL control with `USED_M=1` reached 73.44% test accuracy, compared with 90.00% for binary-global BFL and 99.29% for the 16-unit ANN. These tests do not establish general superiority for RBFL. The models also differ in their parameter and training budgets.

The benchmark IDs describe each variant's configuration:

| ID | Configuration |
|---|---|
| `rbfl_used_m_trainable_feedback` | Parity winner: trainable `USED_M`, persistent Q. |
| `rbfl_used_m_fixed1_feedback` | Control: `USED_M=1`, persistent Q. |
| `rbfl_used_m_trainable_no_feedback` | Ablation: trainable `USED_M`, Q reset at every step. |

The other ablations are listed in [ablation.csv](results/ablation.csv). The IDs were clarified after measurement. Missing splits are marked `NA` in the CSV files; measured values were retained.

## Reproduce

You need Git, a C++17-compatible `g++`, and Python 3. From this directory, run:

```bash
bash scripts/reproduce.sh 5 20
```

The first argument sets the seed count for the new comparison; the second sets the seed count for the historical executable and must be at least 20. The script extracts BFL commit `53c2f46` with `git archive`, compiles the historical code **unchanged**, runs the RBFL tests, and generates the CSV files. If this project is outside the BFL checkout, set `BFL_REPO=/path/to/BFL`.

| File | Contents |
|---|---|
| [benchmark.csv](results/benchmark.csv) | Accuracy and cost for each model, task, and seed. |
| [summary.csv](results/summary.csv) | Means and population standard deviations. |
| [ablation.csv](results/ablation.csv) | All RBFL variants for the temporal task. |
| [training_trace.csv](results/training_trace.csv) | Errors before and after each greedy iteration. |
| [historical_reproduction.csv](results/historical_reproduction.csv) | Unchanged run of the archived BFL versus ANN benchmark. |
| [environment.txt](results/environment.txt) | Compiler, CPU, and flags for the saved run. |

`NA` means that a split does not exist for that task: the historical benchmark has no validation split, and the static tasks have no shifted distribution test. A zero in an existing split is a measurement.

The final temporal test uses a random stream separate from training and validation. Training sequences have 4–8 steps; test sequences have 9–12. The shifted distribution changes the relevant bit's probability from 0.5 to 0.2. The static splits are 600/424 and 600/200/224 over all 1,024 distinct ten-bit vectors. Timing figures can vary across machines and runs.

Compilation uses `-std=c++17 -O3 -march=native -Wall -Wextra -Wpedantic`. A [CMake project](CMakeLists.txt) is also available; after extracting the commit, pass `-DBFL_SNAPSHOT=/path/to/snapshot`. To regenerate the PDF with Tectonic:

```bash
bash scripts/build_paper.sh
```

[REPORT.md](REPORT.md) gives the exact equations, the provenance of each baseline, and the memory and measurement limitations.

## License

Copyright 2026 Facundo Gomez Prates. This repository is licensed under the [Apache License 2.0](LICENSE).
