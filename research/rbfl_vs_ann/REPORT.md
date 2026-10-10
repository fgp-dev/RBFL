# RBFL versus BFL and neural networks

**Winning RBFL variant for temporal parity: `rbfl_used_m_trainable_feedback`.** Its MUX enable signals (`USED_M`) are learned Boolean routes and its Q state persists between time steps. It reached 100% bit and exact sequence accuracy on both final temporal tests for every one of five seeds. This label is limited to the measured temporal parity task. The fixed-`USED_M=1` RBFL control did not beat BFL or the ANN on the static rule. The [paper](PAPER.pdf) focuses on the winning temporal variant. The CSV identifiers were clarified after the measurements; absent validation or shifted-test splits are now marked `NA` without changing measured values.

## Provenance and historical result

The 90.31% figure is the **mean test accuracy over 50 seeds** in BFL commit `53c2f46`, file `results/summary.txt`: `bfl_binary_global` 0.903066 with population SD 0.019364. The exact executable is built from `benchmarks/benchmark_bfl_vs_ann.cpp`, `src/routed_binary.cpp`, `src/bfl.cpp`, and `src/greedy.cpp` at that commit. This project compiles the archived files unchanged, independently of the current checkout state.

The binary-global architecture has 32 routes selected by input bits 0–4. Each route stores a 64-bit word, of which 16 bits are used. The 16 Boolean features are a constant one, all ten inputs, and five adjacent pairwise ANDs. Its prediction is `2 * popcount(active & route_state) >= popcount(active)`. The greedy enumerates zero, one, two and three-bit proposals. For each presented training example it masks a proposal with that example's active-feature mask, then evaluates the whole training group sharing its route. Candidates are ranked lexicographically by `(total errors, newly wrong predictions, effective bit flips)`, initialized with `(current errors, 0, 0)`; a winning nonzero proposal is kept. It trains for 20 shuffled epochs. Thus its `CHANGE` is a proposed configuration flip, **not** the prediction-error mask, and its `USED` is the current example's active-feature mask. It does not use a weighted popcount objective. Its search is global only within the selected route, not across all 600 examples.

The archival benchmark also includes the 16-unit tanh MLP. It has 193 float64 parameters, logistic output loss, 500 full-batch Adam epochs, learning rate 0.03, and fixed initialization based on the split seed. The new benchmark reproduces its equations and hyperparameters in `src/benchmark.cpp`; it is also run unchanged in `historical_reproduction.csv`.

## RBFL implementation

There are eight MUX units. Each A and B route selects one of 20 sources: constants 0/1, the ten current input bits, or the eight **previous-step** state bits. All MUXes read the same old state; only after all eight outputs are computed is the next state installed. For unit `i`, `M=(A & !Q) | (B & Q)` and `Q_next=Q ^ (USED_M & (Q ^ M))`. This implements exactly `(A,B)=(0,0)->0, (0,1)->Q, (1,0)->!Q, (1,1)->1`. The output is one selected new-state bit, optionally inverted. All source indices, the output selector and inversion are discrete binary configuration; there are no float weights or gradients. The default `Q0` is zero. Static examples have one time step. Temporal sequences reset `Q0` between examples but retain Q between time steps.

The control fixes `USED_M=1`. The gating variant learns a Boolean source for each `USED_M` from the same 20 source choices, with no target input. `CHANGE_ERROR = prediction XOR target` is computed only in supervised evaluation. For `USED_G`, the implementation records `USED_A=USED_M & !Q` and `USED_B=USED_M & Q` over each current training trajectory. Active route words are eligible for all alternative-source rewires. Dormant route words are also explicitly proposed and logged as exploratory rewires, so they can become useful after state trajectories change. Each source rewire is applied as XOR of old and new five-bit indices, with cost equal to their Hamming distance. Enable-source, output-source, inversion and optional `Q0` changes are proposed similarly. The exhaustive candidate set is one complete route reassignment or one output/initialization edit per greedy iteration; it does **not** include simultaneous multi-route changes. This search neighborhood differs from BFL's up-to-three-feature-bit flips.

Every candidate is evaluated on **all training sequences from their own `Q0`**, never on cached states. Relative to the current model, `E0=Y XOR P0`, `EΔ=Y XOR PΔ`, and each score is `(sum EΔ, sum (!E0 & EΔ), popcount(effective parameter XOR))`. The comparison is lexicographic. The current model has score `(sum E0, 0, 0)`, so a candidate must improve that triple to be kept; otherwise it is reverted. Up to four such greedy iterations are run. The `no_forget` ablation removes only the second ordering criterion. The `no_feedback` ablation resets Q to `Q0` before every step, removing cross-step memory while keeping the same MUX equation. The `q0_trained` ablation permits initial-state flips. These are search and architecture controls, not identical parameter budgets.

## Data and measurement

The historical target is the threshold rule in the archived benchmark over all 1,024 distinct ten-bit vectors. `std::mt19937(seed)` shuffles the domain; 600 unique inputs go to train and 424 to test. The new `historical` rows use the same split and BFL initialization stream. For `generalization`, the same domain and rule are repartitioned 600/200/224; validation and test have unseen vectors. This is a held-out-input test on one fixed synthetic rule, not a broad distribution shift.

For temporal parity, input bit 0 is fair Bernoulli and the target at step `t` is the XOR of all bit-0 inputs through `t`; nine other input bits are random distractors. Each seed has 160 training sequences and 40 validation sequences of lengths 4–8, then 80 test sequences of lengths 9–12. The shifted test has another 80 length-9–12 sequences with bit 0 drawn with probability 0.2. Samples are generated independently for each partition. Training does not inspect test targets. The memoryless MLP sees each step alone. The Elman RNN has eight tanh hidden units, 161 float64 parameters, sigmoid readout, full BPTT, full-batch Adam, 500 epochs, learning rate 0.01, and per-coordinate gradient clipping at 1.

Accuracy is reported both as exact whole-sequence correctness and as per-output-bit correctness. For static data they coincide. Generalization gap is train bit accuracy minus test bit accuracy. An absent split is `NA` in the CSVs: the historical task has no validation split, and static tasks have no shifted test. Inference timing uses 10,000 warmup calls and five batches of 100,000 calls; each timed call runs a complete sequence without reading a dataset file or computing a target. `ns_per_step` divides by the mean test sequence length. RBFL's current API allocates a prediction trace per call, so its timing includes that allocation. The model bytes are the C++ object/parameter storage, while temporary inference bytes are a lower-bound estimate of stack/heap payload for the longest test sequence; allocator bookkeeping is excluded. BFL's archived prediction uses no scratch buffer. Hardware instruction counters were unavailable in this environment, so no instruction counts are claimed.

## Measured results

See `results/summary.csv` for means and population SD by seed, `results/benchmark.csv` for all individual runs, `results/ablation.csv` for parity controls, and `results/training_trace.csv` for per-iteration training errors. The archived program's exact reproduction is in `results/historical_reproduction.csv`. The archived executable reproduced **90.6132% ± 1.6145%** test accuracy for binary-global over 20 seeds, versus **90.3066% ± 1.9364%** in its archived 50-seed run. The different seed counts account for the mean difference. Its archived ANN produced **99.4340% ± 0.4245%** in the new 20-seed reproduction. For seeds 1–5, the new benchmark's BFL and MLP test accuracies match the archived executable for each seed exactly.

The following are measured means over the five matched seeds of the new benchmark. Accuracy SD is the population SD across seeds. Time is wall time on the machine described in `results/environment.txt`.

| Historical split, model | Train bit | Test bit ± SD | Gap | Model B | Temp B | Total B | Train ms | Inference ns/step |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BFL binary-global | .9613 | .9000 ± .0182 | .0613 | 256 | 0 | 256 | 242.19 | 14.13 |
| `rbfl_used_m_fixed1_feedback` | .7627 | .7344 ± .0119 | .0282 | 44 | 105 | 149 | 18.88 | 45.78 |
| ANN, 16 hidden | 1.0000 | .9929 ± .0047 | .0071 | 1544 | 128 | 1672 | 160.65 | 425.32 |
| Elman RNN, 8 hidden | .9983 | .9731 ± .0140 | .0252 | 1288 | 137 | 1425 | 110.40 | 347.15 |

The independent 600/200/224 partition gives the following bit accuracies. Each example has one target bit, so exact and bit accuracies are identical here.

| Model | Train | Validation | Test ± SD | Gap |
|---|---:|---:|---:|---:|
| BFL binary-global | .9613 | .8930 | .9063 ± .0129 | .0551 |
| `rbfl_used_m_fixed1_feedback` | .7627 | .7350 | .7339 ± .0146 | .0287 |
| ANN, 16 hidden | 1.0000 | .9920 | .9938 ± .0083 | .0063 |
| Elman RNN, 8 hidden | .9983 | .9740 | .9723 ± .0184 | .0260 |

The final parity test was generated from a separate random stream **after architecture and hyperparameters were fixed**. The preliminary stream was used only during development and is not included in these final CSVs. The shifted test is another independent stream with input-bit probability 0.2.

| Temporal model | Train bit | Long-test bit ± SD | Long-test exact | Shifted bit | Model + temp B | Train ms | Inference ns/step |
|---|---:|---:|---:|---:|---:|---:|---:|
| Memoryless ANN, 16 hidden | .7743 | .5056 ± .0223 | .0025 | .5204 | 1544 + 128 | 224.89 | 393.05 |
| Elman RNN, 8 hidden | .9611 | .8108 ± .1931 | .5475 | .8079 | 1288 + 940 | 148.85 | 276.51 |
| `rbfl_used_m_fixed1_feedback` | .7090 | .5882 ± .0096 | .0875 | .6494 | 44 + 116 | 38.08 | 29.70 |
| **`rbfl_used_m_trainable_feedback` (winner on parity)** | **1.0000** | **1.0000 ± .0000** | **1.0000** | **1.0000** | 44 + 116 | 57.37 | 31.88 |
| `rbfl_used_m_trainable_no_feedback` | .5901 | .5492 ± .0161 | .0025 | .5988 | 44 + 116 | 34.87 | 31.22 |

The complete parity ablation is in `results/ablation.csv`. Removing FORGET did not change accuracy in these five seeds; the tested candidates did not produce a consequential error tie. Starting from zero and allowing `Q0` to train also gave the same accuracy under this one-edit neighborhood. The control accepted 2.4 changes per seed on average; learned `USED_M` accepted 3.2. Both had thousands of state transitions over training sequences, while the tests verify that state can be held when `USED_M=0`. Exploratory rewires of inactive branches were evaluated, but none were accepted in these runs. `CHANGE_ERROR` was checked against `prediction XOR target`, and target-free inference is a separate API.

Persistent model storage is 256 B for BFL (2,048 state bits), 44 B for RBFL (84 active trainable bits in the control, 124 with learned enable routes), 1,544 B for the MLP (12,352 bits), and 1,288 B for the RNN (10,304 bits). The RBFL object includes the routes and initial-state storage; it is not counted as only eight Q bits. The temporary and total inference figures are separate in `results/summary.csv`. No equal-parameter or equal-training-time comparison was run: preserving the historical BFL and ANN configurations fixes their sizes and training schedules, while drastically shrinking a neural baseline would change its capacity. These resource differences limit causal claims about architecture efficiency.

On the historical static rule and the new static partition, RBFL **does not improve generalization** over either BFL binary-global or the MLP. On this simple cumulative-parity task, learned gating and feedback together yield perfect observed accuracy on new lengths and the shifted bit distribution, with less model memory and lower measured inference and training time than the tested RNN. Removing feedback while retaining learned gating drops long-test bit accuracy to .5492, so the feedback contributes a large measured gain on this task. Because the task is a single synthetic rule and the RNN has high seed variance, this result is evidence for this parity construction only, not a general claim that RBFL is superior.

## Limitations

The original binary-global model is a feature-majority classifier with route-local state, whereas RBFL is a stateful MUX network; equal input partitions do not make their hypothesis classes equal. The MLP and RNN use standard float training and more persistent bytes. RBFL's one-edit greedy can stop at local minima, and four iterations give it a smaller training budget than the archived BFL search. The parity task favors gating because `USED_M` can implement a conditional Q toggle directly; performance on this task does not establish broad generalization. The static held-out task evaluates a single deterministic target. The reported wall times depend on machine load and implementation choices. The code's byte estimate does not include allocator overhead, and instruction counters were not accessible.
