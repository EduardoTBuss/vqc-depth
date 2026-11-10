# Paper and results

Eduardo Timm Buss, Diogo Pereira Ribeiro, Miguel Rosa, Ramiro Paz, Hélida Santos,
and Renata Reiser. **Circuit Depth in Variational Quantum Classifiers:
Performance, Stability, and a Classical MLP Baseline.** Accepted at
[ERAMIA-RS 2026](https://eramia-rs.sbc.org.br/2026/).

## Per-run metrics

| File | Contents |
|---|---|
| [main_runs.csv](main_runs.csv) | 120 runs across three datasets, four models, and seeds 0–9, all at learning rate 0.05 |
| [mlp_sensitivity_runs.csv](mlp_sensitivity_runs.csv) | 30 MLP runs across three datasets and seeds 0–9 at learning rate 0.001 |

Each row contains test metrics, validation checkpoint metrics, confusion
counts, model size, training settings, durations in seconds, and split hashes.
Group by `dataset` and `model` to reproduce mean and sample standard deviation.
Pair comparisons by `dataset` and `seed` and verify the shared splits with
`split_hash_all`. The exports omit execution timestamps.

The main grid used Linux, Python 3.13.15, PennyLane 0.44.1, PyTorch 2.11.0+cpu,
NumPy 2.1.3, and scikit-learn 1.7.0. The exploratory MLP grid used Windows,
Python 3.11, and PyTorch 2.7.1. The environment difference makes that comparison
descriptive.

The [reproduction commands](../README.md#reproduce-the-study) generate further
artifacts, including training histories and per-sample predictions. Use
[CITATION.cff](../CITATION.cff) to cite the software.
