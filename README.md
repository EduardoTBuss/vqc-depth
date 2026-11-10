# VQC Depth

Reproduction code and per-run metrics for **Circuit Depth in Variational Quantum
Classifiers: Performance, Stability, and a Classical MLP Baseline**, accepted at
[ERAMIA-RS 2026](https://eramia-rs.sbc.org.br/2026/).

The study compares four-qubit variational quantum classifiers (VQCs) with 1, 2,
and 4 trainable layers against a classical multilayer perceptron (MLP). The
main experiment covers three UCI datasets, four models, and ten seeds for a
total of **120 runs**. An additional **30 runs** examine the MLP's sensitivity
to the learning rate.

[Paper and results](paper/) · [Run on Colab](COLAB.md) · [Cite the software](CITATION.cff) · [MIT License](LICENSE)

## Quick start

Use Python 3.11 or later and a CPU. The main study used Python 3.13 with
PennyLane 0.44.1, PyTorch 2.11.0, and scikit-learn 1.7.0. The first run
downloads the requested dataset from UCI and caches it under `data/cache/`.

```sh
git clone https://github.com/EduardoTBuss/vqc-depth.git
cd vqc-depth
python -m venv .venv
```

Activate the environment with `source .venv/bin/activate` on Linux/macOS or
`.\.venv\Scripts\Activate.ps1` in PowerShell.

```sh
python -m pip install -r requirements.txt
python -m pip install -e .
python -m pytest
python -m vqcdepth --part all --outdir outputs/pilot --datasets banknote --models vqc1 --seeds 0 --epochs 2 --no-checkpoint --stop-on-error
```

The test suite uses cached datasets by default. Set the environment variable
`VQCDEPTH_ALLOW_DOWNLOAD=1` to enable the dataset-backed tests on a fresh clone.
The pilot trains one model and writes metrics, tables, and figures under
`outputs/pilot/`.

## Reproduce the study

```sh
python -m vqcdepth --part all --outdir outputs/paper --stop-on-error
python -m vqcdepth --part status --outdir outputs/paper
```

The main grid uses 100 epochs and batch size 16. CPU time depends on the
machine. The local pilot used for planning estimated about 82 minutes for
the sequential grid. The same command resumes from completed run records.
Use separate output directories whenever you change epochs, batch size, or
learning rate. The resume key for those settings stays `dataset__model__seedNN`.

For the exploratory MLP learning-rate check, use its own directory.

```sh
python -m vqcdepth --part all --outdir outputs/mlp-lr-001 --models mlp --lr 0.001 --stop-on-error
```

The paper's sensitivity run used Windows, Python 3.11, and PyTorch 2.7.1.
Compare it descriptively because the execution environment also changed.
See [paper/](paper/) for the recorded environments and per-run metrics.

The CLI supports `status`, `train`, `rebuild`, `tables`, `plot`, and `all`.
Run `python -m vqcdepth --help` for model, dataset, and seed filters. The
`--diff-method` and `--shots` flags support additional experiments. The paper
uses analytic backpropagation on PennyLane's `default.qubit` simulator.

## Experimental protocol

| Dataset | CLI name | Rows | Features used |
|---|---|---:|---|
| [Banknote Authentication](https://doi.org/10.24432/C55P57) | `banknote` | 1,372 | Variance, skewness, curtosis, entropy |
| [Blood Transfusion Service Center](https://doi.org/10.24432/C5GS39) | `blood` | 748 | Recency, Frequency, Monetary, Time |
| [Mammographic Mass](https://doi.org/10.24432/C53K6Z) | `mammographic` | 961 | Age, Shape, Margin, Density |

- **Splits.** Seeds 0–9 each define a stratified 70/15/15 train/validation/test
  split shared by all four models. Median imputation and min–max scaling are
  fitted on the training fold. The scaler maps that fold to `[0, π]`, and
  held-out values retain their transformed values. Mammographic excludes
  BI-RADS and treats Shape and Margin codes numerically. Blood's Monetary
  column is exactly 250 times Frequency.
- **Models.** The circuit encodes the four features once with `RY`. Depth
  counts repetitions of `RZ`, then `RY` on each qubit, followed by CNOTs
  `3→2`, `2→1`, `1→0`. VQC-1/2/4 have 8/16/32 trainable angles. The readout
  maps the probability of qubit 0 being in state `|0⟩` to the positive class
  via `(1 + ⟨Z₀⟩) / 2`. The MLP is `4→8→4→1`, with ReLU hidden layers,
  a sigmoid output, and 81 parameters.
- **Training and evaluation.** All main runs use binary cross-entropy, Adam
  with learning rate 0.05, and a classification threshold of 0.5. Validation
  F1 selects the checkpoint, with validation loss breaking ties. The test
  fold supplies positive-class F1 and ROC AUC.
- **Statistics.** Wilcoxon tests pair models by seed. Holm adjustment applies
  separately to the three depth comparisons and the three VQC–MLP comparisons
  for each dataset and metric. The test folds overlap across seeds, so the
  tests are exploratory and conditional on those resamples.

## Repository and outputs

| Path | Contents |
|---|---|
| [src/vqcdepth/](src/vqcdepth/) | Data loaders, splits, models, training, CLI, and analysis |
| [tests/](tests/) | Circuit, preprocessing, training, and resume checks |
| [notebooks/](notebooks/) | Colab launcher and its source generator |
| [paper/](paper/) | Paper citation, original per-run metrics, and environment details |

Each output directory contains `runs/` (the per-run JSON records used for
resuming), `results/` (rebuilt CSVs), `predictions/`, `checkpoints/`,
`tables/`, `figures/`, and `env.json` (environment metadata). Rebuild and
analysis commands also work with a partial grid. Parallel workers should
train disjoint subsets, with one process generating the shared tables and
figures after training.

## License

The software and its documentation are available under the [MIT License](LICENSE).
UCI provides the three datasets separately under CC BY 4.0. Follow each dataset
link for its citation and license details.
