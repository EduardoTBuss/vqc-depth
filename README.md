# vqc-depth — Does Deeper Mean Better in Variational Quantum Classifiers?

A controlled empirical study with **one independent variable: circuit depth**.
Variational quantum classifiers of **1, 2 and 4 layers** are compared against a
small classical MLP across **3 UCI datasets** and **10 seeds** — 120 runs — with
every model receiving literally the same data splits, and seed-paired
significance testing.

Research code for a short paper submitted to **ERAMIA-RS 2026**.
Author: Eduardo T. Buss.

This repository contains **code only**: no downloaded data, no results and no
manuscript are versioned here.

## The experiment

| axis | values |
|---|---|
| Depth | VQC-1 · VQC-2 · VQC-4 (**the only thing that varies**) |
| Baseline | classical MLP `4 → 8 → 4 → 1`, ReLU, sigmoid |
| Datasets | Banknote (UCI 267) · Blood Transfusion (UCI 176) · Mammographic Mass (UCI 161) |
| Seeds | `range(10)` |
| Qubits | 4 — one per feature, no dimensionality reduction |
| Encoding | angle encoding, `x_j → RY(x_j)`, applied once |
| Layer | `RZ(θ) → RY(θ)` per wire, then a CNOT chain `3→2→1→0` |
| Readout | `⟨Z₀⟩ → p(y=1) = (1 + ⟨Z⟩)/2` |
| Loss / optimiser | BCE / Adam — **identical for quantum and classical** |
| Epochs / batch | 100 / 16 |
| Simulator | `default.qubit`, analytic statevector, no shot noise |

**Primary metric: F1.** Total: 3 × 4 × 10 = **120 runs**, about 82 minutes of
CPU locally.

## Research questions

| | |
|---|---|
| **RQ1** | How does circuit depth affect predictive performance? |
| **RQ2** | How does depth affect training stability and convergence across seeds? |
| **RQ3** | How do VQCs of different depths compare with a classical MLP under the same low-dimensional input? |

**Hypothesis, registered before any run:** added depth may increase
representational capacity, but gains are expected to **saturate or regress** as
optimisation becomes harder.

This is **not** a claim of quantum advantage, not a new technique, and not an
attempt to tune the MLP.

## Datasets — four native features, no PCA

No dimensionality reduction anywhere: each dataset already has, or is reduced by
justified removal to, exactly four features — one per qubit. This removes PCA as
a confounder.

| name | UCI id | n | features | role |
|---|---:|---:|---|---|
| `banknote` | 267 | 1372 | variance, skewness, curtosis, entropy | easy and clean; a sanity floor |
| `blood` | 176 | 748 | Recency, Frequency, Monetary, Time | hard, imbalanced (23.8% positive) |
| `mammographic` | 161 | 961 | Age, Shape, Margin, Density | clinical, medium, has missing values |

Two properties worth stating explicitly, both verified numerically:

- **`blood` has four nominal features but only three informative ones.**
  `Monetary = 250 × Frequency` holds exactly (bitwise, `max|M − 250F| = 0`) and
  the feature matrix has **rank 3**. One qubit therefore encodes a rescaled copy
  of another feature.
- **`mammographic` drops BI-RADS**, which is a radiologist's *assessment*
  rather than a measurement; keeping it would leak the label by proxy. The
  remaining features carry **160 missing values** across 130 rows, imputed with
  the **training-set median**.

## The ansatz

The variational layer is `RZ(θ) → RY(θ)` on every wire, followed by a CNOT
chain running **toward the readout wire** (`3→2→1→0`), repeated 1, 2 or 4 times.
Both choices exist so that **every parameter counts**.

Pulling the readout back through the final entangling block decides which angles
are inert. The pullback is always **diagonal** (a product of Pauli-Z), so the
rotation sitting last in a layer must not commute with Z — `RZ` does, hence it
goes first. And a CNOT *ring* would give `U†Z₀U = Z₁Z₂Z₃`, the identity on wire
0, silently killing that wire's last-layer rotations; the chain gives
`Z₀Z₁Z₂Z₃`, which acts on all four wires.

Effective angle counts, obtained by perturbing each angle and counting those
that move the output:

| ansatz | L=1 | L=2 | L=4 |
|---|---|---|---|
| `RY,RZ` + ring | 3/8 | 11/16 | 27/32 |
| `RZ,RY` + ring | 6/8 | 14/16 | 30/32 |
| `RY` only + chain | 4/4 | 8/8 | 16/16 |
| **`RZ,RY` + chain** (used here) | **8/8** | **16/16** | **32/32** |

The chosen ansatz is the only variant where nominal equals effective at every
depth. That matters for three reasons: the parameter counts reported in the
paper are honest; the progression is exactly 1:2:4, so no depth is handicapped
by a degenerate final layer; and keeping two rotations per wire leaves the state
free of the real-amplitude restriction an `RY`-only ansatz would impose.

`tests/test_ansatz.py` pins these numbers, so changing the layer template fails
loudly instead of silently moving every result.

## Fair comparison protocol

For each `(dataset, seed)`:

1. a single stratified **70/15/15** split;
2. imputer and scaler fitted on the **training set only**, then applied to
   validation and test;
3. **the same indices reused by all four models**.

The split is a function of `(dataset, seed)` alone — never of the model. Every
run records the **SHA-256 hash** of its train/validation/test indices, and a
test fails if two models of the same `(dataset, seed)` disagree, including
across separate processes and sessions. This is what makes the **seed-paired
Wilcoxon** legitimate rather than decorative.

**Model selection uses the validation set. The test set is touched once**, at
the end, with the weights chosen on validation.

Features are scaled with `MinMaxScaler` to `[0, π]` rather than standardised:
under `RY(x)`, values beyond ±π alias, mapping distinct points onto the same
quantum state. Validation and test may fall slightly outside the range and are
deliberately **not clipped** — clipping would pretend the training set had seen
an extremum it never saw.

## Install

```bash
git clone https://github.com/EduardoTBuss/vqc-depth.git
cd vqc-depth
pip install -r requirements.txt
pip install -e .
pytest -q
```

Python ≥ 3.10. **CPU is enough** — four qubits on `default.qubit` will not touch
a GPU.

## Usage

```bash
python -m vqcdepth --part train|rebuild|plot|tables|all|status --outdir DIR
    [--datasets ...] [--models ...] [--seeds ...] [--epochs N] [--no-checkpoint]
```

The output root always comes from `--outdir`; no path is hardcoded, which is
what allows writing straight to Google Drive from Colab.

```bash
# what is still missing (trains nothing)
python -m vqcdepth --part status --outdir ./out

# quick pilot
python -m vqcdepth --part train --outdir ./out \
    --datasets banknote --models vqc1 --seeds 0 1 --epochs 10

# full grid — resumable, re-run at will
python -m vqcdepth --part train --outdir ./out

# CSVs, tables and figures — all work on a PARTIAL grid
python -m vqcdepth --part rebuild --outdir ./out
python -m vqcdepth --part tables  --outdir ./out
python -m vqcdepth --part plot    --outdir ./out
```

The grid is partitionable by dataset, model and seed, so it can be split across
runtimes. To run it on Colab with everything persisted to Drive, see
[COLAB.md](COLAB.md) and `notebooks/ERAMIA_launcher.ipynb`.

## Persistence and resuming

- **One file per run**, id `dataset__model__seedNN` (e.g.
  `banknote__vqc4__seed07`), written atomically (`tmp` + `os.replace`).
- Reopening a session means scanning what exists and running only what is
  missing.
- **No shared CSV is ever appended to**: Drive plus two runtimes is a recipe for
  silent corruption. CSVs are *rebuilt* from the per-run files, and the rebuild
  works on a partial grid.
- Resuming is **per run, not per epoch**. Runs are short enough that intra-run
  resume would be over-engineering; the every-5-epochs checkpoints are an
  **experimental artefact**, not a recovery mechanism.

Each run stores final metrics, the history of **every** epoch, per-sample
predictions, the confusion matrix as numbers, parameter counts, training and
inference times, the split hashes, and an `env.json` describing the environment.

Gradient and parameter norms are recorded per epoch as an exploratory extra —
useful for *observing* whether deeper circuits optimise worse, but **not an
objective of the study**.

## Outside the paper's scope: `--diff-method` and `--shots`

These exist as a capability, not as variables of the study. The published grid
uses backprop on an analytic simulator.

```bash
# reproduce the methodological claim (same gradient, ~10x the cost)
python -m vqcdepth --part train --outdir ./out --diff-method parameter-shift

# shot-noise extension (requires parameter-shift)
python -m vqcdepth --part train --outdir ./out \
    --diff-method parameter-shift --shots 1024
```

On an ideal simulator both methods return *the same* gradient — measured
difference ~1e-16, i.e. float64 noise — at very different costs:

| | backprop | parameter-shift |
|---|---:|---:|
| circuit executions per step (batch 16, L=1) | **16** | 272 |
| same, L=4 | **16** | 1040 |
| time per step, L=4 | **33.8 ms** | 356.1 ms |
| full grid | **~82 min** | ~11.6 h |

Parameter-shift is mandatory on real hardware, where no statevector exists to
differentiate — only measurements. `tests/test_diffmethod_shots.py` keeps the
equivalence under test, so the justification cannot quietly rot.

Non-default configurations get a suffixed `run_id`
(`banknote__vqc4__seed07__shots1024__parametershift`); without it a shot-noise
run would overwrite the analytic run of the same cell and the rebuilt CSV would
mix two experiments in one row. Test-set inference is always analytic, even when
training used shots, so optimisation noise and measurement noise do not blur
into a single metric.

## Layout

```
src/vqcdepth/
├── config.py     every experimental constant; no other module defines one
├── data.py       the three UCI loaders, ids verified at load time
├── splits.py     deterministic 70/15/15, leak-free preprocessing, index hashes
├── circuits.py   the VQCs; batched evaluation via parameter broadcasting
├── models.py     the classical MLP baseline
├── train.py      the training loop, structurally identical for both families
├── storage.py    atomic writes, one file per run, scan-based resuming
├── runner.py     grid orchestration
├── tables.py     CSV rebuild, aggregation, paired Wilcoxon with Holm correction
├── figures.py    all figures, in PDF and PNG
└── cli.py        the command line interface
```

## Reproducibility

- All randomness flows through **explicit generators** seeded by
  `(seed, dataset, model)`. No global RNG is used — no `np.random.seed`, no
  `torch.manual_seed`.
- Python's built-in `hash()` is never used to derive seeds: it is randomised per
  process, which would produce different splits across sessions. Everything goes
  through `hashlib`.
- Split stability across processes is verified by a test that spawns
  subprocesses with different `PYTHONHASHSEED` values.
- `env.json` records package versions and hardware for every run. That file, not
  `requirements.txt`, is what belongs in a reproducibility statement.
- The suite doubles as the gate before the grid: it proves that the vectorised
  circuit matches the per-sample loop in value *and* gradient, that all four
  models of a cell share indices, and that the test set never reaches the
  preprocessing.

## Limitations

Ideal simulator (no noise, no shots), three datasets, four qubits, a single
ansatz family, a baseline MLP deliberately left untuned, and **no claim of
quantum advantage**.

## License

MIT — see [LICENSE](LICENSE).
