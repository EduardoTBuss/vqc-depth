# Running the grid on Google Colab (everything persisted to Drive)

Everything is written to Google Drive, under `MyDrive/ERAMIA-VQC/`:

```
MyDrive/ERAMIA-VQC/
├── runs/          one JSON per run plus its epoch history (the real checkpoint)
├── results/       the rebuilt CSVs
├── predictions/   one parquet per run: sample_id, y_true, y_pred, y_probability
├── checkpoints/   weights every 5 epochs plus the best one
├── figures/       PDF (for LaTeX) and PNG (for looking at)
├── tables/        CSV, Markdown and LaTeX
├── logs/          stdout of each execution
└── env.json       package versions and hardware for the run
```

## How resuming works

Each run writes **its own file**, atomically (`tmp` + `os.replace`). A run that
already has a file is skipped. If the session dies, reconnect and re-run the
same cell: it continues where it stopped, losing at most the run in flight.

**No shared CSV is ever appended to.** Drive plus two runtimes writing to the
same CSV is a recipe for silent corruption. The CSVs are *rebuilt* from the
per-run files with `--part rebuild`, and the rebuild works on a partial grid.

Resuming is **per run, not per epoch**. Runs are short enough that intra-run
resume would be over-engineering; the every-5-epochs checkpoints are an
experimental artefact, not a recovery mechanism.

## Cost

Measured locally, not estimated:

| model | s/epoch (n_train=961) | one 100-epoch run | 30 runs |
|---|---:|---:|---:|
| `vqc1` | 0.370 | 0.6 min | 13.8 min |
| `vqc2` | 0.579 | 1.0 min | 21.6 min |
| `vqc4` | 1.208 | 2.0 min | 45.2 min |
| `mlp`  | 0.034 | 0.06 min | 1.3 min |

**Full grid (120 runs, sequential): ~82 min** on a reasonable local machine.
Colab is usually 1.5–2x slower, so expect **~2 to 2.7 h**. `vqc4` dominates,
at 55% of the total.

One epoch here is a **complete sweep of the training set** in minibatches of 16
— roughly 60 optimiser steps per epoch on Banknote. The pilot in cell 5 measures
this again on whichever machine Colab hands you.

**Do not request a GPU runtime.** Four qubits on `default.qubit` will not touch
it; CPU is the right choice.

## The notebook

[`notebooks/ERAMIA_launcher.ipynb`](notebooks/ERAMIA_launcher.ipynb) is a **thin
launcher**: all logic lives in the versioned `vqcdepth` package. The notebook
only mounts Drive, clones, installs, runs the test gate, measures the pilot and
fires the grid.

Keep a copy in your own Drive and open it from there (*File → Open notebook →
Google Drive*), which avoids re-authorising Colab against GitHub every session.

---

## Cell 1 — Drive

Mounts Drive and creates the output tree.

## Cell 2 — private repository, via Colab Secrets

The repository is private, so cloning needs a token.

**Before running**, create the secret in Colab (key icon in the sidebar):

- Name: **`GH_PAT`**
- Value: a **fine-grained PAT**, scoped to this repository only, with
  **`Contents: Read-only`** and nothing else. The token does not need write
  access, nor visibility of any other repository.
- Enable *Notebook access*.

The token is never typed into a cell, never printed and never committed. The
cell also does two things to avoid leaking it: it captures the output of
`git clone` and **does not print the raw error** (git echoes the URL, which
contains the token, when authentication fails), and it rewrites the remote right
after cloning so the token is not left in `.git/config`.

If the clone fails, the message is deliberately generic: check the PAT, the
`Contents: Read` scope and the repository name.

**If the secret is missing**, the cell prints the step-by-step fix and then
offers a hidden one-off prompt, so a session can proceed without leaving the
notebook to go and create the secret. A token pasted at that prompt is used for
the clone and discarded in the same cell; it is not stored anywhere. Pressing
Enter on the empty prompt stops cleanly.

The secret is read *outside* the `except` block on purpose. Raising `SystemExit`
from inside an exception handler chains the two exceptions, and IPython's
traceback formatter crashes on that chain with
`TypeError: object of type 'NoneType' has no len()` in `find_recursion` — which
hides the one message that says what to do. Keep the raise out of the handler.

## Cell 3 — dependencies

`pip install -r requirements.txt` and `pip install -e .`.

### If a cell asks you to restart the runtime

Colab preinstalls numpy and imports it before the first cell runs. If pip then
installs a different version, the new Python files load against the C extension
already in memory and numpy raises, from inside itself,
`AttributeError: 'numpy.ufunc' object has no attribute '__module__'`. No user
code can recover from that; only a fresh process can.

For that reason `requirements.txt` sets **no upper bound on numpy**: a cap makes
pip downgrade whatever Colab ships and causes exactly this crash, in exchange
for silencing a cosmetic numba warning. If pip changes a loaded library anyway,
cell 3 detects the mismatch, prints which library drifted and stops.

Restarting costs nothing: *Runtime -> Restart session*, then run again. Drive is
remounted, the repository is re-cloned, and the grid resumes from Drive, so
every run that already finished is skipped.

## Cell 4 — gate: the test suite

```python
!VQCDEPTH_ALLOW_DOWNLOAD=1 python -m pytest tests/ -q
```

**If this fails, do not run the grid.** This is where it is proven that the
vectorised circuit returns the same value *and* the same gradient as the
per-sample loop, that all four models of a `(dataset, seed)` cell receive
literally the same indices, that the test set never reached the preprocessing,
and that no angle of the ansatz is inert.

`VQCDEPTH_ALLOW_DOWNLOAD=1` matters: without it the 18 tests that fetch the UCI
datasets are *skipped* on a fresh clone, which has no local cache, and the gate
would be weaker than it looks.

## Cell 5 — pilot, and the cell that measures time

One dataset, one model, two seeds, ten epochs. It confirms that files reach
Drive **and** measures seconds per epoch on this machine. The following cell
reads the freshly written files and extrapolates, so the number comes from a
measurement rather than a guess.

Run it with `--models vqc4` as well to capture the worst case: cost grows with
depth.

The pilot uses `--epochs 10`, so those runs are **not** paper-grade. The third
pilot cell deletes them — runs, predictions and checkpoints — before the real
grid. Left in place, the resume logic would treat them as finished and the final
grid would contain two cells trained for 10 epochs instead of 100.

## Cell 6 — full grid

```python
!python -m vqcdepth --part train --outdir "{DRIVE}" 2>&1 | tee "{LOG}"
```

Resumable: re-run at will.

To split across two runtimes (each run is its own file, so there is no race):

```python
# runtime A
!python -m vqcdepth --part train --outdir "{DRIVE}" --datasets banknote
# runtime B
!python -m vqcdepth --part train --outdir "{DRIVE}" --datasets blood mammographic
```

Worth it only if the pilot shows the grid is long. Free Colab rarely grants two
backends at once.

## Cell 7 — CSVs, tables and figures

```python
!python -m vqcdepth --part rebuild --outdir "{DRIVE}"
!python -m vqcdepth --part tables  --outdir "{DRIVE}"
!python -m vqcdepth --part plot    --outdir "{DRIVE}"
```

All three work on a **partial grid**. There is no need to wait for all 120 runs
before looking at the data.

`tables/wilcoxon_depth.*` is what answers RQ1 (vqc1 vs vqc2 vs vqc4);
`tables/wilcoxon_vs_baseline.*` compares each model against the MLP. In both,
the column that belongs in the paper is **`p_holm`**, not `p_value` — `p_holm`
corrects for the multiple comparisons within each dataset.

## Checking progress at any time

```python
!python -m vqcdepth --part status --outdir "{DRIVE}"
```

Reports how many runs are on disk, how many are missing and which come next.
Trains nothing.
