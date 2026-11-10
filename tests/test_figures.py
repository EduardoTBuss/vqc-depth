"""Figure generation, and the library incompatibilities that silently eat figures.

``generate_all`` deliberately isolates each plotter so that one failure never
blocks the rest.  That is the right behaviour for a partial grid -- and it is
also how four figures went missing without a single error: matplotlib removed
``Axes.boxplot(labels=...)`` in 3.11, every boxplot raised ``TypeError``, and
the run reported ``7/11 figures generated`` and exited 0.

So the contract this file pins is not "plots look nice", it is:

* a plotter may RETURN None (nothing to draw yet) -- normal on a partial grid;
* a plotter may not RAISE -- that is a defect, and must be visible as one.
"""

from __future__ import annotations

import inspect

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from vqcdepth import config, figures, storage, tables  # noqa: E402

from test_tables import _seed_disk  # reuse the on-disk grid builder


@pytest.fixture
def grid_on_disk(outdir):
    """One dataset, all 4 models, all 10 seeds, plus the per-epoch columns and
    the prediction files the curve and ROC/PR plotters need.

    Richer than the tables fixture on purpose: with a minimal history, four
    plotters return None for lack of data and the suite could not tell that
    apart from four plotters crashing -- which is exactly the confusion that
    let the matplotlib regression through.
    """
    rng = np.random.default_rng(3)
    cells = []
    for model in config.MODELS:
        bump = {"vqc1": 0.0, "vqc2": 0.05, "vqc4": 0.09, "mlp": 0.02}[model]
        for seed in config.SEEDS:
            f1 = float(np.clip(0.70 + bump + 0.01 * rng.standard_normal(), 0, 1))
            cells.append(("banknote", model, seed, f1))
    _seed_disk(outdir, cells, with_history=False)

    n_epochs = 6
    for dataset, model, seed, f1 in cells:
        run_id = f"{dataset}__{model}__seed{seed:02d}"
        ep = np.arange(1, n_epochs + 1)
        hist = pd.DataFrame({
            "dataset": dataset, "model": model, "seed": seed, "epoch": ep,
            "train_loss": 0.9 * np.exp(-0.3 * ep),
            "val_loss": 0.9 * np.exp(-0.25 * ep) + 0.01 * rng.standard_normal(n_epochs),
            "val_f1": np.linspace(0.3, f1, n_epochs),
            "gradient_norm": 0.5 * np.exp(-0.2 * ep),
            "parameter_norm": np.linspace(1.0, 1.4, n_epochs),
        })
        storage.atomic_write_text(storage.history_path(outdir, run_id),
                                  hist.to_csv(index=False))

        n = 40
        y_true = (rng.random(n) < 0.45).astype(np.int64)
        # a probability correlated with the truth, so ROC/PR are not degenerate
        prob = np.clip(0.5 + (y_true - 0.5) * f1 + 0.15 * rng.standard_normal(n), 0, 1)
        preds = pd.DataFrame({
            "sample_id": np.arange(n), "dataset": dataset, "model": model,
            "seed": seed, "y_true": y_true,
            "y_pred": (prob >= 0.5).astype(np.int64), "y_probability": prob,
        })
        preds.to_parquet(storage.predictions_path(outdir, run_id), index=False)

    tables.rebuild(outdir)
    return outdir


# --- 1. The matplotlib boxplot keyword ---
def test_boxplot_label_keyword_is_accepted_by_the_installed_matplotlib():
    """matplotlib deprecated `labels` in 3.9, renamed it `tick_labels`, and
    removed `labels` in 3.11.  requirements.txt sets no upper bound, so the
    name has to be resolved against whatever is installed."""
    kw = figures._boxplot_label_kw()
    assert kw in {"labels", "tick_labels"}
    params = inspect.signature(matplotlib.axes.Axes.boxplot).parameters
    assert kw in params, (
        f"_boxplot_label_kw() returned {kw!r}, which matplotlib "
        f"{matplotlib.__version__} does not accept"
    )


def test_boxplot_label_keyword_actually_labels_the_ticks():
    """Resolving the name is not enough; the argument has to do its job."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    try:
        ax.boxplot([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]],
                   **{figures._boxplot_label_kw(): ["a", "b"]})
        assert [t.get_text() for t in ax.get_xticklabels()] == ["a", "b"]
    finally:
        plt.close(fig)


# --- 2. No plotter may raise ---
def test_generate_all_produces_every_figure_and_crashes_none(grid_on_disk):
    produced = figures.generate_all(grid_on_disk)
    assert figures.generate_all.crashed == [], (
        f"these plotters raised instead of returning None: "
        f"{figures.generate_all.crashed}"
    )
    assert len(produced) == len(figures._ALL_PLOTS), (
        f"only {len(produced)}/{len(figures._ALL_PLOTS)} figures were produced"
    )
    for path in produced:
        assert path.exists() and path.stat().st_size > 0
        assert path.with_suffix(".png").exists()


def test_generate_all_on_an_empty_outdir_skips_without_raising(outdir):
    """A grid with nothing in it yet: every plotter returns None, none raises."""
    produced = figures.generate_all(outdir)
    assert produced == []
    assert figures.generate_all.crashed == []


def test_generate_all_on_a_single_seed_skips_bands_without_raising(outdir):
    """One seed per cell: the mean +- sd bands cannot be drawn, but that is a
    skip, not a crash.  This is the state of the grid for most of a Colab run."""
    _seed_disk(outdir, [("banknote", m, 0, 0.8) for m in config.MODELS])
    tables.rebuild(outdir)
    figures.generate_all(outdir)
    assert figures.generate_all.crashed == []


# --- 3. Band clipping: a +-1 sd band may never leave the metric's domain ---
#
# Regression tests for a real bug: with few/noisy seeds, mean +- std can fall
# outside what the metric can physically take (F1 > 1, a norm < 0). The bands
# must be clipped for the plot, without touching the underlying data.

def _write_minimal_csvs(outdir, raw_rows, history_rows):
    """Write results/raw_results.csv and results/training_history.csv
    directly, bypassing tables.rebuild(): these two plotters only read those
    two CSVs (see `_load_raw_results` / `_load_history`), so a minimal,
    hand-built pair is enough and keeps the fixture legible."""
    import pandas as pd
    from vqcdepth import storage

    raw_df = pd.DataFrame(raw_rows)
    raw_df["status"] = "completed"
    (outdir / "results").mkdir(parents=True, exist_ok=True)
    raw_df.to_csv(outdir / "results" / "raw_results.csv", index=False)
    pd.DataFrame(history_rows).to_csv(
        outdir / "results" / "training_history.csv", index=False
    )


def _capture_figure(monkeypatch, plot_fn, outdir):
    """`_save()` closes the figure before returning the PDF path, so the only
    way to inspect the artists afterwards is to intercept `plt.close` and keep
    the figure alive one extra moment."""
    captured = []
    real_close = plt.close

    def fake_close(fig=None):
        captured.append(fig if fig is not None else plt.gcf())

    monkeypatch.setattr(figures.plt, "close", fake_close)
    path = plot_fn(outdir)
    monkeypatch.undo()
    assert captured, "plotter did not call plt.close(fig) -- fixture assumption broke"
    fig = captured[0]
    try:
        yield path, fig
    finally:
        real_close(fig)


def test_val_f1_band_is_clipped_to_the_unit_interval(outdir, monkeypatch):
    """9 seeds at val_f1=1.0 and 1 seed at 0.5 gives mean=0.95, sd~0.158, so
    mean+sd~1.11: an impossible F1. The drawn band must stop at 1.0."""
    vals = [1.0] * 9 + [0.5]
    rows = []
    for seed, f1 in enumerate(vals):
        rows.append({"dataset": "banknote", "model": "vqc1", "seed": seed,
                     "epoch": 1, "val_f1": f1, "val_loss": 0.1,
                     "gradient_norm": 0.1})
    _write_minimal_csvs(outdir, [], rows)

    for path, fig in _capture_figure(monkeypatch, figures.plot_val_f1_curves, outdir):
        assert path is not None
        ax = fig.axes[0]
        poly = ax.collections[0]  # the fill_between band
        ys = poly.get_paths()[0].vertices[:, 1]
        assert ys.max() <= 1.0 + 1e-9, f"band exceeds F1=1: max={ys.max()}"
        assert ys.min() >= 0.0 - 1e-9


def test_gradient_norm_band_is_clipped_to_nonnegative(outdir, monkeypatch):
    """9 seeds at gradient_norm=0.01 and 1 at 1.0 gives mean~0.109, sd~0.313,
    so mean-sd~-0.20: a norm cannot be negative. The band must stop at 0."""
    vals = [0.01] * 9 + [1.0]
    rows = []
    for seed, g in enumerate(vals):
        rows.append({"dataset": "banknote", "model": "vqc1", "seed": seed,
                     "epoch": 1, "val_f1": 0.8, "val_loss": 0.1,
                     "gradient_norm": g})
    _write_minimal_csvs(outdir, [], rows)

    for path, fig in _capture_figure(monkeypatch, figures.plot_gradient_norm_curves, outdir):
        assert path is not None
        ax = fig.axes[0]
        poly = ax.collections[0]
        ys = poly.get_paths()[0].vertices[:, 1]
        assert ys.min() >= 0.0 - 1e-9, f"band goes negative: min={ys.min()}"


def test_f1_vs_depth_whiskers_and_mlp_band_clipped_to_unit_interval(outdir, monkeypatch):
    """Same idea as above, but on the RQ1 figure: both the VQC errorbar
    whiskers and the MLP reference band must stay within [0, 1]."""
    f1_vals = [1.0] * 9 + [0.5]
    rows = []
    for seed, f1 in enumerate(f1_vals):
        for model in ("vqc1", "mlp"):
            rows.append({"dataset": "banknote", "model": model, "seed": seed,
                          "test_f1": f1})
    _write_minimal_csvs(outdir, rows, [])

    for path, fig in _capture_figure(monkeypatch, figures.plot_f1_vs_depth, outdir):
        assert path is not None
        ax = fig.axes[0]

        # errorbar() whiskers and axhspan(): both live in ax.collections.
        all_y = []
        for coll in ax.collections:
            for path_ in coll.get_paths():
                all_y.extend(path_.vertices[:, 1].tolist())
        assert max(all_y) <= 1.0 + 1e-9, f"some band/whisker exceeds F1=1: {max(all_y)}"
        assert min(all_y) >= 0.0 - 1e-9, f"some band/whisker goes below F1=0: {min(all_y)}"
