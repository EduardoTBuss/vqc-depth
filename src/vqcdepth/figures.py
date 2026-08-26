"""Figure generation for the ERAMIA-RS 2026 VQC depth study.

Reads the aggregated CSVs produced by `tables.py` and the prediction parquets
in `<outdir>/predictions/`; it never reads the `runs/` JSONs directly.

Robustness contract: the grid may be incomplete at any time (20 of 120 runs, or
none), so no plotting function may raise because data is missing -- it prints a
warning and returns `None`. It also never draws a misleading plot, e.g. a
standard-deviation band from a single seed pretending to be an aggregate of ten.

Each `plot_*(outdir)` writes the same content to `.pdf` (vector, for the SBC
LaTeX template) and `.png` under `<outdir>/figures/`, returning the PDF `Path`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")  # headless: Colab/CI have no display

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# -- one style per model (color + marker + line style) ------------------
# Color, marker and line style are redundant on purpose: the ERAMIA-RS
# proceedings are printed and may come out in black and white.
MODEL_ORDER: tuple[str, ...] = ("vqc1", "vqc2", "vqc4", "mlp")

MODEL_STYLE: dict[str, dict] = {
    "vqc1": dict(color="#1b9e77", marker="o", linestyle="-", label="VQC-1"),
    "vqc2": dict(color="#d95f02", marker="s", linestyle="--", label="VQC-2"),
    "vqc4": dict(color="#7570b3", marker="^", linestyle="-.", label="VQC-4"),
    "mlp": dict(color="#666666", marker="D", linestyle=":", label="MLP"),
}

MODEL_DEPTH: dict[str, int | None] = {"vqc1": 1, "vqc2": 2, "vqc4": 4, "mlp": None}

# -- publication parameters (IEEE/SBC single column) --------------------
FONT_SIZE = 8
FIGSIZE_SINGLE = (3.4, 2.6)   # single panel
FIGSIZE_ROW3 = (7.0, 2.6)     # 1x3 panels (one per dataset)
DPI_PNG = 200

plt.rcParams.update(
    {
        "font.size": FONT_SIZE,
        "axes.titlesize": FONT_SIZE,
        "axes.labelsize": FONT_SIZE,
        "xtick.labelsize": FONT_SIZE - 1,
        "ytick.labelsize": FONT_SIZE - 1,
        "legend.fontsize": FONT_SIZE - 1,
        "lines.linewidth": 1.2,
        "lines.markersize": 3.5,
        "figure.dpi": 100,
        "pdf.fonttype": 42,  # editable fonts in the PDF (common venue requirement)
        "ps.fonttype": 42,
    }
)


# -- shared infrastructure ----------------------------------------------

def _warn(msg: str) -> None:
    print(f"[figures] warning: {msg}")


def _figures_dir(outdir) -> Path:
    d = Path(outdir) / "figures"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _save(fig, outdir, name: str) -> Path:
    d = _figures_dir(outdir)
    pdf_path = d / f"{name}.pdf"
    png_path = d / f"{name}.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=DPI_PNG)
    plt.close(fig)
    return pdf_path


def _model_style(model: str) -> dict:
    return MODEL_STYLE.get(
        model, dict(color="#000000", marker="x", linestyle="-", label=model)
    )


def _load_raw_results(outdir) -> pd.DataFrame | None:
    path = Path(outdir) / "results" / "raw_results.csv"
    if not path.exists():
        _warn(f"'{path}' does not exist -- run tables.py to rebuild the CSVs.")
        return None
    try:
        df = pd.read_csv(path)
    except (pd.errors.EmptyDataError, OSError) as e:
        _warn(f"'{path}' is unreadable or empty ({e}).")
        return None
    if df.empty:
        _warn(f"'{path}' exists but is empty (0 runs).")
        return None
    if "status" in df.columns:
        df = df[df["status"] == "completed"]
        if df.empty:
            _warn("no run with status 'completed' in raw_results.csv.")
            return None
    return df


def _load_history(outdir) -> pd.DataFrame | None:
    path = Path(outdir) / "results" / "training_history.csv"
    if not path.exists():
        _warn(f"'{path}' does not exist -- run tables.py to rebuild the CSVs.")
        return None
    try:
        df = pd.read_csv(path)
    except (pd.errors.EmptyDataError, OSError) as e:
        _warn(f"'{path}' is unreadable or empty ({e}).")
        return None
    if df.empty:
        _warn(f"'{path}' exists but is empty (0 epochs).")
        return None
    return df


def _datasets_present(df: pd.DataFrame) -> list[str]:
    return sorted(df["dataset"].dropna().unique().tolist())


def _models_present(df: pd.DataFrame, order: tuple[str, ...] = MODEL_ORDER) -> list[str]:
    present = set(df["model"].dropna().unique().tolist())
    return [m for m in order if m in present] + sorted(present - set(order))


# -- 1-3: test-metric boxplots per model, one panel per dataset ---------

def _boxplot_by_model(outdir, metric_col: str, ylabel: str, name: str) -> Path | None:
    df = _load_raw_results(outdir)
    if df is None:
        return None
    if metric_col not in df.columns:
        _warn(f"column '{metric_col}' absent from raw_results.csv -- skipping '{name}'.")
        return None
    df = df.dropna(subset=[metric_col])
    if df.empty:
        _warn(f"no non-null value of '{metric_col}' -- skipping '{name}'.")
        return None

    datasets = _datasets_present(df)
    n = len(datasets)
    fig, axes = plt.subplots(1, n, figsize=(FIGSIZE_ROW3[0] * n / 3, FIGSIZE_ROW3[1]),
                              sharey=True, squeeze=False)
    axes = axes[0]

    any_drawn = False
    for ax, ds in zip(axes, datasets):
        sub = df[df["dataset"] == ds]
        models = _models_present(sub)
        data, ticks, colors = [], [], []
        for m in models:
            vals = sub.loc[sub["model"] == m, metric_col].dropna().values
            if len(vals) == 0:
                continue
            data.append(vals)
            ticks.append(_model_style(m)["label"])
            colors.append(_model_style(m)["color"])
        if not data:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                     transform=ax.transAxes, fontsize=FONT_SIZE - 1)
            ax.set_title(ds)
            continue
        any_drawn = True
        bp = ax.boxplot(data, labels=ticks, showmeans=True, widths=0.6,
                         patch_artist=True)
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.35)
            patch.set_edgecolor(c)
        for key in ("whiskers", "caps"):
            for line in bp[key]:
                line.set_color("#333333")
        for line in bp["medians"]:
            line.set_color("#000000")
        ax.set_title(f"{ds} (n≤{sub.groupby('model')['seed'].nunique().max()})")
        ax.tick_params(axis="x", rotation=30)

    if not any_drawn:
        plt.close(fig)
        _warn(f"no panel with enough data -- skipping '{name}'.")
        return None

    axes[0].set_ylabel(ylabel)
    fig.tight_layout()
    return _save(fig, outdir, name)


def plot_f1_by_model_boxplot(outdir) -> Path | None:
    return _boxplot_by_model(outdir, "test_f1", "F1 (test)", "f1_by_model_boxplot")


def plot_accuracy_by_model_boxplot(outdir) -> Path | None:
    """Boxplot of test_accuracy per model, one panel per dataset.

    Read with care: `blood` is imbalanced (~24% positives), so high accuracy
    there may only reflect predicting the majority class. Prefer F1/AUC/balanced
    accuracy for conclusions on that dataset.
    """
    return _boxplot_by_model(
        outdir, "test_accuracy", "Accuracy (test)", "accuracy_by_model_boxplot"
    )


def plot_auc_by_model_boxplot(outdir) -> Path | None:
    return _boxplot_by_model(outdir, "test_roc_auc", "ROC AUC (test)", "auc_by_model_boxplot")


# -- 4-5: training curves aggregated over seeds (mean +- std) -----------

def _mean_std_curves(
    outdir, value_col: str, ylabel: str, name: str, *, min_seeds: int = 2
) -> Path | None:
    hist = _load_history(outdir)
    if hist is None:
        return None
    if value_col not in hist.columns:
        _warn(f"column '{value_col}' absent from training_history.csv -- skipping '{name}'.")
        return None

    hist = hist.dropna(subset=[value_col, "epoch"])
    if hist.empty:
        _warn(f"no non-null value of '{value_col}' -- skipping '{name}'.")
        return None

    datasets = _datasets_present(hist)
    n = len(datasets)
    fig, axes = plt.subplots(1, n, figsize=(FIGSIZE_ROW3[0] * n / 3, FIGSIZE_ROW3[1]),
                              sharey=True, squeeze=False)
    axes = axes[0]

    any_drawn = False
    for ax, ds in zip(axes, datasets):
        sub_ds = hist[hist["dataset"] == ds]
        models = _models_present(sub_ds)
        drawn_here = False
        for m in models:
            sub = sub_ds[sub_ds["model"] == m]
            n_seeds = sub["seed"].nunique()
            if n_seeds < min_seeds:
                _warn(
                    f"'{name}': {ds}/{m} has only {n_seeds} seed(s) "
                    f"({min_seeds} minimum for a standard-deviation band) -- "
                    "model omitted from this panel."
                )
                continue
            agg = sub.groupby("epoch")[value_col].agg(["mean", "std", "count"])
            agg = agg.sort_index()
            style = _model_style(m)
            ax.plot(agg.index, agg["mean"], color=style["color"],
                     linestyle=style["linestyle"], marker=None, label=style["label"])
            std = agg["std"].fillna(0.0)
            ax.fill_between(agg.index, agg["mean"] - std, agg["mean"] + std,
                             color=style["color"], alpha=0.15, linewidth=0)
            drawn_here = True
        if drawn_here:
            any_drawn = True
        else:
            ax.text(0.5, 0.5, "sem seeds suficientes", ha="center", va="center",
                     transform=ax.transAxes, fontsize=FONT_SIZE - 1)
        ax.set_title(ds)
        ax.set_xlabel("epoch")

    if not any_drawn:
        plt.close(fig)
        _warn(f"no panel with enough seeds -- skipping '{name}'.")
        return None

    axes[0].set_ylabel(ylabel)
    handles, labels = [], []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        for hh, ll in zip(h, l):
            if ll not in labels:
                handles.append(hh)
                labels.append(ll)
    fig.legend(handles, labels, loc="lower center", ncol=len(labels),
               bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout()
    return _save(fig, outdir, name)


def plot_val_loss_curves(outdir) -> Path | None:
    """Val_loss per epoch, mean over seeds +-1 std, four models per axis.

    Central figure of the paper. A model with fewer than 2 seeds is omitted
    rather than drawn with a variance estimated from a single sample.
    """
    return _mean_std_curves(outdir, "val_loss", "Val. loss", "val_loss_curves")


def plot_val_f1_curves(outdir) -> Path | None:
    """Val_f1 per epoch, mean over seeds +-1 std; same reading as
    `val_loss_curves`, on the primary metric of the study."""
    return _mean_std_curves(outdir, "val_f1", "Val. F1", "val_f1_curves")


def plot_gradient_norm_curves(outdir) -> Path | None:
    """Gradient_norm per epoch, mean over seeds +-1 std, per model.

    Exploratory only (RQ1 is about F1 vs depth): a qualitative look at whether
    deeper circuits optimize worse. Not confirmatory without a dedicated test.
    """
    return _mean_std_curves(outdir, "gradient_norm", "Norma do gradiente",
                             "gradient_norm_curves")


# -- 6: F1 vs depth (RQ1) -----------------------------------------------

def plot_f1_vs_depth(outdir) -> Path | None:
    """Test_f1 (mean +- std) vs VQC depth (1, 2, 4), with the MLP as a
    horizontal reference. Answers RQ1 directly."""
    df = _load_raw_results(outdir)
    if df is None:
        return None
    if "test_f1" not in df.columns:
        _warn("column 'test_f1' absent -- skipping 'f1_vs_depth'.")
        return None
    df = df.dropna(subset=["test_f1"])
    if df.empty:
        _warn("no test_f1 value -- skipping 'f1_vs_depth'.")
        return None

    vqc_models = [m for m in ("vqc1", "vqc2", "vqc4") if m in df["model"].unique()]
    if not vqc_models:
        _warn("no VQC model present -- skipping 'f1_vs_depth'.")
        return None

    datasets = _datasets_present(df)
    n = len(datasets)
    fig, axes = plt.subplots(1, n, figsize=(FIGSIZE_ROW3[0] * n / 3, FIGSIZE_ROW3[1]),
                              sharey=True, squeeze=False)
    axes = axes[0]

    any_drawn = False
    for ax, ds in zip(axes, datasets):
        sub = df[df["dataset"] == ds]
        depths, means, stds = [], [], []
        for m in vqc_models:
            vals = sub.loc[sub["model"] == m, "test_f1"].dropna().values
            if len(vals) == 0:
                continue
            depths.append(MODEL_DEPTH[m])
            means.append(np.mean(vals))
            stds.append(np.std(vals, ddof=1) if len(vals) > 1 else 0.0)
        if depths:
            any_drawn = True
            order = np.argsort(depths)
            depths = np.array(depths)[order]
            means = np.array(means)[order]
            stds = np.array(stds)[order]
            style = _model_style("vqc1")
            ax.errorbar(depths, means, yerr=stds, color=style["color"],
                         marker="o", linestyle="-", capsize=3, label="VQC")
            ax.set_xticks(sorted(set(int(d) for d in depths)))
        mlp_vals = sub.loc[sub["model"] == "mlp", "test_f1"].dropna().values
        if len(mlp_vals) > 0:
            any_drawn = True
            mlp_mean = np.mean(mlp_vals)
            mlp_std = np.std(mlp_vals, ddof=1) if len(mlp_vals) > 1 else 0.0
            style = _model_style("mlp")
            ax.axhline(mlp_mean, color=style["color"], linestyle=style["linestyle"],
                        label="MLP (ref.)")
            ax.axhspan(mlp_mean - mlp_std, mlp_mean + mlp_std, color=style["color"],
                        alpha=0.12, linewidth=0)
        if len(depths) == 0 and len(mlp_vals) == 0:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                     transform=ax.transAxes, fontsize=FONT_SIZE - 1)
        ax.set_title(ds)
        ax.set_xlabel("depth (layers)")

    if not any_drawn:
        plt.close(fig)
        _warn("no panel with data -- skipping 'f1_vs_depth'.")
        return None

    axes[0].set_ylabel("F1 (test), mean ± sd")
    handles, labels = [], []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        for hh, ll in zip(h, l):
            if ll not in labels:
                handles.append(hh)
                labels.append(ll)
    fig.legend(handles, labels, loc="lower center", ncol=len(labels),
               bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout()
    return _save(fig, outdir, "f1_vs_depth")


# -- 7: n_parameters vs test_f1 (parameter efficiency) ------------------

def plot_parameters_vs_f1(outdir) -> Path | None:
    """Parameter count (log) vs test_f1, one point per (dataset, model, seed);
    a single panel so the frontier can be compared across datasets directly."""
    df = _load_raw_results(outdir)
    if df is None:
        return None
    needed = {"n_parameters", "test_f1"}
    missing = needed - set(df.columns)
    if missing:
        _warn(f"columns absent {sorted(missing)} -- skipping 'parameters_vs_f1'.")
        return None
    df = df.dropna(subset=["n_parameters", "test_f1"])
    if df.empty:
        _warn("no valid (n_parameters, test_f1) pair -- skipping 'parameters_vs_f1'.")
        return None

    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
    for m in _models_present(df):
        sub = df[df["model"] == m]
        if sub.empty:
            continue
        style = _model_style(m)
        ax.scatter(sub["n_parameters"], sub["test_f1"], color=style["color"],
                    marker=style["marker"], label=style["label"], alpha=0.75,
                    edgecolor="none", s=14)
    ax.set_xscale("log")
    ax.set_xlabel("parameters (log)")
    ax.set_ylabel("F1 (test)")
    ax.legend()
    fig.tight_layout()
    return _save(fig, outdir, "parameters_vs_f1")


# -- 8: computational cost (training time) per model --------------------

def plot_training_time_vs_model(outdir) -> Path | None:
    """Boxplot of training_time_seconds per model, pooled over datasets and
    seeds: relative cost of VQC-1/2/4 against the MLP."""
    df = _load_raw_results(outdir)
    if df is None:
        return None
    if "training_time_seconds" not in df.columns:
        _warn("column 'training_time_seconds' absent -- skipping 'training_time_vs_model'.")
        return None
    df = df.dropna(subset=["training_time_seconds"])
    if df.empty:
        _warn("no training_time_seconds value -- skipping 'training_time_vs_model'.")
        return None

    models = _models_present(df)
    data, ticks, colors = [], [], []
    for m in models:
        vals = df.loc[df["model"] == m, "training_time_seconds"].dropna().values
        if len(vals) == 0:
            continue
        data.append(vals)
        ticks.append(_model_style(m)["label"])
        colors.append(_model_style(m)["color"])
    if not data:
        _warn("no model with data -- skipping 'training_time_vs_model'.")
        return None

    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
    bp = ax.boxplot(data, labels=ticks, showmeans=True, widths=0.6, patch_artist=True)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.35)
        patch.set_edgecolor(c)
    for key in ("whiskers", "caps"):
        for line in bp[key]:
            line.set_color("#333333")
    for line in bp["medians"]:
        line.set_color("#000000")
    ax.set_ylabel("training time (s)")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    return _save(fig, outdir, "training_time_vs_model")


# -- 9-10: ROC and PR from the prediction parquets ----------------------

def _load_predictions_by_model(outdir, dataset: str | None = None) -> dict[str, pd.DataFrame]:
    """Concatenate the predictions of every available run, grouped by model.

    Seeds are pooled by concatenation, not by averaging curves: the standard
    way to build an aggregate ROC/PR from sample-level probabilities.
    """
    pred_dir = Path(outdir) / "predictions"
    if not pred_dir.exists():
        return {}
    out: dict[str, list[pd.DataFrame]] = {}
    # Identification comes from the columns, not from the file name: a run with
    # a non-default configuration carries a suffix (__shots1024__parametershift)
    # and name parsing would silently drop it.  The .csv glob covers the
    # fallback the runner uses when no parquet engine is available.
    files = sorted(pred_dir.glob("*.parquet")) + sorted(pred_dir.glob("*.csv"))
    for f in files:
        try:
            df = (pd.read_parquet(f) if f.suffix == ".parquet"
                  else pd.read_csv(f))
        except Exception as e:
            _warn(f"unreadable predictions, skipped: {f.name} ({e})")
            continue
        needed = {"y_true", "y_probability", "model"}
        if not needed.issubset(df.columns):
            _warn(f"'{f.name}' lacks columns {sorted(needed)}, skipped.")
            continue
        if dataset is not None:
            if "dataset" not in df.columns:
                _warn(f"'{f.name}' has no dataset column, skipped.")
                continue
            df = df[df["dataset"] == dataset]
            if df.empty:
                continue
        for model, chunk in df.groupby("model", observed=True):
            out.setdefault(str(model), []).append(chunk)
    return {m: pd.concat(dfs, ignore_index=True) for m, dfs in out.items()}


def _roc_curve(y_true, y_score, n_thresholds: int = 200):
    thresholds = np.linspace(1.0, 0.0, n_thresholds)
    tpr, fpr = [], []
    P = np.sum(y_true == 1)
    N = np.sum(y_true == 0)
    if P == 0 or N == 0:
        return None
    for t in thresholds:
        pred = (y_score >= t).astype(int)
        tp = np.sum((pred == 1) & (y_true == 1))
        fp = np.sum((pred == 1) & (y_true == 0))
        tpr.append(tp / P)
        fpr.append(fp / N)
    return np.array(fpr), np.array(tpr)


def _pr_curve(y_true, y_score, n_thresholds: int = 200):
    thresholds = np.linspace(1.0, 0.0, n_thresholds)
    precision, recall = [], []
    P = np.sum(y_true == 1)
    if P == 0:
        return None
    for t in thresholds:
        pred = (y_score >= t).astype(int)
        tp = np.sum((pred == 1) & (y_true == 1))
        fp = np.sum((pred == 1) & (y_true == 0))
        denom = tp + fp
        precision.append(tp / denom if denom > 0 else 1.0)
        recall.append(tp / P)
    return np.array(recall), np.array(precision)


def _curve_by_dataset(
    outdir, curve_fn, xlabel: str, ylabel: str, name: str, *, ref_diagonal: bool
) -> Path | None:
    raw = _load_raw_results(outdir)
    if raw is None:
        datasets = None
    else:
        datasets = _datasets_present(raw)

    pred_dir = Path(outdir) / "predictions"
    if not pred_dir.exists() or not any(pred_dir.glob("*.parquet")):
        _warn(f"no prediction parquet in '{pred_dir}' -- skipping '{name}'.")
        return None

    if datasets is None:
        # Without raw_results.csv, take the datasets from the parquet names.
        datasets = sorted({p.stem.split("__")[0] for p in pred_dir.glob("*.parquet")
                            if len(p.stem.split("__")) == 3})
    if not datasets:
        _warn(f"no identifiable dataset in the predictions -- skipping '{name}'.")
        return None

    n = len(datasets)
    fig, axes = plt.subplots(1, n, figsize=(FIGSIZE_ROW3[0] * n / 3, FIGSIZE_ROW3[1]),
                              sharey=True, squeeze=False)
    axes = axes[0]

    any_drawn = False
    for ax, ds in zip(axes, datasets):
        by_model = _load_predictions_by_model(outdir, dataset=ds)
        drawn_here = False
        for m in _models_present(pd.DataFrame({"model": list(by_model.keys())})) if by_model else []:
            df = by_model.get(m)
            if df is None or df.empty:
                continue
            y_true = df["y_true"].to_numpy().astype(int)
            y_score = df["y_probability"].to_numpy().astype(float)
            res = curve_fn(y_true, y_score)
            if res is None:
                _warn(f"'{name}': {ds}/{m} has a single class in y_true -- model omitted.")
                continue
            x, y = res
            style = _model_style(m)
            ax.plot(x, y, color=style["color"], linestyle=style["linestyle"],
                     label=style["label"])
            drawn_here = True
        if drawn_here:
            any_drawn = True
            if ref_diagonal:
                ax.plot([0, 1], [0, 1], color="#bbbbbb", linestyle=":", linewidth=0.8,
                         label="chance")
        else:
            ax.text(0.5, 0.5, "no predictions", ha="center", va="center",
                     transform=ax.transAxes, fontsize=FONT_SIZE - 1)
        ax.set_title(ds)
        ax.set_xlabel(xlabel)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)

    if not any_drawn:
        plt.close(fig)
        _warn(f"no panel with valid predictions -- skipping '{name}'.")
        return None

    axes[0].set_ylabel(ylabel)
    handles, labels = [], []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        for hh, ll in zip(h, l):
            if ll not in labels:
                handles.append(hh)
                labels.append(ll)
    fig.legend(handles, labels, loc="lower center", ncol=len(labels),
               bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout()
    return _save(fig, outdir, name)


def plot_roc_curves(outdir) -> Path | None:
    """ROC per model, one panel per dataset; seeds pooled by concatenating the
    predictions of every run of that (dataset, model) pair."""
    return _curve_by_dataset(outdir, _roc_curve, "FPR", "TPR", "roc_curves",
                              ref_diagonal=True)


def plot_pr_curves(outdir) -> Path | None:
    """Precision-Recall per model, seeds pooled as in `plot_roc_curves`. More
    informative than ROC on imbalanced datasets such as `blood` (~24%
    positives)."""
    return _curve_by_dataset(outdir, _pr_curve, "Recall", "Precision", "pr_curves",
                              ref_diagonal=False)


# -- orchestration ------------------------------------------------------

_ALL_PLOTS: tuple[Callable, ...] = (
    plot_f1_by_model_boxplot,
    plot_accuracy_by_model_boxplot,
    plot_auc_by_model_boxplot,
    plot_val_loss_curves,
    plot_val_f1_curves,
    plot_f1_vs_depth,
    plot_parameters_vs_f1,
    plot_training_time_vs_model,
    plot_roc_curves,
    plot_pr_curves,
    plot_gradient_norm_curves,
)


def generate_all(outdir) -> list[Path]:
    """Run every `plot_*` function and return the PDFs actually produced. Each
    plotter is isolated so one failure never blocks the remaining figures."""
    produced: list[Path] = []
    for fn in _ALL_PLOTS:
        try:
            path = fn(outdir)
        except Exception as e:  # extra guard beyond the internal None returns
            _warn(f"'{fn.__name__}' failed unexpectedly ({type(e).__name__}: {e}); skipping.")
            continue
        if path is not None:
            produced.append(path)
    print(f"[figures] {len(produced)}/{len(_ALL_PLOTS)} figures generated in "
          f"'{Path(outdir) / 'figures'}'.")
    return produced
