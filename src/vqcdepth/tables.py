"""Rebuild of the CSVs from the per-run files + aggregated tables + statistics.

The rebuild is what saves a session when Colab dies before any CSV is written:
the per-run files are already on Drive and the CSVs are derived from them. It
works on a PARTIAL grid on purpose. Nothing here appends to a shared CSV --
every aggregate is a read of the per-run files plus one atomic write.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import config, storage

# Canonical column order of raw_results.csv (schema of paragraph 11).
RAW_COLUMNS = [
    "run_id", "dataset", "model", "seed", "n_layers", "n_qubits", "n_parameters",
    "train_samples", "val_samples", "test_samples",
    "best_epoch", "final_epoch", "best_val_loss", "best_val_f1",
    "test_loss", "test_accuracy", "test_f1", "test_precision", "test_recall",
    "test_specificity", "test_balanced_accuracy", "test_roc_auc", "test_pr_auc",
    "TN", "FP", "FN", "TP",
    "training_time_seconds", "inference_time_seconds", "wall_time_seconds",
    "split_hash_train", "split_hash_val", "split_hash_test", "split_hash_all",
    "n_epochs_configured", "batch_size", "learning_rate",
    "diff_method", "shots", "status", "timestamp",
]

MODEL_ORDER = list(config.MODELS)
DATASET_ORDER = list(config.DATASETS)


def _ordered(df: pd.DataFrame, col: str, order: list) -> pd.DataFrame:
    """Order by a categorical column without dropping values outside `order`."""
    if col not in df.columns or df.empty:
        return df
    known = [v for v in order if v in set(df[col])]
    extra = [v for v in df[col].unique() if v not in order]
    df = df.copy()
    df[col] = pd.Categorical(df[col], categories=known + extra, ordered=True)
    return df


# -- rebuild ----------------------------------------------------------

def collect_raw(outdir) -> pd.DataFrame:
    """Read every run from disk and assemble raw_results."""
    records = [r for r in storage.iter_runs(outdir) if r.get("status") == "completed"]
    if not records:
        return pd.DataFrame(columns=RAW_COLUMNS)
    df = pd.DataFrame(records)
    if "n_imputed" in df.columns:
        # n_imputed carries train/val/test AND 'total'; summing d.values() would
        # count every imputation twice. Only the three partitions are summed.
        df["n_imputed_total"] = df["n_imputed"].apply(
            lambda d: int(sum(d[k] for k in ("train", "val", "test") if k in d))
            if isinstance(d, dict) else 0)
        df = df.drop(columns=["n_imputed"])
    cols = [c for c in RAW_COLUMNS if c in df.columns]
    cols += [c for c in df.columns if c not in cols]
    df = df[cols]
    df = _ordered(_ordered(df, "dataset", DATASET_ORDER), "model", MODEL_ORDER)
    return df.sort_values(["dataset", "model", "seed"]).reset_index(drop=True)


def collect_history(outdir) -> pd.DataFrame:
    """Concatenate the per-epoch history of every run."""
    d = Path(outdir) / "runs"
    if not d.exists():
        return pd.DataFrame()
    frames = []
    for p in sorted(d.glob("*__history.csv")):
        try:
            frames.append(pd.read_csv(p))
        except Exception as e:  # a truncated file must not kill the rebuild
            print(f"  [warning] unreadable history, skipped: {p.name} ({e})")
    if not frames:
        return pd.DataFrame()
    # Drop empty frames before concat: pandas warns on all-NA entries and
    # will change the resulting dtypes in a future version.
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    # A foreign or header-truncated CSV in the folder injects NaN into
    # dataset/model, and _ordered rejects a null category -- which would take
    # down the whole rebuild because of one file.
    for c in ("dataset", "model"):
        if c not in df.columns:
            df[c] = np.nan
    bad = df["dataset"].isna() | df["model"].isna()
    if bad.any():
        print(f"  [warning] {int(bad.sum())} history rows without "
              f"dataset/model were discarded")
        df = df[~bad].copy()
    if df.empty:
        return pd.DataFrame()
    df = _ordered(_ordered(df, "dataset", DATASET_ORDER), "model", MODEL_ORDER)
    return df.sort_values(["dataset", "model", "seed", "epoch"]).reset_index(drop=True)


def aggregate(raw: pd.DataFrame) -> pd.DataFrame:
    """Mean, std, median and IQR per (dataset, model); n = completed seeds."""
    if raw.empty:
        return pd.DataFrame()
    metrics = [c for c in raw.columns
               if c.startswith("test_") or c in ("best_epoch", "best_val_f1",
                                                 "best_val_loss", "n_parameters",
                                                 "training_time_seconds",
                                                 "inference_time_seconds")]
    rows = []
    for (ds, mdl), sub in raw.groupby(["dataset", "model"], observed=True):
        row = {"dataset": ds, "model": mdl, "n_seeds": int(len(sub)),
               "n_parameters": int(sub["n_parameters"].iloc[0])}
        for m in metrics:
            if m == "n_parameters":
                continue
            v = pd.to_numeric(sub[m], errors="coerce").to_numpy(dtype=float)
            row[f"{m}_mean"] = float(np.nanmean(v)) if len(v) else np.nan
            row[f"{m}_std"] = float(np.nanstd(v, ddof=1)) if len(v) > 1 else np.nan
            row[f"{m}_median"] = float(np.nanmedian(v)) if len(v) else np.nan
            if m == config.PRIMARY_METRIC or m == f"test_{config.PRIMARY_METRIC}":
                q1, q3 = np.nanpercentile(v, [25, 75]) if len(v) else (np.nan, np.nan)
                row[f"{m}_iqr"] = float(q3 - q1)
        rows.append(row)
    agg = pd.DataFrame(rows)
    agg = _ordered(_ordered(agg, "dataset", DATASET_ORDER), "model", MODEL_ORDER)
    return agg.sort_values(["dataset", "model"]).reset_index(drop=True)


# -- statistics -------------------------------------------------------

def holm(pvals) -> np.ndarray:
    """Holm-Bonferroni correction; returns adjusted p in the input order.

    There are several comparisons per dataset, so raw p-values at 0.05 would
    not survive review. Holm is uniformly more powerful than Bonferroni and
    assumes no independence.
    """
    p = np.asarray(pvals, dtype=float)
    ok = ~np.isnan(p)
    out = np.full_like(p, np.nan)
    if ok.sum() == 0:
        return out
    pv = p[ok]
    idx = np.argsort(pv)
    m = int(ok.sum())
    adj, running = np.empty(m), 0.0
    for rank, i in enumerate(idx):
        running = max(running, (m - rank) * pv[i])
        adj[i] = min(1.0, running)      # enforce monotonicity
    out[ok] = adj
    return out


def compute_wilcoxon(raw: pd.DataFrame, baseline: str = "mlp",
                     metric: str = "test_f1") -> pd.DataFrame:
    """Seed-paired Wilcoxon between each model and the baseline.

    The pairing is only legitimate because the four models of a given
    (dataset, seed) see exactly the same split, which a test enforces through
    the split hashes.
    """
    from scipy.stats import wilcoxon

    if raw.empty or baseline not in set(raw["model"]):
        return pd.DataFrame()

    rows = []
    for ds, sub in raw.groupby("dataset", observed=True):
        base = sub[sub["model"] == baseline].sort_values("seed")
        family = []
        for mdl in MODEL_ORDER:
            if mdl == baseline or mdl not in set(sub["model"]):
                continue
            other = sub[sub["model"] == mdl].sort_values("seed")
            common = sorted(set(base["seed"]) & set(other["seed"]))
            a = base[base["seed"].isin(common)].sort_values("seed")[metric].to_numpy(float)
            b = other[other["seed"].isin(common)].sort_values("seed")[metric].to_numpy(float)
            n = len(common)
            if n < 5:
                p, delta = np.nan, (float(np.median(b - a)) if n else np.nan)
            else:
                try:
                    _, p = wilcoxon(a, b)
                except ValueError:
                    p = np.nan      # all differences are zero
                delta = float(np.median(b - a))
            family.append({
                "dataset": ds, "metric": metric,
                "model_A": baseline, "model_B": mdl, "n_pairs": n,
                "median_delta": delta,     # >0 = model_B beats the baseline
                "p_value": p,
            })
        for rec, padj in zip(family, holm([r["p_value"] for r in family])):
            rec["p_holm"] = padj
        rows.extend(family)
    return pd.DataFrame(rows)


def compute_depth_wilcoxon(raw: pd.DataFrame, metric: str = "test_f1") -> pd.DataFrame:
    """Depth-vs-depth comparisons (RQ1), with Holm over the 3 per dataset."""
    from scipy.stats import wilcoxon

    if raw.empty:
        return pd.DataFrame()
    pairs = [("vqc1", "vqc2"), ("vqc2", "vqc4"), ("vqc1", "vqc4")]
    rows = []
    for ds, sub in raw.groupby("dataset", observed=True):
        family = []
        for a_name, b_name in pairs:
            A = sub[sub["model"] == a_name].sort_values("seed")
            B = sub[sub["model"] == b_name].sort_values("seed")
            common = sorted(set(A["seed"]) & set(B["seed"]))
            a = A[A["seed"].isin(common)].sort_values("seed")[metric].to_numpy(float)
            b = B[B["seed"].isin(common)].sort_values("seed")[metric].to_numpy(float)
            n = len(common)
            if n < 5:
                p, delta = np.nan, (float(np.median(b - a)) if n else np.nan)
            else:
                try:
                    _, p = wilcoxon(a, b)
                except ValueError:
                    p = np.nan
                delta = float(np.median(b - a))
            family.append({"dataset": ds, "metric": metric,
                           "model_A": a_name, "model_B": b_name,
                           "n_pairs": n, "median_delta": delta, "p_value": p})
        for rec, padj in zip(family, holm([r["p_value"] for r in family])):
            rec["p_holm"] = padj
        rows.extend(family)
    return pd.DataFrame(rows)


# -- tables for the paper ---------------------------------------------

def main_table(agg: pd.DataFrame) -> pd.DataFrame:
    """Main table: mean +- std of test F1 per dataset x model."""
    if agg.empty:
        return pd.DataFrame()
    t = agg.pivot_table(index="dataset", columns="model", values="test_f1_mean",
                        observed=True)
    s = agg.pivot_table(index="dataset", columns="model", values="test_f1_std",
                        observed=True)
    # pivot_table DROPS an all-NaN column: a single-seed cell has std=NaN,
    # disappears from `s`, and s[c] would raise KeyError -- breaking
    # --part tables for most of a Colab session (partial grid). Both axes are
    # reindexed because an all-NaN std yields a frame with no rows or columns.
    s = s.reindex(index=t.index, columns=t.columns)
    out = t.copy().astype(object)
    for c in t.columns:
        vals = []
        for m, d in zip(t[c], s[c]):
            if pd.isna(m):
                vals.append("--")
            elif pd.isna(d):
                vals.append(f"{m:.3f} (n=1)")   # single seed: no spread to show
            else:
                vals.append(f"{m:.3f} +- {d:.3f}")
        out[c] = vals
    return out.reset_index()


def secondary_table(agg: pd.DataFrame) -> pd.DataFrame:
    """Secondary table: parameters / training time / best epoch."""
    if agg.empty:
        return pd.DataFrame()
    cols = ["dataset", "model", "n_parameters", "training_time_seconds_mean",
            "training_time_seconds_std", "best_epoch_mean", "best_epoch_std"]
    return agg[[c for c in cols if c in agg.columns]].copy()


def _to_latex(df: pd.DataFrame, caption: str, label: str) -> str:
    if df.empty:
        return f"% empty table: {label}\n"
    body = df.to_latex(index=False, escape=True, float_format="%.3f")
    return (f"\\begin{{table}}[t]\n\\centering\n\\caption{{{caption}}}\n"
            f"\\label{{tab:{label}}}\n{body}\\end{{table}}\n")


def write_tables(outdir, agg: pd.DataFrame, wdf: pd.DataFrame,
                 ddf: pd.DataFrame) -> list[Path]:
    """Write CSV + Markdown + LaTeX for every table."""
    tdir = Path(outdir) / "tables"
    tdir.mkdir(parents=True, exist_ok=True)
    written = []
    specs = [
        ("main_f1", main_table(agg), "Test F1 (mean +- standard deviation over 10 seeds)"),
        ("secondary_cost", secondary_table(agg), "Parameters, training time and best epoch"),
        ("complete", agg, "All aggregated metrics"),
        ("wilcoxon_vs_baseline", wdf, "Seed-paired Wilcoxon against the MLP (Holm-corrected p)"),
        ("wilcoxon_depth", ddf, "Seed-paired Wilcoxon across depths (Holm-corrected p)"),
    ]
    for name, df, caption in specs:
        if df is None or df.empty:
            continue
        storage.atomic_write_text(tdir / f"{name}.csv", df.to_csv(index=False))
        storage.atomic_write_text(tdir / f"{name}.tex", _to_latex(df, caption, name))
        written += [tdir / f"{name}.{e}" for e in ("csv", "tex")]
        # Markdown needs 'tabulate', an optional pandas dependency: losing it
        # must not cost the CSV and the LaTeX, which are what the paper uses.
        try:
            storage.atomic_write_text(tdir / f"{name}.md", df.to_markdown(index=False))
            written.append(tdir / f"{name}.md")
        except ImportError:
            print(f"  [warning] 'tabulate' not installed; {name}.md not generated "
                  f"(csv and tex are fine). pip install tabulate")
    return written


# -- CLI entry points -------------------------------------------------

def rebuild(outdir) -> dict:
    """--part rebuild: rebuild the CSVs from the per-run files."""
    storage.ensure_layout(outdir)
    raw = collect_raw(outdir)
    hist = collect_history(outdir)
    agg = aggregate(raw)

    rdir = Path(outdir) / "results"
    storage.atomic_write_text(rdir / "raw_results.csv", raw.to_csv(index=False))
    if not hist.empty:
        storage.atomic_write_text(rdir / "training_history.csv", hist.to_csv(index=False))
    if not agg.empty:
        storage.atomic_write_text(rdir / "summary.csv", agg.to_csv(index=False))

    total = len(config.DATASETS) * len(config.MODELS) * len(config.SEEDS)
    print(f"\n  runs read: {len(raw)}/{total}  |  epochs in history: {len(hist)}")
    if not raw.empty:
        print("\n  runs per dataset/model (expected "
              f"{len(config.SEEDS)} each):")
        print(raw.groupby(["dataset", "model"], observed=True).size().to_string())
        missing = _missing(raw)
        if missing:
            print(f"\n  {len(missing)} runs missing. First: "
                  f"{', '.join(missing[:8])}{' ...' if len(missing) > 8 else ''}")
    print(f"\n  CSVs in {rdir}")
    return {"raw": raw, "history": hist, "summary": agg}


def _missing(raw: pd.DataFrame) -> list[str]:
    have = set(raw["run_id"]) if "run_id" in raw.columns else set()
    todo = [f"{d}__{m}__seed{s:02d}"
            for d in config.DATASETS for s in config.SEEDS for m in config.MODELS]
    return [r for r in todo if r not in have]


def make_tables(outdir) -> dict:
    """--part tables: aggregates + Wilcoxon + the files that go to the paper."""
    data = rebuild(outdir)
    raw, agg = data["raw"], data["summary"]
    wdf = compute_wilcoxon(raw) if not raw.empty else pd.DataFrame()
    ddf = compute_depth_wilcoxon(raw) if not raw.empty else pd.DataFrame()
    written = write_tables(outdir, agg, wdf, ddf)
    if written:
        print(f"\n  {len(written)} table files in {Path(outdir) / 'tables'}")
    else:
        print("\n  nothing to tabulate yet (no completed run).")
    if not agg.empty:
        print("\n" + main_table(agg).to_string(index=False))
    return {**data, "wilcoxon": wdf, "wilcoxon_depth": ddf}
