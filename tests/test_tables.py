"""Aggregation, rebuild-from-disk and the multiple-comparison correction.

``holm`` decides which claims of the paper get a star, so it is checked against
hand-computed values, not against another implementation of itself.  ``rebuild``
has to work on a PARTIAL grid, because the researcher looks at 20 of 120 runs
long before the grid ends.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vqcdepth import config, storage, tables


# ======================================================================
# 1. Holm-Bonferroni
# ======================================================================
def test_holm_matches_a_hand_computed_family():
    """p = [0.01, 0.04, 0.03, 0.005], m = 4.

    sorted: 0.005 -> 4*0.005 = 0.020
            0.010 -> 3*0.010 = 0.030
            0.030 -> 2*0.030 = 0.060
            0.040 -> 1*0.040 = 0.040 -> raised to 0.060 by monotonicity
    back in input order: [0.03, 0.06, 0.06, 0.02]
    """
    got = tables.holm([0.01, 0.04, 0.03, 0.005])
    np.testing.assert_allclose(got, [0.03, 0.06, 0.06, 0.02], rtol=1e-12, atol=0)


def test_holm_of_a_single_p_value_is_the_p_value_itself():
    np.testing.assert_allclose(tables.holm([0.037]), [0.037], rtol=1e-12)


def test_holm_is_monotone_in_the_sorted_order_of_the_raw_p_values():
    """Step-down monotonicity: a larger raw p can never get a smaller adjusted p."""
    rng = np.random.default_rng(0)
    for _ in range(50):
        p = rng.random(rng.integers(2, 9))
        adj = tables.holm(p)
        order = np.argsort(p)
        assert np.all(np.diff(adj[order]) >= -1e-15), (
            f"monotonicity violated:\n raw={p[order]}\n adj={adj[order]}"
        )


def test_holm_never_lowers_a_p_value_and_never_exceeds_bonferroni():
    rng = np.random.default_rng(1)
    for _ in range(50):
        p = rng.random(rng.integers(2, 9))
        adj = tables.holm(p)
        assert np.all(adj >= p - 1e-15), "an adjusted p came out below the raw p"
        assert np.all(adj <= np.minimum(1.0, len(p) * p) + 1e-15), (
            "Holm exceeded Bonferroni; it must be uniformly more powerful"
        )


def test_holm_is_capped_at_one():
    got = tables.holm([0.6, 0.7, 0.8])
    assert np.all(got <= 1.0)
    np.testing.assert_allclose(got, [1.0, 1.0, 1.0], rtol=0, atol=1e-12)


def test_holm_keeps_nan_in_place_and_corrects_only_the_rest():
    """A missing p (fewer than 5 pairs, or all differences zero) must not be
    counted in the family size, and must come back as NaN."""
    got = tables.holm([0.01, np.nan, 0.04, 0.03])
    assert np.isnan(got[1]), "a NaN p-value was silently turned into a number"
    # family size is 3, not 4: sorted 0.01 -> 3*0.01 = 0.03, 0.03 -> 2*0.03 = 0.06,
    # 0.04 -> 1*0.04 = 0.04 -> 0.06 by monotonicity
    np.testing.assert_allclose(got[[0, 2, 3]], [0.03, 0.06, 0.06], rtol=1e-12)


def test_holm_of_an_all_nan_family_is_all_nan_and_keeps_its_length():
    got = tables.holm([np.nan, np.nan, np.nan])
    assert got.shape == (3,) and np.all(np.isnan(got))


def test_holm_of_an_empty_family_is_empty():
    assert tables.holm([]).shape == (0,)


def test_holm_handles_exact_ties():
    got = tables.holm([0.02, 0.02, 0.02])
    np.testing.assert_allclose(got, [0.06, 0.06, 0.06], rtol=1e-12)


def test_holm_handles_p_equal_to_zero():
    got = tables.holm([0.0, 0.5])
    np.testing.assert_allclose(got, [0.0, 0.5], rtol=0, atol=1e-15)


# ======================================================================
# 2. rebuild from disk, including a PARTIAL grid
# ======================================================================
def _fake_record(dataset: str, model: str, seed: int, f1: float) -> dict:
    return {
        "run_id": f"{dataset}__{model}__seed{seed:02d}",
        "dataset": dataset, "model": model, "seed": seed,
        "n_layers": config.VQC_LAYERS.get(model),
        "n_qubits": config.N_QUBITS if model in config.VQC_LAYERS else None,
        "n_parameters": config.n_parameters(model),
        "train_samples": 96, "val_samples": 24, "test_samples": 24,
        "best_epoch": 3, "final_epoch": 5,
        "best_val_loss": 0.5, "best_val_f1": f1,
        "test_loss": 0.5, "test_accuracy": f1, "test_f1": f1,
        "test_precision": f1, "test_recall": f1, "test_specificity": f1,
        "test_balanced_accuracy": f1, "test_roc_auc": f1, "test_pr_auc": f1,
        "TN": 5, "FP": 1, "FN": 2, "TP": 16,
        "training_time_seconds": 1.0, "inference_time_seconds": 0.1,
        "wall_time_seconds": 1.2,
        "split_hash_train": "aa", "split_hash_val": "bb",
        "split_hash_test": "cc", "split_hash_all": "dd",
        "n_imputed": {"train": 10, "val": 2, "test": 3, "total": 15},
        "n_epochs_configured": 5, "batch_size": 16, "learning_rate": 0.05,
        "status": "completed", "timestamp": "2026-08-25T12:00:00",
    }


def _seed_disk(outdir, cells, with_history: bool = True) -> None:
    """cells: iterable of (dataset, model, seed, f1)."""
    for dataset, model, seed, f1 in cells:
        rec = _fake_record(dataset, model, seed, f1)
        storage.atomic_write_json(storage.run_path(outdir, rec["run_id"]), rec)
        if with_history:
            hist = pd.DataFrame({
                "dataset": dataset, "seed": seed, "model": model,
                "epoch": [1, 2, 3], "train_loss": [0.7, 0.6, 0.5],
                "val_f1": [0.1, 0.2, f1],
            })
            storage.atomic_write_text(
                storage.history_path(outdir, rec["run_id"]),
                hist.to_csv(index=False))


@pytest.fixture
def full_dataset_on_disk(outdir):
    """One dataset, all 4 models, all 10 seeds -- enough for a Wilcoxon."""
    rng = np.random.default_rng(3)
    cells = []
    for model in config.MODELS:
        bump = {"vqc1": 0.0, "vqc2": 0.05, "vqc4": 0.09, "mlp": 0.02}[model]
        for seed in config.SEEDS:
            cells.append(("banknote", model, seed,
                          float(np.clip(0.70 + bump + 0.01 * rng.standard_normal(),
                                        0, 1))))
    _seed_disk(outdir, cells)
    return outdir, cells


def test_collect_raw_on_an_empty_outdir_returns_an_empty_but_typed_frame(outdir):
    raw = tables.collect_raw(outdir)
    assert raw.empty and list(raw.columns) == tables.RAW_COLUMNS
    assert tables.aggregate(raw).empty
    assert tables.compute_wilcoxon(raw).empty
    assert tables.compute_depth_wilcoxon(raw).empty
    assert tables.main_table(tables.aggregate(raw)).empty


def test_rebuild_on_an_empty_outdir_does_not_explode(outdir):
    out = tables.rebuild(outdir)
    assert out["raw"].empty
    assert (outdir / "results" / "raw_results.csv").exists()


def test_rebuild_on_a_partial_grid_reports_exactly_what_is_missing(outdir, capsys):
    """3 of 120 cells present: the rebuild must succeed and say so."""
    _seed_disk(outdir, [("banknote", "vqc1", 0, 0.7),
                        ("banknote", "mlp", 0, 0.6),
                        ("blood", "vqc4", 3, 0.5)])
    out = tables.rebuild(outdir)
    printed = capsys.readouterr().out

    assert len(out["raw"]) == 3
    total = len(config.DATASETS) * len(config.MODELS) * len(config.SEEDS)
    assert total == 120
    assert f"runs lidos: 3/{total}" in printed
    assert f"faltam {total - 3} runs" in printed
    assert len(tables._missing(out["raw"])) == total - 3
    assert "banknote__vqc1__seed00" not in tables._missing(out["raw"])
    assert (outdir / "results" / "raw_results.csv").exists()
    assert (outdir / "results" / "summary.csv").exists()


def test_rebuild_ignores_incomplete_and_damaged_runs(outdir, capsys):
    _seed_disk(outdir, [("banknote", "vqc1", 0, 0.7)])
    storage.atomic_write_json(storage.run_path(outdir, "banknote__vqc2__seed00"),
                              {**_fake_record("banknote", "vqc2", 0, 0.9),
                               "status": "failed"})
    storage.run_path(outdir, "banknote__vqc4__seed00").write_text(
        '{"broken', encoding="utf-8")

    raw = tables.collect_raw(outdir)
    assert list(raw["run_id"]) == ["banknote__vqc1__seed00"], (
        "a failed or corrupted run entered the aggregated results"
    )
    assert "aviso" in capsys.readouterr().out


def test_collect_raw_flattens_n_imputed_without_double_counting_the_total(outdir):
    _seed_disk(outdir, [("mammographic", "vqc1", 0, 0.7)])
    raw = tables.collect_raw(outdir)
    assert "n_imputed" not in raw.columns
    assert int(raw["n_imputed_total"].iloc[0]) == 15, (
        "n_imputed carries train/val/test AND total; summing every value would "
        "count each imputation twice (30 instead of 15)"
    )


def test_collect_raw_orders_rows_canonically(outdir):
    _seed_disk(outdir, [("blood", "mlp", 1, 0.5), ("banknote", "vqc4", 0, 0.5),
                        ("banknote", "vqc1", 0, 0.5)])
    raw = tables.collect_raw(outdir)
    assert list(raw["dataset"])[0] == "banknote"
    assert list(raw[raw["dataset"] == "banknote"]["model"]) == ["vqc1", "vqc4"]


def test_collect_history_concatenates_and_survives_damaged_files(outdir, capsys):
    """The three shapes a Drive-killed history CSV really takes: zero bytes,
    header only, and a chopped last row."""
    _seed_disk(outdir, [("banknote", "vqc1", 0, 0.7), ("banknote", "mlp", 0, 0.6)])
    storage.atomic_write_text(
        storage.history_path(outdir, "banknote__vqc2__seed00"), "")
    storage.atomic_write_text(
        storage.history_path(outdir, "banknote__vqc4__seed00"),
        "dataset,seed,model,epoch,train_loss\n")
    storage.atomic_write_text(
        storage.history_path(outdir, "blood__vqc1__seed00"),
        "dataset,seed,model,epoch,train_loss\nblood,0,vqc1,1,0.5\nblood,0,vq")

    hist = tables.collect_history(outdir)
    assert len(hist) >= 6, "healthy histories were lost because of a damaged one"
    assert set(hist["model"]) >= {"vqc1", "mlp"}
    assert "aviso" in capsys.readouterr().out


# REGRESSAO: defeito encontrado em 2026-08-25 (rebuild/tabelas com dado degenerado) e CORRIGIDO
# no mesmo dia. Este teste fica para impedir que volte.
def test_collect_history_survives_a_foreign_csv_in_the_runs_directory(outdir):
    _seed_disk(outdir, [("banknote", "vqc1", 0, 0.7)])
    storage.atomic_write_text(
        storage.history_path(outdir, "zzz_foreign"), "not,a\ncsv,")
    hist = tables.collect_history(outdir)
    assert len(hist) == 3


def test_collect_history_on_a_missing_directory_is_empty(tmp_path):
    assert tables.collect_history(tmp_path / "nope").empty


# ======================================================================
# 3. aggregation and statistics
# ======================================================================
def test_aggregate_reproduces_plain_numpy_statistics(full_dataset_on_disk):
    outdir, cells = full_dataset_on_disk
    raw = tables.collect_raw(outdir)
    agg = tables.aggregate(raw)

    assert len(agg) == len(config.MODELS)
    for model in config.MODELS:
        vals = np.array([f1 for _, m, _, f1 in cells if m == model])
        row = agg[agg["model"] == model].iloc[0]
        assert int(row["n_seeds"]) == len(config.SEEDS)
        assert row["test_f1_mean"] == pytest.approx(vals.mean(), rel=1e-12)
        assert row["test_f1_std"] == pytest.approx(vals.std(ddof=1), rel=1e-12)
        assert row["test_f1_median"] == pytest.approx(np.median(vals), rel=1e-12)
        q1, q3 = np.percentile(vals, [25, 75])
        assert row["test_f1_iqr"] == pytest.approx(q3 - q1, rel=1e-12)
        assert int(row["n_parameters"]) == config.n_parameters(model)


def test_aggregate_of_a_single_seed_gives_nan_std_not_zero(outdir):
    _seed_disk(outdir, [("banknote", "vqc1", 0, 0.7)])
    row = tables.aggregate(tables.collect_raw(outdir)).iloc[0]
    assert np.isnan(row["test_f1_std"]), (
        "std over a single observation must be NaN; 0.0 would look like a "
        "perfectly reproducible model"
    )


def test_wilcoxon_against_the_baseline_is_computed_and_holm_corrected(
        full_dataset_on_disk):
    outdir, _ = full_dataset_on_disk
    raw = tables.collect_raw(outdir)
    wdf = tables.compute_wilcoxon(raw)

    assert set(wdf["model_B"]) == {"vqc1", "vqc2", "vqc4"}
    assert set(wdf["model_A"]) == {"mlp"}
    assert (wdf["n_pairs"] == len(config.SEEDS)).all()
    assert (wdf["p_holm"] >= wdf["p_value"] - 1e-15).all()
    np.testing.assert_allclose(
        wdf["p_holm"].to_numpy(), tables.holm(wdf["p_value"].to_numpy()),
        rtol=1e-12)
    # vqc4 was built to beat the baseline, vqc1 to lose to it
    assert wdf.loc[wdf["model_B"] == "vqc4", "median_delta"].iloc[0] > 0
    assert wdf.loc[wdf["model_B"] == "vqc1", "median_delta"].iloc[0] < 0


def test_depth_wilcoxon_compares_the_three_depth_pairs(full_dataset_on_disk):
    outdir, _ = full_dataset_on_disk
    ddf = tables.compute_depth_wilcoxon(tables.collect_raw(outdir))
    assert list(zip(ddf["model_A"], ddf["model_B"])) == [
        ("vqc1", "vqc2"), ("vqc2", "vqc4"), ("vqc1", "vqc4")]
    assert ddf["p_value"].notna().all()


def test_wilcoxon_refuses_to_produce_a_p_value_from_too_few_pairs(outdir):
    """4 seeds cannot yield a meaningful signed-rank test: p must be NaN, and
    the effect size must still be reported."""
    cells = [("banknote", m, s, 0.6 + 0.05 * (m == "vqc4") + 0.001 * s)
             for m in ("mlp", "vqc4") for s in range(4)]
    _seed_disk(outdir, cells)
    wdf = tables.compute_wilcoxon(tables.collect_raw(outdir))
    row = wdf.iloc[0]
    assert int(row["n_pairs"]) == 4
    assert np.isnan(row["p_value"]) and np.isnan(row["p_holm"])
    assert np.isfinite(row["median_delta"])


def test_wilcoxon_pairs_only_the_seeds_present_in_both_models(outdir):
    cells = [("banknote", "mlp", s, 0.6) for s in range(10)]
    cells += [("banknote", "vqc4", s, 0.7) for s in range(6)]
    _seed_disk(outdir, cells)
    wdf = tables.compute_wilcoxon(tables.collect_raw(outdir))
    row = wdf[wdf["model_B"] == "vqc4"].iloc[0]
    assert int(row["n_pairs"]) == 6, (
        "unmatched seeds were paired anyway; the comparison would mix cells"
    )


def test_wilcoxon_without_the_baseline_returns_an_empty_frame(outdir):
    _seed_disk(outdir, [("banknote", "vqc1", s, 0.6) for s in range(10)])
    assert tables.compute_wilcoxon(tables.collect_raw(outdir)).empty


# ======================================================================
# 4. paper artefacts
# ======================================================================
def test_main_table_formats_mean_and_std_per_model(full_dataset_on_disk):
    outdir, _ = full_dataset_on_disk
    agg = tables.aggregate(tables.collect_raw(outdir))
    t = tables.main_table(agg)
    assert "dataset" in t.columns
    for model in config.MODELS:
        assert model in t.columns
    cell = t.loc[0, "vqc4"]
    assert "+-" in cell and len(cell.split("+-")[0].strip().split(".")[1]) == 3


def test_write_tables_emits_csv_tex_and_markdown(full_dataset_on_disk):
    outdir, _ = full_dataset_on_disk
    raw = tables.collect_raw(outdir)
    agg = tables.aggregate(raw)
    written = tables.write_tables(outdir, agg, tables.compute_wilcoxon(raw),
                                  tables.compute_depth_wilcoxon(raw))
    assert written, "no table file was written"
    names = {p.name for p in written}
    assert {"main_f1.csv", "main_f1.tex"} <= names
    for p in written:
        assert p.exists() and p.stat().st_size > 0
    assert not list((outdir / "tables").glob(".tmp*")), "temp file left behind"


def test_make_tables_runs_end_to_end_on_a_partial_grid(outdir):
    """Two seeds per cell: the researcher looks at the tables long before the
    grid ends."""
    cells = [("banknote", m, s, 0.6 + 0.02 * s + 0.05 * (m == "vqc4"))
             for m in ("vqc1", "vqc4", "mlp") for s in (0, 1)]
    _seed_disk(outdir, cells)
    out = tables.make_tables(outdir)
    assert len(out["raw"]) == 6
    assert out["wilcoxon"].empty or np.isnan(out["wilcoxon"]["p_value"]).all()
    assert (outdir / "tables" / "main_f1.csv").exists()
    assert (outdir / "tables" / "complete.csv").exists()


# REGRESSAO: defeito encontrado em 2026-08-25 (rebuild/tabelas com dado degenerado) e CORRIGIDO
# no mesmo dia. Este teste fica para impedir que volte.
def test_make_tables_survives_a_cell_with_a_single_seed(outdir):
    _seed_disk(outdir, [("banknote", "vqc1", 0, 0.7), ("banknote", "mlp", 0, 0.6)])
    tables.make_tables(outdir)
    assert (outdir / "tables" / "main_f1.csv").exists()
