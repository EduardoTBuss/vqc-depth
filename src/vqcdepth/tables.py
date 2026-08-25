"""Reconstrucao dos CSVs a partir dos runs + tabelas agregadas + estatistica.

O rebuild e' o que salva a sessao quando o Colab morre antes de gerar CSV:
os arquivos por run ja estao no Drive, e daqui saem os CSVs. Funciona com
GRID PARCIAL de proposito - o pesquisador precisa olhar 20 de 120 runs sem
esperar o grid inteiro.

Nao ha CSV compartilhado sendo apendado em lugar nenhum: tudo aqui e' leitura
dos arquivos por run e escrita atomica do agregado.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import config, storage

# Ordem canonica das colunas do raw_results.csv (schema do paragrafo 11).
RAW_COLUMNS = [
    "run_id", "dataset", "model", "seed", "n_layers", "n_qubits", "n_parameters",
    "train_samples", "val_samples", "test_samples",
    "best_epoch", "final_epoch", "best_val_loss", "best_val_f1",
    "test_loss", "test_accuracy", "test_f1", "test_precision", "test_recall",
    "test_specificity", "test_balanced_accuracy", "test_roc_auc", "test_pr_auc",
    "TN", "FP", "FN", "TP",
    "training_time_seconds", "inference_time_seconds", "wall_time_seconds",
    "split_hash_train", "split_hash_val", "split_hash_test", "split_hash_all",
    "n_epochs_configured", "batch_size", "learning_rate", "status", "timestamp",
]

MODEL_ORDER = list(config.MODELS)
DATASET_ORDER = list(config.DATASETS)


def _ordered(df: pd.DataFrame, col: str, order: list) -> pd.DataFrame:
    """Ordena por uma coluna categorica sem perder valores fora da ordem."""
    if col not in df.columns or df.empty:
        return df
    known = [v for v in order if v in set(df[col])]
    extra = [v for v in df[col].unique() if v not in order]
    df = df.copy()
    df[col] = pd.Categorical(df[col], categories=known + extra, ordered=True)
    return df


# -- rebuild ----------------------------------------------------------

def collect_raw(outdir) -> pd.DataFrame:
    """Le todos os runs do disco e monta o raw_results."""
    records = [r for r in storage.iter_runs(outdir) if r.get("status") == "completed"]
    if not records:
        return pd.DataFrame(columns=RAW_COLUMNS)
    df = pd.DataFrame(records)
    # n_imputed e' um dict aninhado; achata para nao poluir o CSV
    if "n_imputed" in df.columns:
        # ATENCAO: n_imputed traz train/val/test E TAMBEM 'total'. Somar
        # d.values() contaria dobrado - somar so as tres particoes.
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
    """Concatena o historico de TODAS as epocas de todos os runs."""
    d = Path(outdir) / "runs"
    if not d.exists():
        return pd.DataFrame()
    frames = []
    for p in sorted(d.glob("*__history.csv")):
        try:
            frames.append(pd.read_csv(p))
        except Exception as e:  # arquivo truncado nao derruba o rebuild
            print(f"  [aviso] historico ilegivel, ignorado: {p.name} ({e})")
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    # Um CSV estranho na pasta (ou truncado no cabecalho) injeta NaN em
    # dataset/model, e _ordered rejeita categoria nula - derrubando o rebuild
    # inteiro por causa de um arquivo.
    for c in ("dataset", "model"):
        if c not in df.columns:
            df[c] = np.nan
    ruins = df["dataset"].isna() | df["model"].isna()
    if ruins.any():
        print(f"  [aviso] {int(ruins.sum())} linhas de historico sem "
              f"dataset/model foram descartadas")
        df = df[~ruins].copy()
    if df.empty:
        return pd.DataFrame()
    df = _ordered(_ordered(df, "dataset", DATASET_ORDER), "model", MODEL_ORDER)
    return df.sort_values(["dataset", "model", "seed", "epoch"]).reset_index(drop=True)


def aggregate(raw: pd.DataFrame) -> pd.DataFrame:
    """Media, desvio, mediana e IQR por (dataset, model). n = seeds completas."""
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


# -- estatistica ------------------------------------------------------

def holm(pvals) -> np.ndarray:
    """Correcao de Holm-Bonferroni; devolve p ajustados na ordem de entrada.

    Sao varias comparacoes por dataset (cada modelo contra o baseline).
    Reportar p bruto a 0,05 com multiplas comparacoes e' exatamente o que
    revisor marca. Holm e' uniformemente mais poderoso que Bonferroni e nao
    assume independencia.
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
        adj[i] = min(1.0, running)      # monotonia forcada
    out[ok] = adj
    return out


def compute_wilcoxon(raw: pd.DataFrame, baseline: str = "mlp",
                     metric: str = "test_f1") -> pd.DataFrame:
    """Wilcoxon pareado POR SEED entre cada modelo e o baseline.

    O pareamento so e' legitimo porque os 4 modelos do mesmo (dataset, seed)
    viram exatamente o mesmo split - e ha um teste que garante isso pelos
    hashes. Sem essa garantia, este teste seria invalido.
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
                    p = np.nan      # todas as diferencas nulas
                delta = float(np.median(b - a))
            family.append({
                "dataset": ds, "metric": metric,
                "model_A": baseline, "model_B": mdl, "n_pairs": n,
                "median_delta": delta,     # >0 = model_B melhor que o baseline
                "p_value": p,
            })
        for rec, padj in zip(family, holm([r["p_value"] for r in family])):
            rec["p_holm"] = padj
        rows.extend(family)
    return pd.DataFrame(rows)


def compute_depth_wilcoxon(raw: pd.DataFrame, metric: str = "test_f1") -> pd.DataFrame:
    """Comparacoes entre profundidades - e' isto que responde a RQ1.

    vqc1 vs vqc2, vqc2 vs vqc4, vqc1 vs vqc4. Holm sobre as 3 por dataset.
    """
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


# -- tabelas para o paper ---------------------------------------------

def main_table(agg: pd.DataFrame) -> pd.DataFrame:
    """Tabela PRINCIPAL: F1 medio +- dp por dataset x modelo."""
    if agg.empty:
        return pd.DataFrame()
    t = agg.pivot_table(index="dataset", columns="model", values="test_f1_mean",
                        observed=True)
    s = agg.pivot_table(index="dataset", columns="model", values="test_f1_std",
                        observed=True)
    # pivot_table DESCARTA coluna inteiramente NaN: uma celula com uma unica
    # seed tem std=NaN, some de `s`, e s[c] levantaria KeyError. Isso quebraria
    # --part tables durante quase toda a execucao no Colab (grid parcial).
    # reindexa LINHAS e COLUNAS: se toda std for NaN (grid so com celulas
    # de uma seed), pivot_table devolve um frame sem linhas nem colunas.
    s = s.reindex(index=t.index, columns=t.columns)
    out = t.copy().astype(object)
    for c in t.columns:
        vals = []
        for m, d in zip(t[c], s[c]):
            if pd.isna(m):
                vals.append("--")
            elif pd.isna(d):
                vals.append(f"{m:.3f} (n=1)")   # uma seed so: sem desvio
            else:
                vals.append(f"{m:.3f} +- {d:.3f}")
        out[c] = vals
    return out.reset_index()


def secondary_table(agg: pd.DataFrame) -> pd.DataFrame:
    """Tabela SECUNDARIA: parametros / tempo de treino / melhor epoca."""
    if agg.empty:
        return pd.DataFrame()
    cols = ["dataset", "model", "n_parameters", "training_time_seconds_mean",
            "training_time_seconds_std", "best_epoch_mean", "best_epoch_std"]
    return agg[[c for c in cols if c in agg.columns]].copy()


def _to_latex(df: pd.DataFrame, caption: str, label: str) -> str:
    if df.empty:
        return f"% tabela vazia: {label}\n"
    body = df.to_latex(index=False, escape=True, float_format="%.3f")
    return (f"\\begin{{table}}[t]\n\\centering\n\\caption{{{caption}}}\n"
            f"\\label{{tab:{label}}}\n{body}\\end{{table}}\n")


def write_tables(outdir, agg: pd.DataFrame, wdf: pd.DataFrame,
                 ddf: pd.DataFrame) -> list[Path]:
    """Grava CSV + Markdown + LaTeX de cada tabela."""
    tdir = Path(outdir) / "tables"
    tdir.mkdir(parents=True, exist_ok=True)
    written = []
    specs = [
        ("main_f1", main_table(agg), "F1 no teste (media +- desvio sobre 10 seeds)"),
        ("secondary_cost", secondary_table(agg), "Parametros, tempo de treino e melhor epoca"),
        ("complete", agg, "Todas as metricas agregadas"),
        ("wilcoxon_vs_baseline", wdf, "Wilcoxon pareado por seed contra a MLP (p corrigido por Holm)"),
        ("wilcoxon_depth", ddf, "Wilcoxon pareado por seed entre profundidades (p corrigido por Holm)"),
    ]
    for name, df, caption in specs:
        if df is None or df.empty:
            continue
        storage.atomic_write_text(tdir / f"{name}.csv", df.to_csv(index=False))
        storage.atomic_write_text(tdir / f"{name}.tex", _to_latex(df, caption, name))
        written += [tdir / f"{name}.{e}" for e in ("csv", "tex")]
        # markdown depende de 'tabulate', que e' opcional no pandas: se faltar,
        # NAO vale perder o CSV e o LaTeX (que sao os que vao para o paper).
        try:
            storage.atomic_write_text(tdir / f"{name}.md", df.to_markdown(index=False))
            written.append(tdir / f"{name}.md")
        except ImportError:
            print(f"  [aviso] 'tabulate' ausente; {name}.md nao gerado "
                  f"(csv e tex ok). pip install tabulate")
    return written


# -- entradas do CLI --------------------------------------------------

def rebuild(outdir) -> dict:
    """--part rebuild: reconstroi os CSVs a partir dos arquivos por run."""
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
    print(f"\n  runs lidos: {len(raw)}/{total}  |  epocas no historico: {len(hist)}")
    if not raw.empty:
        print("\n  runs por dataset/modelo (esperado "
              f"{len(config.SEEDS)} cada):")
        print(raw.groupby(["dataset", "model"], observed=True).size().to_string())
        faltando = _missing(raw)
        if faltando:
            print(f"\n  faltam {len(faltando)} runs. Primeiros: "
                  f"{', '.join(faltando[:8])}{' ...' if len(faltando) > 8 else ''}")
    print(f"\n  CSVs em {rdir}")
    return {"raw": raw, "history": hist, "summary": agg}


def _missing(raw: pd.DataFrame) -> list[str]:
    have = set(raw["run_id"]) if "run_id" in raw.columns else set()
    todo = [f"{d}__{m}__seed{s:02d}"
            for d in config.DATASETS for s in config.SEEDS for m in config.MODELS]
    return [r for r in todo if r not in have]


def make_tables(outdir) -> dict:
    """--part tables: agregados + Wilcoxon + arquivos para o paper."""
    data = rebuild(outdir)
    raw, agg = data["raw"], data["summary"]
    wdf = compute_wilcoxon(raw) if not raw.empty else pd.DataFrame()
    ddf = compute_depth_wilcoxon(raw) if not raw.empty else pd.DataFrame()
    written = write_tables(outdir, agg, wdf, ddf)
    if written:
        print(f"\n  {len(written)} arquivos de tabela em {Path(outdir) / 'tables'}")
    else:
        print("\n  nada para tabular ainda (nenhum run completo).")
    if not agg.empty:
        print("\n" + main_table(agg).to_string(index=False))
    return {**data, "wilcoxon": wdf, "wilcoxon_depth": ddf}
