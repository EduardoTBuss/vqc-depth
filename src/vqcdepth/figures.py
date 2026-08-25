"""Geração de figuras do estudo ERAMIA-RS 2026 (profundidade de VQC).

Consome os CSVs agregados produzidos por `tables.py`
(`<outdir>/results/raw_results.csv`, `<outdir>/results/training_history.csv`)
e os parquets de predição em `<outdir>/predictions/<run_id>.parquet`. Nunca
lê os JSONs de `runs/` diretamente — isso é responsabilidade de `tables.py`.

Contrato de robustez
---------------------
Este módulo roda em CIMA de um grid que pode estar incompleto a qualquer
momento (o pesquisador chama `generate_all` com 20 dos 120 runs prontos, ou
até com nenhum). Nenhuma função de plotagem pode levantar exceção por falta
de dado: se não houver dado suficiente para uma figura específica, a função
IMPRIME um aviso e retorna `None`. Nunca desenha um gráfico enganoso (ex.:
banda de desvio-padrão com uma única seed fingindo ser um agregado de 10).

Cada `plot_*(outdir)` salva o MESMO conteúdo em `.pdf` (vetorial, para o
LaTeX da SBC) e `.png` (inspeção rápida no Colab) sob `<outdir>/figures/` e
devolve o `Path` do PDF, ou `None` se não gerou nada.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")  # headless: Colab/CI não têm display

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── estilo único por modelo (cor + marcador + estilo de linha) ─────────
# Redundância cor/marcador/linha propositalmente: os anais do ERAMIA-RS
# saem impressos e podem sair em preto-e-branco.
MODEL_ORDER: tuple[str, ...] = ("vqc1", "vqc2", "vqc4", "mlp")

MODEL_STYLE: dict[str, dict] = {
    "vqc1": dict(color="#1b9e77", marker="o", linestyle="-", label="VQC-1"),
    "vqc2": dict(color="#d95f02", marker="s", linestyle="--", label="VQC-2"),
    "vqc4": dict(color="#7570b3", marker="^", linestyle="-.", label="VQC-4"),
    "mlp": dict(color="#666666", marker="D", linestyle=":", label="MLP"),
}

MODEL_DEPTH: dict[str, int | None] = {"vqc1": 1, "vqc2": 2, "vqc4": 4, "mlp": None}

# ── parâmetros de publicação (coluna única IEEE/SBC) ────────────────────
FONT_SIZE = 8
FIGSIZE_SINGLE = (3.4, 2.6)   # painel único
FIGSIZE_ROW3 = (7.0, 2.6)     # 1x3 painéis (um por dataset)
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
        "pdf.fonttype": 42,  # fontes editáveis no PDF (exigência comum de venue)
        "ps.fonttype": 42,
    }
)


# ── infraestrutura comum ─────────────────────────────────────────────────

def _warn(msg: str) -> None:
    print(f"[figures] aviso: {msg}")


def _figures_dir(outdir) -> Path:
    d = Path(outdir) / "figures"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _save(fig, outdir, name: str) -> Path:
    """Salva `fig` em PDF e PNG sob <outdir>/figures/<name>.{pdf,png}."""
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
        _warn(f"'{path}' não existe — rode tables.py para reconstruir os CSVs.")
        return None
    try:
        df = pd.read_csv(path)
    except (pd.errors.EmptyDataError, OSError) as e:
        _warn(f"'{path}' ilegível ou vazio ({e}).")
        return None
    if df.empty:
        _warn(f"'{path}' existe mas está vazio (0 runs).")
        return None
    if "status" in df.columns:
        df = df[df["status"] == "completed"]
        if df.empty:
            _warn("nenhum run com status 'completed' em raw_results.csv.")
            return None
    return df


def _load_history(outdir) -> pd.DataFrame | None:
    path = Path(outdir) / "results" / "training_history.csv"
    if not path.exists():
        _warn(f"'{path}' não existe — rode tables.py para reconstruir os CSVs.")
        return None
    try:
        df = pd.read_csv(path)
    except (pd.errors.EmptyDataError, OSError) as e:
        _warn(f"'{path}' ilegível ou vazio ({e}).")
        return None
    if df.empty:
        _warn(f"'{path}' existe mas está vazio (0 épocas).")
        return None
    return df


def _datasets_present(df: pd.DataFrame) -> list[str]:
    return sorted(df["dataset"].dropna().unique().tolist())


def _models_present(df: pd.DataFrame, order: tuple[str, ...] = MODEL_ORDER) -> list[str]:
    present = set(df["model"].dropna().unique().tolist())
    return [m for m in order if m in present] + sorted(present - set(order))


# ── 1-3: boxplots de métrica de teste por modelo, um painel por dataset ─

def _boxplot_by_model(outdir, metric_col: str, ylabel: str, name: str) -> Path | None:
    df = _load_raw_results(outdir)
    if df is None:
        return None
    if metric_col not in df.columns:
        _warn(f"coluna '{metric_col}' ausente em raw_results.csv — pulando '{name}'.")
        return None
    df = df.dropna(subset=[metric_col])
    if df.empty:
        _warn(f"nenhum valor não-nulo de '{metric_col}' — pulando '{name}'.")
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
            ax.text(0.5, 0.5, "sem dados", ha="center", va="center",
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
        n_seeds = sub.groupby("model")["seed"].nunique().min() if len(models) else 0
        ax.set_title(f"{ds} (n≤{sub.groupby('model')['seed'].nunique().max()})")
        ax.tick_params(axis="x", rotation=30)

    if not any_drawn:
        plt.close(fig)
        _warn(f"nenhum painel com dados suficientes — pulando '{name}'.")
        return None

    axes[0].set_ylabel(ylabel)
    fig.tight_layout()
    return _save(fig, outdir, name)


def plot_f1_by_model_boxplot(outdir) -> Path | None:
    """Boxplot de test_f1 por modelo, um painel por dataset presente no grid."""
    return _boxplot_by_model(outdir, "test_f1", "F1 (teste)", "f1_by_model_boxplot")


def plot_accuracy_by_model_boxplot(outdir) -> Path | None:
    """Boxplot de test_accuracy por modelo, um painel por dataset.

    Cuidado ao interpretar: `blood` é desbalanceado (~24% positivos); acurácia
    alta ali pode só refletir o classificador prevendo a classe majoritária.
    Prefira F1/AUC/balanced accuracy para conclusões nesse dataset.
    """
    return _boxplot_by_model(
        outdir, "test_accuracy", "Acurácia (teste)", "accuracy_by_model_boxplot"
    )


def plot_auc_by_model_boxplot(outdir) -> Path | None:
    """Boxplot de test_roc_auc por modelo, um painel por dataset."""
    return _boxplot_by_model(outdir, "test_roc_auc", "ROC AUC (teste)", "auc_by_model_boxplot")


# ── 4-5: curvas de treino agregadas por seed (média ± dp), 1 painel/dataset

def _mean_std_curves(
    outdir, value_col: str, ylabel: str, name: str, *, min_seeds: int = 2
) -> Path | None:
    hist = _load_history(outdir)
    if hist is None:
        return None
    if value_col not in hist.columns:
        _warn(f"coluna '{value_col}' ausente em training_history.csv — pulando '{name}'.")
        return None

    hist = hist.dropna(subset=[value_col, "epoch"])
    if hist.empty:
        _warn(f"nenhum valor não-nulo de '{value_col}' — pulando '{name}'.")
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
                    f"'{name}': {ds}/{m} tem só {n_seeds} seed(s) "
                    f"(mínimo {min_seeds} para banda de desvio-padrão) — "
                    "modelo omitido nesse painel."
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
        ax.set_xlabel("época")

    if not any_drawn:
        plt.close(fig)
        _warn(f"nenhum painel com seeds suficientes — pulando '{name}'.")
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
    """Val_loss x época, média entre seeds ±1 dp, 4 modelos no mesmo eixo.

    Figura central do artigo: mostra convergência, estabilidade, saturação
    de profundidade e overfitting em um só painel por dataset. Exige pelo
    menos 2 seeds por (dataset, modelo) para desenhar a banda de dp; com
    menos que isso o modelo é omitido do painel (nunca desenhado como se
    tivesse variância estimada de 1 amostra).
    """
    return _mean_std_curves(outdir, "val_loss", "Val. loss", "val_loss_curves")


def plot_val_f1_curves(outdir) -> Path | None:
    """Val_f1 x época, média entre seeds ±1 dp, 4 modelos no mesmo eixo.

    Complementar a `val_loss_curves`: mesma leitura, na métrica primária do
    estudo (F1).
    """
    return _mean_std_curves(outdir, "val_f1", "Val. F1", "val_f1_curves")


def plot_gradient_norm_curves(outdir) -> Path | None:
    """Gradient_norm x época, média entre seeds ±1 dp, por modelo.

    BÔNUS EXPLORATÓRIO — não é objetivo declarado do artigo (RQ1 é sobre
    F1 x profundidade). Serve só para inspecionar, de forma qualitativa, se
    circuitos mais profundos otimizam pior (possível barren plateau). Não
    reportar isso como resultado confirmatório sem teste estatístico dedicado.
    """
    return _mean_std_curves(outdir, "gradient_norm", "Norma do gradiente",
                             "gradient_norm_curves")


# ── 6: F1 vs profundidade (RQ1) ──────────────────────────────────────────

def plot_f1_vs_depth(outdir) -> Path | None:
    """Test_f1 (média ± dp) vs. profundidade do VQC (1, 2, 4); MLP como
    referência horizontal. Figura que responde diretamente à RQ1 do estudo.
    """
    df = _load_raw_results(outdir)
    if df is None:
        return None
    if "test_f1" not in df.columns:
        _warn("coluna 'test_f1' ausente — pulando 'f1_vs_depth'.")
        return None
    df = df.dropna(subset=["test_f1"])
    if df.empty:
        _warn("nenhum valor de test_f1 — pulando 'f1_vs_depth'.")
        return None

    vqc_models = [m for m in ("vqc1", "vqc2", "vqc4") if m in df["model"].unique()]
    if not vqc_models:
        _warn("nenhum modelo VQC presente — pulando 'f1_vs_depth'.")
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
            ax.text(0.5, 0.5, "sem dados", ha="center", va="center",
                     transform=ax.transAxes, fontsize=FONT_SIZE - 1)
        ax.set_title(ds)
        ax.set_xlabel("profundidade (camadas)")

    if not any_drawn:
        plt.close(fig)
        _warn("nenhum painel com dados — pulando 'f1_vs_depth'.")
        return None

    axes[0].set_ylabel("F1 (teste), média ± dp")
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


# ── 7: n_parameters vs test_f1 (eficiência paramétrica) ──────────────────

def plot_parameters_vs_f1(outdir) -> Path | None:
    """N. de parâmetros (log) x test_f1: mostra eficiência paramétrica.

    Um ponto por (dataset, modelo, seed); cor/marcador por modelo, painel
    único (não por dataset) para comparar diretamente a fronteira
    parâmetros-desempenho entre datasets.
    """
    df = _load_raw_results(outdir)
    if df is None:
        return None
    needed = {"n_parameters", "test_f1"}
    missing = needed - set(df.columns)
    if missing:
        _warn(f"colunas ausentes {sorted(missing)} — pulando 'parameters_vs_f1'.")
        return None
    df = df.dropna(subset=["n_parameters", "test_f1"])
    if df.empty:
        _warn("nenhum par (n_parameters, test_f1) válido — pulando 'parameters_vs_f1'.")
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
    ax.set_xlabel("Nº de parâmetros (log)")
    ax.set_ylabel("F1 (teste)")
    ax.legend()
    fig.tight_layout()
    return _save(fig, outdir, "parameters_vs_f1")


# ── 8: custo computacional (tempo de treino) por modelo ──────────────────

def plot_training_time_vs_model(outdir) -> Path | None:
    """Boxplot de training_time_seconds por modelo (agregado entre datasets
    e seeds) — custo computacional relativo entre VQC-1/2/4 e MLP.
    """
    df = _load_raw_results(outdir)
    if df is None:
        return None
    if "training_time_seconds" not in df.columns:
        _warn("coluna 'training_time_seconds' ausente — pulando 'training_time_vs_model'.")
        return None
    df = df.dropna(subset=["training_time_seconds"])
    if df.empty:
        _warn("nenhum valor de training_time_seconds — pulando 'training_time_vs_model'.")
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
        _warn("nenhum modelo com dados — pulando 'training_time_vs_model'.")
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
    ax.set_ylabel("Tempo de treino (s)")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    return _save(fig, outdir, "training_time_vs_model")


# ── 9-10: ROC e PR a partir dos parquets de predição ─────────────────────

def _load_predictions_by_model(outdir, dataset: str | None = None) -> dict[str, pd.DataFrame]:
    """Concatena predições de todos os runs disponíveis, agrupadas por modelo.

    Se `dataset` for dado, filtra pelo dataset (run_id = dataset__model__seedNN).
    Seeds são agregadas por CONCATENAÇÃO das predições (pooling), não por
    média de curvas — é a forma padrão de gerar uma ROC/PR "agregada" quando
    já se tem y_true/y_pred/y_probability amostra a amostra.
    """
    pred_dir = Path(outdir) / "predictions"
    if not pred_dir.exists():
        return {}
    out: dict[str, list[pd.DataFrame]] = {}
    for p in sorted(pred_dir.glob("*.parquet")):
        parts = p.stem.split("__")
        if len(parts) != 3:
            _warn(f"nome de arquivo de predição inesperado, ignorado: {p.name}")
            continue
        ds, model, _seed = parts
        if dataset is not None and ds != dataset:
            continue
        try:
            df = pd.read_parquet(p)
        except Exception as e:
            _warn(f"predições ilegíveis, ignoradas: {p.name} ({e})")
            continue
        needed = {"y_true", "y_probability"}
        if not needed.issubset(df.columns):
            _warn(f"'{p.name}' sem colunas {sorted(needed)}, ignorado.")
            continue
        out.setdefault(model, []).append(df)
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
        _warn(f"nenhum parquet de predição em '{pred_dir}' — pulando '{name}'.")
        return None

    if datasets is None:
        # sem raw_results.csv, descobre datasets pelos nomes dos parquets
        datasets = sorted({p.stem.split("__")[0] for p in pred_dir.glob("*.parquet")
                            if len(p.stem.split("__")) == 3})
    if not datasets:
        _warn(f"nenhum dataset identificável nas predições — pulando '{name}'.")
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
                _warn(f"'{name}': {ds}/{m} tem só uma classe em y_true — modelo omitido.")
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
                         label="acaso")
        else:
            ax.text(0.5, 0.5, "sem predições", ha="center", va="center",
                     transform=ax.transAxes, fontsize=FONT_SIZE - 1)
        ax.set_title(ds)
        ax.set_xlabel(xlabel)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)

    if not any_drawn:
        plt.close(fig)
        _warn(f"nenhum painel com predições válidas — pulando '{name}'.")
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
    """ROC por modelo, um painel por dataset; seeds agregadas por pooling
    das predições (concatenação de y_true/y_probability de todos os runs
    daquele par dataset/modelo antes de calcular a curva).
    """
    return _curve_by_dataset(outdir, _roc_curve, "FPR", "TPR", "roc_curves",
                              ref_diagonal=True)


def plot_pr_curves(outdir) -> Path | None:
    """Precision-Recall por modelo, um painel por dataset; seeds agregadas
    por pooling (ver `plot_roc_curves`). Mais informativa que ROC em datasets
    desbalanceados como `blood` (~24% positivos).
    """
    return _curve_by_dataset(outdir, _pr_curve, "Recall", "Precisão", "pr_curves",
                              ref_diagonal=False)


# ── orquestração ──────────────────────────────────────────────────────

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
    """Roda todas as funções `plot_*` e devolve os PDFs efetivamente gerados.

    Nunca levanta exceção por causa de um grid incompleto: cada plotter é
    isolado por `try/except` aqui para que uma falha inesperada em UMA
    figura não impeça as demais de serem geradas.
    """
    produced: list[Path] = []
    for fn in _ALL_PLOTS:
        try:
            path = fn(outdir)
        except Exception as e:  # defesa extra além dos retornos None internos
            _warn(f"'{fn.__name__}' falhou inesperadamente ({type(e).__name__}: {e}); pulando.")
            continue
        if path is not None:
            produced.append(path)
    print(f"[figures] {len(produced)}/{len(_ALL_PLOTS)} figuras geradas em "
          f"'{Path(outdir) / 'figures'}'.")
    return produced
