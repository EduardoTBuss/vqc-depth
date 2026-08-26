"""Interface de linha de comando.

    python -m vqcdepth --part train|rebuild|plot|tables|all --outdir DIR
        [--datasets ...] [--models ...] [--seeds ...] [--epochs N] [--no-checkpoint]

A raiz de saida SEMPRE vem de --outdir: nenhum caminho e' fixado no codigo, o
que e' o que permite escrever direto no Google Drive a partir do Colab.

Particionavel por dataset/modelo/seed para dividir o grid entre runtimes. Como
cada run e' um arquivo proprio, dois runtimes escrevendo no mesmo Drive nao se
corrompem.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import config


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m vqcdepth",
        description="Estudo de profundidade de VQC (ERAMIA-RS 2026).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--part", default="all",
                   choices=["train", "rebuild", "plot", "tables", "all", "status"],
                   help="etapa a executar")
    p.add_argument("--outdir", required=True,
                   help="raiz de saida (no Colab, uma pasta do Drive)")
    p.add_argument("--datasets", nargs="+", default=None,
                   choices=list(config.DATASETS), metavar="DS",
                   help=f"subconjunto de datasets {config.DATASETS}")
    p.add_argument("--models", nargs="+", default=None,
                   choices=list(config.MODELS), metavar="M",
                   help=f"subconjunto de modelos {config.MODELS}")
    p.add_argument("--seeds", nargs="+", type=int, default=None, metavar="S",
                   help=f"subconjunto de seeds (padrao: {list(config.SEEDS)})")
    p.add_argument("--epochs", type=int, default=config.N_EPOCHS,
                   help="epocas COMPLETAS de treino (1 epoca = varredura inteira)")
    p.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    p.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    p.add_argument("--diff-method", default=config.DIFF_METHOD,
                   choices=["backprop", "parameter-shift"],
                   help="método de gradiente. NÃO é variável do estudo: o grid "
                        "do artigo usa backprop, verificado idêntico a "
                        "parameter-shift a ~1e-16 e ~10x mais barato")
    p.add_argument("--shots", type=int, default=config.SHOTS,
                   help="amostras por medição (padrão: analítico, sem ruído). "
                        "Fora do escopo do artigo; exige --diff-method "
                        "parameter-shift e multiplica o custo")
    p.add_argument("--no-checkpoint", action="store_true",
                   help="nao gravar checkpoints por epoca (o resume e' por RUN, "
                        "entao isto nao afeta a retomada)")
    p.add_argument("--stop-on-error", action="store_true",
                   help="abortar no primeiro run que falhar (padrao: seguir)")
    p.add_argument("--dry-run", action="store_true",
                   help="listar o que seria executado e sair")
    return p


def _print_status(grid, outdir) -> None:
    from . import runner
    todo = runner.pending(grid, outdir)
    done = len(grid) - len(todo)
    print(f"\n  outdir: {Path(outdir).resolve()}")
    print(f"  grid selecionado: {len(grid)} runs")
    print(f"  ja concluidos:    {done}")
    print(f"  faltando:         {len(todo)}")
    if todo:
        head = [c.run_id for c in todo[:10]]
        print("  proximos: " + ", ".join(head) + (" ..." if len(todo) > 10 else ""))


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    outdir = args.outdir

    from . import runner, storage
    storage.ensure_layout(outdir)

    grid = runner.build_grid(
        datasets=args.datasets, models=args.models, seeds=args.seeds,
        n_epochs=args.epochs, batch_size=args.batch_size,
        learning_rate=args.lr, save_checkpoints=not args.no_checkpoint,
        diff_method=args.diff_method, shots=args.shots,
    )
    if args.shots is not None or args.diff_method != config.DIFF_METHOD:
        print(f"\n  [atenção] configuração FORA do desenho do artigo: "
              f"diff_method={args.diff_method} shots={args.shots}. "
              f"Os run_id levam sufixo, então estes resultados NÃO "
              f"sobrescrevem o grid analítico.")

    if args.part == "status" or args.dry_run:
        _print_status(grid, outdir)
        return 0

    if args.part in ("train", "all"):
        _print_status(grid, outdir)
        runner.run_grid(grid, outdir, stop_on_error=args.stop_on_error)

    if args.part in ("rebuild", "all"):
        from . import tables
        tables.rebuild(outdir)

    if args.part in ("tables", "all"):
        from . import tables
        tables.make_tables(outdir)

    if args.part in ("plot", "all"):
        try:
            from . import figures
            paths = figures.generate_all(outdir)
            print(f"\n  {len(paths)} figuras em {Path(outdir) / 'figures'}")
        except ImportError as e:
            print(f"\n  [aviso] figuras indisponiveis ({e}); instale matplotlib")

    return 0


if __name__ == "__main__":
    sys.exit(main())
