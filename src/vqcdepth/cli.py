"""Command line interface.

    python -m vqcdepth --part train|rebuild|plot|tables|all --outdir DIR
        [--datasets ...] [--models ...] [--seeds ...] [--epochs N] [--no-checkpoint]

The output root always comes from --outdir; no path is hardcoded, which is what
allows writing straight into Google Drive from Colab.

The grid can be partitioned by dataset/model/seed to split it across runtimes.
Since each run is its own file, two runtimes writing to the same Drive folder
do not corrupt each other.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import config


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m vqcdepth",
        description="VQC depth study (ERAMIA-RS 2026).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--part", default="all",
                   choices=["train", "rebuild", "plot", "tables", "all", "status"],
                   help="stage to run")
    p.add_argument("--outdir", required=True,
                   help="output root (on Colab, a Drive folder)")
    p.add_argument("--datasets", nargs="+", default=None,
                   choices=list(config.DATASETS), metavar="DS",
                   help=f"subset of datasets {config.DATASETS}")
    p.add_argument("--models", nargs="+", default=None,
                   choices=list(config.MODELS), metavar="M",
                   help=f"subset of models {config.MODELS}")
    p.add_argument("--seeds", nargs="+", type=int, default=None, metavar="S",
                   help=f"subset of seeds (default: {list(config.SEEDS)})")
    p.add_argument("--epochs", type=int, default=config.N_EPOCHS,
                   help="COMPLETE training epochs (1 epoch = full sweep)")
    p.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    p.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    p.add_argument("--diff-method", default=config.DIFF_METHOD,
                   choices=["backprop", "parameter-shift"],
                   help="gradient method. Not a variable of the study: the "
                        "paper grid uses backprop, verified identical to "
                        "parameter-shift to ~1e-16 and ~10x cheaper")
    p.add_argument("--shots", type=int, default=config.SHOTS,
                   help="samples per measurement (default: analytic, noiseless). "
                        "Out of scope for the paper; requires --diff-method "
                        "parameter-shift and multiplies the cost")
    p.add_argument("--no-checkpoint", action="store_true",
                   help="do not write per-epoch checkpoints (resume is per RUN, "
                        "so this does not affect resuming)")
    p.add_argument("--stop-on-error", action="store_true",
                   help="abort on the first failing run (default: keep going)")
    p.add_argument("--dry-run", action="store_true",
                   help="list what would be executed and exit")
    return p


def _print_status(grid, outdir) -> None:
    from . import runner
    todo = runner.pending(grid, outdir)
    done = len(grid) - len(todo)
    print(f"\n  outdir: {Path(outdir).resolve()}")
    print(f"  grid selected: {len(grid)} runs")
    print(f"  already done:  {done}")
    print(f"  missing:       {len(todo)}")
    if todo:
        head = [c.run_id for c in todo[:10]]
        print("  next: " + ", ".join(head) + (" ..." if len(todo) > 10 else ""))


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
        print(f"\n  [warning] configuration OUTSIDE the paper design: "
              f"diff_method={args.diff_method} shots={args.shots}. "
              f"These run_id values carry a suffix, so the results do NOT "
              f"overwrite the analytic grid.")

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
            print(f"\n  {len(paths)} figures in {Path(outdir) / 'figures'}")
        except ImportError as e:
            print(f"\n  [warning] figures unavailable ({e}); install matplotlib")

    return 0


if __name__ == "__main__":
    sys.exit(main())
