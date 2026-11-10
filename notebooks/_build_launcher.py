"""Generates ERAMIA_launcher.ipynb. Edit here, never the .ipynb by hand."""
import json
from pathlib import Path

DRIVE = "/content/drive/MyDrive/ERAMIA-VQC"

md = lambda s: {"cell_type": "markdown", "metadata": {}, "source": s.strip().splitlines(True)}
code = lambda s: {"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": s.strip().splitlines(True)}

cells = [
md(f"""
# vqc-depth — circuit depth grid on Colab

A **thin launcher**: all logic lives in the `vqcdepth` package, versioned at
`github.com/EduardoTBuss/vqc-depth`. This notebook only mounts Drive, installs,
runs the test gate, measures a pilot and fires the grid.

**Use a CPU runtime.** Four qubits on `default.qubit` will not touch a GPU.

Authorize **Google Drive** in cell 1 to save results and resume later.

**Expected cost (measured locally): ~82 min there -> ~2 to 2.7 h here.** Cell 5
re-measures it on the machine you actually got, before the grid is fired.

Everything is written to `{DRIVE}`. Each run is **its own file**, written
atomically: if the session dies, reopen and re-run the same cell — it continues
where it stopped, losing at most the run in flight.

One epoch is a **complete sweep of the training set** in minibatches of 16,
roughly 60 optimiser steps per epoch on Banknote. Cell 5 measures the real cost
on whichever machine Colab hands you.
"""),

md("## 1 - Drive and the run() helper"),
code(f"""
from google.colab import drive
import os, subprocess, sys

drive.mount('/content/drive')

DRIVE = '{DRIVE}'
for sub in ['runs', 'results', 'predictions', 'checkpoints', 'figures', 'tables', 'logs']:
    os.makedirs(f'{{DRIVE}}/{{sub}}', exist_ok=True)

print('persisting to:', DRIVE)
print('already on Drive:', len(os.listdir(f'{{DRIVE}}/runs')), 'files in runs/')


def run(args, log=None, env=None, check=True):
    \"\"\"Run a command, stream its output live, optionally copy it to a log file,
    and STOP the notebook if it fails.

    This exists because `!command` does not stop anything: IPython runs it, and
    execution moves to the next cell whatever the exit code was. With a plain
    `!`, a failing test gate or a crashed grid looks exactly like a successful
    one, and 'Run all' happily carries on. Every command that matters goes
    through here instead. It also replaces `... | tee file`, which depends on a
    shell and fails silently when the target directory is missing.
    \"\"\"
    print('$', ' '.join(str(a) for a in args), flush=True)
    _env = {{**os.environ, **(env or {{}})}}
    fh = open(log, 'w', encoding='utf-8') if log else None
    proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1, env=_env)
    try:
        for line in proc.stdout:
            print(line, end='', flush=True)
            if fh:
                fh.write(line)
    finally:
        proc.wait()
        if fh:
            fh.close()
    if check and proc.returncode != 0:
        print(f'\\n>>> command FAILED with exit code {{proc.returncode}}.')
        print('>>> Stopping here on purpose: do not run later cells on a broken run.')
        raise SystemExit(proc.returncode)
    return proc.returncode
"""),

md("""
## 2 - Clone the public repository

The notebook prints the Git commit used for the session.
"""),
code("""
import tempfile

REPO_URL = 'https://github.com/EduardoTBuss/vqc-depth.git'
REPO_DIR = tempfile.mkdtemp(prefix='vqc-depth-', dir='/content')
run(['git', 'clone', '--quiet', REPO_URL, REPO_DIR])
os.chdir(REPO_DIR)
run(['git', 'log', '--oneline', '-1'])
"""),

md("## 3 - Dependencies"),
code("""
run([sys.executable, '-m', 'pip', 'install', '-q', '-r', 'requirements.txt'])
run([sys.executable, '-m', 'pip', 'install', '-q', '-e', '.'])

# An editable install drops a .pth file into site-packages, and .pth files are
# only read by the `site` module when the interpreter STARTS. This kernel is
# already running, so without the two lines below the package stays invisible
# and `import vqcdepth` raises ModuleNotFoundError even though pip reported
# success. The package uses a src layout, so the working directory does not
# help either. site.main() re-processes the .pth now; prepending src/ is a
# fallback that does not depend on how pip chose to install.
import importlib, site, sys
from pathlib import Path

site.main()
_src = str(Path.cwd() / 'src')
if _src not in sys.path:
    sys.path.insert(0, _src)
importlib.invalidate_caches()

import vqcdepth
print('vqcdepth', vqcdepth.__version__)
print('loaded from', Path(vqcdepth.__file__).parent)

# Did pip change a library that this kernel had ALREADY imported? Colab imports
# numpy before the first cell runs, so a pip upgrade or downgrade leaves the new
# Python files loading against the C extension still in memory. numpy then dies
# with things like
#     AttributeError: 'numpy.ufunc' object has no attribute '__module__'
# from inside numpy itself. No user code can recover from that: the process has
# to be replaced. Catch it here, with an instruction, instead of three cells
# later with a traceback that points at the wrong place.
import importlib.metadata as _meta

_stale = []
for _dist, _mod in [('numpy', 'numpy'), ('pandas', 'pandas'),
                    ('scipy', 'scipy'), ('scikit-learn', 'sklearn'),
                    ('matplotlib', 'matplotlib')]:
    _live = sys.modules.get(_mod)
    if _live is None:
        continue                      # not imported yet: the next import is clean
    try:
        _disk = _meta.version(_dist)
    except _meta.PackageNotFoundError:
        continue
    if getattr(_live, '__version__', _disk) != _disk:
        _stale.append((_dist, _live.__version__, _disk))

if _stale:
    print()
    print('=' * 72)
    print('RESTART THE RUNTIME. pip replaced a library this session had already')
    print('imported, so the version in memory and the version on disk disagree:')
    for _d, _mem, _dsk in _stale:
        print(f'    {_d}: {_mem} in memory, {_dsk} on disk')
    print()
    print('  Runtime  ->  Restart session,  then run the notebook again.')
    print()
    print('Nothing is lost by restarting: Drive is remounted in cell 1, the')
    print('repository is re-cloned in cell 2, and the grid resumes from Drive,')
    print('so every run already finished is skipped.')
    print('=' * 72)
    raise SystemExit

# If requirements.txt silently failed to install, this is where it surfaces,
# rather than two cells later in the middle of a training run.
for _m in ('pennylane', 'torch', 'sklearn', 'pandas', 'pyarrow', 'scipy',
           'matplotlib', 'tabulate', 'ucimlrepo'):
    importlib.import_module(_m)
print('all runtime dependencies import cleanly')
"""),

md("""
## 4 - Gate: the test suite

**If this fails, do not run the grid.** It checks that the vectorised circuit
returns the same value and the same gradient as the per-sample loop, that all
four models of a `(dataset, seed)` cell receive literally the same indices, that
the test set never reached the preprocessing, and that no angle of the ansatz is
inert.

`VQCDEPTH_ALLOW_DOWNLOAD=1` matters: without it the tests that fetch the UCI
datasets are skipped on a fresh clone, and the gate is weaker than it looks.
"""),
code("""
# VQCDEPTH_ALLOW_DOWNLOAD goes through the environment rather than as a shell
# prefix: the prefix is not portable and, worse, fails silently.
run([sys.executable, '-m', 'pytest', 'tests/', '-q'],
    env={'VQCDEPTH_ALLOW_DOWNLOAD': '1'})
print('Gate passed. The grid may run.')
"""),

md("""
## 5 - Pilot: the cell that measures time

Two short runs of the most expensive model. Confirms that files reach Drive and
measures seconds per epoch in this session. Each pilot uses its own folder
on Drive, separate from the main grid.
"""),
code("""
import time
from pathlib import Path

pilots = Path(DRIVE) / 'pilots'
pilots.mkdir(exist_ok=True)
pilot_dir = Path(tempfile.mkdtemp(prefix='pilot-', dir=pilots))
LOG = f'{DRIVE}/logs/pilot_{time.strftime("%Y%m%d_%H%M%S")}.log'

run([sys.executable, '-m', 'vqcdepth', '--part', 'train', '--outdir', str(pilot_dir),
     '--datasets', 'banknote', '--models', 'vqc4', '--seeds', '0', '1',
     '--epochs', '5', '--no-checkpoint', '--stop-on-error'], log=LOG)
"""),
code("""
import json

ts = []
for path in (pilot_dir / 'runs').glob('banknote__vqc4__seed0[01].json'):
    record = json.loads(path.read_text(encoding='utf-8'))
    ts.append(record['training_time_seconds'] / record['final_epoch'])

if len(ts) != 2:
    raise RuntimeError('Expected two pilot results. Run the previous cell again.')

s = sum(ts) / len(ts)
# vqc4 represented about 55% of grid cost in the original pilot.
total_h = (s * 100 * 10 * (961 + 523 + 672) / 961) / 0.55 / 3600
print(f'vqc4/banknote: {s:.2f} s/epoch, about {s*100/60:.1f} min per 100-epoch run')
print(f'Estimated full grid time (120 runs): {total_h:.1f} h')
"""),

md("""
## 6 - Full grid: 120 runs, resumable

If the session drops, reconnect and re-run this cell: it continues where it
stopped.
"""),
code("""
import time
LOG = f'{DRIVE}/logs/grid_{time.strftime("%Y%m%d_%H%M%S")}.log'

run([sys.executable, '-m', 'vqcdepth', '--part', 'train', '--outdir', DRIVE,
     '--stop-on-error'], log=LOG)
"""),
md("""
### 6b - Optional: split the grid

Use disjoint subsets across runtimes and generate the shared tables and figures once training finishes.
"""),
code("""
# for _ds in ['banknote', 'blood', 'mammographic']:
#     run([sys.executable, '-m', 'vqcdepth', '--part', 'train',
#          '--outdir', DRIVE, '--datasets', _ds])
"""),

md("""
## 7 - CSVs, tables and figures

All of these work on a **partial grid**.
"""),
code("""
for _part in ['rebuild', 'tables', 'plot']:
    run([sys.executable, '-m', 'vqcdepth', '--part', _part, '--outdir', DRIVE])
"""),
code("""
import pandas as pd
raw = pd.read_csv(f'{DRIVE}/results/raw_results.csv')
print(f'completed runs: {len(raw)}/120\\n')
print(raw.groupby(['dataset', 'model'], observed=True)['test_f1']
         .agg(['count', 'mean', 'std']).round(4).to_string())
"""),
code("""
import pandas as pd
print(pd.read_csv(f'{DRIVE}/tables/main_f1.csv').to_string(index=False))
print('\\nSeed-paired Wilcoxon across depths (p_holm is the one for the paper):')
print(pd.read_csv(f'{DRIVE}/tables/wilcoxon_depth.csv').round(4).to_string(index=False))
"""),
code("""
from IPython.display import Image, display
import os
for name in ['f1_vs_depth', 'val_f1_curves', 'val_loss_curves', 'f1_by_model_boxplot']:
    p = f'{DRIVE}/figures/{name}.png'
    if os.path.exists(p):
        print(name); display(Image(p))
"""),
md("""
## 8 - Done

Everything lives under `/content/drive/MyDrive/ERAMIA-VQC`:

- `results/raw_results.csv` - one row per run, every metric
- `results/training_history.csv` - **every** epoch
- `predictions/*.parquet` - per-sample predictions (ROC/PR can be rebuilt without retraining)
- `tables/*.csv|.md|.tex` - ready for LaTeX
- `figures/*.pdf` - vector, for the paper

After a disconnect, run the setup cells again and resume from cell 6.
Completed runs are kept and an interrupted run restarts from its first epoch.
"""),
]

nb = {"cells": cells,
      "metadata": {"colab": {"provenance": [], "toc_visible": True},
                   "kernelspec": {"name": "python3", "display_name": "Python 3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}

out = Path(__file__).with_name("ERAMIA_launcher.ipynb")
out.write_text(json.dumps(nb, indent=1, ensure_ascii=True), encoding="utf-8")
print(f"written: {out}  ({len(cells)} cells)")
