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

**Two things are yours before *Runtime -> Run all*, and nothing else works without
them:**
1. **Secret `GH_PAT`** (key icon in the sidebar): a GitHub *fine-grained PAT*
   scoped to the `vqc-depth` repository, **`Contents: Read-only`** and nothing
   else. Turn *Notebook access* on. Cell 2 stops here if it is missing.
2. Cell 1 asks for **Google Drive authorisation**.

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
## 2 - Private repository, via Colab Secrets

**Before running**, create the secret in Colab (key icon in the sidebar):
- Name: `GH_PAT`
- Value: a **fine-grained PAT** scoped to this repository only, with
  **`Contents: Read-only`** and nothing else.
- Enable *Notebook access*.

The token is never typed into a cell, never printed and never committed. The
clone error is suppressed on purpose: git echoes the URL, which contains the
token. The remote is rewritten right after cloning so the token is not left in
`.git/config`.
"""),
code("""
import getpass, subprocess

GH_USER, REPO = 'EduardoTBuss', 'vqc-depth'

# The secret is read OUTSIDE the except block on purpose. Raising SystemExit
# from inside a handler chains the exceptions, and IPython's traceback
# formatter crashes on that chain with a TypeError in find_recursion, which
# buries the only message that actually tells you what to do.
_pat, _why = None, ''
try:
    from google.colab import userdata
    _pat = userdata.get('GH_PAT')
except Exception as e:
    _why = f'{type(e).__name__}: {e}'

if not _pat:
    print('=' * 72)
    print('The Colab secret GH_PAT is not available.')
    if _why:
        print(_why)
    print('')
    print('Preferred fix, done once per Google account (about a minute):')
    print('  1. key icon in the left sidebar  ->  + Add new secret')
    print('  2. Name: GH_PAT   (exactly this, case sensitive)')
    print('  3. Value: a GitHub fine-grained PAT scoped to the vqc-depth repo,')
    print('     Repository permissions -> Contents: Read-only. Create one at')
    print('     https://github.com/settings/personal-access-tokens/new')
    print('  4. turn "Notebook access" ON for this notebook')
    print('  5. re-run this cell')
    print('')
    print('Or paste the token once, right now, to keep going without the')
    print('secret. It is hidden while typing and never stored or printed.')
    print('Press Enter on an empty prompt to stop instead.')
    print('=' * 72)
    try:
        _pat = getpass.getpass('GH_PAT (hidden): ').strip()
    except Exception:
        _pat = None

if not _pat:
    print('')
    print('No token available, so nothing was cloned. Set the secret as')
    print('described above and re-run this cell.')
    raise SystemExit

import os, shutil
REPO_DIR = f'/content/{REPO}'
shutil.rmtree(REPO_DIR, ignore_errors=True)
_url = f'https://{GH_USER}:{_pat}@github.com/{GH_USER}/{REPO}.git'
r = subprocess.run(['git', 'clone', '--quiet', _url, REPO_DIR],
                   capture_output=True, text=True)
del _pat, _url
if r.returncode != 0:
    print('git clone failed. The real message is suppressed because git echoes')
    print('the URL, and the URL carries the token. Check, in this order:')
    print('  - the token has not expired')
    print('  - Repository access covers EduardoTBuss/vqc-depth specifically')
    print('  - Repository permissions -> Contents is at least Read-only')
    print('  - the repository name is still vqc-depth')
    raise SystemExit

os.chdir(REPO_DIR)
print('working directory:', os.getcwd())
run(['git', 'remote', 'set-url', 'origin',
     f'https://github.com/{GH_USER}/{REPO}.git'])
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

# If requirements.txt silently failed to install, this is where it surfaces,
# rather than two cells later in the middle of a training run.
for _m in ('pennylane', 'torch', 'sklearn', 'pandas', 'pyarrow', 'scipy',
           'matplotlib', 'tabulate', 'ucimlrepo'):
    importlib.import_module(_m)
print('all runtime dependencies import cleanly')
"""),

md("""
## 4 - Gate: the test suite

**If this fails, do not run the grid.** It proves that the vectorised circuit
returns the same value and the same gradient as the per-sample loop, that all
four models of a `(dataset, seed)` cell receive literally the same indices, that
the test set never reached the preprocessing, and that no angle of the ansatz is
inert.

`VQCDEPTH_ALLOW_DOWNLOAD=1` matters: without it the 18 tests that fetch the UCI
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
measures seconds per epoch here, rather than trusting a number measured
elsewhere.
"""),
code("""
import time
LOG = f'{DRIVE}/logs/pilot_{time.strftime("%Y%m%d_%H%M%S")}.log'

run([sys.executable, '-m', 'vqcdepth', '--part', 'train', '--outdir', DRIVE,
     '--datasets', 'banknote', '--models', 'vqc4', '--seeds', '0', '1',
     '--epochs', '5'], log=LOG)
"""),
code("""
# Extrapolation from what was just measured; nothing here is guessed.
import json, glob

ts = []
for p in glob.glob(f'{DRIVE}/runs/banknote__vqc4__seed0[01].json'):
    r = json.load(open(p))
    ts.append(r['training_time_seconds'] / r['final_epoch'])

if not ts:
    print('no pilot run found -- check the previous cell')
else:
    s = sum(ts) / len(ts)
    # vqc4 is ~55% of the grid cost; scale by training-set size across datasets
    total_h = (s * 100 * 10 * (961 + 523 + 672) / 961) / 0.55 / 3600
    print(f'vqc4/banknote: {s:.2f} s/epoch  ->  one 100-epoch run = {s*100/60:.1f} min')
    print(f'ESTIMATE for the full grid (120 runs): {total_h:.1f} h')
    print('\\nIf that exceeds ~4 h, split the grid using cell 6b.')
"""),
code("""
# The pilot ran with --epochs 5, so these runs are NOT paper-grade. Delete them
# before the real grid, otherwise the resume logic treats them as finished.
import glob, os, shutil

for pat in ['runs/banknote__vqc4__seed0[01]*', 'predictions/banknote__vqc4__seed0[01]*']:
    for p in glob.glob(f'{DRIVE}/{pat}'):
        os.remove(p); print('removed', os.path.basename(p))
for d in glob.glob(f'{DRIVE}/checkpoints/banknote__vqc4__seed0[01]'):
    shutil.rmtree(d); print('removed', os.path.basename(d))

run([sys.executable, '-m', 'vqcdepth', '--part', 'status', '--outdir', DRIVE])
"""),

md("""
## 6 - Full grid: 120 runs, resumable

If the session drops, reconnect and re-run this cell: it continues where it
stopped.
"""),
code("""
import time
LOG = f'{DRIVE}/logs/grid_{time.strftime("%Y%m%d_%H%M%S")}.log'

run([sys.executable, '-m', 'vqcdepth', '--part', 'train', '--outdir', DRIVE],
    log=LOG)
"""),
md("""
### 6b - Optional: split the grid

Each run is its own file, so there is no race between runtimes.
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

**Nothing is lost if Colab dies mid-run**: re-run cell 6.

Bring these files back and the next session produces the figures, the tables and the text.
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
