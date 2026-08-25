"""Gera o ERAMIA_launcher.ipynb. Editar AQUI, nunca o .ipynb na mao."""
import json
from pathlib import Path

DRIVE = "/content/drive/MyDrive/ERAMIA-VQC"

md = lambda s: {"cell_type": "markdown", "metadata": {}, "source": s.strip().splitlines(True)}
code = lambda s: {"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": s.strip().splitlines(True)}

cells = [
md(f"""
# ERAMIA-RS 2026 - Estudo de profundidade de VQC

Launcher **fino**: toda a logica vive no pacote `vqcdepth`, versionado em
`github.com/EduardoTBuss/vqc-depth` (repositorio **privado**). Este notebook so
monta o Drive, instala, roda o gate de testes, mede o piloto e dispara o grid.

**Runtime: CPU.** Nao peca GPU - sao 4 qubits em `default.qubit`, a GPU nao
seria tocada.

Tudo e' escrito em `{DRIVE}`. Cada run e' **um arquivo proprio**, escrito de
forma atomica: se a sessao morrer, basta reabrir e rodar a mesma celula - ela
continua exatamente de onde parou, perdendo no maximo o run em voo.

**1 epoca = varredura COMPLETA do treino em minibatches de 16.** Isso NAO e' a
mesma definicao do repositorio `qnn-benchmark`, onde epoca era um unico passo
de minibatch. Nao extrapole tempo de la para ca - **meca no piloto (celula 5)**.
"""),

md("## 1. Drive"),
code(f"""
from google.colab import drive
import os

drive.mount('/content/drive')

DRIVE = '{DRIVE}'
for sub in ['runs', 'results', 'predictions', 'checkpoints', 'figures', 'tables', 'logs']:
    os.makedirs(f'{{DRIVE}}/{{sub}}', exist_ok=True)

print('persistindo em:', DRIVE)
print(sorted(os.listdir(DRIVE)))
"""),

md("""
## 2. Repositorio privado via Colab Secrets

O repo e' **privado**, entao o clone precisa de um token.

**Antes de rodar**, crie o segredo no Colab (icone da chave, na barra lateral):
- Nome: `GH_PAT`
- Valor: um **fine-grained PAT** do GitHub restrito **a este repositorio**, com
  permissao **apenas de leitura em Contents** (`Contents: Read-only`).
- Ative o toggle "Notebook access" para este notebook.

O token **nunca** e' digitado numa celula, **nunca** e' impresso e **nunca** e'
commitado. A URL com credencial e' usada so dentro do `git clone` e o remote e'
reescrito logo depois, para o token nao ficar em `.git/config`.
"""),
code("""
from google.colab import userdata
import subprocess, os

GH_USER = 'EduardoTBuss'
REPO    = 'vqc-depth'

try:
    _pat = userdata.get('GH_PAT')
except Exception as e:
    raise SystemExit(
        'Segredo GH_PAT ausente. Crie em Secrets (icone de chave) e libere '
        f'o acesso para este notebook. Detalhe: {e}')

!rm -rf /content/{REPO}
_url = f'https://{GH_USER}:{_pat}@github.com/{GH_USER}/{REPO}.git'
r = subprocess.run(['git', 'clone', '--quiet', _url, f'/content/{REPO}'],
                   capture_output=True, text=True)
# a mensagem de erro do git pode conter o token: nunca imprimir cru
if r.returncode != 0:
    raise SystemExit('git clone falhou (mensagem suprimida por conter o token). '
                     'Verifique o PAT, o escopo Contents:Read e o nome do repo.')
del _pat, _url

%cd /content/{REPO}
!git remote set-url origin https://github.com/{GH_USER}/{REPO}.git
!git log --oneline -1
"""),

md("## 3. Dependencias"),
code("""
!pip install -q -r requirements.txt
!pip install -q -e .
import vqcdepth; print('vqcdepth', vqcdepth.__version__)
"""),

md("""
## 4. Gate: a suite de testes

**Se isto falhar, NAO rode o grid.** Aqui esta a prova de que a vetorizacao do
circuito da o mesmo valor e o mesmo gradiente do laco por amostra, de que os 4
modelos do mesmo (dataset, seed) recebem literalmente os mesmos indices, e de
que nao ha vazamento do teste para o pre-processamento.
"""),
code("""
!python -m pytest tests/ -q
"""),

md("""
## 5. Piloto - **e a celula que mede o tempo real**

1 dataset x 1 modelo x 2 seeds x 10 epocas. Serve para duas coisas:
confirmar que os arquivos aparecem no Drive **e medir segundos por epoca**.

O numero que sair daqui e' o unico que autoriza qualquer estimativa sobre o
grid completo. Rode tambem com `--models vqc4` para pegar o pior caso, porque
o custo cresce com a profundidade.
"""),
code("""
import time
LOG = f'{DRIVE}/logs/piloto_{time.strftime("%Y%m%d_%H%M%S")}.log'

!python -m vqcdepth --part train --outdir "{DRIVE}" \\
    --datasets banknote --models vqc1 --seeds 0 1 --epochs 10 2>&1 | tee "{LOG}"
"""),
code("""
# Extrapolacao a partir do que ACABOU de ser medido - nada e' chutado aqui.
import json, glob, os

def medir(padrao):
    ts = []
    for p in glob.glob(f'{DRIVE}/runs/{padrao}.json'):
        with open(p) as f:
            r = json.load(f)
        ts.append(r['training_time_seconds'] / r['final_epoch'])
    return ts

ts = medir('banknote__vqc1__seed0[01]')
if not ts:
    print('nenhum run do piloto encontrado - confira a celula anterior')
else:
    s_ep = sum(ts) / len(ts)
    print(f'banknote/vqc1: {s_ep:.2f} s por epoca (media de {len(ts)} runs)')
    print(f'  1 run de 100 epocas: {s_ep * 100 / 60:.1f} min')
    print('\\nATENCAO: vqc2 e vqc4 sao mais caros, e blood/mammographic sao')
    print('menores que banknote. Meca vqc4 antes de estimar o grid inteiro.')

print('\\nruns no Drive:', len(glob.glob(f'{DRIVE}/runs/*.json')))
"""),
code("""
# Piloto rodou com --epochs 10: esses runs NAO sao grau-de-paper.
# Apague antes do grid real, senao o resume os considera "prontos".
import glob, os, shutil

for p in glob.glob(f'{DRIVE}/runs/banknote__vqc1__seed0[01]*'):
    os.remove(p); print('removido', os.path.basename(p))
for p in glob.glob(f'{DRIVE}/predictions/banknote__vqc1__seed0[01]*'):
    os.remove(p); print('removido', os.path.basename(p))
for d in glob.glob(f'{DRIVE}/checkpoints/banknote__vqc1__seed0[01]'):
    shutil.rmtree(d); print('removido', os.path.basename(d))

!python -m vqcdepth --part status --outdir "{DRIVE}"
"""),

md("""
## 6. Grid completo - retomavel

Pode reexecutar a vontade: o que ja terminou e' pulado. Se a sessao cair,
reconecte e rode esta celula de novo.
"""),
code("""
import time
LOG = f'{DRIVE}/logs/grid_{time.strftime("%Y%m%d_%H%M%S")}.log'

!python -m vqcdepth --part train --outdir "{DRIVE}" 2>&1 | tee "{LOG}"
"""),
md("""
### Dividir entre dois runtimes (opcional)

Cada run e' um arquivo, entao dois runtimes escrevendo no mesmo Drive nao se
corrompem. So vale a pena se o piloto mostrar que o grid inteiro e' longo.

```python
# runtime A
!python -m vqcdepth --part train --outdir "{DRIVE}" --datasets banknote
# runtime B
!python -m vqcdepth --part train --outdir "{DRIVE}" --datasets blood mammographic
```
"""),

md("""
## 7. CSVs, tabelas e figuras

Funciona **com grid parcial** - nao espere as 120 execucoes para olhar.
"""),
code("""
!python -m vqcdepth --part rebuild --outdir "{DRIVE}"
!python -m vqcdepth --part tables  --outdir "{DRIVE}"
!python -m vqcdepth --part plot    --outdir "{DRIVE}"
"""),
code("""
import pandas as pd
raw = pd.read_csv(f'{DRIVE}/results/raw_results.csv')
print(raw.groupby(['dataset', 'model'], observed=True)['test_f1']
         .agg(['count', 'mean', 'std']).round(4).to_string())
"""),
code("""
from IPython.display import Image, display
import os
for nome in ['f1_vs_depth', 'val_f1_curves', 'f1_by_model_boxplot']:
    p = f'{DRIVE}/figures/{nome}.png'
    if os.path.exists(p):
        print(nome); display(Image(p))
"""),
]

nb = {
    "cells": cells,
    "metadata": {
        "colab": {"provenance": [], "toc_visible": True},
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}

out = Path(__file__).with_name("ERAMIA_launcher.ipynb")
out.write_text(json.dumps(nb, indent=1, ensure_ascii=True), encoding="utf-8")
print(f"escrito: {out}  ({len(cells)} celulas)")
